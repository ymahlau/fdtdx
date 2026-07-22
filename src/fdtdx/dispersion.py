"""Dispersive material models for FDTDX.

Provides a generic Auxiliary Differential Equation (ADE) dispersion
abstraction for linear materials. The first concrete pole types are
Lorentz and Drude, combined freely as a "Drude-Lorentz" model.

Physics
-------
Each pole contributes a 2nd-order ODE for the normalized polarization
``p = P / eps_0`` (same units as E):

.. math::
    \\ddot{p}_p + \\gamma_p \\dot{p}_p + \\omega_{0,p}^2 p_p = K_p E

Lorentz pole (resonance :math:`\\omega_0`, damping :math:`\\gamma`,
strength :math:`\\Delta\\varepsilon`):

.. math::
    \\chi_p(\\omega) = \\frac{\\Delta\\varepsilon \\cdot \\omega_0^2}{\\omega_0^2 - \\omega^2 - i\\gamma\\omega}

Drude pole (plasma frequency :math:`\\omega_p`, damping :math:`\\gamma`;
special case of Lorentz with :math:`\\omega_0 = 0`):

.. math::
    \\chi_p(\\omega) = -\\frac{\\omega_p^2}{\\omega^2 + i\\gamma\\omega}

The unified pole parameterization stores ``(omega_0, gamma, coupling_sq)``
where ``coupling_sq`` is the effective squared coupling frequency
:math:`K = \\Delta\\varepsilon \\omega_0^2` (Lorentz) or :math:`\\omega_p^2`
(Drude), both in (rad/s)^2.

Discrete update
---------------
Central differences at integer time ``n``:

.. math::
    p_p^{n+1} = c_1 p_p^{n} + c_2 p_p^{n-1} + c_3 E^{n}

with coefficients derived from the unified pole parameters and time
step ``dt``:

.. math::
    c_1 = \\frac{2 - \\omega_0^2 \\Delta t^2}{1 + \\gamma \\Delta t / 2}, \\quad
    c_2 = -\\frac{1 - \\gamma \\Delta t / 2}{1 + \\gamma \\Delta t / 2}, \\quad
    c_3 = \\frac{K \\Delta t^2}{1 + \\gamma \\Delta t / 2}

Two *independent* conditions constrain the coefficients; both are enforced in
:func:`compute_pole_coefficients_per_axis` (only on axes where the pole
actually couples, since a zero-coupling axis keeps its polarization
identically zero):

* **Forward unit-circle (Jury) stability.** The roots of
  :math:`z^2 - c_1 z - c_2 = 0` lie inside the unit circle iff
  :math:`|c_2| < 1` *and* :math:`|c_1| < 1 - c_2`. The first holds for every
  :math:`\\gamma \\Delta t > 0` (:math:`c_2 = 0` at :math:`\\gamma \\Delta t = 2`
  and :math:`|c_2| \\to 1` only as :math:`\\gamma \\Delta t \\to 0` or
  :math:`\\infty`), so it is not the binding constraint. The second is
  algebraically equivalent to :math:`\\omega_0 \\Delta t < 2` (independent of
  :math:`\\gamma`), which is therefore the forward-stability bound.
* **Reverse-update conditioning.** The backward (reverse-time) recurrence is
  formed by dividing through by :math:`c_2`, so :math:`c_2` must stay bounded
  away from zero. This is a *separate* requirement, :math:`\\gamma \\Delta t < 2`
  (:math:`c_2 = 0` exactly at :math:`\\gamma \\Delta t = 2`); in the physical
  regime :math:`\\gamma \\Delta t \\ll 1` so :math:`c_2 \\approx -1` and the
  inversion is well conditioned. It is a conditioning bound for reversibility,
  not a forward unit-circle criterion.

For CCPR poles the polarization couples to :math:`E^{n+1}` through
:math:`c_4`, so the E-field update divides by a per-cell implicit factor
:math:`1 + \\varepsilon_\\infty^{-1} \\sum_p c_{4,p}\\ (+\\ c\\,\\sigma\\,\\eta_0\\,\\varepsilon_\\infty^{-1} / 2)`.
This must stay positive in every cell; as it approaches :math:`0^+` the
transient gain (:math:`\\approx 1/\\text{divisor}`) explodes and accuracy
collapses. Since :math:`c_4 \\propto \\Delta t \\propto` ``courant_factor``,
the divisor can be kept safe by lowering ``courant_factor``. This per-cell
condition is checked at initialization by
:func:`fdtdx.materials.validate_dispersive_divisor_stability` (Lorentz and
Drude poles have :math:`c_4 = 0`, so their divisor is always
:math:`\\geq 1`).

Anisotropic (per-axis) dispersion
---------------------------------
Every pole parameter accepts either a scalar (isotropic, applied to all
three axes) or a 3-tuple ``(x, y, z)`` giving a different value per grid
axis. This yields a diagonally anisotropic susceptibility tensor
:math:`\\chi(\\omega) = \\mathrm{diag}(\\chi_x, \\chi_y, \\chi_z)` — enough to
model uniaxial/biaxial crystals and hyperbolic media (e.g. hBN) whose
optical axes align with the grid. A pole that only acts on one axis is
expressed by zeroing its strength on the others, e.g.
``LorentzPole(resonance_frequency=w0, damping=g, delta_epsilon=(2.25, 0.0, 0.0))``:
with zero coupling the polarization on that axis stays identically zero.

Oriented (off-diagonal) dispersion
----------------------------------
A pole may additionally carry an ``orientation`` unit vector ``u``: it then
acts as a single 1D oscillator along ``u`` and contributes the coupling
tensor :math:`K\\, u u^T` — off-diagonal for non-axis-aligned directions.
This models rotated/tilted crystals and monoclinic media (shear phonon
polaritons), where each IR-active phonon oscillates along its own,
generally non-orthogonal, direction. :meth:`DispersionModel.rotated`
converts a per-axis model into oriented poles for the common case of a
crystal rotated relative to the grid. Oriented dispersion runs through the
fully anisotropic update path and currently supports only the
``checkpointed`` gradient method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.constants import eps0
from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field


def _broadcast_axis_param(value: float | complex | tuple) -> tuple:
    """Normalize a pole parameter to a per-axis 3-tuple ``(x, y, z)``.

    Scalars are broadcast to all three axes; 3-tuples pass through unchanged.
    """
    if isinstance(value, tuple):
        if len(value) != 3:
            raise ValueError(
                f"Per-axis pole parameters must be a scalar or a 3-tuple (x, y, z), got a tuple of length {len(value)}."
            )
        return value
    return (value, value, value)


def _is_uniform(axes: tuple) -> bool:
    return bool(axes[0] == axes[1] == axes[2])


def _as_rotation_matrix(rotation: tuple) -> np.ndarray:
    """Build and validate a 3x3 rotation matrix from a nested tuple or Euler angles."""
    if isinstance(rotation, tuple) and len(rotation) == 3 and not any(isinstance(v, tuple) for v in rotation):
        alpha, beta, gamma = (float(v) for v in rotation)
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
        ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
        rz = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]])
        r_mat = rz @ ry @ rx
    else:
        r_mat = np.asarray(rotation, dtype=np.float64)
        if r_mat.shape != (3, 3):
            raise ValueError(
                f"rotation must be a 3x3 nested tuple or a 3-tuple of Euler angles, got shape {r_mat.shape}."
            )
    if not np.allclose(r_mat @ r_mat.T, np.eye(3), atol=1e-9) or not np.isclose(np.linalg.det(r_mat), 1.0, atol=1e-9):
        raise ValueError("rotation must be a proper rotation matrix (orthogonal with determinant +1).")
    return r_mat


def _signed_permutation(r_mat: np.ndarray, tol: float = 1e-12) -> tuple[int, int, int] | None:
    """Detect a signed axis permutation: returns ``perm`` with grid axis ``perm[a]``
    receiving crystal axis ``a``, or ``None`` if the rotation is not a permutation."""
    perm = []
    for a in range(3):
        col = r_mat[:, a]
        nonzero = np.flatnonzero(np.abs(col) > tol)
        if len(nonzero) != 1 or not np.isclose(abs(col[nonzero[0]]), 1.0, atol=tol):
            return None
        perm.append(int(nonzero[0]))
    return (perm[0], perm[1], perm[2])


def _permute_pole_axes(p: "Pole", perm: tuple[int, int, int]) -> "Pole":
    """Remap a per-axis pole's parameters under a signed axis permutation (sign is
    irrelevant: the coupling enters as ``u u^T``)."""

    def _remap(value):
        axes = _broadcast_axis_param(value)
        out = [axes[0]] * 3
        for a in range(3):
            out[perm[a]] = axes[a]
        return (out[0], out[1], out[2])

    if isinstance(p, LorentzPole):
        return LorentzPole(
            resonance_frequency=_remap(p.resonance_frequency),
            damping=_remap(p.damping),
            delta_epsilon=_remap(p.delta_epsilon),
        )
    if isinstance(p, DrudePole):
        return DrudePole(plasma_frequency=_remap(p.plasma_frequency), damping=_remap(p.damping))
    if isinstance(p, CCPRPole):
        return CCPRPole(pole=_remap(p.pole), residue=_remap(p.residue))
    raise TypeError(
        f"Cannot rotate pole of type {type(p).__name__}; construct oriented poles directly for custom pole types."
    )


def _oriented_pole_for_axis(p: "Pole", axis: int, direction: tuple[float, float, float]) -> "Pole":
    """Extract the 1D oscillator of a per-axis pole along ``axis`` as an oriented pole."""
    if isinstance(p, LorentzPole):
        w = _broadcast_axis_param(p.resonance_frequency)
        g = _broadcast_axis_param(p.damping)
        de = _broadcast_axis_param(p.delta_epsilon)
        return LorentzPole(
            resonance_frequency=float(w[axis]),
            damping=float(g[axis]),
            delta_epsilon=float(de[axis]),
            orientation=direction,
        )
    if isinstance(p, DrudePole):
        wp = _broadcast_axis_param(p.plasma_frequency)
        g = _broadcast_axis_param(p.damping)
        return DrudePole(plasma_frequency=float(wp[axis]), damping=float(g[axis]), orientation=direction)
    if isinstance(p, CCPRPole):
        q = _broadcast_axis_param(p.pole)
        r = _broadcast_axis_param(p.residue)
        # Raises at construction if the residue has a real part (dE/dt coupling).
        return CCPRPole(pole=complex(q[axis]), residue=complex(r[axis]), orientation=direction)
    raise TypeError(
        f"Cannot rotate pole of type {type(p).__name__}; construct oriented poles directly for custom pole types."
    )


@autoinit
class Pole(TreeClass, ABC):
    """Abstract base class for a single 2nd-order ADE pole.

    Concrete subclasses store physically-meaningful parameters
    (e.g. ``delta_epsilon`` for Lorentz, ``omega_p`` for Drude) and
    expose the unified ``(omega_0, gamma, coupling_sq)`` triplet the
    FDTD loop needs via per-axis properties. New pole types can
    subclass :class:`Pole` as long as they fit the 2nd-order
    ODE form.

    Every parameter may differ per grid axis (diagonally anisotropic
    dispersion); the canonical accessors are the ``*_axes`` properties
    returning ``(x, y, z)`` tuples. The scalar accessors (``omega_0`` etc.)
    are a convenience for isotropic poles and raise for per-axis ones.
    Alternatively a pole may carry an :attr:`orientation` unit vector,
    turning it into a single 1D oscillator along that direction
    (off-diagonal coupling tensor :math:`K\\, u u^T`).
    """

    #: Optional oscillator direction ``u`` (normalized on construction).
    #: ``None`` (default) applies the pole isotropically or per-axis. When
    #: set, the pole is a single 1D oscillator along ``u`` with coupling
    #: tensor ``K * u u^T``; all other pole parameters must be scalars.
    orientation: tuple[float, float, float] | None = frozen_field(default=None)

    def _validate_orientation(self):
        """Normalize and validate :attr:`orientation`. Concrete pole classes call
        this from ``__post_init__`` (which ``autoinit`` only invokes when defined
        directly on the class, not inherited)."""
        if self.orientation is None:
            return
        vec = self.orientation
        if not isinstance(vec, tuple) or len(vec) != 3:
            raise ValueError(f"Pole orientation must be a 3-tuple (x, y, z), got {vec!r}.")
        arr = np.asarray(vec, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Pole orientation components must be finite, got {vec!r}.")
        scale = float(np.max(np.abs(arr)))
        if scale == 0.0:
            raise ValueError("Pole orientation must be a non-zero vector.")
        # scale first so the squared terms cannot overflow for large components
        scaled = arr / scale
        norm = float(np.linalg.norm(scaled))
        object.__setattr__(
            self, "orientation", (float(scaled[0]) / norm, float(scaled[1]) / norm, float(scaled[2]) / norm)
        )
        for name in ("omega_0", "gamma", "coupling_sq"):
            if not _is_uniform(getattr(self, f"{name}_axes")):
                raise ValueError(
                    f"Oriented poles are single 1D oscillators and require scalar parameters, "
                    f"but '{name}' differs per axis. Use one oriented pole per direction instead."
                )
        if any(b != 0.0 for b in self.coupling_edot_axes):
            raise NotImplementedError(
                "Oriented CCPR poles with a dE/dt coupling (non-zero Re(residue)) are not supported."
            )

    def _uniform_or_raise(self, axes: tuple, name: str) -> float:
        if not _is_uniform(axes):
            raise ValueError(
                f"{type(self).__name__} has per-axis parameters; use the per-axis "
                f"accessor '{name}_axes' instead of the scalar '{name}'."
            )
        return axes[0]

    @property
    @abstractmethod
    def omega_0_axes(self) -> tuple[float, float, float]:
        """Per-axis resonance angular frequency (rad/s). Zero for pure Drude poles."""
        raise NotImplementedError

    @property
    @abstractmethod
    def gamma_axes(self) -> tuple[float, float, float]:
        """Per-axis damping rate (rad/s)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def coupling_sq_axes(self) -> tuple[float, float, float]:
        """Per-axis effective squared coupling frequency ``K`` (rad^2/s^2).

        ``delta_epsilon * omega_0**2`` for a Lorentz pole and
        ``omega_p**2`` for a Drude pole.

        This is the coefficient ``a`` of the ``E`` driving term in the unified
        2nd-order ODE ``p'' + gamma p' + omega_0**2 p = a E + b E'``.
        """
        raise NotImplementedError

    @property
    def coupling_edot_axes(self) -> tuple[float, float, float]:
        """Per-axis coefficient ``b`` of the ``dE/dt`` driving term (rad/s).

        Zero for Lorentz and Drude poles (their susceptibility numerator has no
        ``omega`` term). A non-zero value is what distinguishes a general
        complex-conjugate pole-residue (CCPR) pole — it corresponds to a
        non-zero real part of the residue and adds the ``b E'`` term to the ADE.
        Defaults to all-zero so existing pole types need not override it.
        """
        return (0.0, 0.0, 0.0)

    @property
    def is_oriented(self) -> bool:
        """Whether the pole is a 1D oscillator along an :attr:`orientation` vector."""
        return self.orientation is not None

    @property
    def is_isotropic(self) -> bool:
        """Whether the pole acts identically on the three axes (and is not oriented)."""
        return (
            self.orientation is None
            and _is_uniform(self.omega_0_axes)
            and _is_uniform(self.gamma_axes)
            and _is_uniform(self.coupling_sq_axes)
            and _is_uniform(self.coupling_edot_axes)
        )

    @property
    def omega_0(self) -> float:
        """Resonance angular frequency (rad/s). Zero for pure Drude poles.

        Raises ``ValueError`` for per-axis poles; use :attr:`omega_0_axes`.
        """
        return self._uniform_or_raise(self.omega_0_axes, "omega_0")

    @property
    def gamma(self) -> float:
        """Damping rate (rad/s).

        Raises ``ValueError`` for per-axis poles; use :attr:`gamma_axes`.
        """
        return self._uniform_or_raise(self.gamma_axes, "gamma")

    @property
    def coupling_sq(self) -> float:
        """Effective squared coupling frequency ``K`` (rad^2/s^2).

        Raises ``ValueError`` for per-axis poles; use :attr:`coupling_sq_axes`.
        """
        return self._uniform_or_raise(self.coupling_sq_axes, "coupling_sq")

    @property
    def coupling_edot(self) -> float:
        """Coefficient ``b`` of the ``dE/dt`` driving term (rad/s).

        Raises ``ValueError`` for per-axis poles; use :attr:`coupling_edot_axes`.
        """
        return self._uniform_or_raise(self.coupling_edot_axes, "coupling_edot")


@autoinit
class LorentzPole(Pole):
    """Lorentz pole parameterised by its physical constants.

    The contribution to the susceptibility is

    .. math::
        \\chi(\\omega) = \\frac{\\Delta\\varepsilon \\cdot \\omega_0^2}{\\omega_0^2 - \\omega^2 - i\\gamma\\omega}.

    Each parameter is either a scalar (isotropic) or a per-axis 3-tuple
    ``(x, y, z)`` for diagonally anisotropic dispersion. An axis without a
    resonance is expressed by a zero ``delta_epsilon`` entry on that axis.
    """

    #: Resonance angular frequency (rad/s). Must be > 0.
    #: Scalar or per-axis 3-tuple.
    resonance_frequency: float | tuple[float, float, float] = frozen_field()

    #: Damping rate (rad/s). Must be >= 0. Scalar or per-axis 3-tuple.
    damping: float | tuple[float, float, float] = frozen_field()

    #: Oscillator strength (dimensionless); the zero-frequency
    #: contribution to the susceptibility. Scalar or per-axis 3-tuple.
    delta_epsilon: float | tuple[float, float, float] = frozen_field()

    def __post_init__(self):
        self._validate_orientation()

    @property
    def omega_0_axes(self) -> tuple[float, float, float]:
        w = _broadcast_axis_param(self.resonance_frequency)
        return (float(w[0]), float(w[1]), float(w[2]))

    @property
    def gamma_axes(self) -> tuple[float, float, float]:
        g = _broadcast_axis_param(self.damping)
        return (float(g[0]), float(g[1]), float(g[2]))

    @property
    def coupling_sq_axes(self) -> tuple[float, float, float]:
        w = self.omega_0_axes
        de = _broadcast_axis_param(self.delta_epsilon)
        return (float(de[0]) * w[0] ** 2, float(de[1]) * w[1] ** 2, float(de[2]) * w[2] ** 2)


@autoinit
class DrudePole(Pole):
    """Drude pole parameterised by its physical constants.

    The contribution to the susceptibility is

    .. math::
        \\chi(\\omega) = -\\frac{\\omega_p^2}{\\omega^2 + i\\gamma\\omega},

    equivalent to a Lorentz pole with ``omega_0 = 0``.

    Each parameter is either a scalar (isotropic) or a per-axis 3-tuple
    ``(x, y, z)`` for diagonally anisotropic dispersion — e.g.
    ``plasma_frequency=(wp, 0.0, 0.0)`` gives a metallic (hyperbolic)
    response only along x.
    """

    #: Plasma angular frequency (rad/s). Must be > 0.
    #: Scalar or per-axis 3-tuple.
    plasma_frequency: float | tuple[float, float, float] = frozen_field()

    #: Damping rate (rad/s). Must be >= 0. Scalar or per-axis 3-tuple.
    damping: float | tuple[float, float, float] = frozen_field()

    def __post_init__(self):
        self._validate_orientation()

    @property
    def omega_0_axes(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    @property
    def gamma_axes(self) -> tuple[float, float, float]:
        g = _broadcast_axis_param(self.damping)
        return (float(g[0]), float(g[1]), float(g[2]))

    @property
    def coupling_sq_axes(self) -> tuple[float, float, float]:
        wp = _broadcast_axis_param(self.plasma_frequency)
        return (float(wp[0]) ** 2, float(wp[1]) ** 2, float(wp[2]) ** 2)


@autoinit
class CCPRPole(Pole):
    r"""General complex-conjugate pole-residue (CCPR) pole.

    A single conjugate pair contributes to the susceptibility (in the
    ``exp(-i omega t)`` convention, Laplace variable ``s = -i omega``):

    .. math::
        \chi_p(\omega) = \frac{r}{-i\omega - q} + \frac{r^*}{-i\omega - q^*}

    with **complex** pole ``q`` and **complex** residue ``r``. Summing the pair
    with its conjugate guarantees a real time-domain response. Combined over a
    common denominator this equals the unified 2nd-order form

    .. math::
        \chi_p(\omega) = \frac{a - i\omega b}{\omega_0^2 - \omega^2 - i\gamma\omega}

    with

    .. math::
        \omega_0^2 = |q|^2, \quad \gamma = -2\,\mathrm{Re}(q), \quad
        a = -2\,\mathrm{Re}(r q^*), \quad b = 2\,\mathrm{Re}(r).

    Lorentz and Drude poles are the special case ``b = 0`` (purely imaginary
    residue). A non-zero ``b`` (``= coupling_edot``) is the extra degree of
    freedom that lets CCPR fit metals (gold, silver) and arbitrary
    vector-fitted permittivity data.

    A stable, passive (lossy) medium requires ``Re(q) < 0`` (so ``gamma > 0``).

    Both ``pole`` and ``residue`` are either scalars (isotropic) or per-axis
    3-tuples ``(x, y, z)`` for diagonally anisotropic dispersion (e.g. a
    vector-fitted uniaxial material with a different ``(q, r)`` set per axis).
    """

    #: Complex pole ``q`` (rad/s). ``Re(q) < 0`` for a stable, lossy medium.
    #: Scalar or per-axis 3-tuple.
    pole: complex | tuple[complex, complex, complex] = frozen_field()

    #: Complex residue ``r`` (rad/s). Scalar or per-axis 3-tuple.
    residue: complex | tuple[complex, complex, complex] = frozen_field()

    def __post_init__(self):
        self._validate_orientation()

    @property
    def omega_0_axes(self) -> tuple[float, float, float]:
        q = _broadcast_axis_param(self.pole)
        return (float(abs(complex(q[0]))), float(abs(complex(q[1]))), float(abs(complex(q[2]))))

    @property
    def gamma_axes(self) -> tuple[float, float, float]:
        q = _broadcast_axis_param(self.pole)
        return (
            float(-2.0 * complex(q[0]).real),
            float(-2.0 * complex(q[1]).real),
            float(-2.0 * complex(q[2]).real),
        )

    @property
    def coupling_sq_axes(self) -> tuple[float, float, float]:
        q = _broadcast_axis_param(self.pole)
        r = _broadcast_axis_param(self.residue)
        return (
            float(-2.0 * (complex(r[0]) * complex(q[0]).conjugate()).real),
            float(-2.0 * (complex(r[1]) * complex(q[1]).conjugate()).real),
            float(-2.0 * (complex(r[2]) * complex(q[2]).conjugate()).real),
        )

    @property
    def coupling_edot_axes(self) -> tuple[float, float, float]:
        r = _broadcast_axis_param(self.residue)
        return (
            float(2.0 * complex(r[0]).real),
            float(2.0 * complex(r[1]).real),
            float(2.0 * complex(r[2]).real),
        )

    @classmethod
    def from_critical_point(
        cls,
        amplitude: float,
        phase: float,
        resonance_frequency: float,
        damping: float,
    ) -> "CCPRPole":
        r"""Build a CCPR pole from critical-point (modified-Lorentz) parameters.

        The critical-point model term (``exp(-i omega t)`` convention) is

        .. math::
            \chi_p(\omega) = A\,\Omega\left[
                \frac{e^{i\phi}}{\Omega - \omega - i\Gamma}
                + \frac{e^{-i\phi}}{\Omega + \omega + i\Gamma}\right],

        which is the parameterization commonly reported for fitted metal
        permittivities. This maps to the complex pole/residue

        .. math::
            q = -\Gamma - i\Omega, \qquad r = i\,A\,\Omega\,e^{i\phi}.

        Args:
            amplitude: Dimensionless amplitude :math:`A`.
            phase: Phase :math:`\phi` (radians).
            resonance_frequency: Resonance :math:`\Omega` (rad/s).
            damping: Broadening :math:`\Gamma` (rad/s), ``> 0`` for loss.

        Returns:
            CCPRPole: Equivalent pole with the ``(q, r)`` above.
        """
        import cmath

        q = complex(-damping, -resonance_frequency)
        r = 1j * amplitude * resonance_frequency * cmath.exp(1j * phase)
        return cls(pole=q, residue=r)


@autoinit
class DispersionModel(TreeClass):
    """Linear susceptibility built from a sum of 2nd-order ADE poles.

    The high-frequency permittivity :math:`\\varepsilon_\\infty` is NOT
    stored here - it lives in the parent :class:`~fdtdx.materials.Material`
    as the existing ``permittivity`` field. This keeps a single source of
    truth for the ``inv_permittivities`` array.
    """

    #: Tuple of poles making up the susceptibility model.
    poles: tuple[Pole, ...] = frozen_field(default=())

    @property
    def num_poles(self) -> int:
        """Number of poles in this model."""
        return len(self.poles)

    @property
    def is_isotropic(self) -> bool:
        """Whether every pole applies the same parameters to all three axes."""
        return all(p.is_isotropic for p in self.poles)

    @property
    def has_off_diagonal_coupling(self) -> bool:
        """Whether any pole is oriented (contributing an off-diagonal coupling tensor)."""
        return any(p.is_oriented for p in self.poles)

    def susceptibility_tensor(self, omega: complex | float) -> np.ndarray:
        """Evaluate the full 3x3 complex susceptibility tensor :math:`\\chi_{ij}(\\omega)`.

        Oriented poles contribute :math:`\\chi_p(\\omega)\\, u_p u_p^T`;
        per-axis and isotropic poles contribute diagonal terms. Uses the
        ``exp(-i omega t)`` Fourier convention.

        Args:
            omega: Angular frequency (rad/s).

        Returns:
            Complex numpy array of shape ``(3, 3)``.
        """
        w = complex(omega)
        total = np.zeros((3, 3), dtype=np.complex128)
        for p in self.poles:
            omega_0 = p.omega_0_axes
            gamma = p.gamma_axes
            coupling_sq = p.coupling_sq_axes
            coupling_edot = p.coupling_edot_axes
            if p.is_oriented:
                assert p.orientation is not None
                denom = omega_0[0] ** 2 - w * w - 1j * gamma[0] * w
                numer = coupling_sq[0] - 1j * w * coupling_edot[0]
                u = np.asarray(p.orientation, dtype=np.float64)
                total += (numer / denom) * np.outer(u, u)
            else:
                for ax in range(3):
                    denom = omega_0[ax] ** 2 - w * w - 1j * gamma[ax] * w
                    numer = coupling_sq[ax] - 1j * w * coupling_edot[ax]
                    total[ax, ax] += numer / denom
        return total

    def permittivity_tensor(
        self,
        omega: complex | float,
        eps_inf: float | tuple = 1.0,
    ) -> np.ndarray:
        """Full 3x3 complex relative permittivity tensor :math:`\\varepsilon_\\infty + \\chi(\\omega)`.

        Args:
            omega: Angular frequency (rad/s).
            eps_inf: High-frequency permittivity — scalar, 3-tuple (diagonal),
                flat 9-tuple or nested 3x3. Defaults to 1.0.

        Returns:
            Complex numpy array of shape ``(3, 3)``.
        """
        eps_arr = np.asarray(eps_inf, dtype=np.complex128)
        if eps_arr.ndim == 0:
            eps_mat = np.eye(3, dtype=np.complex128) * complex(eps_arr)
        elif eps_arr.shape == (3,):
            eps_mat = np.diag(eps_arr)
        elif eps_arr.shape == (9,):
            eps_mat = eps_arr.reshape(3, 3)
        elif eps_arr.shape == (3, 3):
            eps_mat = eps_arr
        else:
            raise ValueError(f"eps_inf must be a scalar, 3-tuple, flat 9-tuple or 3x3, got shape {eps_arr.shape}.")
        return eps_mat + self.susceptibility_tensor(omega)

    def rotated(self, rotation: tuple) -> "DispersionModel":
        """Return a copy of this model with the crystal axes rotated.

        Args:
            rotation: Either a 3x3 rotation matrix as a nested tuple
                ``((r11, r12, r13), ...)`` or a 3-tuple of Euler angles
                ``(alpha, beta, gamma)`` in radians, applied extrinsically as
                ``R = Rz(gamma) @ Ry(beta) @ Rx(alpha)``.

        Returns:
            DispersionModel: Isotropic poles are unchanged; oriented poles have
            their direction rotated; per-axis poles are decomposed into up to
            three oriented poles (one per axis with non-zero coupling), so the
            pole count — and with it the simulation's pole-slot memory — can
            grow. For a rotation that is a signed axis permutation (e.g. 90
            degree rotations), per-axis poles are instead remapped in place and
            keep the cheaper diagonal representation.
        """
        r_mat = _as_rotation_matrix(rotation)
        perm = _signed_permutation(r_mat)
        new_poles: list[Pole] = []
        for p in self.poles:
            if p.is_oriented:
                assert p.orientation is not None
                u = r_mat @ np.asarray(p.orientation, dtype=np.float64)
                new_poles.append(p.aset("orientation", (float(u[0]), float(u[1]), float(u[2]))))
            elif p.is_isotropic:
                new_poles.append(p)
            elif perm is not None:
                new_poles.append(_permute_pole_axes(p, perm))
            else:
                for ax in range(3):
                    if p.coupling_sq_axes[ax] == 0.0 and p.coupling_edot_axes[ax] == 0.0:
                        continue
                    direction = (float(r_mat[0, ax]), float(r_mat[1, ax]), float(r_mat[2, ax]))
                    new_poles.append(_oriented_pole_for_axis(p, ax, direction))
        return DispersionModel(poles=tuple(new_poles))

    def susceptibility_axes(self, omega: complex | float) -> tuple[complex, complex, complex]:
        """Evaluate the per-axis complex susceptibility :math:`(\\chi_x, \\chi_y, \\chi_z)`.

        Uses the ``exp(-i omega t)`` Fourier convention (damping appears
        with a ``-i gamma omega`` term in the Lorentz denominator). For an
        isotropic model all three entries are equal.

        Args:
            omega: Angular frequency (rad/s).

        Returns:
            tuple: :math:`\\chi_a(\\omega) = \\sum_p \\chi_{p,a}(\\omega)` for
            each axis ``a`` in ``(x, y, z)``.
        """
        if self.has_off_diagonal_coupling:
            raise ValueError(
                "DispersionModel has oriented poles; use susceptibility_tensor(omega) for the full 3x3 tensor."
            )
        w = complex(omega)
        totals = [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
        for p in self.poles:
            omega_0 = p.omega_0_axes
            gamma = p.gamma_axes
            coupling_sq = p.coupling_sq_axes
            coupling_edot = p.coupling_edot_axes
            for ax in range(3):
                denom = omega_0[ax] ** 2 - w * w - 1j * gamma[ax] * w
                numer = coupling_sq[ax] - 1j * w * coupling_edot[ax]
                totals[ax] = totals[ax] + numer / denom
        return (totals[0], totals[1], totals[2])

    def susceptibility(self, omega: complex | float) -> complex:
        """Evaluate the complex susceptibility :math:`\\chi(\\omega)`.

        Uses the ``exp(-i omega t)`` Fourier convention (damping appears
        with a ``-i gamma omega`` term in the Lorentz denominator).

        Raises ``ValueError`` for models with per-axis poles; use
        :meth:`susceptibility_axes` for those.

        Args:
            omega: Angular frequency (rad/s).

        Returns:
            complex: :math:`\\chi(\\omega) = \\sum_p \\chi_p(\\omega)`.
        """
        if not self.is_isotropic:
            raise ValueError(
                "DispersionModel has per-axis poles; use susceptibility_axes(omega) for the (x, y, z) values."
            )
        return self.susceptibility_axes(omega)[0]

    def permittivity_axes(
        self,
        omega: complex | float,
        eps_inf: float | tuple[float, float, float] = 1.0,
    ) -> tuple[complex, complex, complex]:
        """Per-axis complex relative permittivity :math:`\\varepsilon_a(\\omega) = \\varepsilon_{\\infty,a} + \\chi_a(\\omega)`.

        Args:
            omega: Angular frequency (rad/s).
            eps_inf: High-frequency permittivity — scalar or per-axis
                3-tuple (the diagonal of the ε∞ tensor). Defaults to 1.0.

        Returns:
            tuple: Relative permittivity at ``omega`` per axis ``(x, y, z)``.
        """
        chi = self.susceptibility_axes(omega)
        e = _broadcast_axis_param(eps_inf)
        return (complex(e[0]) + chi[0], complex(e[1]) + chi[1], complex(e[2]) + chi[2])

    def permittivity(self, omega: complex | float, eps_inf: float = 1.0) -> complex:
        """Complex relative permittivity :math:`\\varepsilon(\\omega) = \\varepsilon_\\infty + \\chi(\\omega)`.

        Raises ``ValueError`` for models with per-axis poles; use
        :meth:`permittivity_axes` for those.

        Args:
            omega: Angular frequency (rad/s).
            eps_inf: High-frequency permittivity. Defaults to 1.0 (vacuum).

        Returns:
            complex: Relative permittivity at ``omega``.
        """
        return eps_inf + self.susceptibility(omega)


def compute_pole_coefficients_per_axis(
    poles: tuple[Pole, ...],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the per-axis discrete-time ADE recurrence coefficients.

    For each pole and grid axis, returns ``(c1, c2, c3, c4)`` with
    (``D = 1 + gamma dt / 2``)

    .. math::
        c_1 = \\frac{2 - \\omega_0^2 \\Delta t^2}{D}, \\quad
        c_2 = -\\frac{1 - \\gamma \\Delta t / 2}{D}, \\quad
        c_3 = \\frac{a \\Delta t^2 - b \\Delta t}{D}, \\quad
        c_4 = \\frac{b \\Delta t}{D},

    where ``a = coupling_sq`` is the ``E`` coupling and ``b = coupling_edot`` is
    the ``dE/dt`` coupling. The recurrence uses a forward difference for the
    ``dE/dt`` term so it stays compatible with the reversible time stepping:

    :math:`p_p^{n+1} = c_1 p_p^n + c_2 p_p^{n-1} + c_3 E^n + c_4 E^{n+1}`.

    For isotropic poles the three axis columns are identical. For Lorentz and
    Drude poles ``b = 0``, so ``c4 = 0`` and ``c3`` reduces to the classic
    :math:`K \\Delta t^2 / D`.

    Args:
        poles: Tuple of poles (may be empty).
        dt: Simulation time step (seconds).

    Returns:
        Four ``numpy`` arrays of shape ``(len(poles), 3)`` with ``c1``, ``c2``,
        ``c3``, ``c4`` per pole and axis. For an empty pole tuple, returns four
        ``(0, 3)`` arrays.
    """
    n = len(poles)
    c1 = np.zeros((n, 3), dtype=np.float64)
    c2 = np.zeros((n, 3), dtype=np.float64)
    c3 = np.zeros((n, 3), dtype=np.float64)
    c4 = np.zeros((n, 3), dtype=np.float64)
    for i, p in enumerate(poles):
        if p.is_oriented:
            raise ValueError(
                f"Pole {i} ({type(p).__name__}) is oriented; use compute_pole_coefficients_tensor instead."
            )
        omega_0 = p.omega_0_axes
        gamma = p.gamma_axes
        coupling_sq = p.coupling_sq_axes
        coupling_edot = p.coupling_edot_axes
        for ax in range(3):
            gamma_dt = gamma[ax] * dt
            omega0_dt = omega_0[ax] * dt
            # The stability bounds only bind on axes where the pole actually
            # couples. A zero-coupling axis (e.g. a Lorentz pole with
            # delta_epsilon = 0 there, the documented way to express an absent
            # resonance) has c3 = c4 = 0, so its polarization stays identically
            # zero and its unused omega_0 / gamma are irrelevant.
            axis_active = coupling_sq[ax] != 0.0 or coupling_edot[ax] != 0.0
            if axis_active and gamma_dt >= 2.0:
                axis_note = "" if p.is_isotropic else f" on axis {'xyz'[ax]}"
                raise ValueError(
                    f"Pole {i} ({type(p).__name__}) has gamma * dt = {gamma_dt:.4g} >= 2{axis_note}; "
                    "the reversible ADE update requires gamma * dt < 2 (physically gamma * dt << 1). "
                    "Lower the damping or reduce the time step."
                )
            if axis_active and omega0_dt >= 2.0:
                axis_note = "" if p.is_isotropic else f" on axis {'xyz'[ax]}"
                raise ValueError(
                    f"Pole {i} ({type(p).__name__}) has omega_0 * dt = {omega0_dt:.4g} >= 2{axis_note}; "
                    "the ADE recurrence roots leave the unit circle (requires omega_0 * dt < 2, "
                    "physically omega_0 * dt << 1). Lower the resonance frequency or reduce the time step."
                )
            denom = 1.0 + 0.5 * gamma_dt
            c1[i, ax] = (2.0 - (omega_0[ax] ** 2) * (dt**2)) / denom
            c2[i, ax] = -(1.0 - 0.5 * gamma_dt) / denom
            c3[i, ax] = (coupling_sq[ax] * dt**2 - coupling_edot[ax] * dt) / denom
            c4[i, ax] = (coupling_edot[ax] * dt) / denom
    return c1, c2, c3, c4


def compute_pole_coefficients(
    poles: tuple[Pole, ...],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the discrete-time ADE recurrence coefficients of isotropic poles.

    Scalar-per-pole variant of :func:`compute_pole_coefficients_per_axis` (see
    there for the coefficient definitions). Raises ``ValueError`` when any
    pole has per-axis parameters — use the per-axis function for those.

    Args:
        poles: Tuple of isotropic poles (may be empty).
        dt: Simulation time step (seconds).

    Returns:
        Four ``numpy`` arrays of shape ``(len(poles),)`` with ``c1``, ``c2``,
        ``c3``, ``c4``. For an empty pole tuple, returns four empty arrays.
    """
    for i, p in enumerate(poles):
        if not p.is_isotropic:
            raise ValueError(
                f"Pole {i} ({type(p).__name__}) has per-axis parameters or an orientation; "
                "use compute_pole_coefficients_per_axis or compute_pole_coefficients_tensor instead."
            )
    c1, c2, c3, c4 = compute_pole_coefficients_per_axis(poles, dt)
    return c1[:, 0], c2[:, 0], c3[:, 0], c4[:, 0]


def compute_pole_coefficients_tensor(
    poles: tuple[Pole, ...],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ADE recurrence coefficients with full 3x3 coupling tensors.

    Generalizes :func:`compute_pole_coefficients_per_axis` to oriented poles:
    the recurrence coefficients ``c1``/``c2`` stay per-axis (uniform for an
    oriented pole, whose ``omega_0``/``gamma`` are scalars), while the field
    couplings ``c3``/``c4`` become row-major 3x3 tensors per pole —
    ``(K dt^2 / D) u u^T`` for a pole oriented along ``u``, diagonal for
    per-axis and isotropic poles.

    Args:
        poles: Tuple of poles (may be empty). Oriented poles must have a
            non-negative coupling ``K`` (passivity of the ``K u u^T`` tensor).
        dt: Simulation time step (seconds).

    Returns:
        Four ``numpy`` arrays: ``c1``, ``c2`` of shape ``(len(poles), 3)`` and
        ``c3``, ``c4`` of shape ``(len(poles), 9)``.
    """
    n = len(poles)
    c1 = np.zeros((n, 3), dtype=np.float64)
    c2 = np.zeros((n, 3), dtype=np.float64)
    c3 = np.zeros((n, 9), dtype=np.float64)
    c4 = np.zeros((n, 9), dtype=np.float64)
    for i, p in enumerate(poles):
        gamma = p.gamma_axes
        omega_0 = p.omega_0_axes
        coupling_sq = p.coupling_sq_axes
        coupling_edot = p.coupling_edot_axes
        for ax in range(3):
            gamma_dt = gamma[ax] * dt
            if gamma_dt >= 2.0:
                raise ValueError(
                    f"Pole {i} ({type(p).__name__}) has gamma * dt = {gamma_dt:.4g} >= 2; "
                    "the ADE update requires gamma * dt < 2 (physically gamma * dt << 1). "
                    "Lower the damping or reduce the time step."
                )
            denom = 1.0 + 0.5 * gamma_dt
            c1[i, ax] = (2.0 - (omega_0[ax] ** 2) * (dt**2)) / denom
            c2[i, ax] = -(1.0 - 0.5 * gamma_dt) / denom
        if p.is_oriented:
            if coupling_sq[0] < 0.0:
                raise ValueError(
                    f"Pole {i} ({type(p).__name__}) has negative coupling K = {coupling_sq[0]:.4g}; "
                    "oriented poles require K >= 0 so the coupling tensor K u u^T stays positive "
                    "semi-definite (passivity)."
                )
            assert p.orientation is not None
            u = np.asarray(p.orientation, dtype=np.float64)
            denom = 1.0 + 0.5 * gamma[0] * dt
            c3[i] = ((coupling_sq[0] * dt**2 / denom) * np.outer(u, u)).reshape(-1)
            # c4 is identically zero for oriented poles (validated at construction).
        else:
            for ax in range(3):
                denom = 1.0 + 0.5 * gamma[ax] * dt
                c3[i, 4 * ax] = (coupling_sq[ax] * dt**2 - coupling_edot[ax] * dt) / denom
                c4[i, 4 * ax] = (coupling_edot[ax] * dt) / denom
    return c1, c2, c3, c4


def _tensor_from_components(arr: jax.Array) -> jax.Array:
    """Expand a component array ``(1|3|9, *spatial)`` to a matrix field ``(3, 3, *spatial)``."""
    if arr.shape[0] == 9:
        return arr.reshape(3, 3, *arr.shape[1:])
    diag = jnp.broadcast_to(arr, (3, *arr.shape[1:]))
    return jnp.zeros((3, 3, *arr.shape[1:]), dtype=arr.dtype).at[jnp.arange(3), jnp.arange(3)].set(diag)


def _invert_3x3_matrix_field(mat: jax.Array) -> jax.Array:
    """Per-cell inverse of a matrix field ``(3, 3, *spatial)``."""
    moved = jnp.moveaxis(mat, (0, 1), (-2, -1))
    return jnp.moveaxis(jnp.linalg.inv(moved), (-2, -1), (0, 1))


def _eps_matrix_from_inv(inv_eps: jax.Array) -> jax.Array:
    """Per-cell permittivity matrix ``(3, 3, *spatial)`` from stored inverse components."""
    if inv_eps.shape[0] == 9:
        return _invert_3x3_matrix_field(_tensor_from_components(inv_eps))
    return _tensor_from_components(1.0 / inv_eps)


def _expand_recurrence_to_coupling(c: jax.Array, coupling_components: int) -> jax.Array:
    """Expand a recurrence coefficient's component axis to match a 9-component coupling axis.

    A per-axis coefficient ``(P, 3, ...)`` becomes ``(P, 9, ...)`` where the
    row-major coupling entry ``3i+j`` uses the oscillator of row ``i``. Size-1
    axes broadcast as-is.
    """
    if coupling_components != 9 or c.shape[1] != 3:
        return c
    return jnp.repeat(c, 3, axis=1)


def susceptibility_from_coefficients(
    c1: jax.Array,
    c2: jax.Array,
    c3: jax.Array,
    omega: float,
    dt: float,
    c4: jax.Array | None = None,
) -> jax.Array:
    """Evaluate the per-cell complex susceptibility :math:`\\chi(\\omega)` from
    the stored ADE recurrence coefficients.

    The coefficient arrays have shape ``(num_poles, ...)`` where the trailing
    axes are the spatial (and optional component) dimensions. The inversion
    (with ``D = 1 + \\gamma \\Delta t / 2``)

    .. math::
        \\gamma \\Delta t     &= \\frac{2 (1 + c_2)}{1 - c_2},\\\\
        \\omega_0^2 \\Delta t^2 &= 2 - c_1 D,\\\\
        a \\Delta t^2          &= (c_3 + c_4) D,\\\\
        b \\Delta t            &= c_4 D

    is applied pointwise, then each pole contributes

    .. math::
        \\chi_p(\\omega) = \\frac{a - i\\omega b}{\\omega_0^2 - \\omega^2 - i \\gamma \\omega}

    and the result is summed over the leading pole axis. Cells where the
    coefficients are all zero (no pole) contribute exactly zero. When ``c4`` is
    ``None`` (Lorentz/Drude) the ``b`` term vanishes and this reduces to the
    classic real-numerator Lorentzian.

    Args:
        c1: ADE coefficient array of shape ``(num_poles, ...)``.
        c2: ADE coefficient array of shape ``(num_poles, ...)``.
        c3: ADE coefficient array of shape ``(num_poles, ...)``.
        omega: Angular frequency (rad/s) at which to evaluate the
            susceptibility.
        dt: Simulation time step (seconds) used to derive the coefficients.
        c4: Optional ADE coefficient array (the ``dE/dt`` coupling), shape
            ``(num_poles, ...)``. ``None`` is treated as all-zero.

    Returns:
        Complex ``jax.Array`` with shape ``c1.shape[1:]`` — the total
        :math:`\\chi(\\omega)` summed over all poles, in every cell.
    """
    c1 = jnp.asarray(c1)
    c2 = jnp.asarray(c2)
    c3 = jnp.asarray(c3)
    c4 = jnp.zeros_like(c3) if c4 is None else jnp.asarray(c4)
    if c1.ndim >= 2 and c3.ndim >= 2 and c3.shape[1] == 9:
        # 9-component coupling (oriented poles): the recurrence coefficients
        # expand so entry (i, j) uses the oscillator of row i; the result is
        # the per-entry chi_ij with shape (9, *spatial).
        c1 = _expand_recurrence_to_coupling(c1, 9)
        c2 = _expand_recurrence_to_coupling(c2, 9)

    pole_mask = (c1 != 0.0) | (c3 != 0.0) | (c4 != 0.0)

    one_minus_c2 = 1.0 - c2
    safe_denom = jnp.where(one_minus_c2 == 0.0, 1.0, one_minus_c2)
    gamma_dt = 2.0 * (1.0 + c2) / safe_denom
    gamma_dt = jnp.where(pole_mask, gamma_dt, 0.0)

    half_factor = 1.0 + 0.5 * gamma_dt
    omega0_sq_dt2 = 2.0 - c1 * half_factor
    omega0_sq_dt2 = jnp.where(pole_mask, omega0_sq_dt2, 0.0)
    # a*dt^2 = (c3 + c4)*D, b*dt = c4*D (see compute_pole_coefficients).
    a_dt2 = jnp.where(pole_mask, (c3 + c4) * half_factor, 0.0)
    b_dt = jnp.where(pole_mask, c4 * half_factor, 0.0)

    omega_dt = omega * dt
    numer = a_dt2 - 1j * omega_dt * b_dt
    denom = omega0_sq_dt2 - omega_dt * omega_dt - 1j * gamma_dt * omega_dt
    safe_denom_cplx = jnp.where(pole_mask, denom, 1.0 + 0.0j)
    chi_per_pole = jnp.where(pole_mask, numer / safe_denom_cplx, 0.0 + 0.0j)

    return jnp.sum(chi_per_pole, axis=0)


def compute_eps_spectrum_from_coefficients(
    c1: jax.Array | np.ndarray,
    c2: jax.Array | np.ndarray,
    c3: jax.Array | np.ndarray,
    inv_eps_inf: jax.Array | np.ndarray,
    omegas: np.ndarray,
    dt: float,
    weights: np.ndarray | None = None,
    c4: jax.Array | np.ndarray | None = None,
) -> np.ndarray:
    """Spatially-averaged complex permittivity spectrum for a block of cells.

    For each angular frequency in ``omegas``, evaluates the per-cell
    complex permittivity :math:`\\varepsilon(\\omega) = \\varepsilon_\\infty + \\chi(\\omega)`
    where :math:`\\chi` is reconstructed from the ADE recurrence coefficients,
    and averages over the spatial axes (uniformly or with supplied weights).

    This is the broadband generalization of the single-frequency
    :func:`effective_inv_permittivity` used for carrier-frequency impedance
    matching — callers that need a frequency-dependent impedance (e.g. for
    a convolution-based broadband source correction) use this to build the
    :math:`\\varepsilon(\\omega)` spectrum that feeds
    :func:`compute_impedance_corrected_temporal_profile`.

    Args:
        c1: ADE coefficient array of shape ``(num_poles, num_components, *spatial)``
            as stored on :class:`~fdtdx.fdtd.container.ArrayContainer`, with
            ``num_components in (1, 3)`` (the material-component axis; size 3
            for per-axis anisotropic dispersion). Anisotropic components are
            averaged, mirroring the ``inv_eps_inf`` reduction.
        c2: ADE coefficient array, same shape as ``c1``.
        c3: ADE coefficient array, same shape as ``c1``.
        inv_eps_inf: Per-cell inverse of the high-frequency permittivity,
            shape ``(num_components, *spatial)`` with
            ``num_components in (1, 3, 9)``. For anisotropic tensors
            (9 components) only the diagonal entries are used.
        omegas: 1D array of angular frequencies (rad/s) to evaluate at.
        dt: Simulation time step (seconds) used to derive the coefficients.
        weights: Optional spatial weights with the same shape as the
            trailing axes of ``c1``. If ``None``, uniform averaging.

    Returns:
        Complex numpy array of shape ``(len(omegas),)`` — the volume-averaged
        :math:`\\varepsilon(\\omega)` at each requested frequency.
    """
    c1_np = np.asarray(c1)
    c2_np = np.asarray(c2)
    c3_np = np.asarray(c3)
    c4_np = np.zeros_like(c3_np) if c4 is None else np.asarray(c4)
    inv_eps_np = np.asarray(inv_eps_inf)
    omegas_np = np.asarray(omegas, dtype=np.float64)
    if c3_np.ndim >= 2 and c3_np.shape[1] == 9 and c1_np.shape[1] == 3:
        # 9-component coupling (oriented poles): entry 3i+j uses oscillator row i.
        c1_np = np.repeat(c1_np, 3, axis=1)
        c2_np = np.repeat(c2_np, 3, axis=1)

    # Reverse-engineer pole parameters from the ADE coefficients (same inversion
    # as susceptibility_from_coefficients, duplicated in numpy for setup-time use).
    pole_mask = (c1_np != 0.0) | (c3_np != 0.0) | (c4_np != 0.0)
    one_minus_c2 = 1.0 - c2_np
    safe_one_minus_c2 = np.where(one_minus_c2 == 0.0, 1.0, one_minus_c2)
    gamma_dt = np.where(pole_mask, 2.0 * (1.0 + c2_np) / safe_one_minus_c2, 0.0)
    half_factor = 1.0 + 0.5 * gamma_dt
    omega0_sq_dt2 = np.where(pole_mask, 2.0 - c1_np * half_factor, 0.0)
    # a*dt^2 = (c3 + c4)*D, b*dt = c4*D (numerator = a*dt^2 - i*omega*dt*b*dt).
    a_dt2 = np.where(pole_mask, (c3_np + c4_np) * half_factor, 0.0)
    b_dt = np.where(pole_mask, c4_np * half_factor, 0.0)

    # Reduce inv_eps_inf → scalar eps_inf per spatial cell.
    num_components = inv_eps_np.shape[0]
    if num_components == 9:
        # inv_eps_inf stores the inverse tensor and diag(eps) != 1/diag(eps^-1)
        # when off-diagonal terms exist: invert each cell's 3x3 before averaging.
        spatial = inv_eps_np.shape[1:]
        inv_mats = np.moveaxis(inv_eps_np.reshape(3, 3, -1), -1, 0)
        eps_diag_mean = np.trace(np.linalg.inv(inv_mats), axis1=-2, axis2=-1) / 3.0
        eps_inf_per_cell = eps_diag_mean.reshape(spatial)
    elif num_components in (1, 3):
        eps_inf_per_cell = np.mean(1.0 / inv_eps_np, axis=0)
    else:
        raise ValueError(f"Unexpected inv_eps_inf leading dimension {num_components}; expected 1, 3, or 9.")

    # Broadcast: omegas over (M,); coefficient arrays have shape (P, C, *spatial)
    # with C in (1, 3). After [None, ...] prepend: (M, P, C, *spatial).
    omega_dt = (omegas_np * dt).reshape((-1,) + (1,) * c1_np.ndim)
    numer = a_dt2[None, ...] - 1j * omega_dt * b_dt[None, ...]
    denom = omega0_sq_dt2[None, ...] - omega_dt**2 - 1j * gamma_dt[None, ...] * omega_dt
    safe_denom = np.where(pole_mask[None, ...], denom, 1.0 + 0.0j)
    chi_per_pole = np.where(pole_mask[None, ...], numer / safe_denom, 0.0 + 0.0j)
    chi_per_cell = chi_per_pole.sum(axis=1)  # sum over pole axis → (M, C, *spatial)
    # Average the material-component axis (identity for C = 1), mirroring the
    # eps_inf reduction above — this scalar spectrum feeds an impedance filter
    # that has no notion of polarization. For a 9-component coupling only the
    # diagonal entries carry impedance information.
    if chi_per_cell.shape[1] == 9:
        chi_per_cell = chi_per_cell[:, (0, 4, 8)].mean(axis=1)
    else:
        chi_per_cell = chi_per_cell.mean(axis=1)  # → (M, *spatial)

    eps_per_cell = eps_inf_per_cell[None, ...] + chi_per_cell  # (M, *spatial)

    if weights is None:
        flat = eps_per_cell.reshape(eps_per_cell.shape[0], -1)
        return flat.mean(axis=1)

    weights_np = np.asarray(weights, dtype=np.float64).reshape(-1)
    flat = eps_per_cell.reshape(eps_per_cell.shape[0], -1)
    weight_sum = weights_np.sum()
    if weight_sum == 0.0:
        return flat.mean(axis=1)
    return (flat * weights_np).sum(axis=1) / weight_sum


def compute_impedance_corrected_temporal_profile(
    raw_samples: np.ndarray,
    dt: float,
    eps_spectrum: np.ndarray,
    eps_center: complex,
) -> np.ndarray:
    """FIR-filter a raw source temporal profile for broadband impedance matching.

    Given the unfiltered E-side temporal profile ``s(n·dt)`` and the complex
    permittivity spectrum ``eps_spectrum = ε(ω_k)`` at the rFFT frequencies
    of a zero-padded version of ``s``, returns the H-side temporal profile
    ``s_H(n·dt)`` whose spectrum satisfies
    :math:`\\tilde{s}_H(\\omega) = \\tilde{s}(\\omega) \\cdot G(\\omega)` with

    .. math::
        G(\\omega) = \\frac{\\eta(\\omega_c)}{\\eta(\\omega)}
                   = \\sqrt{\\frac{\\varepsilon(\\omega)}{\\varepsilon(\\omega_c)}}

    (assuming a non-dispersive permeability). Injecting the prescribed E and
    H fields as ``E(x,t) = E_spatial(x)·s(t)`` and
    ``H(x,t) = (H_spatial(x)/η(ω_c))·s_H(t)`` then reproduces a physical
    plane wave at every frequency in the pulse bandwidth, not just at
    ``ω_c``. In the non-dispersive limit ``ε(ω) ≡ ε_c`` and ``G`` is the
    identity so ``s_H == s``.

    Implementation: zero-pads to ``M = 2·(len(eps_spectrum) - 1)`` for
    linear convolution, takes a real FFT, multiplies by ``G``, and transforms
    back with :func:`numpy.fft.irfft` (which enforces a real output via
    Hermitian symmetry of the positive-frequency spectrum).

    Args:
        raw_samples: Real 1-D array of the unfiltered temporal profile
            sampled at integer time steps, ``s[n] = s(n·dt)``.
        dt: Simulation time step (seconds). Present for API symmetry; the
            actual time step is encoded in ``eps_spectrum``.
        eps_spectrum: Complex 1-D array of length ``M/2 + 1`` giving
            :math:`\\varepsilon(\\omega)` at
            :math:`\\omega_k = 2\\pi \\cdot k / (M \\cdot \\Delta t)` for
            ``k = 0, ..., M/2``.
        eps_center: Scalar complex :math:`\\varepsilon(\\omega_c)` at the
            source carrier frequency.

    Returns:
        Real 1-D array of length ``len(raw_samples)`` containing ``s_H[n]``.
    """
    del dt
    raw = np.asarray(raw_samples, dtype=np.float64)
    n = raw.shape[0]
    m = (eps_spectrum.shape[0] - 1) * 2
    if m < n:
        raise ValueError(
            f"eps_spectrum of length {eps_spectrum.shape[0]} corresponds to "
            f"M={m} FFT points, which is smaller than the raw profile length {n}."
        )

    padded = np.zeros(m, dtype=np.float64)
    padded[:n] = raw
    spectrum = np.fft.rfft(padded)

    ratio = np.asarray(eps_spectrum, dtype=np.complex128) / complex(eps_center)
    filter_response = np.sqrt(ratio)
    # DC bin: eps(0) can be ill-defined for Drude poles (1/0 in the physical
    # continuum). A real s(t) has a real S(0) anyway, and a real-valued
    # correction there is enough — use G(0)=1 so the filter is the identity
    # at DC. The Nyquist bin must also be real for irfft to produce a real
    # output; take the real part to be safe.
    filter_response[0] = 1.0 + 0.0j
    filter_response[-1] = complex(np.real(filter_response[-1]), 0.0)

    filtered_spectrum = spectrum * filter_response
    filtered = np.fft.irfft(filtered_spectrum, n=m)
    return filtered[:n].astype(np.float64)


def effective_inv_permittivity(
    inv_eps: jax.Array,
    c1: jax.Array | None,
    c2: jax.Array | None,
    c3: jax.Array | None,
    omega: float,
    dt: float,
    c4: jax.Array | None = None,
) -> jax.Array:
    """Per-cell real inverse permittivity :math:`1/\\text{Re}(\\varepsilon_\\infty + \\chi(\\omega))`.

    Sources in FDTDX use a real wave impedance, so only the real part of
    ``ε∞ + χ(ω)`` enters the injected amplitude. The imaginary part describes
    absorption, which is already handled by the ADE update loop (injecting it
    into the source amplitude would double-count).

    Cells with no pole (``c1 = c2 = c3 = 0``) contribute :math:`\\chi = 0` so
    their ``inv_eps`` is returned unchanged.

    Args:
        inv_eps: Per-cell :math:`1/\\varepsilon_\\infty` array. Typically
            has shape ``(num_components, ...)``; any shape broadcast-compatible
            with ``c1.shape[1:]`` works.
        c1: ADE coefficient array of shape ``(num_poles, ...)`` or ``None``.
        c2: ADE coefficient array of shape ``(num_poles, ...)`` or ``None``.
        c3: ADE coefficient array of shape ``(num_poles, ...)`` or ``None``.
        omega: Angular frequency (rad/s) at which to evaluate.
        dt: Simulation time step (seconds).

    Returns:
        Real-valued ``jax.Array`` with the same shape and dtype as
        ``inv_eps``. If any of ``c1``/``c2``/``c3`` is ``None``, returns
        ``inv_eps`` unchanged.
    """
    inv_eps_arr = jnp.asarray(inv_eps)
    coupling_c = jnp.asarray(c3).shape[1] if c3 is not None and jnp.asarray(c3).ndim >= 2 else 1
    if inv_eps_arr.shape[0] == 9 or coupling_c == 9:
        # Tensor path: reconstruct the real permittivity matrix per cell, add
        # the (possibly off-diagonal) susceptibility, and invert per cell.
        # Elementwise 1/inv_eps would divide by the zero off-diagonal entries.
        eps_mat = jnp.real(_eps_matrix_from_inv(inv_eps_arr))
        if c1 is not None and c2 is not None and c3 is not None:
            chi = susceptibility_from_coefficients(c1=c1, c2=c2, c3=c3, omega=omega, dt=dt, c4=c4)
            eps_mat = eps_mat + jnp.real(_tensor_from_components(chi))
        inv_eff = _invert_3x3_matrix_field(eps_mat)
        return inv_eff.reshape(9, *inv_eff.shape[2:]).astype(inv_eps_arr.dtype)

    if c1 is None or c2 is None or c3 is None:
        return inv_eps

    chi = susceptibility_from_coefficients(c1=c1, c2=c2, c3=c3, omega=omega, dt=dt, c4=c4)
    eps_inf = 1.0 / inv_eps_arr
    eps_eff = eps_inf + jnp.real(chi)
    return (1.0 / eps_eff).astype(inv_eps_arr.dtype)


def effective_complex_inv_permittivity(
    inv_eps: jax.Array,
    omega: float,
    dt: float,
    c1: jax.Array | None = None,
    c2: jax.Array | None = None,
    c3: jax.Array | None = None,
    electric_conductivity: jax.Array | None = None,
    conductivity_spacing: float | None = None,
    c4: jax.Array | None = None,
) -> jax.Array:
    r"""Per-cell COMPLEX inverse permittivity :math:`1 / (\varepsilon_\infty + \chi(\omega) + i\sigma/(\varepsilon_0\omega))`.

    Unlike :func:`effective_inv_permittivity` — which returns the real
    ``1/Re(eps)`` for source impedance / energy normalization and deliberately
    drops the imaginary part — this keeps the *full complex* permittivity so the
    mode solver sees the material loss, yielding a complex effective index and a
    lossy mode profile. Use it ONLY for the permittivity handed to the mode
    solver, never for impedance / energy (which would double-count the
    absorption already integrated by the ADE loop and the conductivity update).

    Both loss contributions are added in the ``exp(-i omega t)`` convention
    (positive imaginary part = loss):

    * the dispersive susceptibility :math:`\chi(\omega)` reconstructed from the
      ADE coefficients (omitted when ``c1``/``c2``/``c3`` are ``None``), and
    * the conductivity loss :math:`i\,\sigma_\text{phys} / (\varepsilon_0 \omega)`,
      where :math:`\sigma_\text{phys} = \sigma_\text{array} / \Delta` recovers the
      physical S/m value from the resolution-scaled ``electric_conductivity``
      array (``conductivity_spacing`` is the scaling factor
      :math:`\Delta = c_0 \Delta t / S` applied at initialization).

    Args:
        inv_eps: Per-cell ``1/eps_inf`` (real). Shape ``(num_components, ...)``.
        omega: Angular frequency (rad/s).
        dt: Simulation time step (seconds).
        c1: ADE coefficient array of shape ``(num_poles, ...)`` or ``None``.
        c2: ADE coefficient array of shape ``(num_poles, ...)`` or ``None``.
        c3: ADE coefficient array of shape ``(num_poles, ...)`` or ``None``.
        electric_conductivity: Resolution-scaled conductivity array, or ``None``.
        conductivity_spacing: Scaling factor used to recover the physical
            conductivity. Required when ``electric_conductivity`` is given.

    Returns:
        Complex ``jax.Array`` broadcasting ``inv_eps`` against the loss terms.
    """
    inv_eps = jnp.asarray(inv_eps)
    complex_dtype = jnp.complex128 if inv_eps.dtype == jnp.float64 else jnp.complex64
    coupling_c = jnp.asarray(c3).shape[1] if c3 is not None and jnp.asarray(c3).ndim >= 2 else 1
    if inv_eps.shape[0] == 9 or coupling_c == 9:
        # Diagonal reduction for the mode solver: off-diagonal permittivity and
        # susceptibility entries are dropped, so modal geometry in monoclinic /
        # rotated media is a diagonal approximation.
        eps_mat = _eps_matrix_from_inv(inv_eps)
        eps = jnp.stack([eps_mat[0, 0], eps_mat[1, 1], eps_mat[2, 2]], axis=0).astype(complex_dtype)
        if c1 is not None and c2 is not None and c3 is not None:
            chi = susceptibility_from_coefficients(c1=c1, c2=c2, c3=c3, omega=omega, dt=dt, c4=c4)
            if chi.shape[0] == 9:
                chi = jnp.stack([chi[0], chi[4], chi[8]], axis=0)
            eps = eps + chi
        if electric_conductivity is not None:
            if conductivity_spacing is None:
                raise ValueError("conductivity_spacing is required when electric_conductivity is given.")
            sigma = jnp.asarray(electric_conductivity)
            if sigma.shape[0] == 9:
                sigma = jnp.stack([sigma[0], sigma[4], sigma[8]], axis=0)
            eps = eps + 1j * (sigma / conductivity_spacing) / (omega * eps0)
        return 1.0 / eps

    eps = (1.0 / inv_eps).astype(complex_dtype)
    if c1 is not None and c2 is not None and c3 is not None:
        eps = eps + susceptibility_from_coefficients(c1=c1, c2=c2, c3=c3, omega=omega, dt=dt, c4=c4)
    if electric_conductivity is not None:
        if conductivity_spacing is None:
            raise ValueError("conductivity_spacing is required when electric_conductivity is given.")
        sigma_phys = jnp.asarray(electric_conductivity) / conductivity_spacing
        eps = eps + 1j * sigma_phys / (omega * eps0)
    return 1.0 / eps
