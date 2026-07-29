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
Every integrator produces the same two-level rational map

.. math::
    \\chi_d(z) = \\frac{c_4 z^2 + c_3 z + c_5}{z^2 - c_1 z - c_2},
    \\qquad z = e^{-i \\omega \\Delta t}

i.e. the "P-form" recurrence

.. math::
    p_p^{n+1} = c_1 p_p^{n} + c_2 p_p^{n-1} + c_3 E^{n} + c_4 E^{n+1}
                + c_5 E^{n-1}.

Because the discrete Ampere law consumes only
:math:`\\varepsilon_\\infty (E^{n+1} - E^n) + \\sum_p (p_p^{n+1} - p_p^n)`, the
permittivity the grid actually realizes is exactly
:math:`\\varepsilon_\\infty + \\chi_d(z)` — which is what
:func:`susceptibility_from_coefficients` evaluates.

Three schemes are available per pole via :attr:`Pole.integrator`
(``D = 1 + \\gamma \\Delta t / 2``, ``K = 2 / \\Delta t``,
``D_b = K^2 + \\gamma K + \\omega_0^2``):

``"central"``
    Central differences on the 2nd-order ODE with a **forward** difference on
    the :math:`b \\dot E` term. Second-order accurate for :math:`b = 0`, but only
    **first order** whenever :math:`b \\neq 0`, and its leading error is *real*,
    so it perturbs the in-phase coupling :math:`a` rather than merely shifting
    the pole. Kept for reproducing pre-existing results.

    .. math::
        c_1 = \\frac{2 - \\omega_0^2 \\Delta t^2}{D}, \\quad
        c_2 = -\\frac{1 - \\gamma \\Delta t / 2}{D}, \\quad
        c_3 = \\frac{a \\Delta t^2 - b \\Delta t}{D}, \\quad
        c_4 = \\frac{b \\Delta t}{D}, \\quad c_5 = 0

``"centered_edot"`` (default)
    Same oscillator, with :math:`b \\dot E` **centred** so every term of the ODE
    sits at time :math:`n`. Second order for all :math:`b`, and it halves
    :math:`c_4`, which is what keeps the implicit divisor below well conditioned.

    .. math::
        c_3 = \\frac{a \\Delta t^2}{D}, \\quad
        c_4 = \\frac{b \\Delta t}{2 D}, \\quad
        c_5 = -\\frac{b \\Delta t}{2 D}

    (:math:`c_1`, :math:`c_2` as for ``"central"``.) For :math:`b = 0` all five
    coefficients coincide with ``"central"``, so Lorentz and Drude poles are
    unaffected by the choice.

``"bilinear"``
    Trapezoidal / bilinear map :math:`s \\to K (z-1)/(z+1)`. Second order,
    **unconditionally stable** (it maps the open left half-plane exactly onto
    the open unit disk), and exact at DC. Not supported for oriented poles.

    .. math::
        c_1 = \\frac{2 (K^2 - \\omega_0^2)}{D_b}, \\quad
        c_2 = -\\frac{K^2 - \\gamma K + \\omega_0^2}{D_b}, \\quad
        c_3 = \\frac{2 a}{D_b}, \\quad
        c_4 = \\frac{a + b K}{D_b}, \\quad
        c_5 = \\frac{a - b K}{D_b}

    Note :math:`c_4 \\neq 0` even for :math:`b = 0`, so *every* material becomes
    implicit under this scheme.

Observer-canonical realization
------------------------------
A non-zero :math:`c_5` would make the P-form need an :math:`E^{n-1}` array — an
extra full field history per pole. The FDTD loop therefore marches the
observer-canonical realization of the same :math:`\\chi_d(z)`:

.. math::
    x_1^{n+1} &= c_1 x_1^n + x_2^n + \\beta_1 E^n, \\qquad
    \\beta_1 = c_3 + c_1 c_4 \\\\
    x_2^{n+1} &= c_2 x_1^n \\phantom{{}+ x_2^n} + \\beta_0 E^n, \\qquad
    \\beta_0 = c_5 + c_2 c_4 \\\\
    p^{n+1}   &= x_1^{n+1} + c_4 E^{n+1}

which carries every scheme with two state levels, no field history, and a
purely per-cell polarization update. See :func:`to_observer_form`. This is the
*algebraic* reference form; what the loop actually stores and marches is its
delta-basis rewrite below.

Delta basis (what is stored, and why)
-------------------------------------
The observer form above is mathematically exact but **numerically unusable in
float32**. In the physical regime :math:`\\gamma \\Delta t, \\omega_0 \\Delta t
\\ll 1` its coefficients crowd against fixed values,

.. math::
    c_1 \\to 2, \\qquad c_2 \\to -1,

while every quantity of physical interest lives in the *residuals*. The
denominator's constant term is the extreme case: for a **real** pole
(:math:`\\mathrm{Im}\\,q = 0`, the near-DC pole of every vector-fitted metal),
:math:`\\omega_0 = \\gamma / 2` gives a double root and

.. math::
    1 - c_1 - c_2 = \\frac{\\omega_0^2 \\Delta t^2}{D}
    \\quad\\sim\\quad (\\omega_0 \\Delta t)^2 ,

which is *quadratically* small. float32 resolves :math:`c_1 \\approx 2` only to
:math:`1.2 \\times 10^{-7}` absolute, so once
:math:`(\\omega_0 \\Delta t)^2` drops below that — for gold's
:math:`|q| = 1.28 \\times 10^{14}\\,\\mathrm{s}^{-1}` this happens between 2 nm and
1 nm — the pole's entire DC response is round-off noise. Because refining the
grid *shrinks* :math:`\\Delta t`, the error **grows** under refinement: it
overtakes the :math:`O(\\Delta t^2)` truncation error and the simulation gets
less accurate the finer the grid.

The cure is a change of basis. Expanding the same rational function in
:math:`\\zeta = z - 1` instead of :math:`z`,

.. math::
    z^2 - c_1 z - c_2 &= \\zeta^2 + a_1 \\zeta + a_0, \\qquad
    a_1 = 2 - c_1, \\quad a_0 = 1 - c_1 - c_2 \\\\
    \\beta_1 z + \\beta_0 &= b_1 \\zeta + b_0, \\qquad
    b_1 = \\beta_1, \\quad b_0 = \\beta_1 + \\beta_0

puts the small residuals *in* the stored numbers, where float32 gives them full
**relative** precision. They have closed forms that are sums of non-negative
terms, so they are assembled without any subtraction (see
:func:`_scheme_delta_coefficients`):

.. math::
    \\text{central / centered\\_edot:} \\quad
    a_1 = \\frac{\\gamma \\Delta t + \\omega_0^2 \\Delta t^2}{D}, \\quad
    a_0 = \\frac{\\omega_0^2 \\Delta t^2}{D} \\\\
    \\text{bilinear:} \\quad
    a_1 = \\frac{2 \\gamma K + 4 \\omega_0^2}{D_b}, \\quad
    a_0 = \\frac{4 \\omega_0^2}{D_b}

Since :math:`\\zeta` is the forward-difference operator, the time-domain
realization is the observer form written **incrementally**. With the state
:math:`(x_1, y_2) = (x_1, x_1 + x_2)` — an exact linear change of variables,
both still zero-initialized:

.. math::
    \\Delta x_1^n &= y_2^n - a_1 x_1^n + b_1 E^n
        \\qquad (= x_1^{n+1} - x_1^n) \\\\
    y_2^{n+1} &= y_2^n - a_0 x_1^n + b_0 E^n \\\\
    x_1^{n+1} &= x_1^n + \\Delta x_1^n \\\\
    p^{n+1}   &= x_1^{n+1} + c_4 E^{n+1}

This removes two further float32 cancellations for free: :math:`y_2 = x_1 + x_2`
is :math:`O(\\Delta x_1)` where :math:`x_2 \\approx -x_1` was not, and the
increment :math:`p^n - x_1^{n+1} = c_4 E^n - \\Delta x_1^n` that Ampere's law
consumes is now *computed* rather than recovered by subtracting two nearly
equal large numbers. Measured on gold at 1 nm, that increment's error drops from
:math:`8 \\times 10^{-4}` to :math:`9 \\times 10^{-8}` (float32 machine
precision), and the realized :math:`\\varepsilon''` error from +33% to +0.07%.

Cost is nil: five coefficient arrays before and after, two state arrays before
and after, and one extra add per pole per component per step.

The stored arrays therefore hold :math:`(a_1, a_0, b_1, c_4, b_0)` —
``dispersive_a1`` is :math:`2 - c_1` and ``dispersive_a0`` is
:math:`1 - c_1 - c_2`, **not** :math:`c_1` and :math:`c_2`. Zero-padded and
non-dispersive slots stay all-zero and remain inert (:math:`b_1 = b_0 = 0`
keeps the state at zero forever), so they must be produced by zero-padding the
delta coefficients, never by converting padded P-form zeros — the latter would
yield :math:`(a_1, a_0) = (2, 1)`.
``dispersive_y2`` holds :math:`x_1 + x_2`, which is neither :math:`x_2` nor
:math:`p^{n-1}`.

Stability
---------
**Forward unit-circle (Jury) stability** is enforced in
:func:`compute_pole_coefficients_per_axis`, only on axes where the pole actually
couples (a zero-coupling axis keeps its polarization identically zero). The
roots of :math:`z^2 - c_1 z - c_2 = 0` lie inside the unit circle iff
:math:`|c_2| < 1` *and* :math:`|c_1| < 1 - c_2`. For ``"central"`` /
``"centered_edot"`` the first holds for every :math:`\\gamma \\Delta t > 0`
(:math:`c_2 = 0` at :math:`\\gamma \\Delta t = 2` and :math:`|c_2| \\to 1` only
as :math:`\\gamma \\Delta t \\to 0` or :math:`\\infty`), so it is not the binding
constraint; the second is algebraically equivalent to
:math:`\\omega_0 \\Delta t < 2` (independent of :math:`\\gamma`), which is
therefore the forward-stability bound. ``"bilinear"`` satisfies both for every
:math:`\\Delta t` and carries no such bound.

Implicit divisor
----------------
When :math:`c_4 \\neq 0` the polarization couples to :math:`E^{n+1}`, so the
E-field update divides by a per-cell factor

.. math::
    1 + \\varepsilon_\\infty^{-1} \\sum_p c_{4,p}
    \\ (+\\ c\\,\\sigma\\,\\eta_0\\,\\varepsilon_\\infty^{-1} / 2)

which must stay positive in every cell; as it approaches :math:`0^+` the
transient gain (:math:`\\approx 1/\\text{divisor}`) explodes and accuracy
collapses. Checked at initialization by
:func:`fdtdx.materials.validate_dispersive_divisor_stability`.

For a *physical* CCPR fit the total :math:`1/\\omega` tail of
:math:`\\varepsilon(\\omega)` must vanish, i.e.
:math:`\\sum_p b_p = -\\sigma / \\varepsilon_0`. Since the static conductivity
enters the same divisor trapezoidally as
:math:`\\sigma \\Delta t / (2 \\varepsilon_0 \\varepsilon_\\infty)`, the
``"centered_edot"`` :math:`c_4 = b \\Delta t / (2 D)` makes the two terms cancel
and the divisor sit near 1 at every resolution. ``"central"`` puts twice that
on :math:`E^{n+1}`, overshooting the cancellation and driving the divisor
negative — which is why standard Drude-critical-point metal fits cannot be run
above a few nanometres under that scheme. Lorentz and Drude poles have
:math:`c_4 = 0` under both central-difference schemes, so their divisor is
identically 1.

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
fully anisotropic update path.

Gradients
---------
Dispersive simulations currently support only the ``checkpointed`` gradient
method; the ``reversible`` method rejects them (reversing the ADE recurrence is
under active development).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, get_args

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.constants import eps0
from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field

#: Time-integrator variants available per pole. See the module docstring for the
#: coefficient formulas, accuracy order and stability bound of each.
DispersionIntegrator = Literal["central", "centered_edot", "bilinear"]

#: Tuple form of :data:`DispersionIntegrator` for validation and iteration.
DISPERSION_INTEGRATORS: tuple[str, ...] = get_args(DispersionIntegrator)

#: Schemes whose recurrence roots leave the unit circle for ``omega_0 * dt >= 2``
#: and therefore keep that guard. ``"bilinear"`` is unconditionally stable.
_CONDITIONALLY_STABLE: frozenset[str] = frozenset({"central", "centered_edot"})


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
            integrator=p.integrator,
        )
    if isinstance(p, DrudePole):
        return DrudePole(
            plasma_frequency=_remap(p.plasma_frequency),
            damping=_remap(p.damping),
            integrator=p.integrator,
        )
    if isinstance(p, CCPRPole):
        return CCPRPole(pole=_remap(p.pole), residue=_remap(p.residue), integrator=p.integrator)
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
            integrator=p.integrator,
        )
    if isinstance(p, DrudePole):
        wp = _broadcast_axis_param(p.plasma_frequency)
        g = _broadcast_axis_param(p.damping)
        return DrudePole(
            plasma_frequency=float(wp[axis]),
            damping=float(g[axis]),
            orientation=direction,
            integrator=p.integrator,
        )
    if isinstance(p, CCPRPole):
        q = _broadcast_axis_param(p.pole)
        r = _broadcast_axis_param(p.residue)
        # Raises at construction if the residue has a real part (dE/dt coupling).
        return CCPRPole(pole=complex(q[axis]), residue=complex(r[axis]), orientation=direction, integrator=p.integrator)
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

    #: Time-integrator used to discretize this pole's ODE. See the module
    #: docstring for the coefficient formulas. ``"centered_edot"`` (the default)
    #: is second-order accurate for every pole type; ``"central"`` reproduces
    #: the historical scheme and is only first order when the ``dE/dt`` coupling
    #: is non-zero; ``"bilinear"`` is unconditionally stable but makes every
    #: material implicit and does not support oriented poles.
    integrator: DispersionIntegrator = frozen_field(default="centered_edot")

    def _validate_orientation(self):
        """Normalize and validate :attr:`orientation` and :attr:`integrator`.
        Concrete pole classes call this from ``__post_init__`` (which ``autoinit``
        only invokes when defined directly on the class, not inherited)."""
        if self.integrator not in DISPERSION_INTEGRATORS:
            raise ValueError(
                f"Unknown dispersion integrator {self.integrator!r}; expected one of {DISPERSION_INTEGRATORS}."
            )
        if self.orientation is not None and self.integrator == "bilinear":
            raise NotImplementedError(
                "Oriented poles are not supported with integrator='bilinear': its non-zero c4 at every pole would "
                "turn the off-diagonal coupling into a per-cell 3x3 implicit solve. Use 'centered_edot' instead."
            )
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
        integrator: DispersionIntegrator = "centered_edot",
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
            integrator: Time integrator for the resulting pole. Defaults to
                ``"centered_edot"``; see :attr:`Pole.integrator`.

        Returns:
            CCPRPole: Equivalent pole with the ``(q, r)`` above.
        """
        import cmath

        q = complex(-damping, -resonance_frequency)
        r = 1j * amplitude * resonance_frequency * cmath.exp(1j * phase)
        return cls(pole=q, residue=r, integrator=integrator)


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

    def with_integrator(self, integrator: DispersionIntegrator) -> "DispersionModel":
        """Return a copy of this model with every pole switched to ``integrator``.

        Convenience for multi-pole fits, where setting the scheme pole by pole is
        tedious. See :attr:`Pole.integrator` for the available schemes.

        Args:
            integrator: Time integrator to apply to all poles.

        Returns:
            DispersionModel: Copy with every pole's ``integrator`` replaced.
        """
        if integrator not in DISPERSION_INTEGRATORS:
            raise ValueError(f"Unknown dispersion integrator {integrator!r}; expected one of {DISPERSION_INTEGRATORS}.")
        # ``aset`` bypasses ``__post_init__``, so re-check the one cross-field
        # constraint here rather than letting an invalid pole through silently.
        if integrator == "bilinear" and self.has_off_diagonal_coupling:
            raise NotImplementedError(
                "Oriented poles are not supported with integrator='bilinear': its non-zero c4 at every pole would "
                "turn the off-diagonal coupling into a per-cell 3x3 implicit solve. Use 'centered_edot' instead."
            )
        return DispersionModel(poles=tuple(p.aset("integrator", integrator) for p in self.poles))

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


def _scheme_coefficients(
    scheme: str,
    omega_0: float,
    gamma: float,
    coupling_sq: float,
    coupling_edot: float,
    dt: float,
) -> tuple[float, float, float, float, float]:
    """Discrete P-form coefficients ``(c1, c2, c3, c4, c5)`` of one pole on one axis.

    See the module docstring for the per-scheme formulas and their derivation.
    """
    a, b = coupling_sq, coupling_edot
    if scheme == "bilinear":
        # Trapezoidal / bilinear map s -> K (z - 1) / (z + 1). Exact at DC and
        # unconditionally stable; c4 != 0 even at b = 0.
        k = 2.0 / dt
        denom = k * k + gamma * k + omega_0**2
        c1 = 2.0 * (k * k - omega_0**2) / denom
        c2 = -(k * k - gamma * k + omega_0**2) / denom
        c3 = 2.0 * a / denom
        c4 = (a + b * k) / denom
        c5 = (a - b * k) / denom
        return c1, c2, c3, c4, c5

    # Central differences on the 2nd-order ODE; the two variants differ only in
    # how the b * dE/dt term is differenced.
    denom = 1.0 + 0.5 * gamma * dt
    c1 = (2.0 - (omega_0**2) * (dt**2)) / denom
    c2 = -(1.0 - 0.5 * gamma * dt) / denom
    if scheme == "central":
        # Forward difference b (E^{n+1} - E^n) / dt: first order whenever b != 0.
        c3 = (a * dt**2 - b * dt) / denom
        c4 = (b * dt) / denom
        c5 = 0.0
    else:  # "centered_edot"
        # Central difference b (E^{n+1} - E^{n-1}) / (2 dt): second order, and
        # half the implicit c4 of the forward difference.
        c3 = (a * dt**2) / denom
        c4 = (b * dt) / (2.0 * denom)
        c5 = -(b * dt) / (2.0 * denom)
    return c1, c2, c3, c4, c5


def _scheme_delta_coefficients(
    scheme: str,
    omega_0: float,
    gamma: float,
    coupling_sq: float,
    coupling_edot: float,
    dt: float,
) -> tuple[float, float, float, float, float]:
    r"""Delta-basis coefficients ``(a1, a0, b1, c4, b0)`` of one pole on one axis.

    The same discrete transfer function as :func:`_scheme_coefficients`, expanded
    in :math:`\zeta = z - 1` rather than :math:`z`:

    .. math::
        \chi_d(\zeta) = c_4 + \frac{b_1 \zeta + b_0}{\zeta^2 + a_1 \zeta + a_0}

    so that ``a1 = 2 - c1``, ``a0 = 1 - c1 - c2``, ``b1 = beta1`` and
    ``b0 = beta1 + beta0``. See the module docstring for why the FDTD loop stores
    and marches these instead of ``(c1, c2, beta1, c4, beta0)``.

    Every expression below is a sum of same-signed terms, so each coefficient
    carries full *relative* accuracy — the whole point of the basis change.
    Computing ``a0`` as ``1 - c1 - c2`` instead would already have lost
    :math:`(\omega_0 \Delta t)^{-2}` digits before the float32 cast.
    """
    a, b = coupling_sq, coupling_edot
    if scheme == "bilinear":
        k = 2.0 / dt
        denom = k * k + gamma * k + omega_0**2
        a1 = (2.0 * gamma * k + 4.0 * omega_0**2) / denom
        a0 = 4.0 * omega_0**2 / denom
        c3 = 2.0 * a / denom
        c4 = (a + b * k) / denom
        # b0 = c3 + c5 + (1 - a0) c4, and c3 + c5 + c4 = 4a / D_b = 2 c3 exactly:
        # the +-b*K terms of c5 and c4 cancel analytically.
        b0 = 2.0 * c3 - a0 * c4
        return a1, a0, c3 + (2.0 - a1) * c4, c4, b0

    # Central differences. D = 1 + gamma dt / 2.
    denom = 1.0 + 0.5 * gamma * dt
    a1 = (gamma * dt + (omega_0**2) * (dt**2)) / denom
    a0 = ((omega_0**2) * (dt**2)) / denom
    if scheme == "central":
        c3 = (a * dt**2 - b * dt) / denom
        c4 = (b * dt) / denom
    else:  # "centered_edot"
        c3 = (a * dt**2) / denom
        c4 = (b * dt) / (2.0 * denom)
    # For both central-difference variants c3 + c5 + c4 = a dt^2 / D: the b dt
    # terms cancel analytically, so b0 never forms that difference numerically.
    b0 = (a * dt**2) / denom - a0 * c4
    return a1, a0, c3 + (2.0 - a1) * c4, c4, b0


def to_delta_form(
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    c4: np.ndarray,
    c5: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Convert P-form recurrence coefficients to the delta basis.

    Returns ``(a1, a0, b1, c4, b0)`` with ``a1 = 2 - c1``, ``a0 = 1 - c1 - c2``,
    ``b1 = c3 + c1 c4`` and ``b0 = b1 + c5 + c2 c4`` — the arrays the FDTD loop
    stores. See the module docstring for the derivation.

    Prefer :func:`compute_pole_delta_coefficients_per_axis` /
    :func:`compute_pole_delta_coefficients_tensor`, which build the same values
    from the pole parameters via cancellation-free closed forms. This converter
    is exact in float64 but loses :math:`(\omega_0 \Delta t)^{-2}` digits of
    ``a0``, which matters once the result is stored in float32.

    .. warning::
        Do **not** apply this to zero-padded pole slots: all-zero P-form
        coefficients map to ``(a1, a0) = (2, 1)``, whereas padded slots must stay
        all-zero to remain inert. Zero-pad the delta coefficients instead.

    Args:
        c1: P-form coefficient array.
        c2: P-form coefficient array.
        c3: P-form coefficient array.
        c4: P-form coefficient array (the implicit ``E^{n+1}`` coupling).
        c5: P-form coefficient array (the ``E^{n-1}`` coupling).

    Returns:
        ``(a1, a0, b1, c4, b0)`` — ``c4`` unchanged.
    """
    b1 = c3 + c1 * c4
    b0 = b1 + (c5 + c2 * c4)
    return 2.0 - c1, 1.0 - c1 - c2, b1, c4, b0


def to_observer_form(
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    c4: np.ndarray,
    c5: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Convert P-form recurrence coefficients to the observer-canonical form.

    The P-form ``p^{n+1} = c1 p^n + c2 p^{n-1} + c3 E^n + c4 E^{n+1} + c5 E^{n-1}``
    needs an ``E^{n-1}`` array when ``c5 != 0`` — an extra full field history per
    pole. The observer-canonical realization of the identical transfer function
    :math:`\chi_d(z) = (c_4 z^2 + c_3 z + c_5)/(z^2 - c_1 z - c_2)`,

    .. math::
        x_1^{n+1} &= c_1 x_1^n + x_2^n + \beta_1 E^n \\
        x_2^{n+1} &= c_2 x_1^n + \beta_0 E^n \\
        p^{n+1}   &= x_1^{n+1} + c_4 E^{n+1}

    with :math:`\beta_1 = c_3 + c_1 c_4` and :math:`\beta_0 = c_5 + c_2 c_4`,
    carries every scheme with two state levels, no field history, and a purely
    per-cell polarization update.

    Args:
        c1: P-form coefficient array.
        c2: P-form coefficient array.
        c3: P-form coefficient array.
        c4: P-form coefficient array (the implicit ``E^{n+1}`` coupling).
        c5: P-form coefficient array (the ``E^{n-1}`` coupling).

    Returns:
        ``(c1, c2, beta1, c4, beta0)`` — ``c1``, ``c2`` and ``c4`` unchanged.
    """
    beta1 = c3 + c1 * c4
    beta0 = c5 + c2 * c4
    return c1, c2, beta1, c4, beta0


def compute_pole_coefficients_per_axis(
    poles: tuple[Pole, ...],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the per-axis discrete-time ADE recurrence coefficients.

    For each pole and grid axis, returns the P-form coefficients
    ``(c1, c2, c3, c4, c5)`` of

    :math:`p_p^{n+1} = c_1 p_p^n + c_2 p_p^{n-1} + c_3 E^n + c_4 E^{n+1} + c_5 E^{n-1}`

    under that pole's :attr:`Pole.integrator`. See the module docstring for the
    per-scheme formulas; ``a = coupling_sq`` is the ``E`` coupling and
    ``b = coupling_edot`` the ``dE/dt`` coupling.

    For isotropic poles the three axis columns are identical. For Lorentz and
    Drude poles ``b = 0``, so the two central-difference schemes coincide and
    give ``c4 = c5 = 0`` with ``c3 = a dt^2 / D``.

    Use :func:`compute_pole_delta_coefficients_per_axis` for the coefficients the
    FDTD loop actually stores and marches (:func:`to_observer_form` and
    :func:`to_delta_form` convert these in place, at lower precision).

    Args:
        poles: Tuple of poles (may be empty).
        dt: Simulation time step (seconds).

    Returns:
        Five ``numpy`` arrays of shape ``(len(poles), 3)`` with ``c1``, ``c2``,
        ``c3``, ``c4``, ``c5`` per pole and axis. For an empty pole tuple,
        returns five ``(0, 3)`` arrays.
    """
    n = len(poles)
    c1 = np.zeros((n, 3), dtype=np.float64)
    c2 = np.zeros((n, 3), dtype=np.float64)
    c3 = np.zeros((n, 3), dtype=np.float64)
    c4 = np.zeros((n, 3), dtype=np.float64)
    c5 = np.zeros((n, 3), dtype=np.float64)
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
            omega0_dt = omega_0[ax] * dt
            # The stability bound only binds on axes where the pole actually
            # couples. A zero-coupling axis (e.g. a Lorentz pole with
            # delta_epsilon = 0 there, the documented way to express an absent
            # resonance) has c3 = c4 = c5 = 0, so its polarization stays
            # identically zero and its unused omega_0 / gamma are irrelevant.
            axis_active = coupling_sq[ax] != 0.0 or coupling_edot[ax] != 0.0
            # gamma * dt is unconstrained: |c2| < 1 holds for every gamma * dt > 0
            # (c2 merely passes through zero at gamma * dt = 2), so the only
            # binding forward bound is omega_0 * dt < 2.
            if axis_active and omega0_dt >= 2.0 and p.integrator in _CONDITIONALLY_STABLE:
                axis_note = "" if p.is_isotropic else f" on axis {'xyz'[ax]}"
                raise ValueError(
                    f"Pole {i} ({type(p).__name__}) has omega_0 * dt = {omega0_dt:.4g} >= 2{axis_note}; "
                    f"the {p.integrator!r} ADE recurrence roots leave the unit circle (requires "
                    "omega_0 * dt < 2, physically omega_0 * dt << 1). Lower the resonance frequency, "
                    "reduce the time step, or use integrator='bilinear' (unconditionally stable)."
                )
            vals = _scheme_coefficients(p.integrator, omega_0[ax], gamma[ax], coupling_sq[ax], coupling_edot[ax], dt)
            c1[i, ax], c2[i, ax], c3[i, ax], c4[i, ax], c5[i, ax] = vals
    return c1, c2, c3, c4, c5


def compute_pole_coefficients(
    poles: tuple[Pole, ...],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the discrete-time ADE recurrence coefficients of isotropic poles.

    Scalar-per-pole variant of :func:`compute_pole_coefficients_per_axis` (see
    there for the coefficient definitions). Raises ``ValueError`` when any
    pole has per-axis parameters — use the per-axis function for those.

    Args:
        poles: Tuple of isotropic poles (may be empty).
        dt: Simulation time step (seconds).

    Returns:
        Five ``numpy`` arrays of shape ``(len(poles),)`` with ``c1``, ``c2``,
        ``c3``, ``c4``, ``c5``. For an empty pole tuple, returns five empty
        arrays.
    """
    for i, p in enumerate(poles):
        if not p.is_isotropic:
            raise ValueError(
                f"Pole {i} ({type(p).__name__}) has per-axis parameters or an orientation; "
                "use compute_pole_coefficients_per_axis or compute_pole_coefficients_tensor instead."
            )
    c1, c2, c3, c4, c5 = compute_pole_coefficients_per_axis(poles, dt)
    return c1[:, 0], c2[:, 0], c3[:, 0], c4[:, 0], c5[:, 0]


def compute_pole_coefficients_tensor(
    poles: tuple[Pole, ...],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ADE recurrence coefficients with full 3x3 coupling tensors.

    Generalizes :func:`compute_pole_coefficients_per_axis` to oriented poles:
    the recurrence coefficients ``c1``/``c2`` stay per-axis (uniform for an
    oriented pole, whose ``omega_0``/``gamma`` are scalars), while the field
    couplings ``c3``/``c4``/``c5`` become row-major 3x3 tensors per pole —
    proportional to ``u u^T`` for a pole oriented along ``u``, diagonal for
    per-axis and isotropic poles.

    Oriented poles are restricted to ``integrator="central"`` /
    ``"centered_edot"`` and have no ``dE/dt`` coupling, so their ``c4`` and
    ``c5`` tensors are identically zero and the two schemes agree.

    Args:
        poles: Tuple of poles (may be empty). Oriented poles must have a
            non-negative coupling ``K`` (passivity of the ``K u u^T`` tensor).
        dt: Simulation time step (seconds).

    Returns:
        Five ``numpy`` arrays: ``c1``, ``c2`` of shape ``(len(poles), 3)`` and
        ``c3``, ``c4``, ``c5`` of shape ``(len(poles), 9)``.
    """
    n = len(poles)
    c1 = np.zeros((n, 3), dtype=np.float64)
    c2 = np.zeros((n, 3), dtype=np.float64)
    c3 = np.zeros((n, 9), dtype=np.float64)
    c4 = np.zeros((n, 9), dtype=np.float64)
    c5 = np.zeros((n, 9), dtype=np.float64)
    for i, p in enumerate(poles):
        gamma = p.gamma_axes
        omega_0 = p.omega_0_axes
        coupling_sq = p.coupling_sq_axes
        coupling_edot = p.coupling_edot_axes
        for ax in range(3):
            omega0_dt = omega_0[ax] * dt
            # Same forward-stability bound (and same zero-coupling exemption) as
            # compute_pole_coefficients_per_axis.
            axis_active = coupling_sq[ax] != 0.0 or coupling_edot[ax] != 0.0
            if axis_active and omega0_dt >= 2.0 and p.integrator in _CONDITIONALLY_STABLE:
                raise ValueError(
                    f"Pole {i} ({type(p).__name__}) has omega_0 * dt = {omega0_dt:.4g} >= 2; "
                    f"the {p.integrator!r} ADE recurrence roots leave the unit circle (requires "
                    "omega_0 * dt < 2, physically omega_0 * dt << 1). Lower the resonance frequency, "
                    "reduce the time step, or use integrator='bilinear' (unconditionally stable)."
                )
            row = _scheme_coefficients(p.integrator, omega_0[ax], gamma[ax], coupling_sq[ax], coupling_edot[ax], dt)
            c1[i, ax], c2[i, ax] = row[0], row[1]
        if p.is_oriented:
            if coupling_sq[0] < 0.0:
                raise ValueError(
                    f"Pole {i} ({type(p).__name__}) has negative coupling K = {coupling_sq[0]:.4g}; "
                    "oriented poles require K >= 0 so the coupling tensor K u u^T stays positive "
                    "semi-definite (passivity)."
                )
            assert p.orientation is not None
            u = np.asarray(p.orientation, dtype=np.float64)
            # coupling_edot is zero for oriented poles (validated at construction),
            # so c4 and c5 stay zero and only the c3 tensor is populated.
            _, _, c3_scalar, _, _ = _scheme_coefficients(p.integrator, omega_0[0], gamma[0], coupling_sq[0], 0.0, dt)
            c3[i] = (c3_scalar * np.outer(u, u)).reshape(-1)
        else:
            for ax in range(3):
                _, _, c3_ax, c4_ax, c5_ax = _scheme_coefficients(
                    p.integrator, omega_0[ax], gamma[ax], coupling_sq[ax], coupling_edot[ax], dt
                )
                c3[i, 4 * ax] = c3_ax
                c4[i, 4 * ax] = c4_ax
                c5[i, 4 * ax] = c5_ax
    return c1, c2, c3, c4, c5


def compute_pole_delta_coefficients_per_axis(
    poles: tuple[Pole, ...],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Per-axis delta-basis coefficients ``(a1, a0, b1, c4, b0)`` — what the loop stores.

    Delta-basis counterpart of :func:`compute_pole_coefficients_per_axis`:
    the same discrete transfer function expanded in :math:`\zeta = z - 1`, so
    ``a1 = 2 - c1`` and ``a0 = 1 - c1 - c2`` are represented directly rather than
    as differences of near-fixed :math:`O(1)` numbers. See the module docstring
    for why this is what makes float32 dispersion usable.

    Shares every stability guard with the P-form function.

    Args:
        poles: Tuple of non-oriented poles (may be empty).
        dt: Simulation time step (seconds).

    Returns:
        Five ``numpy`` arrays of shape ``(len(poles), 3)`` with ``a1``, ``a0``,
        ``b1``, ``c4``, ``b0`` per pole and axis.
    """
    # Delegate the guards (and the oriented-pole rejection) to the P-form path,
    # then rebuild from the cancellation-free closed forms. Host-side numpy on
    # (num_poles, 3) arrays, so the duplicated work is free.
    compute_pole_coefficients_per_axis(poles, dt)
    n = len(poles)
    out = [np.zeros((n, 3), dtype=np.float64) for _ in range(5)]
    for i, p in enumerate(poles):
        omega_0, gamma = p.omega_0_axes, p.gamma_axes
        coupling_sq, coupling_edot = p.coupling_sq_axes, p.coupling_edot_axes
        for ax in range(3):
            vals = _scheme_delta_coefficients(
                p.integrator, omega_0[ax], gamma[ax], coupling_sq[ax], coupling_edot[ax], dt
            )
            for arr, v in zip(out, vals):
                arr[i, ax] = v
    return out[0], out[1], out[2], out[3], out[4]


def compute_pole_delta_coefficients_tensor(
    poles: tuple[Pole, ...],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Delta-basis coefficients with full 3x3 field-coupling tensors.

    Delta-basis counterpart of :func:`compute_pole_coefficients_tensor` (see
    there for the tensor layout, and the module docstring for the basis change).
    The oscillator coefficients ``a1``/``a0`` stay per-axis while the field
    couplings ``b1``/``c4``/``b0`` become row-major 3x3 tensors per pole.

    Oriented poles have no ``dE/dt`` coupling, so their ``c4`` tensor is
    identically zero and ``b0`` equals ``b1``.

    Args:
        poles: Tuple of poles (may be empty).
        dt: Simulation time step (seconds).

    Returns:
        Five ``numpy`` arrays: ``a1``, ``a0`` of shape ``(len(poles), 3)`` and
        ``b1``, ``c4``, ``b0`` of shape ``(len(poles), 9)``.
    """
    compute_pole_coefficients_tensor(poles, dt)  # guards + passivity checks
    n = len(poles)
    a1 = np.zeros((n, 3), dtype=np.float64)
    a0 = np.zeros((n, 3), dtype=np.float64)
    b1 = np.zeros((n, 9), dtype=np.float64)
    c4 = np.zeros((n, 9), dtype=np.float64)
    b0 = np.zeros((n, 9), dtype=np.float64)
    for i, p in enumerate(poles):
        omega_0, gamma = p.omega_0_axes, p.gamma_axes
        coupling_sq, coupling_edot = p.coupling_sq_axes, p.coupling_edot_axes
        for ax in range(3):
            row = _scheme_delta_coefficients(
                p.integrator, omega_0[ax], gamma[ax], coupling_sq[ax], coupling_edot[ax], dt
            )
            a1[i, ax], a0[i, ax] = row[0], row[1]
        if p.is_oriented:
            assert p.orientation is not None
            u = np.asarray(p.orientation, dtype=np.float64)
            # coupling_edot is zero for oriented poles (validated at construction),
            # so c4 stays zero and b0 coincides with b1.
            _, _, b1_scalar, _, b0_scalar = _scheme_delta_coefficients(
                p.integrator, omega_0[0], gamma[0], coupling_sq[0], 0.0, dt
            )
            uu = np.outer(u, u).reshape(-1)
            b1[i] = b1_scalar * uu
            b0[i] = b0_scalar * uu
        else:
            for ax in range(3):
                _, _, b1_ax, c4_ax, b0_ax = _scheme_delta_coefficients(
                    p.integrator, omega_0[ax], gamma[ax], coupling_sq[ax], coupling_edot[ax], dt
                )
                b1[i, 4 * ax] = b1_ax
                c4[i, 4 * ax] = c4_ax
                b0[i, 4 * ax] = b0_ax
    return a1, a0, b1, c4, b0


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
    a1: jax.Array,
    a0: jax.Array,
    b1: jax.Array,
    omega: float,
    dt: float,
    c4: jax.Array | None = None,
    b0: jax.Array | None = None,
) -> jax.Array:
    """Evaluate the per-cell complex susceptibility from the stored ADE coefficients.

    Evaluates the **discrete** transfer function each pole actually realizes on
    the grid, in the delta basis the coefficients are stored in,

    .. math::
        \\chi_{d,p}(\\omega) = c_4
            + \\frac{b_1 \\zeta + b_0}{\\zeta^2 + a_1 \\zeta + a_0},
        \\qquad \\zeta = e^{-i \\omega \\Delta t} - 1

    and sums over the leading pole axis. This is exact rather than approximate:
    the discrete Ampere law consumes only
    :math:`\\varepsilon_\\infty (E^{n+1} - E^n) + \\sum_p (p_p^{n+1} - p_p^n)`, so
    :math:`\\varepsilon_\\infty + \\sum_p \\chi_{d,p}` *is* the permittivity the
    simulation realizes. It differs from the continuum
    :math:`\\chi(\\omega)` by the scheme's :math:`O((\\omega \\Delta t)^2)`
    truncation, and is scheme-agnostic — it needs no knowledge of which
    integrator produced the coefficients.

    Evaluating in :math:`\\zeta` also keeps the *evaluation* well conditioned:
    at low frequency :math:`\\zeta \\to 0`, so the denominator is a sum of small
    terms rather than a cancellation among :math:`O(1)` ones.

    Cells with no pole have all-zero coefficients and contribute exactly zero
    (the denominator is :math:`\\zeta^2 \\neq 0` there, so no masking is
    required).

    Args:
        a1: ADE coefficient array ``a1 = 2 - c1`` of shape ``(num_poles, ...)``
            where the trailing axes are the spatial (and optional component)
            dimensions.
        a0: ADE coefficient array ``a0 = 1 - c1 - c2``, shape
            ``(num_poles, ...)``.
        b1: Field coupling ``b1 = beta1 = c3 + c1 c4``, shape
            ``(num_poles, ...)``.
        omega: Angular frequency (rad/s) at which to evaluate the
            susceptibility.
        dt: Simulation time step (seconds) used to derive the coefficients.
        c4: Optional implicit ``E^{n+1}`` coupling, shape ``(num_poles, ...)``.
            ``None`` is treated as all-zero.
        b0: Optional field coupling ``b0 = beta1 + beta0``, shape
            ``(num_poles, ...)``. ``None`` means "equal to ``b1``", which is
            exactly the case whenever no pole is implicit (``c4 = c5 = 0``
            implies ``beta0 = 0``).

    Returns:
        Complex ``jax.Array`` with shape ``b1.shape[1:]`` — the total
        susceptibility summed over all poles, in every cell.
    """
    a1 = jnp.asarray(a1)
    a0 = jnp.asarray(a0)
    b1 = jnp.asarray(b1)
    c4 = jnp.zeros_like(b1) if c4 is None else jnp.asarray(c4)
    b0 = b1 if b0 is None else jnp.asarray(b0)
    if a1.ndim >= 2 and b1.ndim >= 2 and b1.shape[1] == 9:
        # 9-component coupling (oriented poles): the recurrence coefficients
        # expand so entry (i, j) uses the oscillator of row i; the result is
        # the per-entry chi_ij with shape (9, *spatial).
        a1 = _expand_recurrence_to_coupling(a1, 9)
        a0 = _expand_recurrence_to_coupling(a0, 9)

    zeta = jnp.exp(-1j * omega * dt) - 1.0
    denom = zeta * zeta + a1 * zeta + a0
    chi_per_pole = c4 + (b1 * zeta + b0) / denom
    return jnp.sum(chi_per_pole, axis=0)


def compute_eps_spectrum_from_coefficients(
    a1: jax.Array | np.ndarray,
    a0: jax.Array | np.ndarray,
    b1: jax.Array | np.ndarray,
    inv_eps_inf: jax.Array | np.ndarray,
    omegas: np.ndarray,
    dt: float,
    weights: np.ndarray | None = None,
    c4: jax.Array | np.ndarray | None = None,
    b0: jax.Array | np.ndarray | None = None,
) -> np.ndarray:
    """Spatially-averaged complex permittivity spectrum for a block of cells.

    For each angular frequency in ``omegas``, evaluates the per-cell complex
    permittivity :math:`\\varepsilon_\\infty + \\chi_d(\\omega)` — with
    :math:`\\chi_d` the discrete transfer function of
    :func:`susceptibility_from_coefficients` — and averages over the spatial
    axes (uniformly or with supplied weights).

    This is the broadband generalization of the single-frequency
    :func:`effective_inv_permittivity` used for carrier-frequency impedance
    matching — callers that need a frequency-dependent impedance (e.g. for
    a convolution-based broadband source correction) use this to build the
    :math:`\\varepsilon(\\omega)` spectrum that feeds
    :func:`compute_impedance_corrected_temporal_profile`.

    Args:
        a1: ADE coefficient array ``a1 = 2 - c1`` of shape
            ``(num_poles, num_components, *spatial)`` as stored on
            :class:`~fdtdx.fdtd.container.ArrayContainer`, with
            ``num_components in (1, 3)`` (the material-component axis; size 3
            for per-axis anisotropic dispersion). Anisotropic components are
            averaged, mirroring the ``inv_eps_inf`` reduction.
        a0: ADE coefficient array ``a0 = 1 - c1 - c2``, same shape as ``a1``.
        b1: Field coupling array ``b1 = beta1``, same shape as ``a1``.
        inv_eps_inf: Per-cell inverse of the high-frequency permittivity,
            shape ``(num_components, *spatial)`` with
            ``num_components in (1, 3, 9)``. For anisotropic tensors
            (9 components) only the diagonal entries are used.
        omegas: 1D array of angular frequencies (rad/s) to evaluate at.
        dt: Simulation time step (seconds) used to derive the coefficients.
        weights: Optional spatial weights with the same shape as the
            trailing axes of ``c1``. If ``None``, uniform averaging.
        c4: Optional implicit ``E^{n+1}`` coupling array. ``None`` is all-zero.
        b0: Optional field coupling array ``b0 = beta1 + beta0``. ``None`` means
            "equal to ``b1``" (exact whenever no pole is implicit).

    Returns:
        Complex numpy array of shape ``(len(omegas),)`` — the volume-averaged
        :math:`\\varepsilon(\\omega)` at each requested frequency.
    """
    a1_np = np.asarray(a1)
    a0_np = np.asarray(a0)
    b1_np = np.asarray(b1)
    c4_np = np.zeros_like(b1_np) if c4 is None else np.asarray(c4)
    b0_np = b1_np if b0 is None else np.asarray(b0)
    inv_eps_np = np.asarray(inv_eps_inf)
    omegas_np = np.asarray(omegas, dtype=np.float64)
    if b1_np.ndim >= 2 and b1_np.shape[1] == 9 and a1_np.shape[1] == 3:
        # 9-component coupling (oriented poles): entry 3i+j uses oscillator row i.
        a1_np = np.repeat(a1_np, 3, axis=1)
        a0_np = np.repeat(a0_np, 3, axis=1)

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
    # Evaluates the same discrete chi_d(zeta) as susceptibility_from_coefficients;
    # pole-free slots have all-zero coefficients and a denominator of zeta^2 != 0,
    # so they contribute exactly zero without masking.
    zeta = (np.exp(-1j * (omegas_np * dt)) - 1.0).reshape((-1,) + (1,) * a1_np.ndim)
    denom = zeta * zeta + a1_np[None, ...] * zeta + a0_np[None, ...]
    chi_per_pole = c4_np[None, ...] + (b1_np[None, ...] * zeta + b0_np[None, ...]) / denom
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
    a1: jax.Array | None,
    a0: jax.Array | None,
    b1: jax.Array | None,
    omega: float,
    dt: float,
    c4: jax.Array | None = None,
    b0: jax.Array | None = None,
) -> jax.Array:
    """Per-cell real inverse permittivity :math:`1/\\text{Re}(\\varepsilon_\\infty + \\chi(\\omega))`.

    Sources in FDTDX use a real wave impedance, so only the real part of
    ``ε∞ + χ(ω)`` enters the injected amplitude. The imaginary part describes
    absorption, which is already handled by the ADE update loop (injecting it
    into the source amplitude would double-count).

    Cells with no pole (all-zero coefficients) contribute :math:`\\chi = 0` so
    their ``inv_eps`` is returned unchanged.

    Args:
        inv_eps: Per-cell :math:`1/\\varepsilon_\\infty` array. Typically
            has shape ``(num_components, ...)``; any shape broadcast-compatible
            with ``a1.shape[1:]`` works.
        a1: Delta-basis coefficient ``a1 = 2 - c1``, shape
            ``(num_poles, ...)`` or ``None``.
        a0: Delta-basis coefficient ``a0 = 1 - c1 - c2``, shape
            ``(num_poles, ...)`` or ``None``.
        b1: Delta-basis field coupling ``b1 = beta1``, shape
            ``(num_poles, ...)`` or ``None``.
        omega: Angular frequency (rad/s) at which to evaluate.
        dt: Simulation time step (seconds).
        c4: Optional implicit ``E^{n+1}`` coupling array.
        b0: Optional delta-basis field coupling ``b0``; ``None`` means equal to ``b1``.

    Returns:
        Real-valued ``jax.Array`` with the same shape and dtype as
        ``inv_eps``. If any of ``a1``/``a0``/``b1`` is ``None``, returns
        ``inv_eps`` unchanged.
    """
    inv_eps_arr = jnp.asarray(inv_eps)
    coupling_c = jnp.asarray(b1).shape[1] if b1 is not None and jnp.asarray(b1).ndim >= 2 else 1
    if inv_eps_arr.shape[0] == 9 or coupling_c == 9:
        # Tensor path: reconstruct the real permittivity matrix per cell, add
        # the (possibly off-diagonal) susceptibility, and invert per cell.
        # Elementwise 1/inv_eps would divide by the zero off-diagonal entries.
        eps_mat = jnp.real(_eps_matrix_from_inv(inv_eps_arr))
        if a1 is not None and a0 is not None and b1 is not None:
            chi = susceptibility_from_coefficients(a1=a1, a0=a0, b1=b1, omega=omega, dt=dt, c4=c4, b0=b0)
            eps_mat = eps_mat + jnp.real(_tensor_from_components(chi))
        inv_eff = _invert_3x3_matrix_field(eps_mat)
        return inv_eff.reshape(9, *inv_eff.shape[2:]).astype(inv_eps_arr.dtype)

    if a1 is None or a0 is None or b1 is None:
        return inv_eps

    chi = susceptibility_from_coefficients(a1=a1, a0=a0, b1=b1, omega=omega, dt=dt, c4=c4, b0=b0)
    eps_inf = 1.0 / inv_eps_arr
    eps_eff = eps_inf + jnp.real(chi)
    return (1.0 / eps_eff).astype(inv_eps_arr.dtype)


def effective_complex_inv_permittivity(
    inv_eps: jax.Array,
    omega: float,
    dt: float,
    a1: jax.Array | None = None,
    a0: jax.Array | None = None,
    b1: jax.Array | None = None,
    electric_conductivity: jax.Array | None = None,
    conductivity_spacing: float | None = None,
    c4: jax.Array | None = None,
    b0: jax.Array | None = None,
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
      ADE coefficients (omitted when ``a1``/``a0``/``b1`` are ``None``), and
    * the conductivity loss :math:`i\,\sigma_\text{phys} / (\varepsilon_0 \omega)`,
      where :math:`\sigma_\text{phys} = \sigma_\text{array} / \Delta` recovers the
      physical S/m value from the resolution-scaled ``electric_conductivity``
      array (``conductivity_spacing`` is the scaling factor
      :math:`\Delta = c_0 \Delta t / S` applied at initialization).

    Args:
        inv_eps: Per-cell ``1/eps_inf`` (real). Shape ``(num_components, ...)``.
        omega: Angular frequency (rad/s).
        dt: Simulation time step (seconds).
        a1: Delta-basis coefficient ``a1 = 2 - c1``, shape
            ``(num_poles, ...)`` or ``None``.
        a0: Delta-basis coefficient ``a0 = 1 - c1 - c2``, shape
            ``(num_poles, ...)`` or ``None``.
        b1: Delta-basis field coupling ``b1 = beta1``, shape
            ``(num_poles, ...)`` or ``None``.
        electric_conductivity: Resolution-scaled conductivity array, or ``None``.
        conductivity_spacing: Scaling factor used to recover the physical
            conductivity. Required when ``electric_conductivity`` is given.
        c4: Optional implicit ``E^{n+1}`` coupling array.
        b0: Optional delta-basis field coupling ``b0``; ``None`` means equal to ``b1``.

    Returns:
        Complex ``jax.Array`` broadcasting ``inv_eps`` against the loss terms.
    """
    inv_eps = jnp.asarray(inv_eps)
    complex_dtype = jnp.complex128 if inv_eps.dtype == jnp.float64 else jnp.complex64
    coupling_c = jnp.asarray(b1).shape[1] if b1 is not None and jnp.asarray(b1).ndim >= 2 else 1
    if inv_eps.shape[0] == 9 or coupling_c == 9:
        # Diagonal reduction for the mode solver: off-diagonal permittivity and
        # susceptibility entries are dropped, so modal geometry in monoclinic /
        # rotated media is a diagonal approximation.
        eps_mat = _eps_matrix_from_inv(inv_eps)
        eps = jnp.stack([eps_mat[0, 0], eps_mat[1, 1], eps_mat[2, 2]], axis=0).astype(complex_dtype)
        if a1 is not None and a0 is not None and b1 is not None:
            chi = susceptibility_from_coefficients(a1=a1, a0=a0, b1=b1, omega=omega, dt=dt, c4=c4, b0=b0)
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
    if a1 is not None and a0 is not None and b1 is not None:
        eps = eps + susceptibility_from_coefficients(a1=a1, a0=a0, b1=b1, omega=omega, dt=dt, c4=c4, b0=b0)
    if electric_conductivity is not None:
        if conductivity_spacing is None:
            raise ValueError("conductivity_spacing is required when electric_conductivity is given.")
        sigma_phys = jnp.asarray(electric_conductivity) / conductivity_spacing
        eps = eps + 1j * sigma_phys / (omega * eps0)
    return 1.0 / eps
