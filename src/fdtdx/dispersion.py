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

Stability requires :math:`\\gamma \\Delta t < 2`; in the physical regime
of interest :math:`\\gamma \\Delta t \\ll 1`, so :math:`c_2 \\approx -1`,
which makes the backward (reverse-time) recurrence numerically stable
after algebraic inversion.
"""

from __future__ import annotations

import numpy as np

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field


@autoinit
class Pole(TreeClass):
    """Single 2nd-order ADE pole.

    Models Lorentz and Drude poles through a unified
    ``(omega_0, gamma, coupling_sq)`` parameterization.
    """

    #: Resonance angular frequency omega_0 (rad/s). Set to 0 for a pure Drude pole.
    omega_0: float = frozen_field(default=0.0)

    #: Damping rate gamma (rad/s).
    gamma: float = frozen_field(default=0.0)

    #: Effective squared coupling frequency K (rad^2/s^2):
    #: ``delta_epsilon * omega_0**2`` for a Lorentz pole,
    #: ``omega_p**2`` for a Drude pole.
    coupling_sq: float = frozen_field(default=0.0)


def lorentz_pole(omega_0: float, gamma: float, delta_epsilon: float) -> Pole:
    """Build a Lorentz pole.

    Args:
        omega_0: Resonance angular frequency (rad/s). Must be > 0.
        gamma: Damping rate (rad/s). Must be >= 0.
        delta_epsilon: Oscillator strength (dimensionless). The zero-frequency
            contribution to susceptibility is ``delta_epsilon``.

    Returns:
        Pole: A pole with ``coupling_sq = delta_epsilon * omega_0**2``.
    """
    if omega_0 <= 0.0:
        raise ValueError(f"Lorentz pole requires omega_0 > 0, got {omega_0}")
    if gamma < 0.0:
        raise ValueError(f"Lorentz pole requires gamma >= 0, got {gamma}")
    return Pole(
        omega_0=float(omega_0),
        gamma=float(gamma),
        coupling_sq=float(delta_epsilon) * float(omega_0) ** 2,
    )


def drude_pole(omega_p: float, gamma: float) -> Pole:
    """Build a Drude pole.

    Args:
        omega_p: Plasma angular frequency (rad/s). Must be > 0.
        gamma: Damping rate (rad/s). Must be >= 0.

    Returns:
        Pole: A pole with ``omega_0 = 0`` and ``coupling_sq = omega_p**2``.
    """
    if omega_p <= 0.0:
        raise ValueError(f"Drude pole requires omega_p > 0, got {omega_p}")
    if gamma < 0.0:
        raise ValueError(f"Drude pole requires gamma >= 0, got {gamma}")
    return Pole(
        omega_0=0.0,
        gamma=float(gamma),
        coupling_sq=float(omega_p) ** 2,
    )


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

    def susceptibility(self, omega: complex | float) -> complex:
        """Evaluate the complex susceptibility :math:`\\chi(\\omega)`.

        Uses the ``exp(-i omega t)`` Fourier convention (damping appears
        with a ``-i gamma omega`` term in the Lorentz denominator).

        Args:
            omega: Angular frequency (rad/s).

        Returns:
            complex: :math:`\\chi(\\omega) = \\sum_p \\chi_p(\\omega)`.
        """
        w = complex(omega)
        total = 0.0 + 0.0j
        for p in self.poles:
            denom = p.omega_0**2 - w * w - 1j * p.gamma * w
            total = total + p.coupling_sq / denom
        return total

    def permittivity(self, omega: complex | float, eps_inf: float = 1.0) -> complex:
        """Complex relative permittivity :math:`\\varepsilon(\\omega) = \\varepsilon_\\infty + \\chi(\\omega)`.

        Args:
            omega: Angular frequency (rad/s).
            eps_inf: High-frequency permittivity. Defaults to 1.0 (vacuum).

        Returns:
            complex: Relative permittivity at ``omega``.
        """
        return eps_inf + self.susceptibility(omega)


def compute_pole_coefficients(
    poles: tuple[Pole, ...],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the discrete-time ADE recurrence coefficients.

    For each pole, returns ``(c1, c2, c3)`` with

    .. math::
        c_1 = \\frac{2 - \\omega_0^2 \\Delta t^2}{1 + \\gamma \\Delta t / 2}, \\quad
        c_2 = -\\frac{1 - \\gamma \\Delta t / 2}{1 + \\gamma \\Delta t / 2}, \\quad
        c_3 = \\frac{K \\Delta t^2}{1 + \\gamma \\Delta t / 2}.

    The recurrence is
    :math:`p_p^{n+1} = c_1 p_p^n + c_2 p_p^{n-1} + c_3 E^n`.

    Args:
        poles: Tuple of poles (may be empty).
        dt: Simulation time step (seconds).

    Returns:
        Three ``numpy`` arrays of shape ``(len(poles),)`` with ``c1``, ``c2``,
        ``c3``. For an empty pole tuple, returns three empty arrays.
    """
    n = len(poles)
    c1 = np.zeros(n, dtype=np.float64)
    c2 = np.zeros(n, dtype=np.float64)
    c3 = np.zeros(n, dtype=np.float64)
    for i, p in enumerate(poles):
        denom = 1.0 + 0.5 * p.gamma * dt
        c1[i] = (2.0 - (p.omega_0**2) * (dt**2)) / denom
        c2[i] = -(1.0 - 0.5 * p.gamma * dt) / denom
        c3[i] = (p.coupling_sq * dt**2) / denom
    return c1, c2, c3
