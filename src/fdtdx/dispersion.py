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

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field


@autoinit
class Pole(TreeClass, ABC):
    """Abstract base class for a single 2nd-order ADE pole.

    Concrete subclasses store physically-meaningful parameters
    (e.g. ``delta_epsilon`` for Lorentz, ``omega_p`` for Drude) and
    expose the unified ``(omega_0, gamma, coupling_sq)`` triplet the
    FDTD loop needs via abstract properties. New pole types can
    subclass :class:`Pole` as long as they fit the 2nd-order
    ODE form.
    """

    @property
    @abstractmethod
    def omega_0(self) -> float:
        """Resonance angular frequency (rad/s). Zero for pure Drude poles."""
        raise NotImplementedError

    @property
    @abstractmethod
    def gamma(self) -> float:
        """Damping rate (rad/s)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def coupling_sq(self) -> float:
        """Effective squared coupling frequency ``K`` (rad^2/s^2).

        ``delta_epsilon * omega_0**2`` for a Lorentz pole and
        ``omega_p**2`` for a Drude pole.
        """
        raise NotImplementedError


@autoinit
class LorentzPole(Pole):
    """Lorentz pole parameterised by its physical constants.

    The contribution to the susceptibility is

    .. math::
        \\chi(\\omega) = \\frac{\\Delta\\varepsilon \\cdot \\omega_0^2}{\\omega_0^2 - \\omega^2 - i\\gamma\\omega}.
    """

    #: Resonance angular frequency (rad/s). Must be > 0.
    resonance_frequency: float = frozen_field()

    #: Damping rate (rad/s). Must be >= 0.
    damping: float = frozen_field()

    #: Oscillator strength (dimensionless); the zero-frequency
    #: contribution to the susceptibility.
    delta_epsilon: float = frozen_field()

    @property
    def omega_0(self) -> float:
        return float(self.resonance_frequency)

    @property
    def gamma(self) -> float:
        return float(self.damping)

    @property
    def coupling_sq(self) -> float:
        return float(self.delta_epsilon) * float(self.resonance_frequency) ** 2


@autoinit
class DrudePole(Pole):
    """Drude pole parameterised by its physical constants.

    The contribution to the susceptibility is

    .. math::
        \\chi(\\omega) = -\\frac{\\omega_p^2}{\\omega^2 + i\\gamma\\omega},

    equivalent to a Lorentz pole with ``omega_0 = 0``.
    """

    #: Plasma angular frequency (rad/s). Must be > 0.
    plasma_frequency: float = frozen_field()

    #: Damping rate (rad/s). Must be >= 0.
    damping: float = frozen_field()

    @property
    def omega_0(self) -> float:
        return 0.0

    @property
    def gamma(self) -> float:
        return float(self.damping)

    @property
    def coupling_sq(self) -> float:
        return float(self.plasma_frequency) ** 2


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


def susceptibility_from_coefficients(
    c1: jax.Array,
    c2: jax.Array,
    c3: jax.Array,
    omega: float,
    dt: float,
) -> jax.Array:
    """Evaluate the per-cell complex susceptibility :math:`\\chi(\\omega)` from
    the stored ADE recurrence coefficients.

    The coefficient arrays have shape ``(num_poles, ...)`` where the trailing
    axes are the spatial (and optional component) dimensions. The inversion

    .. math::
        \\gamma \\Delta t        &= \\frac{2 (1 + c_2)}{1 - c_2},\\\\
        \\omega_0^2 \\Delta t^2 &= 2 - c_1 (1 + \\gamma \\Delta t / 2),\\\\
        K \\Delta t^2            &= c_3 (1 + \\gamma \\Delta t / 2)

    is applied pointwise, then each pole contributes

    .. math::
        \\chi_p(\\omega) = \\frac{K}{\\omega_0^2 - \\omega^2 - i \\gamma \\omega}

    and the result is summed over the leading pole axis. Cells where
    ``(c1, c2, c3)`` are all zero (no pole) contribute exactly zero.

    Args:
        c1: ADE coefficient array of shape ``(num_poles, ...)``.
        c2: ADE coefficient array of shape ``(num_poles, ...)``.
        c3: ADE coefficient array of shape ``(num_poles, ...)``.
        omega: Angular frequency (rad/s) at which to evaluate the
            susceptibility.
        dt: Simulation time step (seconds) used to derive the coefficients.

    Returns:
        Complex ``jax.Array`` with shape ``c1.shape[1:]`` — the total
        :math:`\\chi(\\omega)` summed over all poles, in every cell.
    """
    c1 = jnp.asarray(c1)
    c2 = jnp.asarray(c2)
    c3 = jnp.asarray(c3)

    pole_mask = (c1 != 0.0) | (c3 != 0.0)

    one_minus_c2 = 1.0 - c2
    safe_denom = jnp.where(one_minus_c2 == 0.0, 1.0, one_minus_c2)
    gamma_dt = 2.0 * (1.0 + c2) / safe_denom
    gamma_dt = jnp.where(pole_mask, gamma_dt, 0.0)

    half_factor = 1.0 + 0.5 * gamma_dt
    omega0_sq_dt2 = 2.0 - c1 * half_factor
    omega0_sq_dt2 = jnp.where(pole_mask, omega0_sq_dt2, 0.0)
    K_dt2 = c3 * half_factor
    K_dt2 = jnp.where(pole_mask, K_dt2, 0.0)

    omega_dt = omega * dt
    denom = omega0_sq_dt2 - omega_dt * omega_dt - 1j * gamma_dt * omega_dt
    safe_denom_cplx = jnp.where(pole_mask, denom, 1.0 + 0.0j)
    chi_per_pole = jnp.where(pole_mask, K_dt2 / safe_denom_cplx, 0.0 + 0.0j)

    return jnp.sum(chi_per_pole, axis=0)


def compute_eps_spectrum_from_coefficients(
    c1: jax.Array | np.ndarray,
    c2: jax.Array | np.ndarray,
    c3: jax.Array | np.ndarray,
    inv_eps_inf: jax.Array | np.ndarray,
    omegas: np.ndarray,
    dt: float,
    weights: np.ndarray | None = None,
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
        c1: ADE coefficient array of shape ``(num_poles, 1, *spatial)`` as
            stored on :class:`~fdtdx.fdtd.container.ArrayContainer` (the
            middle size-1 axis is the material-component broadcast axis).
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
    inv_eps_np = np.asarray(inv_eps_inf)
    omegas_np = np.asarray(omegas, dtype=np.float64)

    # Reverse-engineer pole parameters from the ADE coefficients (same inversion
    # as susceptibility_from_coefficients, duplicated in numpy for setup-time use).
    pole_mask = (c1_np != 0.0) | (c3_np != 0.0)
    one_minus_c2 = 1.0 - c2_np
    safe_one_minus_c2 = np.where(one_minus_c2 == 0.0, 1.0, one_minus_c2)
    gamma_dt = np.where(pole_mask, 2.0 * (1.0 + c2_np) / safe_one_minus_c2, 0.0)
    half_factor = 1.0 + 0.5 * gamma_dt
    omega0_sq_dt2 = np.where(pole_mask, 2.0 - c1_np * half_factor, 0.0)
    K_dt2 = np.where(pole_mask, c3_np * half_factor, 0.0)

    # Reduce inv_eps_inf → scalar eps_inf per spatial cell.
    num_components = inv_eps_np.shape[0]
    if num_components == 9:
        diag = np.stack([inv_eps_np[0], inv_eps_np[4], inv_eps_np[8]], axis=0)
        eps_inf_per_cell = np.mean(1.0 / diag, axis=0)
    elif num_components in (1, 3):
        eps_inf_per_cell = np.mean(1.0 / inv_eps_np, axis=0)
    else:
        raise ValueError(f"Unexpected inv_eps_inf leading dimension {num_components}; expected 1, 3, or 9.")

    # Broadcast: omegas over (M,); coefficient arrays have shape (P, 1, *spatial).
    # After [None, ...] prepend: (M, P, 1, *spatial).
    omega_dt = (omegas_np * dt).reshape((-1,) + (1,) * c1_np.ndim)
    denom = omega0_sq_dt2[None, ...] - omega_dt**2 - 1j * gamma_dt[None, ...] * omega_dt
    safe_denom = np.where(pole_mask[None, ...], denom, 1.0 + 0.0j)
    chi_per_pole = np.where(pole_mask[None, ...], K_dt2[None, ...] / safe_denom, 0.0 + 0.0j)
    chi_per_cell = chi_per_pole.sum(axis=1)  # sum over pole axis → (M, 1, *spatial)
    chi_per_cell = chi_per_cell[:, 0]  # drop component broadcast axis → (M, *spatial)

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
    if c1 is None or c2 is None or c3 is None:
        return inv_eps

    chi = susceptibility_from_coefficients(c1=c1, c2=c2, c3=c3, omega=omega, dt=dt)
    eps_inf = 1.0 / jnp.asarray(inv_eps)
    eps_eff = eps_inf + jnp.real(chi)
    return (1.0 / eps_eff).astype(jnp.asarray(inv_eps).dtype)
