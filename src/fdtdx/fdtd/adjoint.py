"""Frequency-domain adjoint gradients (``GradientConfig(method="adjoint")``).

The forward step is affine in the field state ``s_n = (E^n, H^{n-1/2}, psi_n)``,

    s_{n+1} = L(kappa) s_n + b_n,    kappa = (1/eps, 1/mu),

because source injections are independent of the material arrays (TFSF injections are wrapped in
``stop_gradient``). Phasor detectors accumulate ``P_k = sum_n c_k(n) Q(s_{n+1}, s_n)`` with a linear map
``Q`` and a known time coefficient ``c_k(n)``. In the frequency domain ``A(w) X(w) = B(w)`` with
``A(w) = exp(-i w dt) I - L``, where ``X`` is the forward DTFT (pulsed sources, fields decayed by the end
of the run) or the steady-state phasor (CW sources, fields periodic). The gradient of a loss of the
phasors is then

    dl/dkappa = sum_k Re B(X_k, Lambda_k),    Lambda_k = A(w_k)^{-T} r_k,

with ``B(s, l) = (d(L s)/dkappa)^T l`` the bilinear parameter term of one step and ``r_k`` the adjoint
source obtained from the VJP of the detector samples. ``B`` is local, so ``X_k`` and ``Lambda_k`` are only
needed on the devices (plus a halo of :data:`_HALO` cells for the update stencils).

``Lambda_k`` is computed by stepping ``mu_{m+1} = L^T mu_m + r_m`` with the VJP of the source-free forward
step: its DTFT solves ``A(w)^T mu(w) = r(w)``. The adjoint source ``r_m`` is a short pulse whose spectrum
equals ``r_k`` exactly at every objective frequency; frequencies that are too close to be separated within
the adjoint run are handled in separate runs (groups).

The result equals the exact gradient of the simulated loss when the forward and adjoint fields have
decayed by the end of their runs (pulses) or the forward field is periodic in its DFT window (CW).

The exact mode (``adjoint_mode="exact"``, the default; CW sources always use the frequency mode) instead runs the time-domain adjoint
``lambda_n = L^T lambda_{n+1} + g_n`` backwards over the same ``N`` steps, driven by the cotangents ``g_n`` of
the detector samples. The gradient of the simulated loss is exactly

    dl/dkappa = sum_{n<N} B(s_n, lambda_{n+1}) = sum_j w_j Re B(X_j, conj(Y_j)),

by Parseval over the ``N``-point DFT (``X_j``, ``Y_j`` the DFTs of ``s_n`` and ``lambda_{n+1}``, ``w_j = 2 / N``
for the one-sided bins). The forward fields are band-limited by the sources, so only the bins inside the
source band contribute; one adjoint run serves every objective frequency.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import CustomVJPPrimal, SymbolicZero
from loguru import logger

from fdtdx.config import SimulationConfig
from fdtdx.core.progress import _make_pbar, _wrap_body_with_progress
from fdtdx.fdtd.container import ArrayContainer, FieldState, ObjectContainer, PmlAuxField, SimulationState
from fdtdx.fdtd.forward import forward
from fdtdx.fdtd.update import colocate_detector_fields, get_wrap_padding_axes
from fdtdx.objects.detectors.detector import DetectorState
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.sources.dipole import PointDipoleSource
from fdtdx.objects.sources.profile import GaussianPulseProfile, SingleFrequencyProfile
from fdtdx.typing import SliceTuple3D

#: Cells around a device on which the forward and adjoint DFTs are recorded. The parameter term of one
#: step at a cell reads the fields one cell away (curl stencils), and two cells away for the off-diagonal
#: averaging of fully anisotropic materials.
_HALO = 2

#: The adjoint source pulse of a group must fit in the first half of the adjoint run, leaving the second
#: half for the response to decay: ``12 sigma_t <= T_adj / 2`` (the Gaussian envelope peaks at ``6 sigma_t``).
_ADJOINT_PULSE_FRACTION = 24.0

#: Minimum spacing of two frequencies in one adjoint run, in units of the envelope's spectral width.
_GROUP_SEPARATION = 2.5

#: Envelope bandwidth limit relative to the lowest frequency of a group, keeping the pulse free of DC.
_MAX_RELATIVE_BANDWIDTH = 0.2

#: Diagnostics thresholds. The relative gradient error of a pulsed run is about the square root of the
#: final/peak energy ratio in the devices, so the decay threshold warns from gradient errors of about 1e-4.
_DECAY_WARN_RATIO = 1e-8
_STEADY_WARN_RATIO = 1e-2
_CONDITION_WARN = 1e4

#: Time steps whose design-region fields the exact mode buffers before adding them to its DFT accumulators
#: with one matrix product, instead of updating every accumulator at every step.
_DFT_BLOCK = 32

#: Default band tolerance of the exact mode per simulation dtype, see ``GradientConfig.adjoint_band_tolerance``.
#: Both reach the rounding floor of the dtype; float64 stays above the spectral floor of about 2e-9 that the
#: truncation of a GaussianPulseProfile at t = 0 (6 sigma before its peak) leaves.
_BAND_TOLERANCE = {np.dtype(np.float32): 1e-6, np.dtype(np.float64): 1e-8}

#: Fraction (amplitude) of the forward spectrum outside the band of the exact mode above which it warns. It
#: bounds the relative gradient error (the adjoint spectrum is small outside the band, so the error is
#: usually much smaller), like the square root of the energy ratio of the decay check.
_LEAKAGE_WARN_RATIO = 1e-4


@dataclass(frozen=True)
class _Box:
    """Axis-aligned block of grid cells ``[lo, hi)``."""

    lo: tuple[int, int, int]
    hi: tuple[int, int, int]

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return (slice(self.lo[0], self.hi[0]), slice(self.lo[1], self.hi[1]), slice(self.lo[2], self.hi[2]))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.hi[0] - self.lo[0], self.hi[1] - self.lo[1], self.hi[2] - self.lo[2])


def _support_box(
    grid_slice_tuple: SliceTuple3D,
    halo: int,
    grid_shape: tuple[int, int, int],
    wrap_axes: tuple[bool, bool, bool],
) -> _Box:
    """Dilate a grid slice by ``halo`` cells, clipped to the domain.

    On a periodic (wrap-padded) axis a stencil that leaves the domain re-enters on the other side, so the
    box then spans the whole axis.
    """
    lo, hi = [], []
    for axis, (start, stop) in enumerate(grid_slice_tuple):
        n = grid_shape[axis]
        new_lo, new_hi = start - halo, stop + halo
        if wrap_axes[axis] and (new_lo < 0 or new_hi > n):
            new_lo, new_hi = 0, n
        lo.append(max(new_lo, 0))
        hi.append(min(new_hi, n))
    return _Box(lo=tuple(lo), hi=tuple(hi))  # type: ignore[arg-type]


def _overlaps(a: SliceTuple3D, b: SliceTuple3D) -> bool:
    return all(max(a0, b0) < min(a1, b1) for (a0, a1), (b0, b1) in zip(a, b))


@dataclass(frozen=True)
class _AdjointGroup:
    """Frequencies handled by one adjoint run and the scalar time signals that drive it."""

    #: Indices into :attr:`_AdjointSpec.omegas`.
    freq_indices: tuple[int, ...]
    #: ``beta_k(m)``, shape ``(adjoint_steps, K_g)``: the adjoint source at step ``m`` is
    #: ``sum_k Re(r_k * beta_k(m))``, whose DTFT at ``w_l`` equals ``r_l``.
    source_signals: np.ndarray
    #: ``exp(i w_k tau_m)``, shape ``(adjoint_steps, K_g)``.
    dtft_phases: np.ndarray
    #: Condition number of the spectral mixing matrix.
    condition_number: float
    #: Spectral width (rad/s) of the Gaussian envelope of the adjoint pulse.
    sigma_omega: float


@dataclass(frozen=True)
class _AdjointSpec:
    """Static description of an adjoint gradient computation."""

    regime: Literal["pulse", "cw"]
    #: Objective angular frequencies (rad/s), shape ``(K,)``.
    omegas: np.ndarray
    #: Time step duration (s).
    dt: float
    #: Number of forward time steps.
    num_steps: int
    #: Weight times ``exp(i w_k t_n)`` of the forward design-region DFT, shape ``(num_steps, K)``.
    forward_coefficients: np.ndarray
    #: CW window leakage ``kappa_k = sum_n w_n exp(2 i w_k t_n) / 2``: a steady field gives the windowed DFT
    #: ``X_hat = X + kappa * conj(X)``. Zero for pulses. Shape ``(K,)``.
    forward_leakage: np.ndarray
    #: Per-step averaging weights ``(first half, second half)`` of the CW DFT window, used for the steadiness check.
    window_halves: tuple[np.ndarray, np.ndarray] | None
    #: Device regions dilated by :data:`_HALO`.
    design_boxes: tuple[_Box, ...]
    #: Union of the device grid slices, shape ``(Nx, Ny, Nz)``.
    design_mask: np.ndarray
    groups: tuple[_AdjointGroup, ...]


@dataclass(frozen=True)
class _ExactSpec:
    """Static description of an exact-mode adjoint gradient computation."""

    #: DFT bins ``j`` of the source band; their angular frequencies are ``2 pi j / (N dt)``. Shape ``(B,)``.
    bins: np.ndarray
    #: Parseval weights ``2 / N`` (``1 / N`` at DC and Nyquist) of the bins, shape ``(B,)``.
    pair_weights: np.ndarray
    #: Band membership of the one-sided DFT bins ``0 .. N // 2``, for the leakage diagnostics.
    band_mask: np.ndarray
    #: Number of forward time steps ``N``.
    num_steps: int
    #: Device regions dilated by :data:`_HALO`.
    design_boxes: tuple[_Box, ...]
    #: Union of the device grid slices, shape ``(Nx, Ny, Nz)``.
    design_mask: np.ndarray


def _phase_table(omegas: np.ndarray, times: np.ndarray) -> np.ndarray:
    """``exp(i w t)`` evaluated in float64 with the phase reduced modulo 2 pi."""
    phase = np.mod(np.outer(times, omegas), 2 * np.pi)
    return np.exp(1j * phase)


def _source_regime(objects: ObjectContainer) -> Literal["pulse", "cw"]:
    """CW if every source has a single-frequency profile, pulse if none has one."""
    cw = [isinstance(s.temporal_profile, SingleFrequencyProfile) for s in objects.sources]
    if cw and all(cw):
        return "cw"
    if not any(cw):
        return "pulse"
    raise ValueError(
        "The adjoint gradient method needs either only CW sources (SingleFrequencyProfile) or only pulsed "
        f"sources; got a mix: {[(s.name, type(s.temporal_profile).__name__) for s in objects.sources]}."
    )


def _source_bandwidth(objects: ObjectContainer) -> float | None:
    """Smallest spectral width (rad/s) of the Gaussian pulse sources, ``None`` if there is none."""
    widths = [
        2 * math.pi * s.temporal_profile.spectral_width.get_frequency()
        for s in objects.sources
        if isinstance(s.temporal_profile, GaussianPulseProfile)
    ]
    return min(widths) if widths else None


def _detector_omegas(detector: PhasorDetector) -> np.ndarray:
    """Angular frequencies of a phasor detector, computed on the host (the detector may be traced)."""
    return np.asarray([2 * math.pi * wc.get_frequency() for wc in detector.wave_characters], dtype=np.float64)


def _objective_detectors(objects: ObjectContainer) -> list[PhasorDetector]:
    return [d for d in objects.detectors if isinstance(d, PhasorDetector) and not d.inverse]


def _objective_frequencies(objects: ObjectContainer, config: SimulationConfig) -> np.ndarray:
    """Objective angular frequencies: the override of the gradient config or all phasor-detector frequencies."""
    grad_cfg = config.gradient_config
    assert grad_cfg is not None
    if grad_cfg.adjoint_frequencies is not None:
        omegas = [2 * math.pi * wc.get_frequency() for wc in grad_cfg.adjoint_frequencies]
    else:
        omegas = [float(w) for d in _objective_detectors(objects) for w in _detector_omegas(d)]
    unique: list[float] = []
    for w in sorted(omegas):
        if w <= 0:
            raise ValueError(f"Adjoint frequencies must be positive, got angular frequency {w}.")
        if not unique or not math.isclose(w, unique[-1], rel_tol=1e-9):
            unique.append(w)
    if not unique:
        raise ValueError(
            "The adjoint gradient method computes gradients of phasor-detector objectives, but the simulation "
            "has no PhasorDetector (or subclass, e.g. ModeOverlapDetector, PhasorPoyntingFluxDetector)."
        )
    return np.asarray(unique, dtype=np.float64)


def _frequency_indices(detector: PhasorDetector, omegas: np.ndarray) -> np.ndarray:
    """Index of every detector frequency in ``omegas``, ``-1`` if it is not an objective frequency."""
    result = []
    for w in _detector_omegas(detector):
        matches = np.flatnonzero(np.isclose(omegas, w, rtol=1e-9, atol=0.0))
        result.append(int(matches[0]) if matches.size else -1)
    return np.asarray(result, dtype=np.int64)


def _forward_window(config: SimulationConfig, omegas: np.ndarray, regime: str) -> np.ndarray:
    """Per-step weights of the forward design-region DFT.

    Pulses: the full DTFT (weight 1 on every step). CW: the steady-state phasor ``(2 / N) sum`` over a window,
    by default the last half of the run snapped to whole periods of the lowest objective frequency.
    """
    num_steps = config.time_steps_total
    if regime == "pulse":
        return np.ones(num_steps, dtype=np.float64)
    grad_cfg = config.gradient_config
    assert grad_cfg is not None
    if grad_cfg.adjoint_window is not None:
        on = np.asarray(
            grad_cfg.adjoint_window.calculate_on_list(
                num_total_time_steps=num_steps,
                time_step_duration=config.time_step_duration,
            ),
            dtype=bool,
        )
    else:
        period_steps = 2 * math.pi / (float(np.min(omegas)) * config.time_step_duration)
        num_periods = math.floor(num_steps / 2 / period_steps)
        length = round(num_periods * period_steps) if num_periods >= 1 else num_steps // 2
        on = np.zeros(num_steps, dtype=bool)
        on[num_steps - max(length, 1) :] = True
    if not on.any():
        raise ValueError("The adjoint DFT window of the CW forward run contains no time step.")
    return np.where(on, 2.0 / on.sum(), 0.0)


def _plan_groups(
    omegas: np.ndarray,
    adjoint_steps: int,
    dt: float,
    source_bandwidth: float | None,
) -> tuple[_AdjointGroup, ...]:
    """Partition the objective frequencies into adjoint runs and design their source signals.

    Every run is driven by Gaussian-enveloped carriers at its frequencies. The envelope must fit in the
    first half of the run, which bounds its spectral width from below by ``24 / T_adj``; frequencies closer
    than :data:`_GROUP_SEPARATION` widths go to different runs. The ``2 K`` real basis signals are mixed by
    the inverse of their sampled spectra so that the DTFT of the source at every ``w_l`` is exactly ``r_l``.
    """
    duration = adjoint_steps * dt
    sigma_min = _ADJOINT_PULSE_FRACTION / duration
    separation = _GROUP_SEPARATION * sigma_min

    members: list[list[int]] = []
    for idx in np.argsort(omegas):
        for group in members:
            if all(abs(omegas[idx] - omegas[j]) >= separation for j in group):
                group.append(int(idx))
                break
        else:
            members.append([int(idx)])

    taus = np.arange(adjoint_steps, dtype=np.float64) * dt
    groups = []
    for indices in members:
        indices = sorted(indices, key=lambda i: omegas[i])
        w = omegas[indices]
        spacing = float(np.min(np.diff(w))) if len(w) > 1 else math.inf
        sigma = min(spacing / _GROUP_SEPARATION, _MAX_RELATIVE_BANDWIDTH * float(w[0]))
        if source_bandwidth is not None:
            sigma = min(sigma, source_bandwidth)
        sigma = max(sigma, sigma_min)
        if float(w[0]) / sigma < 4.0:
            logger.warning(
                f"The adjoint run of {adjoint_steps} steps is short for frequency {w[0] / (2 * math.pi):.4e} Hz: "
                "its source pulse has a non-negligible DC content, which leaves static fields that do not decay. "
                "Increase the simulation time or GradientConfig.adjoint_time."
            )
        sigma_t = 1.0 / sigma
        envelope = np.exp(-((taus - 6 * sigma_t) ** 2) / (2 * sigma_t**2))
        # Real basis: envelope * cos(w_k tau), envelope * sin(w_k tau), shape (T_adj, 2K).
        basis = np.concatenate(
            [envelope[:, None] * np.cos(np.outer(taus, w)), envelope[:, None] * np.sin(np.outer(taus, w))],
            axis=1,
        )
        phases = _phase_table(w, taus)  # (T_adj, K)
        spectra = phases.T @ basis  # (K, 2K): DTFT of every basis signal at every frequency
        mixing = np.concatenate([spectra.real, spectra.imag], axis=0)  # (2K, 2K)
        condition = float(np.linalg.cond(mixing))
        if condition > _CONDITION_WARN:
            logger.warning(
                f"The adjoint source mixing matrix for frequencies {w / (2 * math.pi)} Hz is ill-conditioned "
                f"(cond={condition:.2e}); the adjoint gradient may be inaccurate."
            )
        inverse = np.linalg.inv(mixing)  # columns: [U | V]
        num = len(w)
        alpha = basis @ inverse[:, :num]  # (T_adj, K): weight of Re r_k
        alpha_im = basis @ inverse[:, num:]  # (T_adj, K): weight of Im r_k
        groups.append(
            _AdjointGroup(
                freq_indices=tuple(indices),
                source_signals=alpha - 1j * alpha_im,
                dtft_phases=phases,
                condition_number=condition,
                sigma_omega=sigma,
            )
        )
    return tuple(groups)


def _validate(arrays: ArrayContainer, objects: ObjectContainer, config: SimulationConfig) -> None:
    grad_cfg = config.gradient_config
    assert grad_cfg is not None
    if _uses_exact_mode(objects, config):
        frequency_only = {
            "adjoint_frequencies": grad_cfg.adjoint_frequencies,
            "adjoint_window": grad_cfg.adjoint_window,
            "adjoint_time": grad_cfg.adjoint_time,
        }
        used = [name for name, value in frequency_only.items() if value is not None]
        if used:
            raise ValueError(
                f"{', '.join(used)} only apply to the frequency adjoint mode, but pulsed sources use the exact mode "
                "by default. Set GradientConfig.adjoint_mode='frequency' to use them."
            )
    elif grad_cfg.adjoint_band_tolerance is not None:
        raise ValueError(
            "adjoint_band_tolerance only applies to the exact adjoint mode, which needs pulsed sources; CW sources and "
            "adjoint_mode='frequency' use the frequency mode."
        )
    if jnp.iscomplexobj(arrays.fields.E):
        raise NotImplementedError("The adjoint gradient method does not support complex fields (Bloch boundaries) yet.")
    if arrays.dispersive_c1 is not None or arrays.fields.dispersive_P_curr is not None:
        raise NotImplementedError("The adjoint gradient method does not support dispersive materials yet.")
    if not objects.devices:
        raise ValueError("The adjoint gradient method computes gradients inside Devices, but there is no Device.")
    for device in objects.devices:
        for pml in objects.pml_objects:
            if _overlaps(device.grid_slice_tuple, pml.grid_slice_tuple):
                raise ValueError(
                    f"Device '{device.name}' overlaps the PML '{pml.name}', which the adjoint gradient method "
                    "does not support."
                )
        for source in objects.sources:
            if isinstance(source, PointDipoleSource) and _overlaps(device.grid_slice_tuple, source.grid_slice_tuple):
                raise NotImplementedError(
                    f"Source '{source.name}' lies inside Device '{device.name}'. Its injection depends on the "
                    "permittivity, which the adjoint gradient method does not differentiate."
                )


def _uses_exact_mode(objects: ObjectContainer, config: SimulationConfig) -> bool:
    """Whether the exact mode runs: it is selected and the sources are pulsed (CW fields are not band-limited)."""
    grad_cfg = config.gradient_config
    assert grad_cfg is not None
    return grad_cfg.adjoint_mode == "exact" and _source_regime(objects) == "pulse"


def _adjoint_progress(show_progress: bool, total_steps: int) -> tuple[Callable[[jax.Array], None], Callable[[], None]]:
    """Step and close callbacks of the progress bar of the adjoint runs (no-ops without a bar)."""
    pbar = _make_pbar(show_progress=show_progress, total_steps=total_steps, desc="FDTD (adjoint, backward)")
    if pbar is None:
        return (lambda step: None), (lambda: None)
    return pbar.get_callbacks()


def _build_spec(objects: ObjectContainer, config: SimulationConfig) -> _AdjointSpec:
    grad_cfg = config.gradient_config
    assert grad_cfg is not None
    regime = _source_regime(objects)
    omegas = _objective_frequencies(objects, config)
    dt = config.time_step_duration
    num_steps = config.time_steps_total
    weights = _forward_window(config, omegas, regime)
    times = np.arange(num_steps, dtype=np.float64) * dt
    forward_coefficients = weights[:, None] * _phase_table(omegas, times)
    forward_leakage = np.zeros(len(omegas), dtype=np.complex128)
    window_halves = None
    if regime == "cw":
        on = np.flatnonzero(weights)
        middle = on[len(on) // 2]
        first = np.where((np.arange(num_steps) < middle) & (weights > 0), 1.0, 0.0)
        second = np.where((np.arange(num_steps) >= middle) & (weights > 0), 1.0, 0.0)
        # normalized so that the diagnostics compare the mean energy of both halves
        window_halves = (first / max(first.sum(), 1.0), second / max(second.sum(), 1.0))
        forward_leakage = 0.5 * (weights @ _phase_table(2 * omegas, times))

    design_boxes, design_mask = _design_regions(objects)
    adjoint_steps = num_steps if grad_cfg.adjoint_time is None else round(grad_cfg.adjoint_time / dt)
    if adjoint_steps < 1:
        raise ValueError(f"The adjoint run needs at least one time step, got {adjoint_steps}.")
    bandwidth = _source_bandwidth(objects) if regime == "pulse" else None
    groups = _plan_groups(omegas, adjoint_steps, dt, bandwidth)
    return _AdjointSpec(
        regime=regime,
        omegas=omegas,
        dt=dt,
        num_steps=num_steps,
        forward_coefficients=forward_coefficients,
        forward_leakage=forward_leakage,
        window_halves=window_halves,
        design_boxes=design_boxes,
        design_mask=design_mask,
        groups=groups,
    )


def _design_regions(objects: ObjectContainer) -> tuple[tuple[_Box, ...], np.ndarray]:
    """Device regions dilated by :data:`_HALO` and the mask of the device cells."""
    grid_shape = objects.volume.grid_shape
    wrap_axes = get_wrap_padding_axes(objects)
    design_boxes = tuple(_support_box(d.grid_slice_tuple, _HALO, grid_shape, wrap_axes) for d in objects.devices)
    design_mask = np.zeros(grid_shape, dtype=bool)
    for device in objects.devices:
        design_mask[device.grid_slice] = True
    return design_boxes, design_mask


def _source_spectrum(objects: ObjectContainer, config: SimulationConfig) -> np.ndarray:
    """One-sided ``N``-point amplitude spectrum of every source signal relative to its peak, maximized over sources.

    The signals are sampled in float64 whatever the simulation dtype: float32 sampling of the time axis adds a
    phase noise floor that would widen the band.
    """
    num_steps = config.time_steps_total
    spectrum = np.zeros(num_steps // 2 + 1, dtype=np.float64)
    for source in objects.sources:
        try:
            with jax.enable_x64(True), jax.ensure_compile_time_eval():
                _, signal = source.temporal_profile.sample_time_signal(
                    period=source.wave_character.get_period(),
                    time_step_duration=config.time_step_duration,
                    num_time_steps=num_steps,
                    phase_shift=source.wave_character.phase_shift,
                )
        except jax.errors.JAXTypeError as error:
            raise ValueError(
                f"adjoint_mode='exact' samples the time signal of source '{source.name}' when the simulation is "
                "traced, but it depends on traced values. Close over the objects instead of passing them as jit "
                "arguments."
            ) from error
        amplitude = np.abs(np.fft.rfft(np.asarray(signal, dtype=np.float64)))
        if amplitude.max() > 0:
            spectrum = np.maximum(spectrum, amplitude / amplitude.max())
    return spectrum


def _build_exact_spec(objects: ObjectContainer, config: SimulationConfig) -> _ExactSpec:
    grad_cfg = config.gradient_config
    assert grad_cfg is not None
    num_steps = config.time_steps_total
    tolerance = grad_cfg.adjoint_band_tolerance
    if tolerance is None:
        tolerance = _BAND_TOLERANCE.get(np.dtype(config.dtype), _BAND_TOLERANCE[np.dtype(np.float32)])
    spectrum = _source_spectrum(objects, config)
    band_mask = spectrum >= tolerance
    bins = np.flatnonzero(band_mask)
    if bins.size == 0:
        raise ValueError("adjoint_mode='exact' needs at least one source with a nonzero time signal.")
    if bins.size > spectrum.size // 2:
        logger.warning(
            f"The band of the exact adjoint mode holds {bins.size} of the {spectrum.size} DFT frequencies of the run: "
            f"the sources are not band-limited at adjoint_band_tolerance={tolerance:.1e}. The DFTs of the devices need "
            "one field copy per frequency; consider adjoint_mode='frequency' or a larger tolerance."
        )
    pair_weights = np.where((bins == 0) | (2 * bins == num_steps), 1.0, 2.0) / num_steps
    design_boxes, design_mask = _design_regions(objects)
    return _ExactSpec(
        bins=bins,
        pair_weights=pair_weights,
        band_mask=band_mask,
        num_steps=num_steps,
        design_boxes=design_boxes,
        design_mask=design_mask,
    )


def _detector_step_weights(detector: PhasorDetector) -> np.ndarray:
    """Real per-step weight ``a_n`` of a phasor detector, with ``_dft_coefficients(n) = a_n exp(i w t_n)``.

    Computed from static fields on the host, since the detector arrays may be traced.
    """
    weights = np.asarray(detector._calculate_on_list(), dtype=np.float64)
    if detector.apodization is not None:
        times = np.arange(weights.size, dtype=np.float64) * detector._config.time_step_duration
        with jax.ensure_compile_time_eval():
            weights = weights * np.asarray(detector.apodization.get_window(jnp.asarray(times)), dtype=np.float64)
    return weights * float(detector._static_scale())


def _detector_fd_scale(
    detector: PhasorDetector, regime: str, omegas: np.ndarray, dt: float
) -> tuple[float, np.ndarray]:
    """Factors ``(nu, mu_k)`` of the detector in the frequency-domain model.

    Pulses: ``P_k = nu * Q_w X_k`` with the DTFT ``X_k`` and ``nu`` the mean per-step weight over the steps the
    detector spans; ``mu_k = 0``. CW: a steady field ``s_n = Re(X_k exp(-i w_k t_n))`` gives
    ``P_k = nu * Q_w X_k + mu_k * conj(Q_w X_k)`` with ``nu = sum_n a_n / 2`` and the leakage of the window
    ``mu_k = sum_n a_n exp(2 i w_k t_n) / 2``, which vanishes for whole periods.
    """
    weights = _detector_step_weights(detector)
    if regime == "pulse":
        nu = float(weights.sum()) / (detector._num_time_steps_on * detector._dft_stride)
        return nu, np.zeros(len(omegas), dtype=np.complex128)
    times = np.arange(weights.size, dtype=np.float64) * dt
    mu = 0.5 * (weights @ _phase_table(2 * omegas, times))
    return float(weights.sum()) / 2.0, mu


def _check_pulse_detector(detector: PhasorDetector, num_steps: int) -> None:
    on = np.flatnonzero(np.asarray(detector._calculate_on_list(), dtype=bool))
    if detector.apodization is not None or on.size == 0 or on[0] > 0 or on[-1] < num_steps - detector._dft_stride:
        logger.warning(
            f"Detector '{detector.name}' does not record the full run without apodization. With pulsed sources "
            "the adjoint gradient method assumes that phasor detectors take the complete DTFT of the fields."
        )


def _is_zero(cotangent) -> bool:
    return cotangent is None or isinstance(cotangent, SymbolicZero)


def _materialize(cotangent, like: jax.Array) -> jax.Array:
    return jnp.zeros_like(like) if _is_zero(cotangent) else cotangent


def adjoint_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    show_progress: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SimulationState:
    """Run an FDTD simulation whose gradient is computed with the frequency-domain adjoint method.

    The forward pass is a plain simulation that additionally records the DFT of the fields inside every
    :class:`~fdtdx.Device`. The backward pass converts the cotangents of the phasor detectors into adjoint
    sources, runs the transposed simulation and pairs both DFTs. No boundary values are recorded.

    ``GradientConfig.adjoint_mode`` selects how:

    - ``"exact"`` (default): one time-domain adjoint run over the same steps, paired with the forward run
      at every DFT frequency inside the source band. This reproduces the checkpointed gradient for any
      number of objective frequencies; its memory grows with the source bandwidth times the simulation
      time. CW sources are not band-limited and always use the frequency mode.
    - ``"frequency"``: the DFTs are taken at the objective frequencies and the adjoint runs are driven by
      short pulses with the right spectrum there. The memory does not grow with the simulation time, but
      frequencies too close to be separated within one adjoint run need separate runs.

    The loss may depend on the states of phasor detectors (:class:`~fdtdx.PhasorDetector` and its
    subclasses such as :class:`~fdtdx.ModeOverlapDetector` and :class:`~fdtdx.PhasorPoyntingFluxDetector`)
    and on the material arrays. Gradients are only computed inside Devices. With pulsed sources the fields
    must have decayed by the end of the run; with CW sources they must be periodic in the DFT window
    (``GradientConfig.adjoint_window``, by default the last half of the run).

    Args:
        arrays (ArrayContainer): Initial state of the simulation.
        objects (ObjectContainer): Collection of physical objects in the simulation.
        config (SimulationConfig): Simulation parameters; ``config.gradient_config.method == "adjoint"``.
        key (jax.Array): JAX PRNGKey for any stochastic operations.
        show_progress (bool): Display tqdm progress bars for the forward run and, when the gradient is
            computed, for the adjoint runs. Defaults to True.
        progress_callback (Callable[[int, int], None] | None): Optional callback for progress updates of
            the forward run.

    Returns:
        SimulationState: Tuple of the final time step and the ArrayContainer with the final state.

    Raises:
        NotImplementedError: For complex fields, dispersive materials, a dipole source inside a Device,
            or a loss depending on outputs other than phasor-detector states and material arrays.
        ValueError: For simulations without Devices or phasor detectors, Devices overlapping a PML, a mix
            of CW and pulsed sources, or options of the adjoint mode that does not run.
    """
    _validate(arrays, objects, config)
    grad_cfg = config.gradient_config
    assert grad_cfg is not None
    arrays = arrays.reset()

    pbar = _make_pbar(
        show_progress=show_progress,
        total_steps=config.time_steps_total,
        desc="FDTD (adjoint)",
        progress_callback=progress_callback,
    )
    _forward_body = partial(
        forward,
        config=config,
        objects=objects,
        key=key,
        record_detectors=True,
        record_boundaries=False,
        simulate_boundaries=True,
    )
    _forward_body_with_progress, _close_pbar = _wrap_body_with_progress(_forward_body, pbar)

    def make_container(
        E: jax.Array,
        H: jax.Array,
        psi_E: PmlAuxField,
        psi_H: PmlAuxField,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        detector_states: dict[str, DetectorState],
    ) -> ArrayContainer:
        return ArrayContainer(
            fields=FieldState(E=E, H=H, psi_E=psi_E, psi_H=psi_H),
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            detector_states=detector_states,
            recording_state=None,
            electric_conductivity=arrays.electric_conductivity,
            magnetic_conductivity=arrays.magnetic_conductivity,
            initial_inv_permittivities=arrays.initial_inv_permittivities,
        )

    def outputs(state: SimulationState):
        time_step, arr = state
        return (
            time_step,
            arr.fields.E,
            arr.fields.H,
            arr.fields.psi_E,
            arr.fields.psi_H,
            arr.inv_permittivities,
            arr.inv_permeabilities,
            arr.detector_states,
            None,
        )

    def run_forward(arr: ArrayContainer) -> SimulationState:
        return jax.lax.while_loop(
            lambda s: config.time_steps_total > s[0],
            _forward_body_with_progress,
            (jnp.asarray(0, dtype=jnp.int32), arr),
        )

    @jax.custom_vjp
    def adjoint_fdtd_primal(
        E, H, psi_E, psi_H, inv_permittivities, inv_permeabilities, detector_states, recording_state
    ):
        del recording_state
        arr = make_container(E, H, psi_E, psi_H, inv_permittivities, inv_permeabilities, detector_states)
        return outputs(run_forward(arr))

    if _uses_exact_mode(objects, config):
        exact_spec = _build_exact_spec(objects, config)

        def forward_with_dft(arr: ArrayContainer):
            return _exact_forward(arr, exact_spec, _forward_body_with_progress)

        def gradient(design_dft, detector_cotangents, inv_permittivities, inv_permeabilities):
            return _exact_adjoint_gradient(
                spec=exact_spec,
                design_dft=design_dft,
                detector_cotangents=detector_cotangents,
                inv_permittivities=inv_permittivities,
                inv_permeabilities=inv_permeabilities,
                objects=objects,
                config=config,
                key=key,
                template=arrays,
                show_progress=show_progress,
            )

    else:
        spec = _build_spec(objects, config)

        def forward_with_dft(arr: ArrayContainer):
            return _forward_with_design_dft(arr, spec, _forward_body_with_progress)

        def gradient(design_dft, detector_cotangents, inv_permittivities, inv_permeabilities):
            return _adjoint_gradient(
                spec=spec,
                design_dft=design_dft,
                detector_cotangents=detector_cotangents,
                inv_permittivities=inv_permittivities,
                inv_permeabilities=inv_permeabilities,
                objects=objects,
                config=config,
                key=key,
                template=arrays,
                show_progress=show_progress,
            )

    def fdtd_fwd(E, H, psi_E, psi_H, inv_permittivities, inv_permeabilities, detector_states, recording_state):
        E, H, psi_E, psi_H, inv_permittivities, inv_permeabilities, detector_states, recording_state = jax.tree.map(
            lambda p: p.value,
            (E, H, psi_E, psi_H, inv_permittivities, inv_permeabilities, detector_states, recording_state),
            is_leaf=lambda p: isinstance(p, CustomVJPPrimal),
        )
        arr = make_container(E, H, psi_E, psi_H, inv_permittivities, inv_permeabilities, detector_states)
        state, design_dft = forward_with_dft(arr)
        residual = (design_dft, inv_permittivities, inv_permeabilities)
        return outputs(state), residual

    def fdtd_bwd(residual, cot):
        design_dft, inv_permittivities, inv_permeabilities = residual
        (_, ct_E, ct_H, ct_psi_E, ct_psi_H, ct_inv_eps, ct_inv_mu, ct_detectors, _) = cot
        if not all(_is_zero(c) for c in jax.tree.leaves((ct_E, ct_H, ct_psi_E, ct_psi_H), is_leaf=_is_zero)):
            # The final fields of a decayed pulse are out-of-band remnants and a CW run has no final phasor, so
            # neither the frequency-domain pairing nor the band of the exact mode represents them.
            raise NotImplementedError(
                "The loss depends on the fields at the end of the simulation. The adjoint gradient method only "
                "supports losses of phasor-detector states; use method='checkpointed' or 'reversible' instead."
            )
        grad_eps, grad_mu = gradient(design_dft, ct_detectors, inv_permittivities, inv_permeabilities)
        if not _is_zero(ct_inv_eps):
            grad_eps = grad_eps + ct_inv_eps
        if grad_mu is not None and not _is_zero(ct_inv_mu):
            grad_mu = grad_mu + ct_inv_mu
        if grad_mu is None:
            grad_mu = _materialize(ct_inv_mu, jnp.asarray(inv_permeabilities))
        return (None, None, None, None, grad_eps, grad_mu, None, None)

    adjoint_fdtd_primal.defvjp(fdtd_fwd, fdtd_bwd, symbolic_zeros=True)

    (time_step, E, H, psi_E, psi_H, inv_permittivities, inv_permeabilities, detector_states, _) = adjoint_fdtd_primal(
        arrays.fields.E,
        arrays.fields.H,
        arrays.fields.psi_E,
        arrays.fields.psi_H,
        arrays.inv_permittivities,
        arrays.inv_permeabilities,
        arrays.detector_states,
        arrays.recording_state,
    )
    _close_pbar()
    out = make_container(E, H, psi_E, psi_H, inv_permittivities, inv_permeabilities, detector_states)
    return time_step, out.aset("recording_state", arrays.recording_state)


def _superpose(coefficients: jax.Array, patterns: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """``Re(sum_k c_k Z_k)`` for coefficients ``(F,)`` and patterns ``(F, ...)``, without a reduced-precision matmul."""
    return jnp.real(jnp.sum(coefficients.reshape((-1,) + (1,) * (patterns.ndim - 1)) * patterns, axis=0)).astype(dtype)


def _design_fields(E: jax.Array, H: jax.Array, boxes: tuple[_Box, ...]) -> list[tuple[jax.Array, jax.Array]]:
    return [(E[:, *box.slices], H[:, *box.slices]) for box in boxes]


def _energy(regions: list[tuple[jax.Array, jax.Array]]) -> jax.Array:
    return sum((jnp.sum(e * e) + jnp.sum(h * h) for e, h in regions), start=jnp.zeros((), dtype=regions[0][0].dtype))


def _forward_with_design_dft(
    arr: ArrayContainer,
    spec: _AdjointSpec,
    body: Callable[[SimulationState], SimulationState],
) -> tuple[SimulationState, list[tuple[jax.Array, jax.Array]]]:
    """Run the forward simulation and accumulate the DFT of the step inputs on the design boxes."""
    real_dtype = arr.fields.E.dtype
    complex_dtype = jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64
    coefficients = jnp.asarray(spec.forward_coefficients, dtype=complex_dtype)  # (T, K)
    num_freqs = len(spec.omegas)
    dft0 = [
        (jnp.zeros((num_freqs, 3, *box.shape), complex_dtype), jnp.zeros((num_freqs, 3, *box.shape), complex_dtype))
        for box in spec.design_boxes
    ]
    halves = None
    if spec.window_halves is not None:
        halves = jnp.asarray(np.stack(spec.window_halves, axis=0), dtype=real_dtype)  # (2, T)
    # diagnostics: (peak energy, last energy, energy summed over the first / second half of the CW window)
    diag0 = jnp.zeros((4,), dtype=real_dtype)

    def step(carry):
        state, dft, diag = carry
        time_step, current = state
        regions = _design_fields(current.fields.E, current.fields.H, spec.design_boxes)
        coeff = coefficients[time_step][:, None, None, None, None]
        dft = [(acc_e + coeff * e[None], acc_h + coeff * h[None]) for (acc_e, acc_h), (e, h) in zip(dft, regions)]
        energy = _energy(regions)
        diag = diag.at[0].max(energy).at[1].set(energy)
        if halves is not None:
            diag = diag.at[2].add(halves[0, time_step] * energy).at[3].add(halves[1, time_step] * energy)
        return body(state), dft, diag

    state, dft, diag = jax.lax.while_loop(
        lambda c: spec.num_steps > c[0][0],
        step,
        ((jnp.asarray(0, dtype=jnp.int32), arr), dft0, diag0),
    )
    jax.debug.callback(partial(_report_forward_diagnostics, spec.regime), diag)
    if spec.regime == "cw":
        # Remove the leakage of the window, X_hat = X + kappa * conj(X), to recover the steady phasor.
        kappa = jnp.asarray(spec.forward_leakage, dtype=complex_dtype)[:, None, None, None, None]
        norm = 1 - jnp.abs(kappa) ** 2

        def unleak(x: jax.Array) -> jax.Array:
            return (x - kappa * jnp.conj(x)) / norm

        dft = [(unleak(e), unleak(h)) for e, h in dft]
    return state, dft


def _report_forward_diagnostics(regime: str, diag: np.ndarray) -> None:
    peak, last, first_half, second_half = (float(v) for v in np.asarray(diag))
    if regime == "pulse":
        _warn_if_not_decayed(peak, last, "fields", "forward run", "increase the simulation time")
    elif first_half + second_half > 0:
        mismatch = abs(first_half - second_half) / (first_half + second_half) * 2
        if mismatch > _STEADY_WARN_RATIO:
            logger.warning(
                f"Adjoint gradient: the CW fields in the devices are not steady in the DFT window (energy differs "
                f"by {mismatch:.2e} between its halves). Increase the simulation time or adjust adjoint_window."
            )


def _report_adjoint_diagnostics(peak_energy: np.ndarray, last_energy: np.ndarray) -> None:
    peak, last = float(np.asarray(peak_energy)), float(np.asarray(last_energy))
    _warn_if_not_decayed(peak, last, "adjoint fields", "adjoint run", "increase GradientConfig.adjoint_time")


def _warn_if_not_decayed(peak: float, last: float, fields: str, run: str, remedy: str) -> None:
    if peak > 0 and last / peak > _DECAY_WARN_RATIO:
        logger.warning(
            f"Adjoint gradient: the {fields} in the devices have not decayed by the end of the {run} "
            f"(final/peak energy = {last / peak:.2e}). Expect a relative gradient error of about "
            f"{math.sqrt(last / peak):.0e}; {remedy}."
        )


def _source_free_step(
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    template: ArrayContainer,
) -> Callable:
    """The linear part of one forward step: ``(E, H, psi_E, psi_H, 1/eps, 1/mu) -> (E', H', psi_E', psi_H')``."""

    def step(E, H, psi_E, psi_H, inv_permittivities, inv_permeabilities):
        arr = ArrayContainer(
            fields=FieldState(E=E, H=H, psi_E=psi_E, psi_H=psi_H),
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            detector_states={},
            recording_state=None,
            electric_conductivity=template.electric_conductivity,
            magnetic_conductivity=template.magnetic_conductivity,
        )
        _, out = forward(
            state=(jnp.asarray(0, dtype=jnp.int32), arr),
            config=config,
            objects=objects,
            key=key,
            record_detectors=False,
            record_boundaries=False,
            simulate_boundaries=True,
            inject_sources=False,
        )
        return out.fields.E, out.fields.H, out.fields.psi_E, out.fields.psi_H

    return step


def _detector_adjoint_sources(
    detector: PhasorDetector,
    cotangent: DetectorState,
    nu: float,
    mu: np.ndarray,
    omegas: np.ndarray,
    dt: float,
    box: _Box,
    template: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
) -> tuple[jax.Array, jax.Array]:
    """Frequency-domain adjoint sources ``r_k`` of one detector on ``box``, shapes ``(F_d, 3, *box)`` (E, H).

    With the patterns ``Z_k`` of :func:`_detector_patterns`, ``r_k = exp(-i w_k dt) W(Z_k_post) + W(Z_k_pre)``
    with ``W(Z) = nu * Z + conj(mu_k * Z)`` (see :func:`_detector_fd_scale`).
    """
    Z_E_post, Z_H_post, Z_H_pre = _detector_patterns(detector, cotangent, len(omegas), box, template, objects, config)
    complex_dtype = Z_E_post.dtype
    shift = jnp.asarray(np.exp(-1j * omegas * dt), dtype=complex_dtype)[:, None, None, None, None]
    leakage = jnp.asarray(mu, dtype=complex_dtype)[:, None, None, None, None]

    def weigh(Z: jax.Array) -> jax.Array:
        return nu * Z + jnp.conj(leakage * Z)

    return shift * weigh(Z_E_post), shift * weigh(Z_H_post) + weigh(Z_H_pre)


def _detector_patterns(
    detector: PhasorDetector,
    cotangent: DetectorState,
    num_frequencies: int,
    box: _Box,
    template: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Spatial adjoint patterns ``Z_k = Q^T P_bar_k`` of one detector on ``box``, each ``(F_d, 3, *box)`` complex.

    ``Q^T`` is the transposed detector sample applied to the cotangent ``P_bar_k`` of frequency ``k``, split into
    the parts acting on E and H after the step (``post``) and on H before it (``pre``, the time-centered H).
    The sample is real-linear, so ``Z = Q^T Re(P_bar) + i Q^T Im(P_bar)`` and ``Q^T Re(c P_bar) = Re(c Z)``.
    """
    E0 = jnp.zeros_like(template.fields.E)
    H0 = jnp.zeros_like(template.fields.H)

    def samples(E, H, H_prev):
        E_reg, H_reg = colocate_detector_fields(detector, E, H, H_prev, objects, config)
        return detector._phasor_samples(E_reg, H_reg)

    sample_struct, pullback = jax.vjp(samples, E0, H0, H0)
    # cotangent[key] has shape (1, F_d, *sample_shape): drop the latent time axis, keep frequencies leading.
    per_frequency = {
        k: (
            jnp.zeros((num_frequencies, *v.shape), jnp.result_type(v.dtype, jnp.complex64))
            if _is_zero(cotangent.get(k))
            else cotangent[k][0]
        )
        for k, v in sample_struct.items()
    }
    real_dtype = E0.dtype

    def one_frequency(ct):
        re = pullback({k: v.real.astype(real_dtype) for k, v in ct.items()})
        im = pullback({k: v.imag.astype(real_dtype) for k, v in ct.items()})
        return tuple(r[:, *box.slices] + 1j * i[:, *box.slices] for r, i in zip(re, im))

    Z_E_post, Z_H_post, Z_H_pre = jax.lax.map(one_frequency, per_frequency)
    return Z_E_post, Z_H_post, Z_H_pre


def _loss_detectors(
    objects: ObjectContainer, detector_cotangents: dict[str, DetectorState] | None
) -> list[tuple[PhasorDetector, DetectorState]]:
    """The detectors the loss depends on with their cotangents; raises for detectors that are not a linear DFT."""
    result = []
    for detector in objects.detectors:
        cotangent = detector_cotangents.get(detector.name) if detector_cotangents is not None else None
        if cotangent is None or all(_is_zero(c) for c in jax.tree.leaves(cotangent, is_leaf=_is_zero)):
            continue
        if not getattr(detector, "_adjoint_linear", False) or not isinstance(detector, PhasorDetector):
            raise NotImplementedError(
                f"The loss depends on detector '{detector.name}' ({type(detector).__name__}), whose recording is "
                "not a linear DFT of the fields. The adjoint gradient method supports PhasorDetector and its "
                "subclasses (e.g. ModeOverlapDetector, PhasorPoyntingFluxDetector instead of PoyntingFluxDetector)."
            )
        if not detector.inverse:
            result.append((detector, cotangent))
    return result


def _transposed_step(step: Callable, inv_permittivities: jax.Array, inv_permeabilities: jax.Array | float) -> Callable:
    """``mu -> L^T mu`` for ``mu = (E, H, psi_E, psi_H)``: the VJP of the source-free step, which is linear in the state."""

    def transpose(mu):
        zeros = jax.tree.map(jnp.zeros_like, mu)
        _, pullback = jax.vjp(lambda e, h, pe, ph: step(e, h, pe, ph, inv_permittivities, inv_permeabilities), *zeros)
        return pullback(mu)

    return transpose


def _adjoint_gradient(
    spec: _AdjointSpec,
    design_dft: list[tuple[jax.Array, jax.Array]],
    detector_cotangents: dict[str, DetectorState],
    inv_permittivities: jax.Array,
    inv_permeabilities: jax.Array | float,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    template: ArrayContainer,
    show_progress: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """Gradients w.r.t. the inverse permittivity and (if it is an array) permeability, zero outside Devices."""
    grid_shape = objects.volume.grid_shape
    wrap_axes = get_wrap_padding_axes(objects)
    magnetic = isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0

    # --- adjoint sources of every detector the loss depends on, per objective frequency ---
    sources = []  # (box, global frequency indices (F_d,), r_E (F_d, 3, *box), r_H (F_d, 3, *box))
    for detector, cotangent in _loss_detectors(objects, detector_cotangents):
        indices = _frequency_indices(detector, spec.omegas)
        if np.any(indices < 0):
            logger.warning(
                f"Detector '{detector.name}' records frequencies that are not in GradientConfig.adjoint_frequencies; "
                "their contribution to the gradient is ignored."
            )
        if spec.regime == "pulse":
            _check_pulse_detector(detector, spec.num_steps)
        box = _support_box(detector.grid_slice_tuple, 1, grid_shape, wrap_axes)
        detector_omegas = _detector_omegas(detector)
        nu, mu = _detector_fd_scale(detector, spec.regime, detector_omegas, spec.dt)
        r_E, r_H = _detector_adjoint_sources(
            detector, cotangent, nu, mu, detector_omegas, spec.dt, box, template, objects, config
        )
        sources.append((box, indices, r_E, r_H))

    grad_eps = jnp.zeros_like(inv_permittivities)
    grad_mu = jnp.zeros_like(jnp.asarray(inv_permeabilities))
    if not sources:
        return grad_eps, grad_mu if magnetic else None

    step = _source_free_step(objects, config, key, template)
    transpose_step = _transposed_step(step, inv_permittivities, inv_permeabilities)

    real_dtype = template.fields.E.dtype
    complex_dtype = jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64
    mu0 = jax.tree.map(
        jnp.zeros_like,
        (template.fields.E, template.fields.H, template.fields.psi_E, template.fields.psi_H),
    )
    report_step, close_progress = _adjoint_progress(show_progress, sum(g.source_signals.shape[0] for g in spec.groups))
    offset = 0

    for group in spec.groups:
        group_offset = offset
        offset += group.source_signals.shape[0]
        group_sources = []
        for box, indices, r_E, r_H in sources:
            local, position = [], []
            for i, global_index in enumerate(indices):
                if global_index in group.freq_indices:
                    local.append(i)
                    position.append(group.freq_indices.index(int(global_index)))
            if local:
                local_idx = np.asarray(local)
                group_sources.append((box, local_idx, np.asarray(position), r_E[local_idx], r_H[local_idx]))
        if not group_sources:
            continue
        num_freqs = len(group.freq_indices)
        signals = jnp.asarray(group.source_signals, dtype=complex_dtype)  # (T_adj, K_g)
        phases = jnp.asarray(group.dtft_phases, dtype=complex_dtype)  # (T_adj, K_g)
        lam0 = [
            (jnp.zeros((num_freqs, 3, *b.shape), complex_dtype), jnp.zeros((num_freqs, 3, *b.shape), complex_dtype))
            for b in spec.design_boxes
        ]

        def body(m, carry):
            # The source is injected *before* the transposed step: rho_m = nu_m + r_m, nu_{m+1} = L^T rho_m.
            # Its DTFT relates to that of mu_{m+1} = L^T mu_m + r_m by mu(w) = exp(i w dt) rho(w), applied
            # after the loop. Injecting after the step instead defeats the in-place update of the loop
            # carry and makes every step several times slower.
            mu, lam, peak, _ = carry
            report_step(group_offset + m)
            E, H, psi_E, psi_H = mu
            for box, _, position, r_E, r_H in group_sources:
                beta = signals[m][position]  # (F,)
                E = E.at[:, *box.slices].add(_superpose(beta, r_E, real_dtype))
                H = H.at[:, *box.slices].add(_superpose(beta, r_H, real_dtype))
            regions = _design_fields(E, H, spec.design_boxes)
            phase = phases[m][:, None, None, None, None]
            lam = [(le + phase * e[None], lh + phase * h[None]) for (le, lh), (e, h) in zip(lam, regions)]
            energy = _energy(regions)
            return transpose_step((E, H, psi_E, psi_H)), lam, jnp.maximum(peak, energy), energy

        zero = jnp.zeros((), real_dtype)
        _, lam, peak, last = jax.lax.fori_loop(0, group.source_signals.shape[0], body, (mu0, lam0, zero, zero))
        jax.debug.callback(_report_adjoint_diagnostics, peak, last)
        shift = jnp.asarray(np.exp(1j * spec.omegas[list(group.freq_indices)] * spec.dt), dtype=complex_dtype)
        shift = shift[:, None, None, None, None]
        lam = [(shift * le, shift * lh) for le, lh in lam]

        forward_dft = [
            (fe[np.asarray(group.freq_indices)], fh[np.asarray(group.freq_indices)]) for fe, fh in design_dft
        ]
        g_eps, g_mu = _pair(forward_dft, lam, spec.design_boxes, step, inv_permittivities, inv_permeabilities, template)
        grad_eps = grad_eps + g_eps
        grad_mu = grad_mu + g_mu

    close_progress()
    mask = jnp.asarray(spec.design_mask)[None]
    grad_eps = jnp.where(mask, grad_eps, 0)
    if not magnetic:
        return grad_eps, None
    return grad_eps, jnp.where(mask, grad_mu, 0)


def _pair(
    forward_dft: list[tuple[jax.Array, jax.Array]],
    adjoint_dft: list[tuple[jax.Array, jax.Array]],
    boxes: tuple[_Box, ...],
    step: Callable,
    inv_permittivities: jax.Array,
    inv_permeabilities: jax.Array | float,
    template: ArrayContainer,
) -> tuple[jax.Array, jax.Array]:
    """``sum_k Re B(X_k, Lambda_k)`` with ``B`` the VJP of one source-free step w.r.t. the material arrays.

    ``B`` is real-bilinear, so ``Re B(X, L) = B(Re X, Re L) - B(Im X, Im L)``. The design boxes are embedded
    into zero full-domain fields; the result is only valid inside the devices, where the halo covers the
    update stencils.
    """
    E0 = jnp.zeros_like(template.fields.E)
    H0 = jnp.zeros_like(template.fields.H)
    psi_E0 = jax.tree.map(jnp.zeros_like, template.fields.psi_E)
    psi_H0 = jax.tree.map(jnp.zeros_like, template.fields.psi_H)
    num_freqs = forward_dft[0][0].shape[0]

    def embed(parts: list[jax.Array], base: jax.Array) -> jax.Array:
        for part, box in zip(parts, boxes):
            base = base.at[:, *box.slices].set(part)
        return base

    def bilinear(E, H, lam_E, lam_H):
        _, pullback = jax.vjp(
            lambda ie, im: step(E, H, psi_E0, psi_H0, ie, im)[:2], inv_permittivities, inv_permeabilities
        )
        return pullback((lam_E, lam_H))

    def body(k, acc):
        X_E = [fe[k] for fe, _ in forward_dft]
        X_H = [fh[k] for _, fh in forward_dft]
        L_E = [le[k] for le, _ in adjoint_dft]
        L_H = [lh[k] for _, lh in adjoint_dft]
        re = bilinear(
            embed([x.real for x in X_E], E0),
            embed([x.real for x in X_H], H0),
            embed([x.real for x in L_E], E0),
            embed([x.real for x in L_H], H0),
        )
        im = bilinear(
            embed([x.imag for x in X_E], E0),
            embed([x.imag for x in X_H], H0),
            embed([-x.imag for x in L_E], E0),
            embed([-x.imag for x in L_H], H0),
        )
        return jax.tree.map(lambda a, r, i: a + r + i, acc, re, im)

    zero = (jnp.zeros_like(inv_permittivities), jnp.zeros_like(jnp.asarray(inv_permeabilities)))
    return jax.lax.fori_loop(0, num_freqs, body, zero)


def _block_phases(bins: np.ndarray, num_steps: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """DFT phases ``exp(2 pi i j (m + offset) / N)`` of loop step ``m``, factorized over the DFT blocks.

    Returns the phases of the block starts, shape ``(num_blocks, B)``, and of the steps within a block, shape
    ``(_DFT_BLOCK, B)``; their product is the phase of a step. The phases are exact: ``j (m + offset)`` is
    reduced modulo ``N`` in integers.
    """

    def table(steps: np.ndarray) -> np.ndarray:
        return np.exp(2j * np.pi * (np.outer(steps, bins) % num_steps) / num_steps)

    num_blocks = -(-num_steps // _DFT_BLOCK)
    return table(np.arange(num_blocks) * _DFT_BLOCK + offset), table(np.arange(_DFT_BLOCK))


def _blocked_dft(
    body: Callable,
    carry,
    num_steps: int,
    boxes: tuple[_Box, ...],
    phases: tuple[np.ndarray, np.ndarray],
    real_dtype: jnp.dtype,
    probe: bool = False,
):
    """Run ``carry, regions = body(m, carry)`` for ``m < num_steps`` and accumulate the DFT of the regions.

    ``regions`` holds one ``(E, H)`` pair of shape ``(3, *box)`` per box. The regions of :data:`_DFT_BLOCK`
    consecutive steps are buffered and added to the accumulators with one real matrix product per block,
    which is far cheaper than updating every accumulator at every step when there are many frequencies.

    Returns:
        The final carry, the DFTs ``sum_m phase(m) * regions_m`` (complex, ``(B, 3, *box)`` per box for E and
        H), the peak and last energy of the regions and, with ``probe``, the time signal (``(num_steps,)``)
        of a fixed random projection of the regions, else ``None``.
    """
    starts, within = phases
    num_blocks, num_bins = starts.shape
    complex_dtype = jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64
    starts_c = jnp.asarray(starts, dtype=complex_dtype)
    within_c = jnp.asarray(within, dtype=complex_dtype)
    highest = jax.lax.Precision.HIGHEST
    # accumulators hold [Re; Im] stacked along the frequency axis
    acc0 = [tuple(jnp.zeros((2 * num_bins, 3, *box.shape), real_dtype) for _ in range(2)) for box in boxes]
    buf0 = [tuple(jnp.zeros((_DFT_BLOCK, 3, *box.shape), real_dtype) for _ in range(2)) for box in boxes]
    keys = jax.random.split(jax.random.PRNGKey(0), 2 * len(boxes))
    weights = [jax.random.normal(k, (3, *boxes[i // 2].shape), real_dtype) for i, k in enumerate(keys)]
    series0 = jnp.zeros((num_blocks * _DFT_BLOCK,), real_dtype)
    diag0 = jnp.zeros((2,), real_dtype)  # peak energy, last energy

    def block(b, outer):
        carry, acc, buf, diag, series = outer
        start = b * _DFT_BLOCK
        length = jnp.minimum(_DFT_BLOCK, num_steps - start)

        def step(i, inner):
            carry, buf = inner
            carry, regions = body(start + i, carry)
            buf = [(be.at[i].set(e), bh.at[i].set(h)) for (be, bh), (e, h) in zip(buf, regions)]
            return carry, buf

        carry, buf = jax.lax.fori_loop(0, length, step, (carry, buf))
        valid = jnp.arange(_DFT_BLOCK) < length  # rows past the end of the run hold stale fields
        phase = jnp.where(valid[:, None], starts_c[b][None, :] * within_c, 0)
        stacked = jnp.concatenate([phase.real, phase.imag], axis=1)  # (S, 2B)
        acc = [
            tuple(a + jnp.tensordot(stacked, x, axes=(0, 0), precision=highest) for a, x in zip(pair, fields))
            for pair, fields in zip(acc, buf)
        ]
        flat = [x for fields in buf for x in fields]
        energy = jnp.where(valid, sum(jnp.sum(x * x, axis=tuple(range(1, x.ndim))) for x in flat), 0)
        diag = jnp.stack([jnp.maximum(diag[0], jnp.max(energy)), energy[length - 1]])
        if probe:
            projection = sum(jnp.tensordot(x, w, axes=x.ndim - 1, precision=highest) for x, w in zip(flat, weights))
            series = jax.lax.dynamic_update_slice(series, projection, (start,))
        return carry, acc, buf, diag, series

    carry, acc, _, diag, series = jax.lax.fori_loop(0, num_blocks, block, (carry, acc0, buf0, diag0, series0))
    dft = [tuple(a[:num_bins] + 1j * a[num_bins:] for a in pair) for pair in acc]
    return carry, dft, diag, (series[:num_steps] if probe else None)


def _exact_forward(
    arr: ArrayContainer,
    spec: _ExactSpec,
    body: Callable[[SimulationState], SimulationState],
) -> tuple[SimulationState, list[tuple[jax.Array, jax.Array]]]:
    """Run the forward simulation and take the DFT ``X_j = sum_n s_n exp(i w_j t_n)`` of the step inputs."""

    def record(m, state: SimulationState):
        del m
        current = state[1]
        return body(state), _design_fields(current.fields.E, current.fields.H, spec.design_boxes)

    state, dft, diag, series = _blocked_dft(
        record,
        (jnp.asarray(0, dtype=jnp.int32), arr),
        spec.num_steps,
        spec.design_boxes,
        _block_phases(spec.bins, spec.num_steps, offset=0),
        arr.fields.E.dtype,
        probe=True,
    )
    jax.debug.callback(partial(_report_exact_forward_diagnostics, spec.band_mask), diag, series)
    return state, dft


def _report_exact_forward_diagnostics(band_mask: np.ndarray, diag: np.ndarray, series: np.ndarray) -> None:
    peak, last = (float(v) for v in np.asarray(diag))
    _warn_if_not_decayed(peak, last, "fields", "forward run", "increase the simulation time")
    power = np.abs(np.fft.rfft(np.asarray(series, dtype=np.float64))) ** 2
    total = float(power.sum())
    if total > 0:
        leakage = math.sqrt(float(power[~band_mask].sum()) / total)
        if leakage > _LEAKAGE_WARN_RATIO:
            logger.warning(
                f"Adjoint gradient: a fraction {leakage:.1e} of the spectrum of the fields in the devices lies outside "
                "the band of the exact adjoint mode; expect a relative gradient error of up to that size. Let the "
                "fields decay (longer simulation) or lower GradientConfig.adjoint_band_tolerance."
            )


def _exact_adjoint_gradient(
    spec: _ExactSpec,
    design_dft: list[tuple[jax.Array, jax.Array]],
    detector_cotangents: dict[str, DetectorState],
    inv_permittivities: jax.Array,
    inv_permeabilities: jax.Array | float,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    template: ArrayContainer,
    show_progress: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """Exact-mode gradients w.r.t. the inverse permittivity and (if it is an array) permeability, zero outside Devices.

    The adjoint ``lambda_n = L^T lambda_{n+1} + g_n`` starts from zero after the last step. A phasor
    detector samples ``Q(s_{n+1}, s_n)`` with coefficient ``c(n)`` at step ``n``, so it contributes
    ``g_{n+1} = Re(c(n) Z_post + c(n + 1) Z_pre)`` with its patterns ``Z``. The loop over ``m`` runs
    ``n = N - 1 - m`` downwards and records ``lambda_{n+1}`` with ``exp(-i w_j t_n)``, i.e. ``conj(Y_j)``.
    """
    grid_shape = objects.volume.grid_shape
    wrap_axes = get_wrap_padding_axes(objects)
    magnetic = isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0
    real_dtype = template.fields.E.dtype
    complex_dtype = jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64
    num_steps = spec.num_steps
    times = np.arange(num_steps + 1, dtype=np.float64) * config.time_step_duration

    sources = []  # (box, coefficients (N + 1, F_d), Z_E_post, Z_H_post, Z_H_pre)
    for detector, cotangent in _loss_detectors(objects, detector_cotangents):
        box = _support_box(detector.grid_slice_tuple, 1, grid_shape, wrap_axes)
        omegas = _detector_omegas(detector)
        patterns = _detector_patterns(detector, cotangent, len(omegas), box, template, objects, config)
        weights = np.append(_detector_step_weights(detector), 0.0)  # c(N) = 0: no sample after the last step
        coefficients = jnp.asarray(weights[:, None] * _phase_table(omegas, times), dtype=complex_dtype)
        sources.append((box, coefficients, *patterns))

    if not sources:
        return jnp.zeros_like(inv_permittivities), jnp.zeros_like(jnp.asarray(inv_permeabilities)) if magnetic else None
    nu0 = jax.tree.map(
        jnp.zeros_like, (template.fields.E, template.fields.H, template.fields.psi_E, template.fields.psi_H)
    )

    step = _source_free_step(objects, config, key, template)
    transpose_step = _transposed_step(step, inv_permittivities, inv_permeabilities)
    report_step, close_progress = _adjoint_progress(show_progress, num_steps)

    def body(m, nu):
        # Inject g_{n+1} into nu_{n+1} = L^T lambda_{n+2} to form lambda_{n+1}, record it and step back.
        report_step(m)
        n = num_steps - 1 - m
        E, H, psi_E, psi_H = nu
        for box, coefficients, Z_E_post, Z_H_post, Z_H_pre in sources:
            post, pre = coefficients[n], coefficients[n + 1]
            E = E.at[:, *box.slices].add(_superpose(post, Z_E_post, real_dtype))
            H = H.at[:, *box.slices].add(_superpose(post, Z_H_post, real_dtype) + _superpose(pre, Z_H_pre, real_dtype))
        regions = _design_fields(E, H, spec.design_boxes)
        return transpose_step((E, H, psi_E, psi_H)), regions

    # exp(-i w_j t_n) with n = N - 1 - m equals exp(2 pi i j (m + 1) / N)
    _, adjoint_dft, _, _ = _blocked_dft(
        body, nu0, num_steps, spec.design_boxes, _block_phases(spec.bins, num_steps, offset=1), real_dtype
    )
    close_progress()
    weights = jnp.asarray(spec.pair_weights, dtype=complex_dtype)[:, None, None, None, None]
    lam = [(weights * le, weights * lh) for le, lh in adjoint_dft]
    grad_eps, grad_mu = _pair(
        design_dft, lam, spec.design_boxes, step, inv_permittivities, inv_permeabilities, template
    )
    mask = jnp.asarray(spec.design_mask)[None]
    grad_eps = jnp.where(mask, grad_eps, 0)
    if not magnetic:
        return grad_eps, None
    return grad_eps, jnp.where(mask, grad_mu, 0)
