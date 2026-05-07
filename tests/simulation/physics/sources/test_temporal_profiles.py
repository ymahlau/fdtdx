"""Physics simulation tests: temporal profile validation.

  7a. SingleFrequencyProfile reaches steady-state CW with nonzero phasor.
  7b. GaussianPulseProfile excites multiple frequencies within its bandwidth.
  7c. CustomTimeSignalProfile sampled as an amplitude-scaled, phase-shifted,
      delayed Gaussian reproduces GaussianPulseProfile physics up to the
      analytically expected amplitude, phase, and delay factors.

Domain: 1D-like (periodic xy, PML z).
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6
_C = 3e8
_RESOLUTION = 50e-9
_PML_CELLS = 10
_DOMAIN_XY = 3 * _RESOLUTION
_DOMAIN_Z = 4e-6

_SOURCE_Z = _PML_CELLS + 2
_DET_Z = _SOURCE_Z + 10

_SIM_TIME = 120e-15


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_base():
    config = fdtdx.SimulationConfig(
        resolution=_RESOLUTION,
        time=_SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(_DOMAIN_XY, _DOMAIN_XY, _DOMAIN_Z),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            "min_x": "periodic",
            "max_x": "periodic",
            "min_y": "periodic",
            "max_y": "periodic",
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    return objects, constraints, config, volume


def _run(objects, constraints, config):
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=config, key=key)
    return arrays


def _phasor_detector_wave_characters():
    """Return phasor detector WaveCharacters at f_c and f_c ± σ_f."""
    center_frequency = fdtdx.WaveCharacter(wavelength=_WAVELENGTH).get_frequency()
    spectral_width_frequency = fdtdx.WaveCharacter(wavelength=3e-6).get_frequency()
    frequencies = [
        center_frequency - spectral_width_frequency,
        center_frequency,
        center_frequency + spectral_width_frequency,
    ]
    return [fdtdx.WaveCharacter(wavelength=_C / freq) for freq in frequencies]


def _reference_gaussian_profile():
    """Return the built-in Gaussian source profile used as the custom-source reference."""
    center_wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    spectral_width = fdtdx.WaveCharacter(wavelength=3e-6)
    return fdtdx.GaussianPulseProfile(center_wave=center_wave, spectral_width=spectral_width)


def _custom_time_signal_profile(config, amplitude: float, delay: float, phase_shift: float):
    """Sample an amplitude-scaled, phase-shifted, delayed Gaussian pulse."""
    time = jnp.arange(config.time_steps_total, dtype=config.dtype) * config.time_step_duration
    delayed_time = time - delay

    center_frequency = fdtdx.WaveCharacter(wavelength=_WAVELENGTH).get_frequency()
    spectral_width_frequency = fdtdx.WaveCharacter(wavelength=3e-6).get_frequency()
    sigma_t = 1.0 / (2 * jnp.pi * spectral_width_frequency)
    t0 = 6 * sigma_t
    envelope = jnp.exp(-((delayed_time - t0) ** 2) / (2 * sigma_t**2))
    carrier_phase = 2 * jnp.pi * center_frequency * delayed_time + phase_shift
    carrier = jnp.real(jnp.exp(-1j * carrier_phase))
    signal = amplitude * envelope * carrier

    return fdtdx.CustomTimeSignalProfile(
        signal=signal,
        time_step_duration=config.time_step_duration,
        interpolation="linear",
    )


def _build_custom_source_setup(temporal_profile):
    """Build a small plane-source simulation for custom-profile comparison."""
    objects, constraints, config, volume = _build_base()

    center_wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=center_wave,
        temporal_profile=temporal_profile,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_SOURCE_Z,)),
        ]
    )
    objects.append(source)

    phasor_det = fdtdx.PhasorDetector(
        name="phasor",
        partial_grid_shape=(None, None, 1),
        wave_characters=_phasor_detector_wave_characters(),
        components=("Ex",),
        reduce_volume=True,
        scaling_mode="pulse",
        plot=False,
    )
    constraints.extend(
        [
            phasor_det.same_size(volume, axes=(0, 1)),
            phasor_det.place_at_center(volume, axes=(0, 1)),
            phasor_det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_DET_Z,)),
        ]
    )
    objects.append(phasor_det)

    field_det = fdtdx.FieldDetector(
        name="field",
        partial_grid_shape=(None, None, 1),
        components=("Ex",),
        reduce_volume=True,
        plot=False,
    )
    constraints.extend(
        [
            field_det.same_size(volume, axes=(0, 1)),
            field_det.place_at_center(volume, axes=(0, 1)),
            field_det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_DET_Z,)),
        ]
    )
    objects.append(field_det)

    return objects, constraints, config


def _custom_phasors(arrays):
    """Return Ex phasors with shape (num_frequencies,)."""
    return np.asarray(arrays.detector_states["phasor"]["phasor"][0, :, 0])


def _custom_field_trace(arrays):
    """Return the reduced Ex time trace from the field detector."""
    return np.asarray(arrays.detector_states["field"]["fields"][:, 0])


def _analytic_envelope(signal):
    """Return the Hilbert-transform envelope using an FFT analytic signal."""
    spectrum = np.fft.fft(signal)
    weights = np.zeros(signal.shape[0])
    if signal.shape[0] % 2 == 0:
        weights[0] = 1
        weights[signal.shape[0] // 2] = 1
        weights[1 : signal.shape[0] // 2] = 2
    else:
        weights[0] = 1
        weights[1 : (signal.shape[0] + 1) // 2] = 2
    return np.abs(np.fft.ifft(spectrum * weights))


# ── Tests ────────────────────────────────────────────────────────────────────


def test_single_frequency_steady_state():
    """SingleFrequencyProfile (default) produces nonzero steady-state phasor."""
    objects, constraints, config, volume = _build_base()

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
        # Default temporal_profile = SingleFrequencyProfile
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_SOURCE_Z,)),
        ]
    )
    objects.append(source)

    det = fdtdx.PhasorDetector(
        name="phasor",
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave,),
        components=("Ex",),
        reduce_volume=True,
        plot=False,
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(0, 1)),
            det.place_at_center(volume, axes=(0, 1)),
            det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_DET_Z,)),
        ]
    )
    objects.append(det)

    arrays = _run(objects, constraints, config)

    p = complex(arrays.detector_states["phasor"]["phasor"][0, 0].ravel()[0])
    assert abs(p) > 1e-10, f"CW phasor amplitude should be nonzero, got |p|={abs(p):.2e}"


def test_gaussian_pulse_broadband():
    """GaussianPulseProfile spectral amplitude follows the Gaussian envelope.

    GaussianPulseProfile produces the time-domain signal::

        E(t) = exp(-((t - t0)² / (2σ_t²))) · cos(2π f_c t),   t0 = 6σ_t

    whose Fourier transform is Gaussian in frequency with the same σ_f::

        |Â(f)| ∝ exp(-(f - f_c)² / (2σ_f²)),   σ_f = 1/(2π σ_t)

    Five phasor detectors placed at f_c and ±1σ_f, ±2σ_f verify the ratios::

        |A(f_c ± 1σ)| / |A(f_c)| = exp(-0.5) ≈ 0.607
        |A(f_c ± 2σ)| / |A(f_c)| = exp(-2.0) ≈ 0.135

    sim_time = 12σ_t ensures the full pulse passes the detector before the
    simulation ends (the pulse peaks at t0 = 6σ_t and is negligible after
    t0 + 6σ_t).
    """
    objects, constraints, config, volume = _build_base()

    _SIGMA_F = 1e14  # Hz — spectral standard deviation
    _F_CENTER = _C / _WAVELENGTH  # 3e14 Hz

    sigma_t = 1.0 / (2 * np.pi * _SIGMA_F)  # ~1.59 fs
    sim_time = 12 * sigma_t  # t0=6σ_t peak + 6σ_t tail ≈ 19 fs

    config = fdtdx.SimulationConfig(
        resolution=_RESOLUTION,
        time=sim_time,
        dtype=jnp.float32,
    )

    center_wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    # WaveCharacter whose frequency = σ_f  →  λ = c/σ_f = 3 µm
    spectral_width = fdtdx.WaveCharacter(wavelength=_C / _SIGMA_F)

    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=center_wave,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
        temporal_profile=fdtdx.GaussianPulseProfile(
            spectral_width=spectral_width,
            center_wave=center_wave,
        ),
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_SOURCE_Z,)),
        ]
    )
    objects.append(source)

    # Detectors at f_c and ±1σ_f, ±2σ_f (expressed as wavelengths)
    # f_c = 3e14 Hz  →  λ = 1000 nm
    # f_c ± 1σ = 4e14 / 2e14  →  λ = 750 / 1500 nm
    # f_c ± 2σ = 5e14 / 1e14  →  λ = 600 / 3000 nm
    det_specs = [
        ("center", _C / _F_CENTER, 0.0),
        ("plus1s", _C / (_F_CENTER + _SIGMA_F), -0.5),
        ("minus1s", _C / (_F_CENTER - _SIGMA_F), -0.5),
        ("plus2s", _C / (_F_CENTER + 2 * _SIGMA_F), -2.0),
        ("minus2s", _C / (_F_CENTER - 2 * _SIGMA_F), -2.0),
    ]

    for name, wl, _ in det_specs:
        det = fdtdx.PhasorDetector(
            name=name,
            partial_grid_shape=(None, None, 1),
            wave_characters=(fdtdx.WaveCharacter(wavelength=wl),),
            components=("Ex",),
            reduce_volume=True,
            plot=False,
        )
        constraints.extend(
            [
                det.same_size(volume, axes=(0, 1)),
                det.place_at_center(volume, axes=(0, 1)),
                det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_DET_Z,)),
            ]
        )
        objects.append(det)

    arrays = _run(objects, constraints, config)

    amps = {name: abs(complex(arrays.detector_states[name]["phasor"][0, 0].ravel()[0])) for name, _, _ in det_specs}

    assert amps["center"] > 1e-20, "Center phasor is zero — pulse not detected"

    for name, _, log_ratio in det_specs[1:]:
        expected_ratio = float(np.exp(log_ratio))
        measured_ratio = amps[name] / amps["center"]
        rel_err = abs(measured_ratio - expected_ratio) / expected_ratio
        assert rel_err < 0.05, (
            f"{name}: |A|/|A_center| = {measured_ratio:.4f}, "
            f"expected {expected_ratio:.4f} (Gaussian at {log_ratio:+.1f} x σ_f²/2), "
            f"rel_err = {rel_err:.2%}"
        )


def test_custom_time_signal_matches_gaussian_pulse_physics():
    """CustomTimeSignalProfile reproduces GaussianPulseProfile physics up to known factors.

    The custom signal is sampled as an amplitude-scaled, phase-shifted, delayed
    Gaussian on the FDTD time grid. These transformations have analytical
    effects: phasor magnitudes scale with amplitude, the center phasor rotates
    by exp(i 2πfτ - iφ), and the Hilbert-envelope peak arrives delay_steps
    later.
    """
    amplitude = 0.7
    phase_shift = np.pi / 5
    delay_steps = 8
    phasor_mag_tol = 0.01
    center_complex_phasor_tol = 0.01
    time_trace_peak_tol = 0.01
    peak_step_tol = 1

    reference_profile = _reference_gaussian_profile()
    obj_ref, con_ref, cfg_ref = _build_custom_source_setup(reference_profile)
    delay = delay_steps * cfg_ref.time_step_duration
    custom_profile = _custom_time_signal_profile(
        cfg_ref,
        amplitude=amplitude,
        delay=delay,
        phase_shift=phase_shift,
    )
    obj_custom, con_custom, cfg_custom = _build_custom_source_setup(
        custom_profile,
    )

    assert cfg_custom.time_step_duration == cfg_ref.time_step_duration
    assert cfg_custom.time_steps_total == cfg_ref.time_steps_total

    arrays_ref = _run(obj_ref, con_ref, cfg_ref)
    arrays_custom = _run(obj_custom, con_custom, cfg_custom)

    phasor_ref = _custom_phasors(arrays_ref)
    phasor_custom = _custom_phasors(arrays_custom)
    amp_ref = np.abs(phasor_ref)
    amp_custom = np.abs(phasor_custom)
    expected_amp = amplitude * amp_ref

    # Amplitude check: all monitored phasor magnitudes should scale by A.
    assert np.all(amp_ref > 1e-30), f"Reference phasor amplitudes are zero: {amp_ref}"
    mag_rel_err = np.abs(amp_custom - expected_amp) / expected_amp
    assert np.all(mag_rel_err < phasor_mag_tol), (
        f"Phasor magnitude relative errors {mag_rel_err} exceed {phasor_mag_tol}; "
        f"expected={expected_amp}, custom={amp_custom}"
    )

    center_idx = 1
    center_frequency = _phasor_detector_wave_characters()[center_idx].get_frequency()
    expected_center_phasor = (
        amplitude * phasor_ref[center_idx] * np.exp(1j * 2 * np.pi * center_frequency * delay - 1j * phase_shift)
    )
    # Phase check: with FDTDX's exp(+iωt) phasor convention, delay τ and
    # carrier phase φ rotate the center phasor by exp(i2πfτ - iφ).
    center_rel_err = abs(phasor_custom[center_idx] - expected_center_phasor) / abs(expected_center_phasor)
    assert center_rel_err < center_complex_phasor_tol, (
        f"Corrected center-frequency phasor relative error {center_rel_err:.3f} exceeds {center_complex_phasor_tol}"
    )

    trace_ref = _custom_field_trace(arrays_ref)
    trace_custom = _custom_field_trace(arrays_custom)
    assert trace_custom.shape == trace_ref.shape

    envelope_ref = _analytic_envelope(trace_ref)
    envelope_custom = _analytic_envelope(trace_custom)
    peak_ref = float(np.max(envelope_ref))
    peak_custom = float(np.max(envelope_custom))
    expected_peak = amplitude * peak_ref
    assert peak_ref > 1e-30, "Reference propagated time-domain field is zero"

    # Time-domain amplitude check: the propagated field envelope should scale by A.
    peak_rel_err = abs(peak_custom - expected_peak) / expected_peak
    assert peak_rel_err < time_trace_peak_tol, (
        f"Time-trace peak relative error {peak_rel_err:.3f} exceeds {time_trace_peak_tol}; "
        f"expected={expected_peak:.4e}, custom={peak_custom:.4e}"
    )

    # Delay check: the Hilbert envelope removes carrier-phase ambiguity, so its
    # peak should arrive delay_steps later.
    peak_step_ref = int(np.argmax(envelope_ref))
    peak_step_custom = int(np.argmax(envelope_custom))
    peak_step_err = abs((peak_step_custom - peak_step_ref) - delay_steps)
    assert peak_step_err <= peak_step_tol, (
        f"Envelope peak arrival differs from the expected {delay_steps}-step delay "
        f"by {peak_step_err} time steps, "
        f"expected <= {peak_step_tol}; "
        f"reference={peak_step_ref}, custom={peak_step_custom}"
    )
