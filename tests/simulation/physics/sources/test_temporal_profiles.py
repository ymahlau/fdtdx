"""Physics simulation tests: temporal profile validation.

  7a. SingleFrequencyProfile reaches steady-state CW with nonzero phasor.
  7b. GaussianPulseProfile excites multiple frequencies within its bandwidth.

Domain: 1D-like (periodic xy, PML z).
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6
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

    _C = 3e8  # m/s
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
            f"expected {expected_ratio:.4f} (Gaussian at {log_ratio:+.1f} × σ_f²/2), "
            f"rel_err = {rel_err:.2%}"
        )
