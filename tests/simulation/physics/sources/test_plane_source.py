"""Physics simulation tests: UniformPlaneSource validation.

  4a. Amplitude consistency: doubling static_amplitude_factor doubles phasor amplitude.
  4b. Spatial uniformity: amplitude and phase are constant across the xy-plane.

The phase velocity and impedance tests for UniformPlaneSource already exist in
test_plane_wave.py. This file tests additional properties specific to the source.

Domain: Same 1D-like layout as test_plane_wave.py (periodic xy, PML z).
"""

import jax
import jax.numpy as jnp

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6
_RESOLUTION = 50e-9
_PML_CELLS = 10
_DOMAIN_XY = 5 * _RESOLUTION
_DOMAIN_Z = 4e-6

_SOURCE_Z = _PML_CELLS + 2
_DET_Z = _SOURCE_Z + 10

_SIM_TIME = 120e-15


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_and_run(amplitude_factor=1.0, reduce_volume=True):
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

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
        static_amplitude_factor=amplitude_factor,
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
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        reduce_volume=reduce_volume,
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


def test_uniform_plane_source_amplitude_consistency():
    """Doubling static_amplitude_factor doubles the phasor amplitude at the detector."""
    arrays_1x = _build_and_run(amplitude_factor=1.0)
    arrays_2x = _build_and_run(amplitude_factor=2.0)

    # Extract Ex phasor (may have trailing dims even with reduce_volume)
    p1 = complex(arrays_1x.detector_states["phasor"]["phasor"][0, 0].ravel()[0])
    p2 = complex(arrays_2x.detector_states["phasor"]["phasor"][0, 0].ravel()[0])

    assert abs(p1) > 1e-30, "1× phasor amplitude is zero"

    ratio = abs(p2) / abs(p1)
    assert abs(ratio - 2.0) / 2.0 < 0.05, f"Amplitude ratio: {ratio:.3f}, expected 2.0"


def test_uniform_plane_source_xy_uniformity():
    """Output amplitude and phase should be constant across the xy-plane."""
    arrays = _build_and_run(reduce_volume=False)

    # phasor shape: (num_time_steps, num_wavelengths, num_components, nx, ny, nz)
    phasor = arrays.detector_states["phasor"]["phasor"]
    # Take last time step, wavelength 0, Ex component (index 0), single z-slice → (nx, ny)
    ex_phasor = phasor[-1, 0, 0, :, :, 0]

    amplitudes = jnp.abs(ex_phasor)
    phases = jnp.angle(ex_phasor)

    # All amplitudes should be non-zero
    assert jnp.all(amplitudes > 1e-30), "Some Ex phasor amplitudes are zero"

    # Amplitude should be uniform: max deviation < 1% of mean
    mean_amp = jnp.mean(amplitudes)
    max_amp_dev = jnp.max(jnp.abs(amplitudes - mean_amp)) / mean_amp
    assert max_amp_dev < 0.01, f"Amplitude varies across xy: max relative deviation {float(max_amp_dev):.4f}"

    # Phase should be uniform: max deviation < 0.02 rad (~1 degree)
    mean_phase = jnp.mean(phases)
    max_phase_dev = jnp.max(jnp.abs(phases - mean_phase))
    assert max_phase_dev < 0.02, f"Phase varies across xy: max deviation {float(max_phase_dev):.4f} rad"
