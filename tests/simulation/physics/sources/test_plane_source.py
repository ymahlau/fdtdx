"""Physics simulation tests: UniformPlaneSource validation.

  4a. Amplitude consistency: doubling static_amplitude_factor doubles phasor amplitude.

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
_DOMAIN_XY = 3 * _RESOLUTION
_DOMAIN_Z = 4e-6

_SOURCE_Z = _PML_CELLS + 2
_DET_Z = _SOURCE_Z + 10

_SIM_TIME = 120e-15


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_and_run(amplitude_factor=1.0):
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
