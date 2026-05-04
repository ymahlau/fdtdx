"""End-to-end physics checks for mildly stretched rectilinear grids."""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

_WAVELENGTH = 1.0e-6
_RESOLUTION = 50.0e-9
_PML_CELLS = 8
_NX = 3
_NY = 3
_NZ = 60
_DOMAIN_XY = _NX * _RESOLUTION
_DOMAIN_Z = _NZ * _RESOLUTION
_SOURCE_Z = _PML_CELLS + 2
_DET1_Z = _SOURCE_Z + 5
_DET2_Z = _DET1_Z + 5
_SIM_TIME = 90.0e-15


def _stretched_z_edges() -> jnp.ndarray:
    """Return a mildly stretched z grid with the same total physical length."""
    cells = np.arange(_NZ, dtype=float)
    widths = _RESOLUTION * (1.0 + 0.08 * np.sin(2.0 * np.pi * (cells + 0.5) / _NZ))
    widths *= _DOMAIN_Z / widths.sum()
    return jnp.asarray(np.concatenate([[0.0], np.cumsum(widths)]), dtype=jnp.float32)


def _grid(kind: str) -> fdtdx.UniformGrid | fdtdx.RectilinearGrid:
    if kind == "uniform":
        return fdtdx.UniformGrid(spacing=_RESOLUTION)
    if kind == "stretched":
        return fdtdx.RectilinearGrid.custom(
            x_edges=jnp.linspace(0.0, _DOMAIN_XY, _NX + 1),
            y_edges=jnp.linspace(0.0, _DOMAIN_XY, _NY + 1),
            z_edges=_stretched_z_edges(),
        )
    raise ValueError(f"Unknown grid kind: {kind}")


def _add_real_z_plane(obj, volume, objects, constraints, z_coord: float):
    constraints.extend(
        [
            obj.same_size(volume, axes=(0, 1)),
            obj.place_at_center(volume, axes=(0, 1)),
            fdtdx.RealCoordinateConstraint(
                object=obj.name,
                axes=(2,),
                sides=("-",),
                coordinates=(z_coord,),
            ),
        ]
    )
    objects.append(obj)


def _build_and_run(kind: str):
    config = fdtdx.SimulationConfig(
        grid=_grid(kind),
        time=_SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_real_shape=(_DOMAIN_XY, _DOMAIN_XY, _DOMAIN_Z))
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
    )
    _add_real_z_plane(source, volume, objects, constraints, _SOURCE_Z * _RESOLUTION)

    for name, z_idx in (("d1", _DET1_Z), ("d2", _DET2_Z)):
        detector = fdtdx.PhasorDetector(
            name=name,
            partial_grid_shape=(None, None, 1),
            wave_characters=(wave,),
            components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
            reduce_volume=True,
            plot=False,
        )
        _add_real_z_plane(detector, volume, objects, constraints, z_idx * _RESOLUTION)

    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=config, key=key)
    return arrays, obj_container, config


def _phasor(arrays, detector_name: str, component: int) -> complex:
    return complex(arrays.detector_states[detector_name]["phasor"][0, 0, component])


def _detector_center_z(obj_container, config, detector_name: str) -> float:
    detector = next(det for det in obj_container.detectors if det.name == detector_name)
    lower, upper = detector.grid_slice_tuple[2]
    grid = config.realized_grid
    assert grid is not None
    return float(0.5 * (grid.z_edges[lower] + grid.z_edges[upper]))


def _measured_k(arrays, obj_container, config) -> float:
    d1 = _phasor(arrays, "d1", 0)
    d2 = _phasor(arrays, "d2", 0)
    separation = _detector_center_z(obj_container, config, "d2") - _detector_center_z(obj_container, config, "d1")
    delta_phi = (np.angle(d2) - np.angle(d1)) % (2.0 * np.pi)
    return float(delta_phi / separation)


def _impedance(arrays, detector_name: str) -> float:
    ex = _phasor(arrays, detector_name, 0)
    hy = _phasor(arrays, detector_name, 4)
    return float(abs(ex) / abs(hy))


def test_mildly_stretched_grid_matches_uniform_vacuum_plane_wave():
    """A full FDTD run on a mildly stretched grid matches uniform-grid observables."""
    uniform_arrays, uniform_objects, uniform_config = _build_and_run("uniform")
    stretched_arrays, stretched_objects, stretched_config = _build_and_run("stretched")

    k_uniform = _measured_k(uniform_arrays, uniform_objects, uniform_config)
    k_stretched = _measured_k(stretched_arrays, stretched_objects, stretched_config)
    z_uniform = _impedance(uniform_arrays, "d1")
    z_stretched = _impedance(stretched_arrays, "d1")

    assert np.isclose(k_stretched, k_uniform, rtol=0.10), (k_uniform, k_stretched)
    assert np.isclose(z_stretched, z_uniform, rtol=0.10), (z_uniform, z_stretched)
