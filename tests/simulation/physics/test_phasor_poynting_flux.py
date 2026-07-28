"""Physics simulation test: frequency-domain Poynting flux detectors.

Validates the two phasor-based (frequency-domain) Poynting flux detectors against
their time-domain counterparts in a single CW plane-wave run:

* ``PhasorPoyntingFluxDetector.compute_poynting_flux`` must equal the steady-state
  time-average of a co-located ``PoyntingFluxDetector`` (both give
  ``<S> = 1/2 Re(E x H*)`` integrated over the plane).
* ``ClosedSurfacePhasorPoyntingFluxDetector.compute_net_flux`` must equal the
  time-average of a co-located ``ClosedSurfacePoyntingFluxDetector`` -- here the
  box straddles a lossy slab so the net flux is a genuine (nonzero) absorbed power.

All detectors are gated to the SAME integer-periods steady-state window so the
phasor DFT and the time-domain average cover identical steps; otherwise the DFT
would integrate the source turn-on transient and the two would disagree.

Domain (50 nm resolution, z is propagation axis, periodic x/y, PML in z):
  80 cells in z; source at z=12; measurement plane at z=20; lossy slab at z=22-25;
  closed-surface box spans z=20-27 (faces in vacuum on either side of the slab).
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

_WAVELENGTH = 1e-6
_RESOLUTION = 50e-9
_PML_CELLS = 10
_DOMAIN_XY = 3 * _RESOLUTION
_DOMAIN_Z = 4e-6
_SIM_TIME = 120e-15

_SOURCE_Z = _PML_CELLS + 2  # 12
_PLANE_Z = _SOURCE_Z + 8  # 20 (measurement plane / box min-z face)
_SLAB_Z = _PLANE_Z + 2  # 22 (lossy slab, 4 cells thick, inside the box)

_PERIOD = _WAVELENGTH / fdtdx.constants.c
_SWITCH = fdtdx.OnOffSwitch(start_after_periods=25, on_for_periods=10, period=_PERIOD)

_PLANE_TOL = 0.02  # 2 % (observed ~0.02 %)
_BOX_TOL = 0.03  # 3 % (observed ~0.3 %; net flux is a small difference of two faces)


def _build():
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RESOLUTION), time=_SIM_TIME, dtype=jnp.float32)
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(partial_real_shape=(_DOMAIN_XY, _DOMAIN_XY, _DOMAIN_Z))
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={"min_x": "periodic", "max_x": "periodic", "min_y": "periodic", "max_y": "periodic"},
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
    constraints += [
        source.same_size(volume, axes=(0, 1)),
        source.place_at_center(volume, axes=(0, 1)),
        source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_SOURCE_Z,)),
    ]
    objects.append(source)

    # lossy slab so the closed surface encloses a real absorbed power
    slab = fdtdx.UniformMaterialObject(
        partial_grid_shape=(None, None, 4),
        material=fdtdx.Material(permittivity=4.0, electric_conductivity=300.0),
    )
    constraints += [
        slab.same_size(volume, axes=(0, 1)),
        slab.place_at_center(volume, axes=(0, 1)),
        slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_SLAB_Z,)),
    ]
    objects.append(slab)

    def _at_plane(det):
        # all detectors share the same transverse extent and z-min face at _PLANE_Z
        constraints.extend(
            [
                det.same_size(volume, axes=(0, 1)),
                det.place_at_center(volume, axes=(0, 1)),
                det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_PLANE_Z,)),
            ]
        )
        objects.append(det)

    _at_plane(
        fdtdx.PoyntingFluxDetector(name="td_plane", partial_grid_shape=(None, None, 1), direction="+", switch=_SWITCH)
    )
    _at_plane(
        fdtdx.PhasorPoyntingFluxDetector(
            name="fd_plane", partial_grid_shape=(None, None, 1), direction="+", wave_characters=(wave,), switch=_SWITCH
        )
    )
    _at_plane(
        fdtdx.ClosedSurfacePoyntingFluxDetector(
            name="td_box", partial_grid_shape=(None, None, 8), axes=(2,), switch=_SWITCH
        )
    )
    _at_plane(
        fdtdx.ClosedSurfacePhasorPoyntingFluxDetector(
            name="fd_box", partial_grid_shape=(None, None, 8), axes=(2,), wave_characters=(wave,), switch=_SWITCH
        )
    )
    return objects, constraints, config


def test_phasor_poynting_matches_time_domain():
    objects, constraints, config = _build()
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=config, key=key)

    dets = {d.name: d for d in obj_container.detectors}

    def td_avg(name):
        # every recorded step is inside the gated steady-state window
        return float(np.mean(np.asarray(arrays.detector_states[name]["poynting_flux"][:, 0])))

    # --- single plane ---
    td_plane = td_avg("td_plane")
    fd_plane = float(dets["fd_plane"].compute_poynting_flux(arrays.detector_states["fd_plane"])[0])
    assert abs(td_plane) > 0
    assert abs(fd_plane - td_plane) / abs(td_plane) < _PLANE_TOL

    # --- closed surface (net absorbed power across the lossy slab) ---
    td_box = td_avg("td_box")
    fd_box = float(dets["fd_box"].compute_net_flux(arrays.detector_states["fd_box"])[0])
    # the enclosed slab absorbs power -> net outward flux is negative and nonzero
    assert td_box < 0
    assert abs(fd_box - td_box) / abs(td_box) < _BOX_TOL
