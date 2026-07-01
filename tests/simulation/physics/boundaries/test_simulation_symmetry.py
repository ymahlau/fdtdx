"""Physics test: automatic config.symmetry reproduces the full-domain result after unfolding.

This is the automated counterpart of ``test_quarter_domain_symmetry.py``. There, the quarter
domain (PEC min_y, PMC min_z) and its unfolding were built by hand. Here we build the *full*
model once, run it both with ``config.symmetry=(0, 0, 0)`` (reference) and with
``config.symmetry=(0, -1, 1)`` (PEC y-plane, PMC z-plane), and check that
``fdtdx.unfold_detector_states`` of the reduced run reproduces the reference:

* the unfolded Ey phasor field matches the full-domain field element-wise, and
* the unfolded scalar Poynting flux matches the full-domain total power (the x4 quarter-domain
  scaling is applied automatically).

An Ey-polarized +x plane wave is even about both the PEC y-plane (Ey normal) and the PMC z-plane
(Ey tangential), so both walls act as perfect mirrors for this polarization.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx
from fdtdx.constants import c as c0

_WAVELENGTH = 1e-6
_RESOLUTION = 25e-9  # 40 cells / λ
_DOMAIN_YZ = 1e-6  # full transverse extent (40 cells, even -> clean halving)
_DOMAIN_X = 5e-6
_PML_CELLS = 10
_SOURCE_X = _PML_CELLS + 2
_DET_X = _SOURCE_X + 20
_SIM_TIME = 80e-15

_DT = 0.99 * _RESOLUTION / (c0 * np.sqrt(3))
_STEPS_PER_PERIOD = max(1, round(_WAVELENGTH / (c0 * _DT)))
_N_AVG = 10 * _STEPS_PER_PERIOD


_NX = round(_DOMAIN_X / _RESOLUTION)
_NYZ = round(_DOMAIN_YZ / _RESOLUTION)


def _symmetric_edges(n_cells, base):
    """Edge array of ``n_cells`` cells, palindromic widths from ``base`` (center) to ``1.3*base`` (edges).

    Widths satisfy ``w[i] == w[n-1-i]`` (mirror-symmetric about the center, so a symmetric model
    reduces exactly) and the smallest cell equals ``base`` (so ``min_spacing`` — and therefore the
    CFL time step — match the uniform grid, keeping the test fast). The 30% taper makes the grid
    genuinely non-uniform (``is_uniform`` is False).
    """
    idx = np.arange(n_cells)
    center = (n_cells - 1) / 2.0
    widths = base * (1.0 + 0.3 * np.abs(idx - center) / center)
    return jnp.asarray(np.concatenate([[0.0], np.cumsum(widths)]), dtype=jnp.float32)


def _rectilinear_grid():
    """A non-uniform rectilinear grid: uniform x (clean propagation), symmetric non-uniform y, z."""
    return fdtdx.RectilinearGrid.custom(
        x_edges=jnp.arange(_NX + 1) * _RESOLUTION,
        y_edges=_symmetric_edges(_NYZ, _RESOLUTION),
        z_edges=_symmetric_edges(_NYZ, _RESOLUTION),
    )


def _build(symmetry, grid=None):
    config = fdtdx.SimulationConfig(
        grid=grid if grid is not None else fdtdx.UniformGrid(spacing=_RESOLUTION),
        time=_SIM_TIME,
        dtype=jnp.float32,
        symmetry=symmetry,
    )
    objects, constraints = [], []

    # partial_grid_shape (not real_shape) so the volume cell count is fixed independently of the
    # grid's physical extent — lets the rectilinear grid taper its cells without resizing the domain.
    volume = fdtdx.SimulationVolume(partial_grid_shape=(_NX, _NYZ, _NYZ))
    objects.append(volume)

    # Periodic in y,z (infinite uniform slab); PML in x. Under symmetry the min_y/min_z periodic
    # boundaries are replaced by the PEC/PMC walls, leaving periodic only on the far side — exactly
    # the validated quarter-domain configuration.
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            "min_y": "periodic",
            "max_y": "periodic",
            "min_z": "periodic",
            "max_z": "periodic",
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(1, None, None),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(0, 1, 0),
        normalize_by_energy=False,
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(1, 2)),
            source.place_at_center(volume, axes=(1, 2)),
            # Real-coordinate placement (x is uniform in both grids) so this works on the
            # rectilinear grid too, where index-space set_grid_coordinates is rejected.
            fdtdx.RealCoordinateConstraint(
                object=source.name, axes=(0,), sides=("-",), coordinates=(_SOURCE_X * _RESOLUTION,)
            ),
        ]
    )
    objects.append(source)

    ey = fdtdx.PhasorDetector(
        name="ey",
        partial_grid_shape=(1, None, None),
        wave_characters=(wave,),
        components=("Ey",),
        reduce_volume=False,
        plot=False,
    )
    constraints.extend(
        [
            ey.same_size(volume, axes=(1, 2)),
            ey.place_at_center(volume, axes=(1, 2)),
            fdtdx.RealCoordinateConstraint(
                object=ey.name, axes=(0,), sides=("-",), coordinates=(_DET_X * _RESOLUTION,)
            ),
        ]
    )
    objects.append(ey)

    flux = fdtdx.PoyntingFluxDetector(
        name="flux",
        partial_grid_shape=(1, None, None),
        direction="+",
        reduce_volume=True,
        plot=False,
    )
    constraints.extend(
        [
            flux.same_size(volume, axes=(1, 2)),
            flux.place_at_center(volume, axes=(1, 2)),
            fdtdx.RealCoordinateConstraint(
                object=flux.name, axes=(0,), sides=("-",), coordinates=(_DET_X * _RESOLUTION,)
            ),
        ]
    )
    objects.append(flux)

    return objects, constraints, config


def _run(objects, constraints, config):
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=config, key=key)
    return obj_container, arrays, config


def test_unfolded_phasor_matches_full_domain():
    _oc_full, arrays_full, _cfg_full = _run(*_build((0, 0, 0)))
    full_ey = np.asarray(arrays_full.detector_states["ey"]["phasor"][0, 0, 0, 0, :, :])

    oc_sym, arrays_sym, cfg_sym = _run(*_build((0, -1, 1)))
    unfolded = fdtdx.unfold_detector_states(arrays_sym, oc_sym, cfg_sym)
    unf_ey = np.asarray(unfolded.detector_states["ey"]["phasor"][0, 0, 0, 0, :, :])

    assert unf_ey.shape == full_ey.shape, f"{unf_ey.shape} != {full_ey.shape}"
    assert np.max(np.abs(full_ey)) > 1e-30, "reference Ey is zero — wave not launched"

    # The model is perfectly mirror-symmetric, so the PEC/PMC walls reproduce the discarded half
    # cell-for-cell and the unfold is exact (not an approximation) — agreement holds to float
    # precision, far below any physical tolerance.
    ref_amp = float(np.mean(np.abs(full_ey)))
    mean_err = float(np.mean(np.abs(unf_ey - full_ey))) / ref_amp
    max_err = float(np.max(np.abs(unf_ey - full_ey))) / float(np.max(np.abs(full_ey)))
    assert mean_err < 1e-6, f"unfolded vs full Ey mean mismatch = {mean_err:.2e}"
    assert max_err < 1e-6, f"unfolded vs full Ey max mismatch = {max_err:.2e}"


def test_unfolded_flux_recovers_full_power():
    _oc_full, arrays_full, _cfg_full = _run(*_build((0, 0, 0)))
    full_flux = float(np.mean(np.asarray(arrays_full.detector_states["flux"]["poynting_flux"][-_N_AVG:, 0])))

    oc_sym, arrays_sym, cfg_sym = _run(*_build((0, -1, 1)))
    unfolded = fdtdx.unfold_detector_states(arrays_sym, oc_sym, cfg_sym)
    unf_flux = float(np.mean(np.asarray(unfolded.detector_states["flux"]["poynting_flux"][-_N_AVG:, 0])))

    assert abs(full_flux) > 1e-30, "reference flux is zero — wave not launched"
    # Exact decomposition (see phasor test): the x4 quarter-domain rescale recovers the full power
    # to float precision.
    rel_err = abs(unf_flux - full_flux) / abs(full_flux)
    assert rel_err < 1e-5, (
        f"unfolded quarter-domain flux x4 = {unf_flux:.4e} vs full {full_flux:.4e} (rel_err={rel_err:.2e})"
    )


def test_unfolded_phasor_matches_full_domain_rectilinear():
    """Same equivalence as above, but on a symmetric *non-uniform* rectilinear grid.

    Exercises the symmetry + RectilinearGrid path end to end: reduce_symmetric slices the
    non-uniform y/z edges onto the kept half, the reduced FDTD runs the non-uniform curl, and
    unfold reconstructs the full domain. Both runs share the same full non-uniform grid.
    """
    grid = _rectilinear_grid()
    assert not grid.is_uniform  # genuinely non-uniform path

    _oc_full, arrays_full, _cfg_full = _run(*_build((0, 0, 0), grid=grid))
    full_ey = np.asarray(arrays_full.detector_states["ey"]["phasor"][0, 0, 0, 0, :, :])

    oc_sym, arrays_sym, cfg_sym = _run(*_build((0, -1, 1), grid=grid))
    # The pinned grid the reduced run actually used is halved on y and z.
    assert isinstance(cfg_sym.grid, fdtdx.RectilinearGrid)
    assert cfg_sym.grid.shape == (grid.shape[0], grid.shape[1] // 2, grid.shape[2] // 2)

    unfolded = fdtdx.unfold_detector_states(arrays_sym, oc_sym, cfg_sym)
    unf_ey = np.asarray(unfolded.detector_states["ey"]["phasor"][0, 0, 0, 0, :, :])

    assert unf_ey.shape == full_ey.shape, f"{unf_ey.shape} != {full_ey.shape}"
    assert np.max(np.abs(full_ey)) > 1e-30, "reference Ey is zero — wave not launched"

    # As in the uniform case the decomposition is exact; the residual here is float32 roundoff from
    # the (slightly different) reduced grid arrays, still ~1e-7, far below any physical tolerance.
    ref_amp = float(np.mean(np.abs(full_ey)))
    mean_err = float(np.mean(np.abs(unf_ey - full_ey))) / ref_amp
    max_err = float(np.max(np.abs(unf_ey - full_ey))) / float(np.max(np.abs(full_ey)))
    assert mean_err < 1e-5, f"unfolded vs full Ey mean mismatch on rectilinear grid: {mean_err:.2e}"
    assert max_err < 1e-4, f"unfolded vs full Ey max mismatch on rectilinear grid: {max_err:.2e}"
