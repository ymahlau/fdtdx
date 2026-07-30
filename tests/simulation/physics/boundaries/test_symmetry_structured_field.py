"""Physics test: ``config.symmetry`` reproduces the full-domain run for a *structured* field.

``test_simulation_symmetry.py`` compares a reduced run against the full domain for a uniform plane
wave. A transversely uniform field has no transverse derivatives, which makes it blind to how the
mirror walls and the detector co-location stencil treat the cells at the symmetry plane. This test
uses a narrow source instead, so the field diffracts and every transverse derivative is exercised:

* the PEC mirror is exact — the reduced field matches the full-domain half to float precision;
* the PMC mirror is exact in the same sense up to the cell-indexed source rasterization: the
  injected profile is applied per cell, so the components sampled *on* a symmetry plane (which for a
  PMC plane are exactly the even ones) have a footprint centred half a cell off it in the
  full-domain reference. That residue is first order in the cell size — measured 6.7%, 3.3%, 1.6%
  mean at 50, 25, 12.5 nm — while a misplaced wall condition is O(1) and independent of resolution.

The source is a ``UniformPlaneSource`` covering only part of the plane: uniform over its own
footprint, so this stays sensitive to the walls alone and not to a source-profile centering bug
(``tests/unit/objects/sources/test_symmetry_sources.py`` covers those).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx

_SPACING = 50e-9
_WAVELENGTH = 1e-6  # 20 cells per wavelength
_TRANSVERSE_CELLS = 60
_PROPAGATION_CELLS = 40
_SOURCE_CELLS = 20  # even and centered -> the footprint is a mirror-symmetric cell set
_PML_CELLS = 8
_SOURCE_Z = 12
_DETECTOR_Z = 26
_SIM_TIME = 60e-15

# E is polarized along x and propagates in +z, so the mirror that keeps it even is PEC on x (Ex is
# normal there) and PMC on y (Ex is tangential there).
_HALF_X_PEC = (-1, 0, 0)
_HALF_Y_PMC = (0, 1, 0)
_QUARTER = (-1, 1, 0)


def _build(symmetry):
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_SPACING),
        time=_SIM_TIME,
        dtype=jnp.float32,
        symmetry=symmetry,
    )
    volume = fdtdx.SimulationVolume(
        partial_grid_shape=(_TRANSVERSE_CELLS, _TRANSVERSE_CELLS, _PROPAGATION_CELLS)
    )
    objects, constraints = [volume], []
    bound_dict, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML_CELLS), volume
    )
    constraints.extend(boundary_constraints)
    objects.extend(bound_dict.values())

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.UniformPlaneSource(
        name="src",
        partial_grid_shape=(_SOURCE_CELLS, _SOURCE_CELLS, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
        normalize_by_energy=False,
    )
    constraints.extend(
        [
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_SOURCE_Z,)),
        ]
    )
    objects.append(source)

    detector = fdtdx.PhasorDetector(
        name="det",
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave,),
        components=("Ex", "Hy"),
        reduce_volume=False,
        plot=False,
    )
    constraints.extend(
        [
            detector.same_size(volume, axes=(0, 1)),
            detector.same_position(volume, axes=(0, 1)),
            detector.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_DETECTOR_Z,)),
        ]
    )
    objects.append(detector)
    return objects, constraints, config


def _run(symmetry):
    objects, constraints, config = _build(symmetry)
    key = jax.random.PRNGKey(0)
    container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, container, _ = fdtdx.apply_params(arrays, container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=container, config=config, key=key)
    return container, arrays, config


def _phasor(arrays):
    # (num_components, nx, ny)
    return np.asarray(arrays.detector_states["det"]["phasor"][0, 0, :, :, :, 0])


def _kept_half(array, symmetry):
    index: list[slice] = [slice(None)] * 3
    for axis in (0, 1):
        if symmetry[axis] != 0:
            index[1 + axis] = slice(array.shape[1 + axis] // 2, None)
    return array[tuple(index)]


@pytest.fixture(scope="module")
def full_run():
    return _phasor(_run((0, 0, 0))[1])


@pytest.mark.parametrize(
    "symmetry, tolerance",
    [
        # PEC: exact. Before the mirror halo reached the co-location stencil the plane row was
        # recorded at half amplitude, i.e. 50% off, so this tolerance is far below the failure mode.
        (_HALF_X_PEC, 1e-3),
        # PMC: limited only by the O(cell size) source rasterization described in the module
        # docstring. A wall condition placed half a cell off gives ~40%, which this still catches.
        (_HALF_Y_PMC, 0.10),
        (_QUARTER, 0.10),
    ],
)
def test_reduced_run_matches_full_domain_kept_half(full_run, symmetry, tolerance):
    """The reduced run's own cells must agree with the full run, plane row included."""
    _container, arrays, _config = _run(symmetry)
    reduced = _phasor(arrays)
    expected = _kept_half(full_run, symmetry)
    assert reduced.shape == expected.shape
    assert np.abs(expected).max() > 1e-20, "reference field is zero - wave not launched"

    for index, name in enumerate(("Ex", "Hy")):
        scale = np.abs(expected[index]).max()
        error = np.abs(reduced[index] - expected[index]).max() / scale
        assert error < tolerance, f"{name}: reduced run differs from the full domain by {error:.3e}"


@pytest.mark.parametrize("symmetry, tolerance", [(_HALF_X_PEC, 0.02), (_QUARTER, 0.12)])
def test_unfolded_field_matches_full_domain(full_run, symmetry, tolerance):
    """Unfolding the reduced detector output reconstructs the full-domain plane.

    Exercises the mirror index map: the co-located samples sit *on* a symmetry plane normal to x or
    y, so the plane row must not be duplicated and the mirrored half must not be shifted by a cell
    (which is what a plain flip would do — it leaves ~50% errors near the plane).
    """
    container, arrays, config = _run(symmetry)
    unfolded = fdtdx.unfold_detector_states(arrays, container, config)
    reconstructed = np.asarray(unfolded.detector_states["det"]["phasor"][0, 0, :, :, :, 0])
    assert reconstructed.shape == full_run.shape

    for index, name in enumerate(("Ex", "Hy")):
        scale = np.abs(full_run[index]).max()
        # The outermost reconstructed cell repeats its neighbour (its mirror partner lies outside the
        # kept half); it sits inside the PML, so compare the interior.
        error = np.abs(reconstructed[index][1:, 1:] - full_run[index][1:, 1:]).max() / scale
        assert error < tolerance, f"{name}: unfolded field differs from the full domain by {error:.3e}"
