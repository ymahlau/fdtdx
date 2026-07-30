"""Physics test: ``config.symmetry`` reproduces the full-domain run for a *structured* field.

``test_simulation_symmetry.py`` compares a reduced run against the full domain for a uniform plane
wave. A transversely uniform field has no transverse derivatives, which makes it blind to how the
mirror walls and the detector co-location stencil treat the cells at the symmetry plane. This test
uses a narrow source instead, so the field diffracts and every transverse derivative is exercised.

Both mirror types are then exact, each because of where its plane sits:

* an **electric** plane sits on the reduced domain's min edge, where the tangential ``E`` samples
  live and the PEC wall zeroes them;
* a **magnetic** plane sits half a cell *below* the min edge — the source footprint is rasterized per
  cell, so the discrete problem is mirror symmetric about the tangential-``H`` node one cell out,
  where the zero field halo already is the mirror. It gets no wall object at all; zeroing tangential
  ``H`` at the first cell (or filling that halo with its mirror) displaces the plane by half a cell,
  which is a clean first-order-wrong answer: 4.3e-02 raw-field error at 50 nm, halving with the cell
  size instead of vanishing.

Measured here (50 nm / 25 nm), max deviation over the kept half relative to the full-domain peak:
raw fields ≤ 2.8e-04 / 2.1e-06, detector output ≤ 9.3e-06 / 5.6e-07. The raw fields bypass the
co-location stencil, so they isolate the walls; the detector output additionally covers the halo the
stencil reads, which is what recorded *half* the field in the plane row of an electric plane
(``5.0e-01``) before it existed.

Unfolding is looser (~1.2e-02 / 1.7e-02 at 50 nm, first order in the cell size) and cannot be
tightened by either mirror: the co-location average is not symmetric about either candidate plane, so
reconstructing the discarded half from *co-located* samples keeps an O(cell size) residue. The raw
fields it is built from do not.

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
    volume = fdtdx.SimulationVolume(partial_grid_shape=(_TRANSVERSE_CELLS, _TRANSVERSE_CELLS, _PROPAGATION_CELLS))
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


def _kept_half(array, symmetry, first_spatial_axis=1):
    index: list[slice] = [slice(None)] * array.ndim
    for axis in (0, 1):
        if symmetry[axis] != 0:
            ax = first_spatial_axis + axis
            index[ax] = slice(array.shape[ax] // 2, None)
    return array[tuple(index)]


@pytest.fixture(scope="module")
def full_run():
    _container, arrays, _config = _run((0, 0, 0))
    return {
        "phasor": _phasor(arrays),
        "E": np.asarray(arrays.fields.E),
        "H": np.asarray(arrays.fields.H),
    }


@pytest.mark.parametrize("symmetry", [_HALF_X_PEC, _HALF_Y_PMC, _QUARTER])
def test_reduced_raw_fields_match_full_domain_kept_half(full_run, symmetry):
    """The reduced run's own field arrays must agree with the full run, plane row included.

    Reads ``arrays.fields`` directly, so this sees the mirror walls alone — no detector, no
    co-location stencil. A wall condition displaced by half a cell shows up here as ~4e-02.
    """
    _container, arrays, _config = _run(symmetry)
    for name in ("E", "H"):
        reduced = np.asarray(getattr(arrays.fields, name))
        expected = _kept_half(full_run[name], symmetry)
        assert reduced.shape == expected.shape
        scale = np.abs(expected).max()
        assert scale > 1e-20, f"reference {name} is zero - wave not launched"
        error = np.abs(reduced - expected).max() / scale
        assert error < 1e-3, f"{name}: reduced run differs from the full domain by {error:.3e}"


@pytest.mark.parametrize("symmetry", [_HALF_X_PEC, _HALF_Y_PMC, _QUARTER])
def test_reduced_detector_matches_full_domain_kept_half(full_run, symmetry):
    """Same comparison through a detector, which adds the co-location halo at the plane.

    Without the mirror halo an electric plane records exactly half the field in its plane row, i.e.
    5e-01 here; a magnetic plane needs no halo (its plane is half a cell out, where the zero halo
    already is the mirror), and filling one anyway costs ~4e-02.
    """
    _container, arrays, _config = _run(symmetry)
    reduced = _phasor(arrays)
    expected = _kept_half(full_run["phasor"], symmetry)
    assert reduced.shape == expected.shape
    assert np.abs(expected).max() > 1e-20, "reference field is zero - wave not launched"

    for index, name in enumerate(("Ex", "Hy")):
        scale = np.abs(expected[index]).max()
        error = np.abs(reduced[index] - expected[index]).max() / scale
        assert error < 1e-4, f"{name}: reduced run differs from the full domain by {error:.3e}"


@pytest.mark.parametrize("symmetry, tolerance", [(_HALF_X_PEC, 0.02), (_QUARTER, 0.03)])
def test_unfolded_field_matches_full_domain(full_run, symmetry, tolerance):
    """Unfolding the reduced detector output reconstructs the full-domain plane.

    Exercises the per-axis mirror index map. Across the **electric** x-plane the co-located samples
    sit on the plane, so the plane row must not be duplicated and the mirrored half must not be
    shifted by a cell (a plain flip leaves ~8e-02 here). Across the **magnetic** y-plane the plane is
    half a cell out and the same samples mirror one-to-one, so the map is the plain flip instead (the
    on-plane map leaves ~9e-02). Applying either convention to both axes fails one of them.

    The remaining ~1e-02 is the co-location average itself, which is symmetric about neither
    candidate plane; it is first order in the cell size and does not affect the recorded (non-
    unfolded) output, which the tests above hold to 1e-4.
    """
    full_phasor = full_run["phasor"]
    container, arrays, config = _run(symmetry)
    unfolded = fdtdx.unfold_detector_states(arrays, container, config)
    reconstructed = np.asarray(unfolded.detector_states["det"]["phasor"][0, 0, :, :, :, 0])
    assert reconstructed.shape == full_phasor.shape

    for index, name in enumerate(("Ex", "Hy")):
        scale = np.abs(full_phasor[index]).max()
        # The outermost reconstructed cell repeats its neighbour (its mirror partner lies outside the
        # kept half); it sits inside the PML, so compare the interior.
        error = np.abs(reconstructed[index][1:, 1:] - full_phasor[index][1:, 1:]).max() / scale
        assert error < tolerance, f"{name}: unfolded field differs from the full domain by {error:.3e}"
