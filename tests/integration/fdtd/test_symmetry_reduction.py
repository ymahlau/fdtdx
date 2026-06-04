"""Integration tests for automatic symmetry reduction inside place_objects.

Builds a *full* model, enables config.symmetry, and checks that place_objects reduces the
domain: the volume is halved, PEC/PMC walls land on the symmetry planes, straddling objects are
clipped to their upper half, objects entirely in the discarded half are dropped, and mode
sources receive the derived mode-solver symmetry tuple.
"""

import jax
import jax.numpy as jnp
import pytest

import fdtdx


def _by_name(container, name):
    for o in container.objects:
        if o.name == name:
            return o
    return None


def _build(symmetry):
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=50e-9),
        time=20e-15,
        dtype=jnp.float32,
        symmetry=symmetry,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(partial_real_shape=(1e-6, 1e-6, 1e-6))  # 20 x 20 x 20 cells
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=4)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    # Centered core straddling both the y and z planes -> clipped to its upper half.
    core = fdtdx.UniformMaterialObject(
        name="core",
        partial_real_shape=(None, 0.4e-6, 0.4e-6),  # 8 cells in y and z
        material=fdtdx.Material(permittivity=4.0),
    )
    constraints.extend([core.same_size(volume, axes=(0,)), core.place_at_center(volume, axes=(0, 1, 2))])
    objects.append(core)

    # Mode source spanning the full y,z plane, propagating in +x.
    center_wave = fdtdx.WaveCharacter(wavelength=1e-6)
    profile = fdtdx.GaussianPulseProfile(
        center_wave=center_wave,
        spectral_width=fdtdx.WaveCharacter(wavelength=1e-5),
    )
    source = fdtdx.ModePlaneSource(
        name="modesrc",
        partial_grid_shape=(1, None, None),
        wave_character=center_wave,
        temporal_profile=profile,
        direction="+",
        mode_index=0,
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(1, 2)),
            source.place_at_center(volume, axes=(1, 2)),
            source.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(8,)),
        ]
    )
    objects.append(source)

    # Object pinned at the min corner -> entirely in the discarded half on symmetric axes.
    dropme = fdtdx.UniformMaterialObject(
        name="dropme",
        partial_grid_shape=(4, 4, 4),
        material=fdtdx.Material(permittivity=2.0),
    )
    constraints.append(dropme.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(0, 0, 0)))
    objects.append(dropme)

    return objects, constraints, config


def test_symmetry_reduces_volume_and_inserts_walls():
    objects, constraints, config = _build((0, -1, 1))  # PEC y, PMC z
    key = jax.random.PRNGKey(0)
    oc, arrays, _params, config, _info = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )

    # Volume halved on y and z, unchanged on x.
    assert oc.volume.grid_shape == (20, 10, 10)
    assert arrays.fields.E.shape == (3, 20, 10, 10)

    # PEC wall on y, PMC wall on z, each a single cell on the symmetry plane (min edge).
    pec = oc.pec_objects
    pmc = oc.pmc_objects
    assert len(pec) == 1 and len(pmc) == 1
    assert pec[0].axis == 1 and pec[0].direction == "-"
    assert pec[0]._grid_slice_tuple[1] == (0, 1)
    assert pmc[0].axis == 2 and pmc[0].direction == "-"
    assert pmc[0]._grid_slice_tuple[2] == (0, 1)


def test_symmetry_clips_and_drops_objects():
    objects, constraints, config = _build((0, -1, 1))
    key = jax.random.PRNGKey(0)
    oc, _arrays, _params, _config, _info = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )

    # Centered core kept as its upper half on each symmetric axis (8 cells -> 4, starting at plane).
    core = _by_name(oc, "core")
    assert core is not None
    assert core._grid_slice_tuple[1] == (0, 4)
    assert core._grid_slice_tuple[2] == (0, 4)

    # Object entirely in the discarded half is removed.
    assert _by_name(oc, "dropme") is None


def test_mode_symmetry_auto_derived():
    objects, constraints, config = _build((0, -1, 1))
    key = jax.random.PRNGKey(0)
    oc, _arrays, _params, _config, _info = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )

    # +x propagation -> transverse axes (y, z). y is PEC (-> 0), z is PMC (-> 1).
    source = _by_name(oc, "modesrc")
    assert source is not None
    assert source.symmetry == (0, 1)


def test_no_symmetry_is_unchanged():
    objects, constraints, config = _build((0, 0, 0))
    key = jax.random.PRNGKey(0)
    oc, _arrays, _params, _config, _info = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    # Full domain, no walls added, nothing dropped.
    assert oc.volume.grid_shape == (20, 20, 20)
    assert len(oc.pec_objects) == 0 and len(oc.pmc_objects) == 0
    assert _by_name(oc, "dropme") is not None
    assert _by_name(oc, "modesrc").symmetry == (0, 0)


def test_odd_cell_count_on_symmetric_axis_raises():
    # 21 cells on y (odd) cannot be split exactly down the middle -> hard error.
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=50e-9), time=20e-15, dtype=jnp.float32, symmetry=(0, -1, 0)
    )
    volume = fdtdx.SimulationVolume(partial_grid_shape=(20, 21, 20))
    bound_dict, c_list = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=4), volume
    )
    objects = [volume, *bound_dict.values()]
    with pytest.raises(ValueError, match="even number of cells"):
        fdtdx.place_objects(object_list=objects, config=config, constraints=c_list, key=jax.random.PRNGKey(0))
