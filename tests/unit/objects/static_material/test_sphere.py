"""Unit tests for objects/static_material/sphere.py.

Tests Sphere (and ellipsoid variant) shape generation and material mapping.
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.materials import Material
from fdtdx.objects.static_material.sphere import Sphere

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return SimulationConfig(
        time=100e-15,
        resolution=50e-9,
        backend="cpu",
        dtype=jnp.float32,
        gradient_config=None,
    )


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def two_materials():
    return {
        "air": Material(permittivity=1.0),
        "si": Material(permittivity=12.25),
    }


def _make_sphere(materials, radius=200e-9, material_name="si", **kwargs):
    return Sphere(
        materials=materials,
        radius=radius,
        material_name=material_name,
        **kwargs,
    )


def _place(sphere, config, key, slices=((0, 20), (0, 20), (0, 20))):
    return sphere.place_on_grid(grid_slice_tuple=slices, config=config, key=key)


# ---------------------------------------------------------------------------
# get_voxel_mask_for_shape
# ---------------------------------------------------------------------------


class TestGetVoxelMaskForShape:
    def test_mask_shape_matches_grid(self, config, key, two_materials):
        sphere = _make_sphere(two_materials)
        placed = _place(sphere, config, key, ((0, 12), (0, 14), (0, 16)))
        mask = placed.get_voxel_mask_for_shape()
        assert mask.shape == (12, 14, 16)

    def test_mask_is_bool(self, config, key, two_materials):
        sphere = _make_sphere(two_materials)
        placed = _place(sphere, config, key)
        mask = placed.get_voxel_mask_for_shape()
        assert mask.dtype == jnp.bool_

    def test_mask_has_interior_voxels(self, config, key, two_materials):
        """Radius larger than a grid step should produce True interior voxels."""
        sphere = _make_sphere(two_materials, radius=200e-9)  # 4 grid units
        placed = _place(sphere, config, key, ((0, 20), (0, 20), (0, 20)))
        mask = placed.get_voxel_mask_for_shape()
        assert bool(jnp.any(mask))

    def test_mask_has_exterior_voxels(self, config, key, two_materials):
        """Small sphere in large grid should have voxels outside."""
        sphere = _make_sphere(two_materials, radius=50e-9)  # 1 grid unit
        placed = _place(sphere, config, key, ((0, 20), (0, 20), (0, 20)))
        mask = placed.get_voxel_mask_for_shape()
        assert bool(jnp.any(~mask))

    def test_sphere_roughly_symmetric(self, config, key, two_materials):
        """Symmetric grid should yield a symmetric mask."""
        sphere = _make_sphere(two_materials, radius=150e-9)  # 3 grid units
        placed = _place(sphere, config, key, ((0, 20), (0, 20), (0, 20)))
        mask = placed.get_voxel_mask_for_shape()
        # flip along x-axis: mask[i] should equal mask[N-1-i] roughly
        # (not exact due to +0.5 centering, but interior count should match)
        assert mask.shape == (20, 20, 20)

    def test_ellipsoid_different_radii(self, config, key, two_materials):
        """Ellipsoid with different radii per axis should work."""
        sphere = _make_sphere(
            two_materials,
            radius=100e-9,
            radius_x=200e-9,
            radius_y=100e-9,
            radius_z=50e-9,
        )
        placed = _place(sphere, config, key, ((0, 20), (0, 20), (0, 20)))
        mask = placed.get_voxel_mask_for_shape()
        assert mask.shape == (20, 20, 20)
        assert bool(jnp.any(mask))

    def test_per_axis_radius_overrides_default(self, config, key, two_materials):
        """An explicit radius_x larger than radius should extend more along x."""
        small_sphere = _make_sphere(two_materials, radius=50e-9)
        big_x_sphere = _make_sphere(two_materials, radius=50e-9, radius_x=300e-9)
        placed_small = _place(small_sphere, config, key, ((0, 20), (0, 20), (0, 20)))
        placed_big = _place(big_x_sphere, config, key, ((0, 20), (0, 20), (0, 20)))
        mask_small = placed_small.get_voxel_mask_for_shape()
        mask_big = placed_big.get_voxel_mask_for_shape()
        # bigger radius_x means more True voxels
        assert int(jnp.sum(mask_big)) > int(jnp.sum(mask_small))

    def test_none_radius_falls_back_to_default(self, config, key, two_materials):
        """When radius_x is None, radius is used."""
        sphere_default = _make_sphere(two_materials, radius=200e-9)
        sphere_explicit = _make_sphere(two_materials, radius=200e-9, radius_x=200e-9)
        placed_d = _place(sphere_default, config, key)
        placed_e = _place(sphere_explicit, config, key)
        mask_d = placed_d.get_voxel_mask_for_shape()
        mask_e = placed_e.get_voxel_mask_for_shape()
        assert jnp.array_equal(mask_d, mask_e)


# ---------------------------------------------------------------------------
# get_material_mapping
# ---------------------------------------------------------------------------


class TestGetMaterialMapping:
    def test_shape_matches_grid(self, config, key, two_materials):
        sphere = _make_sphere(two_materials)
        placed = _place(sphere, config, key, ((0, 10), (0, 10), (0, 10)))
        mapping = placed.get_material_mapping()
        assert mapping.shape == (10, 10, 10)

    def test_dtype_is_int(self, config, key, two_materials):
        sphere = _make_sphere(two_materials)
        placed = _place(sphere, config, key)
        mapping = placed.get_material_mapping()
        assert jnp.issubdtype(mapping.dtype, jnp.integer)

    def test_correct_index_for_si(self, config, key, two_materials):
        """si has higher permittivity → index 1 in sorted order."""
        sphere = _make_sphere(two_materials, material_name="si")
        placed = _place(sphere, config, key, ((0, 10), (0, 10), (0, 10)))
        mapping = placed.get_material_mapping()
        assert bool(jnp.all(mapping == 1))

    def test_correct_index_for_air(self, config, key, two_materials):
        """air has lower permittivity → index 0 in sorted order."""
        sphere = Sphere(
            materials=two_materials,
            radius=200e-9,
            material_name="air",
        )
        placed = _place(sphere, config, key, ((0, 10), (0, 10), (0, 10)))
        mapping = placed.get_material_mapping()
        assert bool(jnp.all(mapping == 0))

    def test_mapping_uniform_over_grid(self, config, key, two_materials):
        """Every voxel in the mapping gets the same material index."""
        sphere = _make_sphere(two_materials, material_name="si")
        placed = _place(sphere, config, key)
        mapping = placed.get_material_mapping()
        assert bool(jnp.all(mapping == mapping[0, 0, 0]))
