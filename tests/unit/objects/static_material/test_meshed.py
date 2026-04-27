"""Unit and integration tests for objects/static_material/meshed.py.

Tests MeshedObject shape-mask generation, material mapping, and the
meshed_object_from_file factory function.
"""

import jax
import jax.numpy as jnp
import meshio
import numpy as np
import pytest
import trimesh

from fdtdx.config import SimulationConfig
from fdtdx.materials import Material
from fdtdx.objects.static_material.meshed import (
    MeshedObject,
    _extract_surface,
    _voxel_centers,
    meshed_object_from_file,
)

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


@pytest.fixture
def sphere_mesh():
    """Watertight triangular sphere mesh (radius 150 nm) centered at origin."""
    tm = trimesh.creation.icosphere(subdivisions=2, radius=150e-9)
    return tm.vertices, tm.faces


@pytest.fixture
def box_mesh():
    """Watertight triangular box 200x200x200 nm centered at origin."""
    tm = trimesh.creation.box(extents=[200e-9, 200e-9, 200e-9])
    return tm.vertices, tm.faces


def _make_obj(vertices, faces, materials, **kwargs):
    return MeshedObject(vertices=vertices, faces=faces, materials=materials, **kwargs)


def _place(obj, config, key, slices=((0, 20), (0, 20), (0, 20))):
    return obj.place_on_grid(grid_slice_tuple=slices, config=config, key=key)


# ---------------------------------------------------------------------------
# Module-level helper tests
# ---------------------------------------------------------------------------


class TestExtractSurface:
    def test_triangle_passthrough(self):
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        result = _extract_surface([("triangle", faces)])
        assert np.array_equal(result, faces)

    def test_quad_split_to_triangles(self):
        quads = np.array([[0, 1, 2, 3]])
        result = _extract_surface([("quad", quads)])
        assert result.shape == (2, 3)

    def test_tetra_boundary_extraction(self):
        # Minimal tet: 4 vertices, 1 tetra -> all 4 faces are boundary faces
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        tet = np.array([[0, 1, 2, 3]])
        _ = verts  # not used in _extract_surface but kept for clarity
        result = _extract_surface([("tetra", tet)])
        assert result.shape == (4, 3)

    def test_unsupported_cell_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported cell type"):
            _extract_surface([("hexahedron", np.zeros((1, 8), dtype=int))])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="No supported cell types"):
            _extract_surface([])


class TestVoxelCenters:
    def test_shape(self):
        pts = _voxel_centers((4, 5, 6), 50e-9)
        assert pts.shape == (4 * 5 * 6, 3)

    def test_centered_at_origin_for_even_grid(self):
        """For a symmetric grid the point cloud must be symmetric around zero."""
        pts = _voxel_centers((4, 4, 4), 50e-9)
        assert np.allclose(pts.mean(axis=0), 0.0, atol=1e-20)

    def test_voxel_spacing(self):
        """Adjacent voxels in x should differ by exactly one resolution."""
        resolution = 50e-9
        pts = _voxel_centers((3, 1, 1), resolution)
        diffs = np.diff(pts[:, 0])
        assert np.allclose(diffs, resolution)


# ---------------------------------------------------------------------------
# MeshedObject construction
# ---------------------------------------------------------------------------


class TestMeshedObjectConstruction:
    def test_single_material_mode(self, sphere_mesh, two_materials):
        vertices, faces = sphere_mesh
        obj = _make_obj(vertices, faces, two_materials, material_name="si")
        assert isinstance(obj, MeshedObject)
        assert obj.material_name == "si"

    def test_vertices_stored(self, sphere_mesh, two_materials):
        vertices, faces = sphere_mesh
        obj = _make_obj(vertices, faces, two_materials, material_name="si")
        assert obj.vertices is vertices

    def test_faces_stored(self, sphere_mesh, two_materials):
        vertices, faces = sphere_mesh
        obj = _make_obj(vertices, faces, two_materials, material_name="si")
        assert obj.faces is faces


# ---------------------------------------------------------------------------
# get_voxel_mask_for_shape
# ---------------------------------------------------------------------------


class TestGetVoxelMaskForShape:
    def test_mask_shape_matches_grid(self, config, key, sphere_mesh, two_materials):
        vertices, faces = sphere_mesh
        obj = _make_obj(vertices, faces, two_materials, material_name="si")
        placed = _place(obj, config, key, ((0, 12), (0, 14), (0, 16)))
        mask = placed.get_voxel_mask_for_shape()
        assert mask.shape == (12, 14, 16)

    def test_mask_dtype_is_bool(self, config, key, sphere_mesh, two_materials):
        vertices, faces = sphere_mesh
        obj = _make_obj(vertices, faces, two_materials, material_name="si")
        placed = _place(obj, config, key)
        assert placed.get_voxel_mask_for_shape().dtype == jnp.bool_

    def test_mask_has_interior_voxels(self, config, key, sphere_mesh, two_materials):
        vertices, faces = sphere_mesh
        obj = _make_obj(vertices, faces, two_materials, material_name="si")
        placed = _place(obj, config, key)
        assert bool(jnp.any(placed.get_voxel_mask_for_shape()))

    def test_mask_has_exterior_voxels(self, config, key, sphere_mesh, two_materials):
        """Sphere (radius 150 nm) in a 20^3 grid (1000 nm) should leave exterior voxels."""
        vertices, faces = sphere_mesh
        obj = _make_obj(vertices, faces, two_materials, material_name="si")
        placed = _place(obj, config, key)
        assert bool(jnp.any(~placed.get_voxel_mask_for_shape()))

    def test_centered_sphere_mask_is_symmetric(self, config, key, sphere_mesh, two_materials):
        """A sphere centered at origin yields a mask symmetric along all three axes."""
        vertices, faces = sphere_mesh
        obj = _make_obj(vertices, faces, two_materials, material_name="si")
        placed = _place(obj, config, key, ((0, 20), (0, 20), (0, 20)))
        mask = np.array(placed.get_voxel_mask_for_shape())
        assert np.array_equal(mask, mask[::-1, :, :]), "not symmetric along x"
        assert np.array_equal(mask, mask[:, ::-1, :]), "not symmetric along y"
        assert np.array_equal(mask, mask[:, :, ::-1]), "not symmetric along z"

    def test_offcenter_mesh_breaks_symmetry(self, config, key, two_materials):
        """A mesh whose vertices are NOT centered at origin should produce an asymmetric mask."""
        tm = trimesh.creation.icosphere(subdivisions=2, radius=100e-9)
        shifted_vertices = tm.vertices + np.array([150e-9, 0, 0])
        obj = _make_obj(shifted_vertices, tm.faces, two_materials, material_name="si")
        placed = _place(obj, config, key, ((0, 20), (0, 20), (0, 20)))
        mask = np.array(placed.get_voxel_mask_for_shape())
        assert not np.array_equal(mask, mask[::-1, :, :]), "expected asymmetry in x"

    def test_box_mask_fills_interior(self, config, key, box_mesh, two_materials):
        """A 200 nm box in a 20^3 grid (1000 nm) must have a contiguous interior."""
        vertices, faces = box_mesh
        obj = _make_obj(vertices, faces, two_materials, material_name="si")
        placed = _place(obj, config, key, ((0, 20), (0, 20), (0, 20)))
        mask = placed.get_voxel_mask_for_shape()
        assert bool(jnp.any(mask))
        cx, cy, cz = 10, 10, 10
        assert bool(mask[cx, cy, cz]), "central voxel should be inside the box"


# ---------------------------------------------------------------------------
# get_material_mapping
# ---------------------------------------------------------------------------


class TestGetMaterialMapping:
    def test_mapping_shape_matches_grid(self, config, key, sphere_mesh, two_materials):
        vertices, faces = sphere_mesh
        obj = _make_obj(vertices, faces, two_materials, material_name="si")
        placed = _place(obj, config, key, ((0, 10), (0, 12), (0, 8)))
        assert placed.get_material_mapping().shape == (10, 12, 8)

    def test_mapping_dtype_is_int(self, config, key, sphere_mesh, two_materials):
        vertices, faces = sphere_mesh
        obj = _make_obj(vertices, faces, two_materials, material_name="si")
        placed = _place(obj, config, key)
        assert jnp.issubdtype(placed.get_material_mapping().dtype, jnp.integer)

    def test_single_material_si_index(self, config, key, sphere_mesh, two_materials):
        """si (permittivity=12.25 > air 1.0) maps to index 1 in sorted order."""
        vertices, faces = sphere_mesh
        obj = _make_obj(vertices, faces, two_materials, material_name="si")
        placed = _place(obj, config, key)
        assert bool(jnp.all(placed.get_material_mapping() == 1))

    def test_single_material_air_index(self, config, key, sphere_mesh, two_materials):
        """air (permittivity=1.0) maps to index 0 in sorted order."""
        vertices, faces = sphere_mesh
        obj = _make_obj(vertices, faces, two_materials, material_name="air")
        placed = _place(obj, config, key)
        assert bool(jnp.all(placed.get_material_mapping() == 0))

    def test_single_material_mapping_is_uniform(self, config, key, sphere_mesh, two_materials):
        """Single-material mapping must be uniform (every voxel gets the same index)."""
        vertices, faces = sphere_mesh
        obj = _make_obj(vertices, faces, two_materials, material_name="si")
        placed = _place(obj, config, key)
        mapping = placed.get_material_mapping()
        assert bool(jnp.all(mapping == mapping[0, 0, 0]))


# ---------------------------------------------------------------------------
# meshed_object_from_file
# ---------------------------------------------------------------------------


class TestMeshedObjectFromFile:
    def _write_mesh(self, tmp_path, name, vertices, faces):
        path = tmp_path / name
        meshio.Mesh(points=vertices, cells=[("triangle", faces)]).write(str(path))
        return path

    def test_returns_meshed_object(self, tmp_path, sphere_mesh, two_materials):
        vertices, faces = sphere_mesh
        path = self._write_mesh(tmp_path, "sphere.vtk", vertices, faces)
        result = meshed_object_from_file(path, materials=two_materials, material_name="si")
        assert isinstance(result, MeshedObject)

    def test_auto_centers_mesh(self, tmp_path, two_materials):
        """After loading, the bounding-box center of vertices must be at the origin."""
        tm = trimesh.creation.icosphere(subdivisions=1, radius=100e-9)
        offset_vertices = tm.vertices + np.array([300e-9, 200e-9, 100e-9])
        path = self._write_mesh(tmp_path, "offset.vtk", offset_vertices, tm.faces)
        result = meshed_object_from_file(path, materials=two_materials, material_name="si")
        centre = 0.5 * (result.vertices.min(axis=0) + result.vertices.max(axis=0))
        assert np.allclose(centre, 0.0, atol=1e-12)

    def test_kwargs_forwarded(self, tmp_path, sphere_mesh, two_materials):
        vertices, faces = sphere_mesh
        path = self._write_mesh(tmp_path, "sphere.vtk", vertices, faces)
        result = meshed_object_from_file(path, materials=two_materials, material_name="air")
        assert result.material_name == "air"

    def test_mesh_already_centered_unchanged(self, tmp_path, sphere_mesh, two_materials):
        """A mesh already at the origin should stay at the origin after loading."""
        vertices, faces = sphere_mesh
        path = self._write_mesh(tmp_path, "sphere.vtk", vertices, faces)
        result = meshed_object_from_file(path, materials=two_materials, material_name="si")
        centre = 0.5 * (result.vertices.min(axis=0) + result.vertices.max(axis=0))
        assert np.allclose(centre, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMeshedObjectIntegration:
    def test_place_and_get_mask(self, config, key, sphere_mesh, two_materials):
        vertices, faces = sphere_mesh
        obj = _make_obj(vertices, faces, two_materials, material_name="si")
        placed = _place(obj, config, key, ((0, 20), (0, 20), (0, 20)))
        mask = placed.get_voxel_mask_for_shape()
        assert mask.shape == (20, 20, 20)
        assert mask.dtype == jnp.bool_
        assert bool(jnp.any(mask))

    def test_place_and_get_mapping(self, config, key, sphere_mesh, two_materials):
        vertices, faces = sphere_mesh
        obj = _make_obj(vertices, faces, two_materials, material_name="si")
        placed = _place(obj, config, key, ((0, 20), (0, 20), (0, 20)))
        mapping = placed.get_material_mapping()
        assert mapping.shape == (20, 20, 20)
        assert jnp.issubdtype(mapping.dtype, jnp.integer)

    def test_file_round_trip_mask_matches_direct(self, tmp_path, config, key, two_materials):
        """Loading from file should give same mask as constructing directly."""
        tm = trimesh.creation.icosphere(subdivisions=2, radius=150e-9)

        # Direct construction
        obj_direct = _make_obj(tm.vertices, tm.faces, two_materials, material_name="si")
        placed_direct = _place(obj_direct, config, key)
        mask_direct = np.array(placed_direct.get_voxel_mask_for_shape())

        # Via file
        path = tmp_path / "sphere.vtk"
        meshio.Mesh(points=tm.vertices, cells=[("triangle", tm.faces)]).write(str(path))
        obj_file = meshed_object_from_file(path, materials=two_materials, material_name="si")
        placed_file = _place(obj_file, config, key)
        mask_file = np.array(placed_file.get_voxel_mask_for_shape())

        assert np.array_equal(mask_direct, mask_file)
