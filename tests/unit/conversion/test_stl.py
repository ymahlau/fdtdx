"""Unit tests for fdtdx.conversion.stl module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from fdtdx.conversion.stl import export_stl, idx_to_xyz, xyz_to_idx


class TestIdxToXyz:
    """Tests for idx_to_xyz function."""

    def test_single_index_origin(self):
        """Test converting index 0 to (0, 0, 0)."""
        shape = (3, 4, 5)
        x, y, z = idx_to_xyz(np.array([0]), shape)
        assert x[0] == 0
        assert y[0] == 0
        assert z[0] == 0

    def test_single_index_non_origin(self):
        """Test converting a non-zero index."""
        shape = (3, 4, 5)
        # Index 1 should be (0, 0, 1) since z increments first
        x, y, z = idx_to_xyz(np.array([1]), shape)
        assert x[0] == 0
        assert y[0] == 0
        assert z[0] == 1

    def test_multiple_indices(self):
        """Test converting multiple indices at once."""
        shape = (2, 3, 4)
        indices = np.array([0, 1, 4, 12])
        x, y, z = idx_to_xyz(indices, shape)

        # idx=0 -> (0,0,0)
        assert (x[0], y[0], z[0]) == (0, 0, 0)
        # idx=1 -> (0,0,1)
        assert (x[1], y[1], z[1]) == (0, 0, 1)
        # idx=4 -> (0,1,0)
        assert (x[2], y[2], z[2]) == (0, 1, 0)
        # idx=12 -> (1,0,0)
        assert (x[3], y[3], z[3]) == (1, 0, 0)


class TestXyzToIdx:
    """Tests for xyz_to_idx function."""

    def test_origin(self):
        """Test converting (0, 0, 0) to index 0."""
        shape = (3, 4, 5)
        idx = xyz_to_idx(np.array([0]), np.array([0]), np.array([0]), shape)
        assert idx[0] == 0

    def test_non_origin(self):
        """Test converting non-origin coordinates."""
        shape = (3, 4, 5)
        idx = xyz_to_idx(np.array([1]), np.array([2]), np.array([3]), shape)
        # Expected: 1 * (4*5) + 2 * 5 + 3 = 20 + 10 + 3 = 33
        assert idx[0] == 33

    def test_roundtrip(self):
        """Test that idx_to_xyz and xyz_to_idx are inverses."""
        shape = (3, 4, 5)
        original_idx = np.arange(60)  # All indices for shape (3, 4, 5)
        x, y, z = idx_to_xyz(original_idx, shape)
        recovered_idx = xyz_to_idx(x, y, z, shape)
        np.testing.assert_array_equal(original_idx, recovered_idx)


class TestExportStl:
    """Tests for export_stl function."""

    def test_returns_trimesh(self):
        """Test that export_stl returns a trimesh.Trimesh object."""
        matrix = np.ones((2, 2, 2), dtype=bool)
        mesh = export_stl(matrix)
        assert isinstance(mesh, trimesh.Trimesh)

    def test_single_voxel(self):
        """Test exporting a single voxel."""
        matrix = np.ones((1, 1, 1), dtype=bool)
        mesh = export_stl(matrix)
        # A single cube has 8 vertices
        assert mesh.vertices.shape[0] == 8

    def test_empty_matrix(self):
        """Test exporting matrix with no True values."""
        matrix = np.zeros((2, 2, 2), dtype=bool)
        mesh = export_stl(matrix)
        # No faces should be generated
        assert mesh.faces.shape[0] == 0

    def test_voxel_grid_size_scaling(self):
        """Test that voxel_grid_size scales vertices correctly per axis."""
        matrix = np.ones((1, 1, 1), dtype=bool)

        mesh_default = export_stl(matrix, voxel_grid_size=(1, 1, 1))
        mesh_scaled = export_stl(matrix, voxel_grid_size=(2, 3, 4))

        # Scaled mesh should have larger vertex coordinates overall
        assert mesh_scaled.vertices.max() > mesh_default.vertices.max()
        # Per-axis: each axis should scale by its factor
        # vertices shape: (N, 3) where columns are x, y, z
        default_max = mesh_default.vertices.max(axis=0)  # shape (3,)
        scaled_max = mesh_scaled.vertices.max(axis=0)
        assert scaled_max[0] > default_max[0], "x-axis scaling failed"
        assert scaled_max[1] > default_max[1], "y-axis scaling failed"
        assert scaled_max[2] > default_max[2], "z-axis scaling failed"
        # z-axis scale factor (4) should produce larger extent than x-axis (2)
        assert scaled_max[2] > scaled_max[0]

    def test_invalid_dimensions_raises(self):
        """Test that non-3D matrix raises exception."""
        matrix_2d = np.ones((2, 2), dtype=bool)
        with pytest.raises(Exception, match="Invalid matrix shape"):
            export_stl(matrix_2d)

        matrix_4d = np.ones((2, 2, 2, 2), dtype=bool)
        with pytest.raises(Exception, match="Invalid matrix shape"):
            export_stl(matrix_4d)

    def test_file_export(self):
        """Test that STL file is created when filename is provided."""
        matrix = np.ones((2, 2, 2), dtype=bool)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.stl"
            mesh = export_stl(matrix, stl_filename=filepath)

            assert filepath.exists()
            assert filepath.stat().st_size > 0
            assert isinstance(mesh, trimesh.Trimesh)

    def test_sparse_matrix(self):
        """Test exporting sparse matrix (same as existing test)."""
        arr = np.asarray(
            [
                [[0, 1, 0], [0, 0, 0], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ]
        ).astype(bool)

        mesh = export_stl(arr)
        assert mesh.vertices.shape == (20, 3)

    def test_adjacent_voxels_share_faces(self):
        """Test that adjacent voxels don't create internal faces."""
        # Single voxel has 6 faces (12 triangles)
        single = np.ones((1, 1, 1), dtype=bool)
        mesh_single = export_stl(single)
        single_faces = mesh_single.faces.shape[0]

        # Two adjacent voxels should have fewer than 2x single faces
        # because internal faces are removed
        double = np.ones((2, 1, 1), dtype=bool)
        mesh_double = export_stl(double)
        double_faces = mesh_double.faces.shape[0]

        assert double_faces < 2 * single_faces
