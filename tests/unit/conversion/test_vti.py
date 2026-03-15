"""Unit tests for fdtdx.conversion.vti module."""

import struct
import zlib
from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest

from fdtdx.conversion.vti import (
    NUMPY_TO_VTK_DTYPE,
    encode_array,
    export_arrays_snapshot_to_vti,
    export_vti,
)

# ---- NUMPY_TO_VTK_DTYPE mapping ----


class TestNumpyToVtkDtype:
    """Tests for the NUMPY_TO_VTK_DTYPE constant mapping."""

    def test_contains_all_expected_dtypes(self):
        expected = {
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "float32",
            "float64",
        }
        assert set(NUMPY_TO_VTK_DTYPE.keys()) == expected

    def test_vtk_type_names_are_capitalized(self):
        for numpy_dtype, vtk_name in NUMPY_TO_VTK_DTYPE.items():
            assert vtk_name[0].isupper(), f"VTK name for {numpy_dtype} should be capitalized"


# ---- encode_array ----


class TestEncodeArray:
    """Tests for the encode_array function."""

    def test_header_format(self):
        """Header should be 4 unsigned 32-bit ints: [1, uncompressed_size, uncompressed_size, compressed_size]."""
        arr = jnp.ones((3, 3, 3), dtype=jnp.float32)
        result = encode_array(arr)

        header = struct.unpack("<4I", result[:16])
        num_blocks, uncompressed_size, last_block_size, compressed_size = header

        assert num_blocks == 1
        assert uncompressed_size == arr.size * 4  # float32 = 4 bytes each
        assert last_block_size == uncompressed_size
        assert compressed_size == len(result) - 16

    def test_fortran_order_flattening(self):
        """Verify array is flattened in Fortran (column-major) order."""
        arr = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=jnp.float32)
        result = encode_array(arr)

        compressed_data = result[16:]
        decompressed = zlib.decompress(compressed_data)

        expected_flat = arr.flatten(order="F")
        recovered = jnp.frombuffer(decompressed, dtype=jnp.float32)
        assert jnp.array_equal(recovered, expected_flat)

    def test_compression_level_0_no_compression(self):
        """Level 0 should produce valid but uncompressed output."""
        arr = jnp.ones((4, 4, 4), dtype=jnp.float32)
        result = encode_array(arr, compression_level=0)

        compressed_data = result[16:]
        decompressed = zlib.decompress(compressed_data)
        assert decompressed == arr.flatten(order="F").tobytes()

    def test_compression_level_9_smaller_output(self):
        """Level 9 should produce smaller or equal output compared to level 0."""
        arr = jnp.ones((10, 10, 10), dtype=jnp.float32)
        result_0 = encode_array(arr, compression_level=0)
        result_9 = encode_array(arr, compression_level=9)
        assert len(result_9) <= len(result_0)

    def test_different_dtypes(self):
        """Test encoding with different numeric dtypes."""
        for dtype in [jnp.float32, jnp.float64, jnp.int32]:
            arr = jnp.ones((2, 2, 2), dtype=dtype)
            result = encode_array(arr)
            header = struct.unpack("<4I", result[:16])
            expected_size = arr.size * arr.dtype.itemsize
            assert header[1] == expected_size


# ---- export_vti ----


class TestExportVti:
    """Tests for the export_vti function."""

    def test_xml_header(self, tmp_path):
        data = jnp.zeros((4, 4, 4), dtype=jnp.float32)
        path = tmp_path / "test.vti"
        export_vti({"field": data}, path, resolution=1.0)

        content = path.read_bytes()
        assert b'<?xml version="1.0"?>' in content
        assert b'<VTKFile type="ImageData"' in content
        assert b'byte_order="LittleEndian"' in content
        assert b'compressor="vtkZLibDataCompressor"' in content

    def test_xml_footer(self, tmp_path):
        data = jnp.zeros((4, 4, 4), dtype=jnp.float32)
        path = tmp_path / "test.vti"
        export_vti({"field": data}, path, resolution=1.0)

        content = path.read_bytes()
        assert content.endswith(b"\n</AppendedData>\n</VTKFile>")

    def test_extent_default_offset(self, tmp_path):
        data = jnp.zeros((5, 6, 7), dtype=jnp.float32)
        path = tmp_path / "test.vti"
        export_vti({"field": data}, path, resolution=1.0)

        content = path.read_text(encoding="utf-8", errors="ignore")
        assert 'WholeExtent="0 5 0 6 0 7"' in content

    def test_extent_with_offset(self, tmp_path):
        data = jnp.zeros((5, 5, 5), dtype=jnp.float32)
        path = tmp_path / "test.vti"
        export_vti({"field": data}, path, resolution=1.0, offset=(10, 20, 30))

        content = path.read_text(encoding="utf-8", errors="ignore")
        assert 'WholeExtent="10 15 20 25 30 35"' in content

    def test_extent_with_grid_slice(self, tmp_path):
        data = jnp.zeros((5, 5, 5), dtype=jnp.float32)
        path = tmp_path / "test.vti"
        grid_slice = (slice(2, 7), slice(3, 8), slice(4, 9))
        export_vti({"field": data}, path, resolution=1.0, grid_slice=grid_slice)

        content = path.read_text(encoding="utf-8", errors="ignore")
        assert 'WholeExtent="2 7 3 8 4 9"' in content

    def test_grid_slice_with_nonzero_offset_raises(self, tmp_path):
        data = jnp.zeros((5, 5, 5), dtype=jnp.float32)
        path = tmp_path / "test.vti"
        grid_slice = (slice(0, 5), slice(0, 5), slice(0, 5))
        with pytest.raises(AssertionError):
            export_vti({"field": data}, path, resolution=1.0, offset=(1, 0, 0), grid_slice=grid_slice)

    def test_spacing(self, tmp_path):
        data = jnp.zeros((3, 3, 3), dtype=jnp.float32)
        path = tmp_path / "test.vti"
        export_vti({"field": data}, path, resolution=0.5)

        content = path.read_text(encoding="utf-8", errors="ignore")
        assert 'Spacing="0.5 0.5 0.5"' in content

    def test_scalar_field_components(self, tmp_path):
        data = jnp.zeros((3, 3, 3), dtype=jnp.float32)
        path = tmp_path / "test.vti"
        export_vti({"scalar": data}, path, resolution=1.0)

        content = path.read_text(encoding="utf-8", errors="ignore")
        assert 'Name="scalar"' in content
        assert 'NumberOfComponents="1"' in content

    def test_vector_field_components(self, tmp_path):
        data = jnp.zeros((3, 4, 4, 4), dtype=jnp.float32)
        path = tmp_path / "test.vti"
        export_vti({"velocity": data}, path, resolution=1.0)

        content = path.read_text(encoding="utf-8", errors="ignore")
        assert 'Name="velocity"' in content
        assert 'NumberOfComponents="3"' in content

    def test_mixed_scalar_and_vector(self, tmp_path):
        scalar = jnp.zeros((4, 4, 4), dtype=jnp.float32)
        vector = jnp.zeros((3, 4, 4, 4), dtype=jnp.float32)
        path = tmp_path / "test.vti"
        export_vti({"pressure": scalar, "velocity": vector}, path, resolution=1.0)

        content = path.read_text(encoding="utf-8", errors="ignore")
        assert 'Name="pressure"' in content
        assert 'Name="velocity"' in content

    def test_mismatched_shapes_raises(self, tmp_path):
        data1 = jnp.zeros((10, 10, 10), dtype=jnp.float32)
        data2 = jnp.zeros((5, 5, 5), dtype=jnp.float32)
        with pytest.raises(AssertionError, match="same underlying grid"):
            export_vti({"a": data1, "b": data2}, tmp_path / "fail.vti", 1.0)

    def test_2d_array_raises(self, tmp_path):
        data = jnp.zeros((10, 10), dtype=jnp.float32)
        with pytest.raises(AssertionError, match="Only 3d scalar fields"):
            export_vti({"a": data}, tmp_path / "fail.vti", 1.0)

    def test_5d_array_raises(self, tmp_path):
        data = jnp.zeros((2, 3, 4, 5, 6), dtype=jnp.float32)
        with pytest.raises(AssertionError, match="Only 3d scalar fields"):
            export_vti({"a": data}, tmp_path / "fail.vti", 1.0)

    def test_unsupported_dtype_raises(self, tmp_path):
        data = jnp.zeros((3, 3, 3), dtype=jnp.bool_)
        with pytest.raises(AssertionError, match="VTI files only support dtypes"):
            export_vti({"a": data}, tmp_path / "fail.vti", 1.0)

    def test_string_path(self, tmp_path):
        """Test that string paths work (not just Path objects)."""
        data = jnp.zeros((3, 3, 3), dtype=jnp.float32)
        path = str(tmp_path / "string_path.vti")
        export_vti({"field": data}, path, resolution=1.0)
        assert (tmp_path / "string_path.vti").exists()

    def test_data_roundtrip_integrity(self, tmp_path):
        """Verify binary data in the file can be decompressed back to original values."""
        arr = jnp.arange(27, dtype=jnp.float32).reshape(3, 3, 3)
        path = tmp_path / "roundtrip.vti"
        export_vti({"data": arr}, path, resolution=1.0)

        with open(path, "rb") as f:
            raw = f.read()

        # Find the binary data after the underscore marker
        marker = b"    _"
        marker_idx = raw.index(marker) + len(marker)
        footer = b"\n</AppendedData>\n</VTKFile>"
        binary_data = raw[marker_idx : -len(footer)]

        # Parse header and decompress
        compressed = binary_data[16:]
        decompressed = zlib.decompress(compressed)

        recovered = jnp.frombuffer(decompressed, dtype=jnp.float32)
        expected = arr.flatten(order="F")
        assert jnp.array_equal(recovered, expected)


# ---- export_arrays_snapshot_to_vti ----


class TestExportArraysSnapshotToVti:
    """Tests for the export_arrays_snapshot_to_vti convenience function using mocks."""

    def _make_mock_arrays(self, shape=(4, 4, 4), include_conductivity=False, scalar_permeability=False):
        """Create a mock ArrayContainer with the given properties."""
        mock = MagicMock()
        mock.inv_permittivities = jnp.ones(shape, dtype=jnp.float32) * 0.5
        mock.E = jnp.zeros(shape, dtype=jnp.float32)
        mock.H = jnp.zeros(shape, dtype=jnp.float32)

        if scalar_permeability:
            mock.inv_permeabilities = 1.0
        else:
            mock.inv_permeabilities = jnp.ones(shape, dtype=jnp.float32)

        if include_conductivity:
            mock.electric_conductivity = jnp.ones(shape, dtype=jnp.float32) * 0.01
            mock.magnetic_conductivity = jnp.ones(shape, dtype=jnp.float32) * 0.02
        else:
            mock.electric_conductivity = None
            mock.magnetic_conductivity = None

        return mock

    def test_exports_core_fields(self, tmp_path):
        arrays = self._make_mock_arrays()
        path = tmp_path / "snapshot.vti"
        export_arrays_snapshot_to_vti(arrays, path, resolution=1.0)

        content = path.read_text(encoding="utf-8", errors="ignore")
        assert 'Name="permittivity"' in content
        assert 'Name="E"' in content
        assert 'Name="H"' in content

    def test_exports_permeability_when_array(self, tmp_path):
        arrays = self._make_mock_arrays(scalar_permeability=False)
        path = tmp_path / "snapshot.vti"
        export_arrays_snapshot_to_vti(arrays, path, resolution=1.0)

        content = path.read_text(encoding="utf-8", errors="ignore")
        assert 'Name="permeabilities"' in content

    def test_skips_permeability_when_scalar(self, tmp_path):
        arrays = self._make_mock_arrays(scalar_permeability=True)
        path = tmp_path / "snapshot.vti"
        export_arrays_snapshot_to_vti(arrays, path, resolution=1.0)

        content = path.read_text(encoding="utf-8", errors="ignore")
        assert 'Name="permeabilities"' not in content

    def test_exports_conductivities_when_present(self, tmp_path):
        arrays = self._make_mock_arrays(include_conductivity=True)
        path = tmp_path / "snapshot.vti"
        export_arrays_snapshot_to_vti(arrays, path, resolution=1.0)

        content = path.read_text(encoding="utf-8", errors="ignore")
        assert 'Name="electric_conductivity"' in content
        assert 'Name="magnetic_conductivity"' in content

    def test_skips_conductivities_when_none(self, tmp_path):
        arrays = self._make_mock_arrays(include_conductivity=False)
        path = tmp_path / "snapshot.vti"
        export_arrays_snapshot_to_vti(arrays, path, resolution=1.0)

        content = path.read_text(encoding="utf-8", errors="ignore")
        assert 'Name="electric_conductivity"' not in content
        assert 'Name="magnetic_conductivity"' not in content

    def test_permittivity_is_inverse_of_input(self, tmp_path):
        """The function computes 1/inv_permittivities, so values should be inverted."""
        arrays = self._make_mock_arrays()
        arrays.inv_permittivities = jnp.full((3, 3, 3), 0.25, dtype=jnp.float32)
        arrays.E = jnp.zeros((3, 3, 3), dtype=jnp.float32)
        arrays.H = jnp.zeros((3, 3, 3), dtype=jnp.float32)
        arrays.inv_permeabilities = 1.0
        arrays.electric_conductivity = None
        arrays.magnetic_conductivity = None

        path = tmp_path / "snapshot.vti"
        export_arrays_snapshot_to_vti(arrays, path, resolution=1.0)

        # Read binary data and verify permittivity = 1/0.25 = 4.0
        with open(path, "rb") as f:
            raw = f.read()

        marker = b"    _"
        marker_idx = raw.index(marker) + len(marker)
        footer = b"\n</AppendedData>\n</VTKFile>"
        binary_section = raw[marker_idx : -len(footer)]

        # First encoded array is permittivity
        header = struct.unpack("<4I", binary_section[:16])
        compressed_size = header[3]
        compressed = binary_section[16 : 16 + compressed_size]
        decompressed = zlib.decompress(compressed)
        values = jnp.frombuffer(decompressed, dtype=jnp.float32)
        assert jnp.allclose(values, 4.0)
