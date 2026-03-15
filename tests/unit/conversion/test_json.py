"""Unit tests for fdtdx.conversion.json module."""

import json

import jax.numpy as jnp
import pytest

from fdtdx.conversion.json import (
    _export_json,
    _import_obj_from_json,
    export_json,
    export_json_str,
    import_from_json,
)
from fdtdx.core.null import NULL
from fdtdx.core.wavelength import WaveCharacter


class TestExportJsonBasicTypes:
    """Tests for _export_json with basic types."""

    def test_export_none(self):
        """Test exporting None."""
        assert _export_json(None) is None

    def test_export_primitives(self):
        """Test exporting basic Python types."""
        assert _export_json(42) == 42
        assert _export_json(3.14) == 3.14
        assert _export_json("hello") == "hello"
        assert _export_json(True) is True


class TestExportJsonJaxDtypes:
    """Tests for _export_json with JAX dtypes."""

    def test_export_jax_dtypes(self):
        """Test exporting JAX dtypes."""
        for dtype in [jnp.float32, jnp.int32, jnp.complex64]:
            result = _export_json(dtype)
            assert "__dtype__" in result


class TestExportJsonCollections:
    """Tests for _export_json with collections."""

    def test_export_dict(self):
        """Test exporting dictionaries."""
        obj = {"a": 1, "b": 2.5}
        result = _export_json(obj)

        assert result["__module__"] == "builtins"
        assert result["__name__"] == "dict"
        assert result["a"] == 1
        assert result["b"] == 2.5

    def test_export_list(self):
        """Test exporting lists."""
        result = _export_json([1, 2, 3])

        assert result["__name__"] == "list"
        assert result["__value__"] == [1, 2, 3]

    def test_export_tuple(self):
        """Test exporting tuples."""
        result = _export_json((1, 2, 3))

        assert result["__name__"] == "tuple"
        assert result["__value__"] == [1, 2, 3]


class TestExportJsonErrors:
    """Tests for _export_json error cases."""

    def test_export_null_raises(self):
        """Test that exporting NULL raises exception."""
        with pytest.raises(Exception, match="Object should not contain NULL"):
            _export_json(NULL)

    def test_export_non_string_dict_key_raises(self):
        """Test that non-string dict keys raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            _export_json({1: "value"})

    def test_export_nested_class_raises(self):
        """Test that nested classes raise NotImplementedError."""

        class Outer:
            class Inner:
                pass

        with pytest.raises(NotImplementedError):
            _export_json(Outer.Inner())


class TestExportJson:
    """Tests for export_json public function."""

    def test_export_json_basic_type_raises(self):
        """Test that export_json raises for basic types."""
        with pytest.raises(Exception, match="Cannot convert python obj to json"):
            export_json(42)
        with pytest.raises(Exception, match="Cannot convert python obj to json"):
            export_json("string")


class TestExportJsonStr:
    """Tests for export_json_str function."""

    def test_export_json_str_returns_valid_json(self):
        """Test that export_json_str returns valid formatted JSON."""
        obj = WaveCharacter(wavelength=1.55e-6)
        result = export_json_str(obj)

        assert isinstance(result, str)
        assert "\n" in result  # formatted with indentation
        parsed = json.loads(result)
        assert isinstance(parsed, dict)


class TestImportFromJson:
    """Tests for import_from_json function."""

    def test_import_basic_types(self):
        """Test importing basic types."""
        assert _import_obj_from_json(None) is None
        assert _import_obj_from_json(42) == 42
        assert _import_obj_from_json(3.14) == 3.14
        assert _import_obj_from_json("hello") == "hello"

    def test_import_jax_dtype(self):
        """Test importing JAX dtype."""
        result = _import_obj_from_json({"__dtype__": "jax.numpy.float32"})
        assert result == jnp.float32

    def test_import_sequence(self):
        """Test importing sequence."""
        seq_dict = {
            "__module__": "builtins",
            "__name__": "list",
            "__value__": [1, 2, 3],
        }
        assert _import_obj_from_json(seq_dict) == [1, 2, 3]


class TestJsonRoundtrip:
    """Tests for JSON serialization/deserialization roundtrip."""

    def test_roundtrip_wave_character(self):
        """Test roundtrip with WaveCharacter."""
        original = WaveCharacter(wavelength=1.55e-6)
        restored = import_from_json(export_json_str(original))

        assert restored.wavelength == original.wavelength

    def test_roundtrip_with_jax_dtype(self):
        """Test roundtrip with JAX dtype in a dict."""
        original = {"dtype": jnp.float64, "value": 42}
        restored = import_from_json(export_json_str(original))

        assert restored["dtype"] == jnp.float64
        assert restored["value"] == 42

    def test_roundtrip_list(self):
        """Test roundtrip of list."""
        original = [1, 2, 3, 4, 5]
        restored = import_from_json(export_json_str(original))
        assert restored == original

    def test_roundtrip_dict(self):
        """Test roundtrip of dict."""
        original = {"a": 1, "b": 2}
        restored = import_from_json(export_json_str(original))
        assert restored["a"] == 1
        assert restored["b"] == 2

    def test_roundtrip_nested_collections(self):
        """Test roundtrip with nested collections."""
        original = {"items": [1, 2, 3], "value": 42}
        restored = import_from_json(export_json_str(original))
        assert restored["items"] == [1, 2, 3]
        assert restored["value"] == 42
