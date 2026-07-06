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
from fdtdx.materials import Material
from fdtdx.objects.detectors.field_projection import (
    FieldProjectionAngleDetector,
    FieldProjectionCartesianDetector,
    FieldProjectionKSpaceDetector,
)


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

    def test_roundtrip_field_projection_angle_detector(self):
        """Test roundtrip with FieldProjectionAngleDetector."""
        original = FieldProjectionAngleDetector(
            wave_characters=[WaveCharacter(wavelength=0.689e-6)],
            direction="+",
            origin=(0.0, 0.0, 1e-6),
            projection_distance=2.0,
            exact_projection_batch_size=64,
            window_size=(0.1, 0.2),
            interval_space=(2, 3, 1),
            projection_medium_refractive_index=1.5,
        )
        restored = import_from_json(export_json_str(original))

        assert isinstance(restored, FieldProjectionAngleDetector)
        assert restored.wave_characters[0].wavelength == original.wave_characters[0].wavelength
        assert restored.direction == original.direction
        assert restored.origin == original.origin
        assert restored.projection_distance == original.projection_distance
        assert restored.exact_projection_batch_size == original.exact_projection_batch_size
        assert restored.window_size == original.window_size
        assert restored.interval_space == original.interval_space
        assert restored.projection_medium_refractive_index == original.projection_medium_refractive_index

    def test_roundtrip_field_projection_angle_detector_with_material_medium(self):
        """Test roundtrip with a Material projection medium."""
        original = FieldProjectionAngleDetector(
            wave_characters=[WaveCharacter(wavelength=0.689e-6)],
            direction="+",
            projection_medium=Material(permittivity=2.25, permeability=1.2),
        )
        restored = import_from_json(export_json_str(original))

        assert isinstance(restored, FieldProjectionAngleDetector)
        assert isinstance(restored.projection_medium, Material)
        assert restored.projection_medium.permittivity == original.projection_medium.permittivity
        assert restored.projection_medium.permeability == original.projection_medium.permeability

    @pytest.mark.parametrize(
        ("detector_class", "projection_axis"),
        [
            (FieldProjectionCartesianDetector, 0),
            (FieldProjectionKSpaceDetector, 1),
        ],
    )
    def test_roundtrip_field_projection_coordinate_detectors(self, detector_class, projection_axis):
        """Test roundtrip with non-angular field projection detectors."""
        original = detector_class(
            wave_characters=[WaveCharacter(wavelength=0.689e-6)],
            direction="+",
            projection_axis=projection_axis,
            origin=(0.0, 0.0, 1e-6),
            projection_distance=2.0,
            far_field_approx=False,
            exact_projection_batch_size=64,
            window_size=(0.1, 0.2),
            interval_space=(2, 3, 1),
            projection_medium_refractive_index=1.5,
        )
        restored = import_from_json(export_json_str(original))

        assert isinstance(restored, detector_class)
        assert restored.wave_characters[0].wavelength == original.wave_characters[0].wavelength
        assert restored.direction == original.direction
        assert restored.projection_axis == projection_axis
        assert restored.origin == original.origin
        assert restored.projection_distance == original.projection_distance
        assert restored.far_field_approx == original.far_field_approx
        assert restored.exact_projection_batch_size == original.exact_projection_batch_size
        assert restored.window_size == original.window_size
        assert restored.interval_space == original.interval_space
        assert restored.projection_medium_refractive_index == original.projection_medium_refractive_index

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
