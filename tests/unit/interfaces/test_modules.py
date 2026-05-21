"""Unit tests for fdtdx.interfaces.modules"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.interfaces.modules import CompressionModule, DtypeConversion
from fdtdx.interfaces.state import RecordingState

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_state():
    """Create a minimal RecordingState with empty dicts."""
    return RecordingState(data={}, state={})


def _make_input_shapes(**fields):
    """Create a dict of ShapeDtypeStructs; each value is (shape, dtype)."""
    return {name: jax.ShapeDtypeStruct(shape, dtype) for name, (shape, dtype) in fields.items()}


# ─── TestCompressionModuleAbstract ────────────────────────────────────────────


class TestCompressionModuleAbstract:
    """CompressionModule is an ABC — concrete subclasses must implement all methods."""

    def test_cannot_instantiate_directly(self):
        """Direct instantiation of the abstract class raises TypeError."""
        with pytest.raises(TypeError):
            CompressionModule()  # type: ignore

    def test_subclass_must_implement_init_shapes(self):
        """A subclass that omits init_shapes is not instantiable."""

        class IncompleteModule(CompressionModule):
            def compress(self, values, state, key):
                return values, state

            def decompress(self, values, state, key):
                return values

        with pytest.raises(TypeError):
            IncompleteModule()

    def test_subclass_must_implement_compress(self):
        """A subclass that omits compress is not instantiable."""

        class IncompleteModule(CompressionModule):
            def init_shapes(self, input_shape_dtypes):
                return self, input_shape_dtypes, {}

            def decompress(self, values, state, key):
                return values

        with pytest.raises(TypeError):
            IncompleteModule()

    def test_subclass_must_implement_decompress(self):
        """A subclass that omits decompress is not instantiable."""

        class IncompleteModule(CompressionModule):
            def init_shapes(self, input_shape_dtypes):
                return self, input_shape_dtypes, {}

            def compress(self, values, state, key):
                return values, state

        with pytest.raises(TypeError):
            IncompleteModule()


# ─── TestDtypeConversionInitShapes ────────────────────────────────────────────


class TestDtypeConversionInitShapes:
    """Tests for DtypeConversion.init_shapes."""

    def test_output_shapes_have_target_dtype(self):
        """All output shapes use the target dtype."""
        module = DtypeConversion(dtype=jnp.float16)
        input_shapes = _make_input_shapes(
            E=((3, 4, 4), jnp.float32),
            H=((3, 4, 4), jnp.float32),
        )
        _, out_shapes, _ = module.init_shapes(input_shapes)
        assert out_shapes["E"].dtype == jnp.float16
        assert out_shapes["H"].dtype == jnp.float16

    def test_output_shapes_preserve_spatial_shape(self):
        """Output shapes retain the original spatial dimensions."""
        module = DtypeConversion(dtype=jnp.float16)
        input_shapes = _make_input_shapes(E=((3, 8, 6), jnp.float32))
        _, out_shapes, _ = module.init_shapes(input_shapes)
        assert out_shapes["E"].shape == (3, 8, 6)

    def test_state_shapes_is_empty(self):
        """DtypeConversion has no state; state_shapes must be empty."""
        module = DtypeConversion(dtype=jnp.float16)
        input_shapes = _make_input_shapes(E=((4, 4, 4), jnp.float32))
        _, _, state_shapes = module.init_shapes(input_shapes)
        assert state_shapes == {}

    def test_exclude_filter_preserves_dtype(self):
        """Fields matching exclude_filter keep their original dtype."""
        module = DtypeConversion(dtype=jnp.float16, exclude_filter=["H"])
        input_shapes = _make_input_shapes(
            E=((3, 4, 4), jnp.float32),
            H=((3, 4, 4), jnp.float32),
        )
        _, out_shapes, _ = module.init_shapes(input_shapes)
        assert out_shapes["E"].dtype == jnp.float16  # converted
        assert out_shapes["H"].dtype == jnp.float32  # preserved

    def test_exclude_filter_partial_match(self):
        """Exclude filter uses substring matching (any(e in k for e in filter))."""
        module = DtypeConversion(dtype=jnp.float16, exclude_filter=["pml"])
        input_shapes = _make_input_shapes(
            E=((3, 4, 4), jnp.float32),
            pml_sigma=((3, 4, 4), jnp.float32),
        )
        _, out_shapes, _ = module.init_shapes(input_shapes)
        assert out_shapes["E"].dtype == jnp.float16
        assert out_shapes["pml_sigma"].dtype == jnp.float32

    def test_internal_input_shapes_stored(self):
        """After init_shapes, _input_shape_dtypes is populated."""
        module = DtypeConversion(dtype=jnp.float16)
        input_shapes = _make_input_shapes(E=((3, 4, 4), jnp.float32))
        updated, _, _ = module.init_shapes(input_shapes)
        assert updated._input_shape_dtypes["E"].dtype == jnp.float32

    def test_internal_output_shapes_stored(self):
        """After init_shapes, _output_shape_dtypes matches returned out_shapes."""
        module = DtypeConversion(dtype=jnp.float16)
        input_shapes = _make_input_shapes(E=((3, 4, 4), jnp.float32))
        updated, out_shapes, _ = module.init_shapes(input_shapes)
        assert updated._output_shape_dtypes["E"].dtype == out_shapes["E"].dtype

    def test_multiple_fields_all_converted(self):
        """All fields in a multi-field input are converted."""
        module = DtypeConversion(dtype=jnp.bfloat16)
        input_shapes = _make_input_shapes(
            a=((2, 2), jnp.float32),
            b=((4, 4), jnp.float64),
            c=((1,), jnp.float32),
        )
        _, out_shapes, _ = module.init_shapes(input_shapes)
        for key in ["a", "b", "c"]:
            assert out_shapes[key].dtype == jnp.bfloat16


# ─── TestDtypeConversionCompress ──────────────────────────────────────────────


class TestDtypeConversionCompress:
    """Tests for DtypeConversion.compress."""

    def _make_initialized_module(self, dtype, input_shapes, exclude_filter=None):
        kwargs = {"dtype": dtype}
        if exclude_filter is not None:
            kwargs["exclude_filter"] = exclude_filter
        module = DtypeConversion(**kwargs)
        module, _, _ = module.init_shapes(input_shapes)
        return module

    def test_compress_converts_to_target_dtype(self):
        """Output arrays have the target dtype."""
        input_shapes = _make_input_shapes(E=((3, 4, 4), jnp.float32))
        module = self._make_initialized_module(jnp.float16, input_shapes)
        values = {"E": jnp.ones((3, 4, 4), dtype=jnp.float32)}
        out_vals, _ = module.compress(values, _make_state(), jax.random.PRNGKey(0))
        assert out_vals["E"].dtype == jnp.float16

    def test_compress_preserves_shape(self):
        """Output arrays retain the same shape."""
        input_shapes = _make_input_shapes(E=((3, 4, 5), jnp.float32))
        module = self._make_initialized_module(jnp.float16, input_shapes)
        values = {"E": jnp.ones((3, 4, 5), dtype=jnp.float32)}
        out_vals, _ = module.compress(values, _make_state(), jax.random.PRNGKey(0))
        assert out_vals["E"].shape == (3, 4, 5)

    def test_compress_preserves_values_within_precision(self):
        """Compressed values match originals within float16 precision."""
        input_shapes = _make_input_shapes(E=((2, 2), jnp.float32))
        module = self._make_initialized_module(jnp.float16, input_shapes)
        original = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        out_vals, _ = module.compress({"E": original}, _make_state(), jax.random.PRNGKey(0))
        assert jnp.allclose(out_vals["E"].astype(jnp.float32), original, rtol=1e-2)

    def test_compress_state_returned_unchanged(self):
        """compress returns the state object unchanged."""
        input_shapes = _make_input_shapes(E=((3, 4, 4), jnp.float32))
        module = self._make_initialized_module(jnp.float16, input_shapes)
        state = _make_state()
        values = {"E": jnp.ones((3, 4, 4), dtype=jnp.float32)}
        _, returned_state = module.compress(values, state, jax.random.PRNGKey(0))
        assert returned_state is state

    def test_compress_exclude_filter_skips_field(self):
        """Fields matching exclude_filter are not dtype-converted."""
        input_shapes = _make_input_shapes(
            E=((3, 4, 4), jnp.float32),
            H=((3, 4, 4), jnp.float32),
        )
        module = self._make_initialized_module(jnp.float16, input_shapes, exclude_filter=["H"])
        values = {
            "E": jnp.ones((3, 4, 4), dtype=jnp.float32),
            "H": jnp.ones((3, 4, 4), dtype=jnp.float32),
        }
        out_vals, _ = module.compress(values, _make_state(), jax.random.PRNGKey(0))
        assert out_vals["E"].dtype == jnp.float16  # converted
        assert out_vals["H"].dtype == jnp.float32  # not converted

    def test_compress_multiple_fields(self):
        """All non-excluded fields are converted when multiple fields are present."""
        input_shapes = _make_input_shapes(
            a=((2,), jnp.float32),
            b=((4,), jnp.float32),
        )
        module = self._make_initialized_module(jnp.float16, input_shapes)
        values = {
            "a": jnp.ones((2,), dtype=jnp.float32),
            "b": jnp.ones((4,), dtype=jnp.float32),
        }
        out_vals, _ = module.compress(values, _make_state(), jax.random.PRNGKey(0))
        assert out_vals["a"].dtype == jnp.float16
        assert out_vals["b"].dtype == jnp.float16


# ─── TestDtypeConversionDecompress ────────────────────────────────────────────


class TestDtypeConversionDecompress:
    """Tests for DtypeConversion.decompress."""

    def _make_initialized_module(self, dtype, input_shapes):
        module = DtypeConversion(dtype=dtype)
        module, _, _ = module.init_shapes(input_shapes)
        return module

    def test_decompress_restores_original_dtype(self):
        """Decompressed arrays have the original float32 dtype."""
        input_shapes = _make_input_shapes(E=((3, 4, 4), jnp.float32))
        module = self._make_initialized_module(jnp.float16, input_shapes)
        compressed = {"E": jnp.ones((3, 4, 4), dtype=jnp.float16)}
        out_vals = module.decompress(compressed, _make_state(), jax.random.PRNGKey(0))
        assert out_vals["E"].dtype == jnp.float32

    def test_decompress_preserves_shape(self):
        """Decompressed arrays retain their spatial shape."""
        input_shapes = _make_input_shapes(E=((3, 5, 7), jnp.float32))
        module = self._make_initialized_module(jnp.float16, input_shapes)
        compressed = {"E": jnp.ones((3, 5, 7), dtype=jnp.float16)}
        out_vals = module.decompress(compressed, _make_state(), jax.random.PRNGKey(0))
        assert out_vals["E"].shape == (3, 5, 7)

    def test_decompress_values_within_precision(self):
        """Roundtrip compress → decompress recovers original values within float16 tolerance."""
        input_shapes = _make_input_shapes(E=((2, 2), jnp.float32))
        module = self._make_initialized_module(jnp.float16, input_shapes)
        original = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        key = jax.random.PRNGKey(0)
        state = _make_state()
        compressed, _ = module.compress({"E": original}, state, key)
        recovered = module.decompress(compressed, state, key)
        assert jnp.allclose(recovered["E"], original, rtol=1e-2)

    def test_decompress_multiple_fields(self):
        """Decompression handles multiple fields independently."""
        input_shapes = _make_input_shapes(
            a=((2,), jnp.float32),
            b=((3,), jnp.bfloat16),
        )
        module = self._make_initialized_module(jnp.float16, input_shapes)
        compressed = {
            "a": jnp.ones((2,), dtype=jnp.float16),
            "b": jnp.ones((3,), dtype=jnp.float16),
        }
        out_vals = module.decompress(compressed, _make_state(), jax.random.PRNGKey(0))
        assert out_vals["a"].dtype == jnp.float32
        assert out_vals["b"].dtype == jnp.bfloat16

    def test_roundtrip_preserves_dtype(self):
        """Compress followed by decompress returns to the original dtype."""
        input_shapes = _make_input_shapes(
            E=((4, 4), jnp.float32),
            H=((4, 4), jnp.float32),
        )
        module = self._make_initialized_module(jnp.float16, input_shapes)
        values = {
            "E": jnp.ones((4, 4), dtype=jnp.float32) * 0.5,
            "H": jnp.ones((4, 4), dtype=jnp.float32) * 1.5,
        }
        key = jax.random.PRNGKey(42)
        state = _make_state()
        compressed, _ = module.compress(values, state, key)
        recovered = module.decompress(compressed, state, key)
        assert recovered["E"].dtype == jnp.float32
        assert recovered["H"].dtype == jnp.float32
        assert jnp.allclose(recovered["E"], values["E"], rtol=1e-2)
        assert jnp.allclose(recovered["H"], values["H"], rtol=1e-2)

    def test_decompress_to_non_float32_original_dtype(self):
        """Decompression restores non-float32 original dtypes (e.g., bfloat16)."""
        input_shapes = _make_input_shapes(E=((3,), jnp.bfloat16))
        module = self._make_initialized_module(jnp.float16, input_shapes)
        compressed = {"E": jnp.ones((3,), dtype=jnp.float16)}
        out_vals = module.decompress(compressed, _make_state(), jax.random.PRNGKey(0))
        assert out_vals["E"].dtype == jnp.bfloat16


# ─── TestDtypeConversionComplexValidation ────────────────────────────────────


class TestDtypeConversionComplexValidation:
    """Tests for DtypeConversion validation of complex-to-real conversions."""

    def test_complex_input_to_real_target_raises(self):
        """Converting complex64 input to float16 target raises ValueError."""
        module = DtypeConversion(dtype=jnp.float16)
        input_shapes = _make_input_shapes(E=((3, 4, 4), jnp.complex64))
        with pytest.raises(ValueError, match="silently discard the imaginary"):
            module.init_shapes(input_shapes)

    def test_complex128_input_to_float64_target_raises(self):
        """Converting complex128 input to float64 target raises ValueError."""
        module = DtypeConversion(dtype=jnp.float64)
        input_shapes = _make_input_shapes(E=((3, 4, 4), jnp.complex128))
        with pytest.raises(ValueError, match="silently discard the imaginary"):
            module.init_shapes(input_shapes)

    def test_complex_input_to_complex_target_allowed(self):
        """Converting complex64 to complex128 is allowed."""
        module = DtypeConversion(dtype=jnp.complex128)
        input_shapes = _make_input_shapes(E=((3, 4, 4), jnp.complex64))
        _, out_shapes, _ = module.init_shapes(input_shapes)
        assert out_shapes["E"].dtype == jnp.complex128

    def test_complex_input_excluded_field_allowed(self):
        """Complex input with a real target is allowed when the field is excluded."""
        module = DtypeConversion(dtype=jnp.float16, exclude_filter=["E"])
        input_shapes = _make_input_shapes(E=((3, 4, 4), jnp.complex64))
        _, out_shapes, _ = module.init_shapes(input_shapes)
        assert out_shapes["E"].dtype == jnp.complex64

    def test_mixed_real_and_complex_raises_for_complex_field(self):
        """With mixed inputs, only the complex field triggers the error."""
        module = DtypeConversion(dtype=jnp.float16)
        input_shapes = _make_input_shapes(
            E=((3, 4, 4), jnp.complex64),
            sigma=((3, 4, 4), jnp.float32),
        )
        with pytest.raises(ValueError, match="E"):
            module.init_shapes(input_shapes)

    def test_real_input_to_real_target_allowed(self):
        """Standard real-to-real conversion is unaffected by the validation."""
        module = DtypeConversion(dtype=jnp.float16)
        input_shapes = _make_input_shapes(E=((3, 4, 4), jnp.float32))
        _, out_shapes, _ = module.init_shapes(input_shapes)
        assert out_shapes["E"].dtype == jnp.float16
