import jax
import jax.numpy as jnp
import pytest

from fdtdx.core.jax.utils import check_shape_dtype, check_specs


def test_check_specs_single_array_correct_shape():
    """Test check_specs with a single array that has the correct shape."""
    arr = jnp.array([[1, 2], [3, 4]])
    expected_shape = (2, 2)
    # Should not raise an exception
    check_specs(arr, expected_shape)


def test_check_specs_single_array_wrong_shape():
    """Test check_specs with a single array that has the wrong shape."""
    arr = jnp.array([[1, 2], [3, 4]])
    expected_shape = (3, 2)
    with pytest.raises(Exception):
        check_specs(arr, expected_shape)


def test_check_specs_dict_arrays_correct_shapes():
    """Test check_specs with a dictionary of arrays that have correct shapes."""
    arrays = {"a": jnp.array([[1, 2], [3, 4]]), "b": jnp.array([1, 2, 3])}
    expected_shapes = {"a": (2, 2), "b": (3,)}
    # Should not raise an exception
    check_specs(arrays, expected_shapes)


def test_check_specs_dict_arrays_wrong_shapes():
    """Test check_specs with a dictionary of arrays that have wrong shapes."""
    arrays = {"a": jnp.array([[1, 2], [3, 4]]), "b": jnp.array([1, 2, 3])}
    expected_shapes = {
        "a": (2, 2),
        "b": (4,),  # Wrong shape
    }
    with pytest.raises(Exception):
        check_specs(arrays, expected_shapes)


def test_check_specs_mismatched_structures_array_vs_dict():
    """Test check_specs with array given but dict expected."""
    arr = jnp.array([[1, 2], [3, 4]])
    expected_shapes = {"a": (2, 2)}
    with pytest.raises(Exception):
        check_specs(arr, expected_shapes)


def test_check_specs_mismatched_structures_dict_vs_tuple():
    """Test check_specs with dict given but tuple expected."""
    arrays = {"a": jnp.array([1, 2])}
    expected_shape = (2,)
    with pytest.raises(Exception):
        check_specs(arrays, expected_shape)


def test_check_specs_dict_different_lengths():
    """Test check_specs with dict arrays and expected shapes of different lengths."""
    arrays = {"a": jnp.array([1, 2]), "b": jnp.array([3, 4])}
    expected_shapes = {"a": (2,)}
    with pytest.raises(Exception):
        check_specs(arrays, expected_shapes)


def test_check_shape_dtype_single_array_correct():
    """Test check_shape_dtype with a single array that has correct shape and dtype."""
    arr = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
    expected_shape_dtype = jax.ShapeDtypeStruct((2, 2), jnp.float32)
    # Should not raise an exception
    check_shape_dtype(arr, expected_shape_dtype)


def test_check_shape_dtype_single_array_wrong_dtype():
    """Test check_shape_dtype with a single array that has wrong dtype."""
    arr = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
    expected_shape_dtype = jax.ShapeDtypeStruct((2, 2), jnp.int32)
    with pytest.raises(Exception):
        check_shape_dtype(arr, expected_shape_dtype)


def test_check_shape_dtype_dict_arrays_correct():
    """Test check_shape_dtype with a dictionary of arrays that have correct shapes and dtypes."""
    arrays = {"a": jnp.array([[1, 2], [3, 4]], dtype=jnp.float32), "b": jnp.array([1, 2, 3], dtype=jnp.int32)}
    expected_shape_dtypes = {"a": jax.ShapeDtypeStruct((2, 2), jnp.float32), "b": jax.ShapeDtypeStruct((3,), jnp.int32)}
    # Should not raise an exception
    check_shape_dtype(arrays, expected_shape_dtypes)


def test_check_shape_dtype_dict_arrays_wrong_shape():
    """Test check_shape_dtype with a dictionary of arrays that have wrong shapes."""
    arrays = {"a": jnp.array([[1, 2], [3, 4]], dtype=jnp.float32), "b": jnp.array([1, 2, 3], dtype=jnp.int32)}
    expected_shape_dtypes = {
        "a": jax.ShapeDtypeStruct((2, 2), jnp.float32),
        "b": jax.ShapeDtypeStruct((4,), jnp.int32),  # Wrong shape
    }
    with pytest.raises(Exception):
        check_shape_dtype(arrays, expected_shape_dtypes)


def test_check_shape_dtype_single_array_wrong_shape():
    """Test check_shape_dtype with a single array that has wrong shape but correct dtype."""
    arr = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
    expected_shape_dtype = jax.ShapeDtypeStruct((3, 2), jnp.float32)
    with pytest.raises(Exception):
        check_shape_dtype(arr, expected_shape_dtype)


def test_check_shape_dtype_mismatched_structures_array_vs_dict():
    """Test check_shape_dtype with array given but dict expected."""
    arr = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
    expected_shape_dtypes = {"a": jax.ShapeDtypeStruct((2, 2), jnp.float32)}
    with pytest.raises(Exception):
        check_shape_dtype(arr, expected_shape_dtypes)


def test_check_shape_dtype_mismatched_structures_dict_vs_single():
    """Test check_shape_dtype with dict given but single ShapeDtypeStruct expected."""
    arrays = {"a": jnp.array([1, 2], dtype=jnp.float32)}
    expected = jax.ShapeDtypeStruct((2,), jnp.float32)
    with pytest.raises(Exception):
        check_shape_dtype(arrays, expected)


def test_check_shape_dtype_dict_different_lengths():
    """Test check_shape_dtype with dict arrays and expected of different lengths."""
    arrays = {"a": jnp.array([1], dtype=jnp.float32), "b": jnp.array([2], dtype=jnp.float32)}
    expected = {"a": jax.ShapeDtypeStruct((1,), jnp.float32)}
    with pytest.raises(Exception):
        check_shape_dtype(arrays, expected)
