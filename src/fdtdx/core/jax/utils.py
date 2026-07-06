from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core


def contains_bool(values: Any) -> bool:
    """Return whether a scalar or nested container contains boolean values."""
    if isinstance(values, (bool, np.bool_)):
        return True
    if isinstance(values, jax.Array):
        return bool(np.issubdtype(values.dtype, np.bool_))
    if isinstance(values, jax_core.Tracer):
        return False
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.bool_):
            return True
        if values.dtype != object:
            return False
    if isinstance(values, (str, bytes)):
        return False
    try:
        iterator = iter(values)
    except TypeError:
        return False
    return any(contains_bool(value) for value in iterator)


def is_jax_tracer(values: Any) -> bool:
    """Return whether ``values`` is a JAX tracer that cannot be concretely value-checked."""
    return isinstance(values, jax_core.Tracer)


def is_real_numeric_dtype(dtype: Any) -> bool:
    """Return whether a dtype is numeric but not complex."""
    return np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.complexfloating)


def concrete_jax_bool(value: jax.Array) -> bool:
    """Convert an eager scalar JAX boolean to Python bool, returning false for traced booleans."""
    try:
        return bool(value)
    except jax.errors.TracerBoolConversionError:
        return False


def concrete_jax_array_has_nonfinite(values: jax.Array) -> bool:
    """Return whether an eager JAX array contains NaN or infinite values."""
    return concrete_jax_bool(jnp.any(~jnp.isfinite(values)))


def real_numeric_array(name: str, values: Any, message: str, *, keep_jax: bool) -> jax.Array | np.ndarray:
    """Validate real finite numeric array input while preserving JAX arrays when requested.

    Concrete Python/NumPy inputs are fully checked immediately. Eager JAX arrays are checked without converting them
    through NumPy on JAX-heavy paths; traced arrays keep dtype validation but defer value checks to compiled execution.
    """
    if contains_bool(values):
        raise ValueError(message)
    if is_jax_tracer(values):
        if not is_real_numeric_dtype(values.dtype):
            raise ValueError(message)
        if keep_jax:
            return jnp.asarray(values, dtype=float)
        raise ValueError(message)
    if isinstance(values, jax.Array):
        if not is_real_numeric_dtype(values.dtype):
            raise ValueError(message)
        array = jnp.asarray(values, dtype=float)
        if concrete_jax_array_has_nonfinite(array):
            raise ValueError(message)
        return array if keep_jax else np.asarray(array)
    try:
        raw_array = np.asarray(values)
    except jax.errors.TracerArrayConversionError as err:
        if keep_jax:
            return jnp.asarray(values)
        raise ValueError(message) from err
    except (TypeError, ValueError) as err:
        raise ValueError(message) from err
    if not is_real_numeric_dtype(raw_array.dtype):
        raise ValueError(message)
    array = raw_array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError(message)
    return jnp.asarray(array) if keep_jax else array


def finite_1d_array(name: str, values: Any, expected_size: int) -> np.ndarray:
    """Validate a concrete one-dimensional real numeric array of fixed size."""
    message = f"{name} must contain finite numeric values."
    array = real_numeric_array(name, values, message, keep_jax=False)
    if array.shape != (expected_size,):
        raise ValueError(f"{name} must contain {expected_size} values.")
    return cast(np.ndarray, array)


def finite_numeric_array(name: str, values: Any) -> jax.Array:
    """Validate coordinate-like numeric input and return a JAX array."""
    return cast(
        jax.Array, real_numeric_array(name, values, f"{name} must contain finite numeric values.", keep_jax=True)
    )


def finite_scalar(name: str, value: Any) -> float:
    """Validate a concrete finite numeric scalar, rejecting bools and non-scalars."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite numeric value.")
    if isinstance(value, (complex, np.complexfloating)):
        raise ValueError(f"{name} must be a finite numeric value.")
    if not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite numeric value.")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{name} must be a finite numeric value.") from err
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite numeric value.")
    return scalar


def finite_scalar_or_1d_array(name: str, value: Any, expected_size: int) -> float | np.ndarray:
    """Validate a scalar or one value per wave character."""
    message = f"{name} must be a finite numeric scalar or contain {expected_size} values."
    array = real_numeric_array(name, value, message, keep_jax=False)
    if array.ndim == 0:
        return float(array)
    if array.shape != (expected_size,):
        raise ValueError(message)
    return cast(np.ndarray, array)


def check_specs(
    arrays: dict[str, jax.Array] | jax.Array,
    expected_shapes: dict[str, tuple[int, ...]] | tuple[int, ...],
):
    if isinstance(arrays, dict) != isinstance(expected_shapes, dict):
        arr_str = f"{arrays.shape}" if isinstance(arrays, jax.Array) else str(arrays.keys())
        raise Exception(f"Got different structures in arrays: {expected_shapes=}, \n{arr_str=}")
    if not isinstance(arrays, dict):
        arrays = {"dummy": arrays}
    if not isinstance(expected_shapes, dict):
        expected_shapes = {"dummy": expected_shapes}
    arrays = cast(dict[str, jax.Array], arrays)

    if len(arrays) != len(expected_shapes):
        raise Exception(
            f"Arrays and expected dict have different lengths: {arrays.keys()=} \n\n but {expected_shapes=}"
        )
    for k, arr in arrays.items():
        exp_shape = expected_shapes[k]
        if arr.shape != exp_shape:
            raise Exception(f"Wrong shape: {exp_shape} != {arr.shape}")


def check_shape_dtype(
    arrays: dict[str, jax.Array] | jax.Array,
    expected_shape_dtypes: dict[str, jax.ShapeDtypeStruct] | jax.ShapeDtypeStruct,
):
    if isinstance(arrays, jax.Array) == isinstance(expected_shape_dtypes, dict):
        arr_str = f"{arrays.shape}" if isinstance(arrays, jax.Array) else str(arrays.keys())
        raise Exception(f"Got different structures in arrays: {expected_shape_dtypes=}, \n{arr_str=}")
    if not isinstance(arrays, dict):
        arrays = {"dummy": arrays}
    if not isinstance(expected_shape_dtypes, dict):
        expected_shape_dtypes = {"dummy": expected_shape_dtypes}
    expected_shape_dtypes = cast(dict[str, jax.ShapeDtypeStruct], expected_shape_dtypes)
    arrays = cast(dict[str, jax.Array], arrays)
    if len(arrays) != len(expected_shape_dtypes):
        raise Exception(
            f"Arrays and expected dict have different lengths: {arrays.keys()=} \n\n but {expected_shape_dtypes=}"
        )
    for k, arr in arrays.items():
        exp_shape_dtype = expected_shape_dtypes[k]
        if arr.dtype != exp_shape_dtype.dtype:
            raise Exception(f"Wrong dtype: {exp_shape_dtype.dtype} != {arr.dtype}")
        if arr.shape != exp_shape_dtype.shape:
            raise Exception(f"Wrong shape: {exp_shape_dtype.shape} != {arr.shape}")
