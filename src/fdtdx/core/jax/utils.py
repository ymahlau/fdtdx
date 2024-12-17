import jax


def check_shape_dtype(
    partial_value_dict: dict[str, jax.Array],
    shape_dtype_dict: dict[str, jax.ShapeDtypeStruct],
):
    """Validates that arrays match their expected shapes and dtypes.

    Checks each array in partial_value_dict against its corresponding shape and dtype
    specification in shape_dtype_dict. This is useful for validating that arrays
    match their expected specifications before using them in computations.

    Args:
        partial_value_dict: Dictionary mapping names to JAX arrays to validate
        shape_dtype_dict: Dictionary mapping names to ShapeDtypeStruct objects
            specifying the expected shape and dtype for each array

    Raises:
        Exception: If any array's shape or dtype doesn't match its specification
            in shape_dtype_dict. The error message indicates which array failed
            and how its shape/dtype differed from expected.

    Example:
        >>> shapes = {"x": jax.ShapeDtypeStruct((2,3), jnp.float32)}
        >>> arrays = {"x": jnp.zeros((2,3), dtype=jnp.float32)}
        >>> check_shape_dtype(arrays, shapes)  # Passes
        >>> bad = {"x": jnp.zeros((3,2))}  # Wrong shape
        >>> check_shape_dtype(bad, shapes)  # Raises Exception
    """
    for k, arr in partial_value_dict.items():
        shape_dtype = shape_dtype_dict[k]
        if arr.dtype != shape_dtype.dtype:
            raise Exception(f"Wrong dtype: {shape_dtype.dtype} != {arr.dtype}")
        if arr.shape != shape_dtype.shape:
            raise Exception(f"Wrong shape: {shape_dtype.shape} != {arr.shape}")
