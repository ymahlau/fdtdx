from typing import Literal
import jax


def check_shape_dtype(
    arrays: dict[str, jax.Array],
    expected_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
    method: Literal["exact", "arrays_only"] = "exact"
):
    if method == "exact" and len(arrays) != len(expected_shape_dtypes):
        raise Exception(
            f"Arrays and expected dict have different lengths: "
            f"{arrays.keys()=} \n\n but {expected_shape_dtypes=}"
        )
    for k, arr in arrays.items():
        exp_shape_dtype = expected_shape_dtypes[k]
        if arr.dtype != exp_shape_dtype.dtype:
            raise Exception(f"Wrong dtype: {exp_shape_dtype.dtype} != {arr.dtype}")
        if arr.shape != exp_shape_dtype.shape:
            raise Exception(f"Wrong shape: {exp_shape_dtype.shape} != {arr.shape}")
