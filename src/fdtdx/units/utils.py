import math
from typing import Any, Sequence

import jax
import numpy as np

from fdtdx.core.fraction import Fraction
from fdtdx.core.jax.utils import is_traced
from fdtdx.units.typing import SI, PhysicalArrayLike


def handle_n_scales(
    scales: Sequence[int],
) -> tuple[int, list[float]]:
    """
    This uses some static approximations to best maintain numerical
    stability while minimizing overhead from too many conversions.

    Args:
        scales (Sequence[int]): Scales (10^s)

    Returns:
        tuple[int, list[float]]: Tuple of the new scale, and factors to multiply with original scales
    """
    if len(scales) == 1:
        return scales[0], [1.0]
    if len(scales) == 2:
        new_scale, f1, f2 = handle_different_scales(scales[0], scales[1])
        return new_scale, [f1, f2]

    min_scale, max_scale = min(scales), max(scales)
    new_scale, _, _ = handle_different_scales(min_scale, max_scale)
    factors = [10 ** (f - new_scale) for f in scales]
    return new_scale, factors


def handle_different_scales(
    s1: int,
    s2: int,
) -> tuple[int, float, float]:  # new_scale, factor1, factor2
    """
    This uses some static approximations to best maintain numerical
    stability while minimizing overhead from too many conversions.

    Args:
        s1 (int): Scale of first dimension (10^s1)
        s2 (int): Scale of second dimension (10^s2)

    Returns:
        tuple[int, float, float]: Tuple of the new scale, and two factors to multiply with original scale
    """
    if s1 == s2:
        return (s1, 1, 1)
    # compute possible conversion factors
    s2_factor = 10 ** (s2 - s1)
    s1_factor = 10 ** (s1 - s2)

    # if difference between the scales is 9 or more, choose the larger scale. At this level of difference,
    # numerical accuracy cannot be guaranteed and we only care about preventing complete failure to runtime errors.
    if abs(s1 - s2) >= 9:
        if s1 > s2:
            return (s1, 1, s2_factor)
        else:
            return (s2, s1_factor, 1)

    # if either of the scales is zero, use the other scale
    if s1 == 0:
        return (s2, s1_factor, 1)
    if s2 == 0:
        return (s1, 1, s2_factor)

    # if scales have the same sign, use the larger scale
    if np.sign(s1) == np.sign(s2):
        if abs(s1) > abs(s2):
            return (s1, 1, s2_factor)
        # here s2 has to be large since signs are the same
        return (s2, s1_factor, 1)

    # different signs, now choose smaller absolute value
    if abs(s1) < abs(s2):
        return (s1, 1, s2_factor)
    elif abs(s1) > abs(s2):
        return (s2, s1_factor, 1)

    # different signs, same abs value, use 0 scale
    s1_to_zero = 10 ** (s1)
    s2_to_zero = 10 ** (s2)
    return (0, s1_to_zero, s2_to_zero)


def best_scale(
    arr: PhysicalArrayLike,
    previous_scale: int,
) -> tuple[PhysicalArrayLike, int]:
    """
    This uses some static approximations to find the scale of an ArrayLike that has the best numerical accuracy.

    Args:
        arr (ArrayLike): Scale of first dimension (10^s1)
        previous_scale (int): Previously used scale for the array

    Returns:
        tuple[PhysicalArrayLike, int]: Tuple of the numerical arraylike, and the power of 10, which the array was multiplied
        with.
    """
    # we cannot optimize traced values, leave unchanged
    if is_traced(arr):
        return arr, 0

    def scalar_helper(abs_val):
        if math.isnan(abs_val):
            raise Exception(f"Detected NaN-value in Unitful array: {arr}")
        # best scale is zero, so eliminate previous scale
        if not math.isfinite(abs_val) or abs_val == 0:
            return arr, previous_scale
        log_offset = -round(math.log(abs_val, 10))
        # TODO: Improve this logic (current fallback is a workaround and may change dtypes).
        try:
            new_arr = arr * (10**log_offset)
        except OverflowError:
            # fallback: avoid huge Python int -> parsed as int32 in JAX
            new_arr = arr * (10.0**log_offset)
        return new_arr, log_offset

    # scalar logic: absolute value as close to one
    if isinstance(arr, float | complex | np.number):
        abs_val = abs(arr)
        return scalar_helper(abs_val)

    assert isinstance(arr, jax.Array | np.ndarray), "Internal error, please report"

    # array of size 1 can be handled the same as
    if arr.size == 1:
        abs_val = abs(arr.item())
        return scalar_helper(abs_val)

    np_arr = np.asarray(arr)

    # all values are zero: best scale is zero, so eliminate previous scale
    if np.any(np.isnan(arr)) or not np.all(np.isfinite(np_arr)) or np.all(np_arr == 0):
        return arr, previous_scale

    # Calculate median of absolute non-zero values.
    # Masking works here only because we are not working with jax arrays
    # non_zero_mask = np_arr != 0
    # nonzero_values = np_arr[non_zero_mask]
    # median_abs = np.median(np.abs(nonzero_values)).item()
    # log_offset = -round(math.log(median_abs, 10))

    base_val = np.max(np.abs(np_arr)).item()
    log_offset = -round(math.log(base_val, 10))

    scaled_arr = arr * (10.0**log_offset)
    return scaled_arr, log_offset


def dim_after_multiplication(
    dim1: dict[SI, int | Fraction],
    dim2: dict[SI, int | Fraction],
) -> dict[SI, int | Fraction]:
    unit_dict = dim1.copy()
    for k, v in dim2.items():
        if k in unit_dict:
            unit_dict[k] += v
            if unit_dict[k] == 0:
                del unit_dict[k]
        else:
            unit_dict[k] = v
    return unit_dict


def is_struct_optimizable(a: Any) -> bool:
    if is_traced(a):
        return False
    if isinstance(a, PhysicalArrayLike):
        return True
    if isinstance(a, Sequence):
        return all([is_struct_optimizable(a_i) for a_i in a])
    return False
