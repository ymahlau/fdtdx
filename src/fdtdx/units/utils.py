import equinox
import jax
import jax.numpy as jnp
from jax import core
from jaxtyping import ArrayLike
import numpy as np
from fastcore.foundation import copy_func

from fdtdx.typing import PhysicalArrayLike

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
    noop_in_jit: bool = True,
) -> tuple[PhysicalArrayLike, int]:
    """ 
    This uses some static approximations to find the scale of an ArrayLike that has the best numerical accuracy.
    These approximations are only performed outside of jit. In a jitted function, this reduces to a NoOp, except if the
    noop_in_jit is set to False.
    
    Args:
        arr (ArrayLike): Scale of first dimension (10^s1)
        noop_in_jit (bool): If True, the function reduces to Noop in jit context. Defaults to True. 
    
    Returns:
        tuple[ArrayLike, int]: Tuple of the numerical arraylike, and the power of 10, which the array was multiplied 
        with.
    """
    if noop_in_jit and isinstance(arr, core.Tracer):
        return arr, 0
    
    # Convert to array for easier manipulation
    arr_jax = jnp.abs(jnp.asarray(arr))
    
    if noop_in_jit:
        # Handle edge case: all values are zero
        non_zero_mask = arr_jax != 0
        if jnp.sum(non_zero_mask) == 0:
            return arr, 0
        
        # Calculate median of absolute non-zero values.
        # Masking works here only because we are not in jitted context
        nonzero_values = arr_jax[non_zero_mask]
        median_abs = jnp.median(nonzero_values)
    else:
        # Replace zeros with NaN
        arr_for_median = jnp.where(arr_jax == 0, jnp.nan, arr_jax)
        
        # Use nanmedian to ignore NaN values
        median_abs = jnp.nanmedian(arr_for_median)
        
        # Handle case where all values were zero (result would be NaN)
        median_abs = jnp.where(jnp.isnan(median_abs), 1.0, median_abs)
    
    # Find the power of 10 that brings median to around 1.0
    # log10(median_abs * scale_factor) ≈ 0
    # so scale_factor = 10^(-log10(median_abs))
    log_median: float = jnp.log10(median_abs).item()
    target_power = -round(log_median)
    scale_factor = 10.0 ** target_power
    
    # Apply scaling
    scaled_arr = arr * scale_factor
    
    return scaled_arr, target_power


def patch_fn_to_module(
    f,
    mod,
    fn_name: str | None = None,
) -> None:
    fn_copy = copy_func(f)
    if fn_name is None:
        fn_name = f.__name__
    assert fn_name is not None
    fn_copy.__qualname__ = f"{mod.__name__}.{fn_name}"
    original_name = '_orig_' + fn_name
    if hasattr(mod, fn_name) and not hasattr(mod, original_name):
        setattr(mod, original_name, getattr(mod, fn_name))
    setattr(mod, fn_name, fn_copy)
