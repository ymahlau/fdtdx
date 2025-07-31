
import numpy as np


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