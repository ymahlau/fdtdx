import math

import jax

from fdtdx.core.jax.pytrees import extended_autoinit, frozen_field
from fdtdx.objects.device.parameters.transform import SameShapeContinousParameterTransform


@extended_autoinit
class StandardToInversePermittivityRange(SameShapeContinousParameterTransform):
    """Maps standard [0,1] range to inverse permittivity range.

    Linearly maps values from [0,1] to the range between minimum and maximum
    inverse permittivity values allowed by the material configuration.
    """

    def __call__(
        self,
        params: dict[str, jax.Array] | jax.Array,
        **kwargs,
    ) -> dict[str, jax.Array] | jax.Array:
        del kwargs
        # determine minimum and maximum allowed permittivity
        max_inv_perm, min_inv_perm = -math.inf, math.inf
        if isinstance(self._materials, dict):
            for k, v in self._materials.items():
                p = 1 / v.permittivity
                if p > max_inv_perm:
                    max_inv_perm = p
                if p < min_inv_perm:
                    min_inv_perm = p

        # transform
        if isinstance(params, dict):
            result = {}
            for k, v in params.items():
                mapped = v * (max_inv_perm - min_inv_perm) + min_inv_perm
                result[k] = mapped
        else:
            result = params * (max_inv_perm - min_inv_perm) + min_inv_perm
        return result


@extended_autoinit
class StandardToCustomRange(SameShapeContinousParameterTransform):
    """Maps standard [0,1] range to custom range [min_value, max_value].

    Linearly maps values from [0,1] to a custom range specified by min_value
    and max_value parameters.

    Attributes:
        min_value: Minimum value of target range
        max_value: Maximum value of target range
    """

    min_value: float = frozen_field(default=0)
    max_value: float = frozen_field(default=1)

    def __call__(
        self,
        params: dict[str, jax.Array] | jax.Array,
        **kwargs,
    ) -> dict[str, jax.Array] | jax.Array:
        del kwargs
        if isinstance(params, dict):
            result = {}
            for k, v in params.items():
                mapped = v * (self.max_value - self.min_value) + self.min_value
                result[k] = mapped
        else:
            result = params * (self.max_value - self.min_value) + self.min_value
        return result


@extended_autoinit
class StandardToPlusOneMinusOneRange(StandardToCustomRange):
    """Maps standard [0,1] range to [-1,1] range.

    Special case of StandardToCustomRange that maps to [-1,1] range.
    Used for symmetric value ranges around zero.

    Attributes:
        min_value: Fixed to -1
        max_value: Fixed to 1
    """

    min_value: float = frozen_field(default=-1, init=False)
    max_value: float = frozen_field(default=1, init=False)
