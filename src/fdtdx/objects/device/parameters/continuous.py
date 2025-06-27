import math
from typing import Sequence

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field, frozen_private_field
from fdtdx.core.misc import PaddingConfig, advanced_padding
from fdtdx.objects.device.parameters.transform import SameShapeTypeParameterTransform
from fdtdx.typing import ParameterType


@autoinit
class StandardToInversePermittivityRange(SameShapeTypeParameterTransform):
    """Maps standard [0,1] range to inverse permittivity range.

    Linearly maps values from [0,1] to the range between minimum and maximum
    inverse permittivity values allowed by the material configuration.
    """

    _fixed_input_type: ParameterType | Sequence[ParameterType] | None = frozen_private_field(
        default=ParameterType.CONTINUOUS
    )

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
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
        result = {}
        for k, v in params.items():
            mapped = v * (max_inv_perm - min_inv_perm) + min_inv_perm
            result[k] = mapped
        return result


@autoinit
class StandardToCustomRange(SameShapeTypeParameterTransform):
    """Maps standard [0,1] range to custom range [min_value, max_value].

    Linearly maps values from [0,1] to a custom range specified by min_value
    and max_value parameters.

    Attributes:
        min_value (float, optional): Minimum value of target range. Defaults to zero.
        max_value (float, optional): Maximum value of target range. Defaults to one.
    """

    min_value: float = frozen_field(default=0)
    max_value: float = frozen_field(default=1)
    _fixed_input_type: ParameterType | Sequence[ParameterType] | None = frozen_private_field(
        default=ParameterType.CONTINUOUS
    )

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs
        result = {}
        for k, v in params.items():
            mapped = v * (self.max_value - self.min_value) + self.min_value
            result[k] = mapped
        return result


@autoinit
class StandardToPlusOneMinusOneRange(StandardToCustomRange):
    """Maps standard [0,1] range to [-1,1] range.

    Special case of StandardToCustomRange that maps to [-1,1] range.
    Used for symmetric value ranges around zero.
    """

    min_value: float = frozen_private_field(default=-1)
    max_value: float = frozen_private_field(default=1)


@autoinit
class GaussianSmoothing2D(SameShapeTypeParameterTransform):
    """
    Applies Gaussian smoothing to 2D parameter arrays.

    This transform convolves the input with a 2D Gaussian kernel,
    which helps reduce noise and smooth the data.

    Attributes:
        std_discrete (int): Integer specifying the standard deviation of the
                     Gaussian kernel in discrete units.
    """

    std_discrete: int = frozen_field()

    _fixed_input_type: ParameterType | Sequence[ParameterType] | None = frozen_private_field(
        default=ParameterType.CONTINUOUS
    )
    _all_arrays_2d: bool = frozen_private_field(default=True)

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs
        return {k: self._apply_smoothing(v) for k, v in params.items()}

    def _apply_smoothing(self, x: jax.Array) -> jax.Array:
        vertical_axis = x.shape.index(1)
        x_squeezed = x.squeeze(vertical_axis)
        # Check if the array is 2D
        if x_squeezed.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {x_squeezed.shape}")

        # Create Gaussian kernel
        kernel_size = 6 * self.std_discrete + 1  # Ensure kernel covers 3 std on each side
        kernel = self._create_gaussian_kernel(kernel_size, self.std_discrete)

        # pad array with edge values
        padding_cfg = PaddingConfig(widths=(kernel_size // 2,), modes=("edge",))
        padded_arr, orig_slice = advanced_padding(x_squeezed, padding_cfg)

        result = jax.scipy.signal.convolve(
            padded_arr,
            kernel,
            mode="same",
        )
        result = result[*orig_slice]

        # Reshape back to original dimensions
        return result.reshape(x.shape)

    def _create_gaussian_kernel(self, size: int, sigma: float) -> jax.Array:
        # Create a coordinate grid
        coords = jnp.arange(-(size // 2), size // 2 + 1)
        x, y = jnp.meshgrid(coords, coords)

        # Create the Gaussian kernel
        kernel = jnp.exp(-(x**2 + y**2) / (2 * sigma**2))

        # Normalize the kernel to sum to 1
        kernel = kernel / jnp.sum(kernel)

        return kernel
