import math
from typing import Sequence

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field, frozen_private_field
from fdtdx.core.misc import PaddingConfig, advanced_padding
from fdtdx.objects.device.parameters.transform import ParameterTransformation, SameShapeTypeParameterTransform
from fdtdx.typing import ParameterType


@autoinit
class StandardToInversePermittivityRange(ParameterTransformation):
    """Maps standard [0,1] range to inverse permittivity range.

    Linearly maps values from [0,1] to the range between minimum and maximum
    inverse permittivity values allowed by the material configuration.

    For anisotropic materials, each axis (x, y, z) is interpolated independently
    within its own min/max range, producing output with shape ``(3, *input_shape)``.
    """

    _fixed_input_type: ParameterType | Sequence[ParameterType] | None = frozen_private_field(
        default=ParameterType.CONTINUOUS
    )

    def _get_input_shape_impl(
        self,
        output_shape: dict[str, tuple[int, ...]],
    ) -> dict[str, tuple[int, ...]]:
        # Input shape is same as output for isotropic, or without leading 3 for anisotropic
        # Since we don't have access to materials yet at this point in some cases,
        # we return the output shape as-is (input param has spatial shape, output may have component dim)
        return output_shape

    def _get_output_type_impl(
        self,
        input_type: dict[str, ParameterType],
    ) -> dict[str, ParameterType]:
        # Output type is same as input (continuous -> continuous)
        return input_type

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs

        is_isotropic = all(mat.is_isotropic_permittivity for mat in self._materials.values())
        is_diagonally_anisotropic = all(mat.is_diagonally_anisotropic_permittivity for mat in self._materials.values())

        if is_isotropic:
            # Isotropic case: all materials have same permittivity on all axes
            max_inv_perm, min_inv_perm = -math.inf, math.inf
            for v in self._materials.values():
                # For isotropic, all components are equal, just use first
                p = 1 / v.permittivity[0]
                if p > max_inv_perm:
                    max_inv_perm = p
                if p < min_inv_perm:
                    min_inv_perm = p

            result = {}
            for k, v in params.items():
                mapped = v * (max_inv_perm - min_inv_perm) + min_inv_perm
                result[k] = mapped
            return result
        elif is_diagonally_anisotropic:
            # Compute min/max for each axis separately
            max_inv_perm = [-math.inf, -math.inf, -math.inf]
            min_inv_perm = [math.inf, math.inf, math.inf]
            for v in self._materials.values():
                # v.permittivity is tuple (εx, εy, εz)
                for axis in range(3):
                    p = 1 / v.permittivity[axis]
                    if p > max_inv_perm[axis]:
                        max_inv_perm[axis] = p
                    if p < min_inv_perm[axis]:
                        min_inv_perm[axis] = p

            max_inv_perm_arr = jnp.asarray(max_inv_perm)[:, None, None, None]
            min_inv_perm_arr = jnp.asarray(min_inv_perm)[:, None, None, None]

            # Transform: broadcast input to (3, ...) and interpolate each axis
            result = {}
            for k, v in params.items():
                # v has shape (Nx, Ny, Nz), expand to (3, Nx, Ny, Nz)
                v_expanded = v[None, ...]  # (1, Nx, Ny, Nz)
                mapped = v_expanded * (max_inv_perm_arr - min_inv_perm_arr) + min_inv_perm_arr
                result[k] = mapped
            return result
        else:  # fully anisotropic
            # Compute min/max for each tensor element separately
            max_inv_perm = [
                -math.inf,
                -math.inf,
                -math.inf,
                -math.inf,
                -math.inf,
                -math.inf,
                -math.inf,
                -math.inf,
                -math.inf,
            ]
            min_inv_perm = [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf]
            for v in self._materials.values():
                # v.permittivity is tuple (εxx, εxy, εxz, εyx, εyy, εyz, εzx, εzy, εzz)
                inv_perm = jnp.linalg.inv(jnp.array(v.permittivity).reshape(3, 3)).flatten()
                for i in range(9):
                    if inv_perm[i] > max_inv_perm[i]:
                        max_inv_perm[i] = inv_perm[i]
                    if inv_perm[i] < min_inv_perm[i]:
                        min_inv_perm[i] = inv_perm[i]

            max_inv_perm_arr = jnp.asarray(max_inv_perm)[:, None, None, None]
            min_inv_perm_arr = jnp.asarray(min_inv_perm)[:, None, None, None]

            # Transform: broadcast input to (9, ...) and interpolate each element
            result = {}
            for k, v in params.items():
                # v has shape (Nx, Ny, Nz), expand to (9, Nx, Ny, Nz)
                v_expanded = v[None, ...]  # (1, Nx, Ny, Nz)
                mapped = v_expanded * (max_inv_perm_arr - min_inv_perm_arr) + min_inv_perm_arr
                result[k] = mapped
            return result


@autoinit
class StandardToCustomRange(SameShapeTypeParameterTransform):
    """Maps standard [0,1] range to custom range [min_value, max_value].

    Linearly maps values from [0,1] to a custom range specified by min_value
    and max_value parameters.
    """

    #: Minimum value of target range. Defaults to zero.
    min_value: float = frozen_field(default=0)

    #: Maximum value of target range. Defaults to one.
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
    """

    #: Integer specifying the standard deviation of the Gaussian kernel in discrete units.
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
