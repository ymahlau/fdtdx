from typing import Tuple

import chex
import jax
import jax.numpy as jnp


def _depthwise_conv3d(
    inputs: jax.Array,
    kernel: jax.Array,
    *,
    strides: Tuple[int, int, int],
    padding: str,
    channel_axis: int,
    dimension_numbers: Tuple[str, str, str],
) -> jax.Array:
    """Computes a depthwise conv3d in Jax.

    Args:
      inputs: an NDHWC or NCDHW tensor (depending on dimension_numbers), with N=1.
      kernel: a [D', H', W', 1, C] tensor.
      strides: optional stride for the kernel.
      padding: "SAME" or "VALID".
      channel_axis: the index of the channel axis.
      dimension_numbers: see jax.lax.conv_general_dilated.

    Returns:
      The depthwise convolution of inputs with kernel, with the same
      dimension_numbers as the input.
    """
    return jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=kernel,
        window_strides=strides,
        padding=padding,
        feature_group_count=inputs.shape[channel_axis],
        dimension_numbers=dimension_numbers,
    )


def _channels_last(image: jax.Array, channel_axis: int) -> bool:
    """Determines if channels are in the last dimension of the image tensor.

    Args:
      image: the input image tensor.
      channel_axis: the index of the channel axis.

    Returns:
      True if channels are in the last dimension, False otherwise.
    """
    return channel_axis == -1 or channel_axis == image.ndim - 1


def gaussian_blur_3d(
    image: jax.Array,
    sigma: float,
    kernel_size: float,
    *,
    padding: str = "SAME",
    channel_axis: int = -1,
) -> jax.Array:
    """Applies gaussian blur in 3D (convolution with a Gaussian kernel).

    Args:
      image: the input image, as a [0-1] float tensor. Should have 3 or 4
        dimensions with three spatial dimensions.
      sigma: the standard deviation (in pixels) of the gaussian kernel.
      kernel_size: the size (in pixels) of the cubic gaussian kernel. Will be
        "rounded" to the next odd integer.
      padding: either "SAME" or "VALID", passed to the underlying convolution.
      channel_axis: the index of the channel axis.

    Returns:
      The blurred image.
    """
    chex.assert_rank(image, {4, 5})
    data_format = "NDHWC" if _channels_last(image, channel_axis) else "NCDHW"
    dimension_numbers = (data_format, "DHWIO", data_format)
    num_channels = image.shape[channel_axis]
    radius = int(kernel_size / 2)
    kernel_size_ = 2 * radius + 1
    x = jnp.arange(-radius, radius + 1).astype(image.dtype)
    blur_filter = jnp.exp(-(x**2) / (2.0 * sigma**2))
    blur_filter = blur_filter / jnp.sum(blur_filter)
    blur_d = jnp.reshape(blur_filter, [kernel_size_, 1, 1, 1, 1])
    blur_h = jnp.reshape(blur_filter, [1, kernel_size_, 1, 1, 1])
    blur_w = jnp.reshape(blur_filter, [1, 1, kernel_size_, 1, 1])
    blur_h = jnp.tile(blur_h, [1, 1, 1, 1, num_channels])
    blur_w = jnp.tile(blur_w, [1, 1, 1, 1, num_channels])
    blur_d = jnp.tile(blur_d, [1, 1, 1, 1, num_channels])

    expand_batch_dim = image.ndim == 4
    if expand_batch_dim:
        image = image[jnp.newaxis, ...]
    blurred = _depthwise_conv3d(
        image,
        kernel=blur_h,
        strides=(1, 1, 1),
        padding=padding,
        channel_axis=channel_axis,
        dimension_numbers=dimension_numbers,
    )
    blurred = _depthwise_conv3d(
        blurred,
        kernel=blur_w,
        strides=(1, 1, 1),
        padding=padding,
        channel_axis=channel_axis,
        dimension_numbers=dimension_numbers,
    )
    blurred = _depthwise_conv3d(
        blurred,
        kernel=blur_d,
        strides=(1, 1, 1),
        padding=padding,
        channel_axis=channel_axis,
        dimension_numbers=dimension_numbers,
    )
    if expand_batch_dim:
        blurred = jnp.squeeze(blurred, axis=0)
    return blurred
