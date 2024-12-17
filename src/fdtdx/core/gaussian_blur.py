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
    """Computes a depthwise 3D convolution using JAX's conv_general_dilated.

    A depthwise convolution performs the convolution operation separately for each input
    channel using a separate filter for each channel. This is more efficient than a full
    3D convolution when you want to process each channel independently.

    Args:
        inputs: A 5D array in NDHWC or NCDHW format (batch, depth, height, width, channels)
            or (batch, channels, depth, height, width). The batch dimension must be 1.
        kernel: A 5D array of shape [D', H', W', 1, C] containing the convolution kernels,
            where D',H',W' are the kernel dimensions and C matches the number of input channels.
        strides: A tuple of 3 integers specifying the stride of the convolution in each spatial
            dimension (depth, height, width).
        padding: Either "SAME" or "VALID" - the padding algorithm to use.
        channel_axis: Integer specifying which axis contains the channels (-1 for last axis).
        dimension_numbers: A tuple of 3 strings specifying the input, kernel and output data formats.
            See jax.lax.conv_general_dilated documentation for details.

    Returns:
        jax.Array: The result of the depthwise convolution, with same data format as input.
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

    This helper function checks whether the channel dimension is the last dimension
    in the tensor, either by being explicitly -1 or by being equal to the last dimension
    index (ndim - 1). This is useful for determining the data format of image tensors
    since some operations expect channels-last (NDHWC) format while others work with
    channels-first (NCDHW) format.

    Args:
        image: The input image tensor to check. Can be any rank tensor.
        channel_axis: The index of the channel axis in the tensor. Can be negative
            indexing (e.g. -1 for last dimension).

    Returns:
        bool: True if channels are in the last dimension (channels-last format),
            False otherwise (channels-first format).
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
    """Applies 3D Gaussian blur by convolving the image with separable Gaussian kernels.

    This function implements an efficient 3D Gaussian blur by decomposing the 3D Gaussian
    kernel into three 1D kernels and applying them sequentially along each axis. This is
    mathematically equivalent to a full 3D convolution but much more computationally efficient.

    Args:
        image: The input image as a float tensor with values in [0,1]. Should have 4 or 5
            dimensions: [batch?, depth, height, width, channels] or [batch?, channels, depth,
            height, width]. The batch dimension is optional.
        sigma: Standard deviation (in pixels) of the Gaussian kernel. Controls the amount
            of blurring - larger values produce more blurring.
        kernel_size: Size (in pixels) of the cubic Gaussian kernel. Will be rounded up to
            the nearest odd integer to ensure the kernel is symmetric. Should be at least
            2 * ceil(3 * sigma) + 1 to avoid truncating the Gaussian significantly.
        padding: Either "SAME" or "VALID". With "SAME" padding the output has the same
            spatial dimensions as the input. With "VALID" padding the output is smaller
            due to no padding being added.
        channel_axis: The axis containing the channels in the input tensor. Use -1 for
            channels-last format (default) or 1 for channels-first format.

    Returns:
        jax.Array: The blurred image tensor with the same shape and data format as the input.
            With "SAME" padding the spatial dimensions match the input. With "VALID" padding
            they are reduced by (kernel_size - 1) in each dimension.
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
