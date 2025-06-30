import jax


def straight_through_estimator(x: jax.Array, y: jax.Array) -> jax.Array:
    """Straight Through Estimator for gradient estimation with discrete variables.

    This function applies the straight through estimator (STE) by taking the gradient
    with respect to the continuous input `x`, while using the discrete values `y`
    in the forward pass. STE is commonly used in training neural networks with
    discrete/quantized values where standard backpropagation is not possible.

    The implementation uses JAX's stop_gradient to control gradient flow:
        output = x - stop_gradient(x) + stop_gradient(y)

    This ensures during the forward pass we use y, but during backprop the
    gradient flows through x.

    Args:
        x (jax.Array): jax.Array, the original continuous values before quantization/discretization.
            Gradients will be computed with respect to these values.
        y (jax.Array): jax.Array, the discrete/quantized values used in the forward pass.
            Must have the same shape as x.

    Returns:
        jax.Array: The result of applying the straight through estimator, which
        is the same shape as `x` and `y`. In the forward pass this equals y,
        but gradients flow through x.
    """

    return x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(y)
