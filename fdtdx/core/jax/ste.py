import jax

def straight_through_estimator(x: jax.Array, y: jax.Array):
    """
    Straight Through Estimator for gradient estimation with discrete variables.

    This function applies the straight through estimator by taking the gradient
    with respect to the continuous input `x`, while using the discrete values `y`
    in the forward pass.

    Args:
        x: jax.Array, the original continuous values.
        y: jax.Array, the discrete values obtained after quantization.

    Returns:
        jax.Array: The result of applying the straight through estimator, which
        is the same shape as `x` and `y`.
    """

    return x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(y)