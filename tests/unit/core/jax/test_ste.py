"""Unit tests for fdtdx.core.jax.ste module."""

import jax
import jax.numpy as jnp

from fdtdx.core.jax.ste import straight_through_estimator


class TestStraightThroughEstimator:
    """Tests for straight_through_estimator."""

    # ---- Forward pass: output equals y ----

    def test_forward_scalar(self):
        """Forward value equals y for scalar inputs."""
        x = jnp.array(1.0)
        y = jnp.array(5.0)
        result = straight_through_estimator(x, y)
        assert jnp.allclose(result, y)

    def test_forward_1d(self):
        """Forward value equals y for 1-D arrays."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([10.0, 20.0, 30.0])
        result = straight_through_estimator(x, y)
        assert jnp.allclose(result, y)

    def test_forward_identical_inputs(self):
        """When x == y, output equals both."""
        x = jnp.array([2.0, 3.0])
        result = straight_through_estimator(x, x)
        assert jnp.allclose(result, x)

    # ---- Gradient flow: grad w.r.t. x, not y ----

    def test_gradient_wrt_x_is_identity(self):
        """Gradient of STE w.r.t. x is 1 (identity)."""
        x = jnp.array(3.0)
        y = jnp.array(7.0)

        def f(x_):
            return straight_through_estimator(x_, y)

        grad_val = jax.grad(f)(x)
        assert jnp.allclose(grad_val, 1.0)

    def test_gradient_wrt_y_is_zero(self):
        """Gradient of STE w.r.t. y is 0 (stop_gradient)."""
        x = jnp.array(3.0)
        y = jnp.array(7.0)

        def f(y_):
            return straight_through_estimator(x, y_)

        grad_val = jax.grad(f)(y)
        assert jnp.allclose(grad_val, 0.0)

    def test_gradient_weighted_sum(self):
        """Gradient flows correctly through a weighted sum."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([10.0, 20.0, 30.0])
        weights = jnp.array([2.0, 3.0, 4.0])

        def f(x_):
            return jnp.sum(weights * straight_through_estimator(x_, y))

        grad_val = jax.grad(f)(x)
        # d/dx [w * (x - sg(x) + sg(y))] = w * 1 = w
        assert jnp.allclose(grad_val, weights)

    # ---- Output shape / dtype ----

    def test_output_shape_preserved(self):
        """Output shape matches input shapes."""
        x = jnp.ones((2, 3, 4))
        y = jnp.zeros((2, 3, 4))
        result = straight_through_estimator(x, y)
        assert result.shape == (2, 3, 4)

    # ---- Integer discretization scenario ----

    def test_integer_discretization(self):
        """Typical STE use case: x is continuous, y is discretized version."""
        x = jnp.array([1.3, 2.7, 3.1])
        y = jnp.round(x)  # [1.0, 3.0, 3.0]
        result = straight_through_estimator(x, y)
        assert jnp.allclose(result, y)

        # But gradient still flows through x
        def f(x_):
            return jnp.sum(straight_through_estimator(x_, jnp.round(x_)))

        grad_val = jax.grad(f)(x)
        # round has zero gradient, so only the x - sg(x) part contributes = 1
        assert jnp.allclose(grad_val, jnp.ones_like(x))
