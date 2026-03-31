"""Tests for objects/device/parameters/projection.py - projection transformations."""

import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.materials import Material
from fdtdx.objects.device.parameters.projection import (
    SubpixelSmoothedProjection,
    TanhProjection,
    smoothed_projection,
    tanh_projection,
)


class TestTanhProjection:
    """Tests for tanh_projection function."""

    def test_beta_zero_clips_to_01(self):
        """Test that beta=0 clips values to [0, 1]."""
        x = jnp.array([-0.5, 0.0, 0.5, 1.0, 1.5])

        result = tanh_projection(x, beta=0.0, eta=0.5)

        # Should clip to [0, 1]
        assert jnp.all(result >= 0)
        assert jnp.all(result <= 1)
        assert jnp.allclose(result[0], 0.0)  # -0.5 clipped to 0
        assert jnp.allclose(result[-1], 1.0)  # 1.5 clipped to 1

    def test_beta_inf_step_function(self):
        """Test that beta=inf produces step function."""
        x = jnp.array([0.0, 0.4, 0.5, 0.6, 1.0])
        eta = 0.5

        result = tanh_projection(x, beta=jnp.inf, eta=eta)

        # Should be step function at eta
        assert jnp.allclose(result[0], 0.0)  # Below eta
        assert jnp.allclose(result[1], 0.0)  # Below eta
        assert jnp.allclose(result[3], 1.0)  # Above eta
        assert jnp.allclose(result[4], 1.0)  # Above eta

    def test_finite_beta_smooth_transition(self):
        """Test that finite beta produces smooth transition."""
        x = jnp.linspace(0, 1, 11)

        result = tanh_projection(x, beta=2.0, eta=0.5)

        # Should have values in [0, 1]
        assert jnp.all(result >= 0)
        assert jnp.all(result <= 1)
        # Should be monotonically increasing
        assert jnp.all(jnp.diff(result) >= 0)

    def test_eta_shifts_threshold(self):
        """Test that eta shifts the threshold point."""
        x = jnp.array([0.2, 0.3, 0.4])

        result_eta_low = tanh_projection(x, beta=10.0, eta=0.25)
        result_eta_high = tanh_projection(x, beta=10.0, eta=0.35)

        # With eta=0.25, more values should be > 0.5
        # With eta=0.35, fewer values should be > 0.5
        assert jnp.sum(result_eta_low > 0.5) >= jnp.sum(result_eta_high > 0.5)

    def test_2d_input(self):
        """Test with 2D input array."""
        x = jnp.array([[0.2, 0.8], [0.4, 0.6]])

        result = tanh_projection(x, beta=5.0, eta=0.5)

        assert result.shape == (2, 2)
        assert jnp.all(result >= 0)
        assert jnp.all(result <= 1)

    def test_higher_beta_sharper_transition(self):
        """Test that higher beta produces sharper transition."""
        x = jnp.linspace(0, 1, 101)

        result_low_beta = tanh_projection(x, beta=1.0, eta=0.5)
        result_high_beta = tanh_projection(x, beta=10.0, eta=0.5)

        # Higher beta should have more values close to 0 or 1
        low_beta_extreme = jnp.sum((result_low_beta < 0.1) | (result_low_beta > 0.9))
        high_beta_extreme = jnp.sum((result_high_beta < 0.1) | (result_high_beta > 0.9))

        assert high_beta_extreme > low_beta_extreme

    def test_gradient_beta_inf_no_nan(self):
        """Gradient of tanh_projection at beta=inf must not produce NaN.

        On GPU, XLA evaluates ALL branches of a lax.switch for SIMD execution, so
        the tanh formula branch runs with beta=inf, computing tanh(inf*(x-eta)) which
        is NaN when x==eta (0*inf in IEEE). The fix uses jnp.where + safe_beta so
        the non-selected branch never evaluates with inf or 0 as beta.

        Since the fix switches to jnp.where (which always evaluates both branches
        during AD, same as GPU), this test exercises that code path on all platforms.
        x=0.5 is exactly at eta — the pathological input.
        """
        import jax

        x = jnp.array([0.3, 0.5, 0.7])
        grad = jax.grad(lambda v: tanh_projection(v, beta=jnp.inf, eta=0.5).sum())(x)
        assert not jnp.any(jnp.isnan(grad)), f"NaN in gradient: {grad}"

    def test_gradient_beta_zero_no_nan(self):
        """Gradient of tanh_projection at beta=0 must not produce NaN.

        Without safe_beta, the tanh formula evaluates 0/0 = NaN when beta=0.
        The fix substitutes safe_beta=1 for the non-selected branch.
        """
        import jax

        x = jnp.array([0.3, 0.5, 0.7])
        grad = jax.grad(lambda v: tanh_projection(v, beta=0.0, eta=0.5).sum())(x)
        assert not jnp.any(jnp.isnan(grad)), f"NaN in gradient: {grad}"

    def test_gradient_smoothed_projection_beta_inf_no_nan(self):
        """Gradient of smoothed_projection at beta=inf must not produce NaN (end-to-end)."""
        import jax

        rho = jnp.linspace(0, 1, 100).reshape(10, 10)
        grad = jax.grad(lambda v: smoothed_projection(v, beta=jnp.inf, eta=0.5, resolution=10.0).sum())(rho)
        assert not jnp.any(jnp.isnan(grad)), f"NaN in gradient: {grad}"


class TestSmoothedProjection:
    """Tests for smoothed_projection function."""

    def test_basic_2d_input(self):
        """Test basic 2D input processing."""
        rho = jnp.ones((10, 10)) * 0.5

        result = smoothed_projection(rho, beta=5.0, eta=0.5, resolution=10.0)

        assert result.shape == (10, 10)
        assert jnp.all(result >= 0)
        assert jnp.all(result <= 1)

    def test_gradient_field(self):
        """Test with gradient field (has interfaces)."""
        # Create a gradient field from 0 to 1
        x = jnp.linspace(0, 1, 20)
        y = jnp.linspace(0, 1, 20)
        X, Y = jnp.meshgrid(x, y)
        rho = X  # Gradient along x

        result = smoothed_projection(rho, beta=10.0, eta=0.5, resolution=20.0)

        assert result.shape == (20, 20)
        # Result should still be in [0, 1]
        assert jnp.all(result >= 0)
        assert jnp.all(result <= 1)

    def test_uniform_field(self):
        """Test with uniform field (no interfaces)."""
        rho = jnp.ones((10, 10)) * 0.3

        result = smoothed_projection(rho, beta=10.0, eta=0.5, resolution=10.0)

        # Uniform field below eta should project to ~0
        assert jnp.mean(result) < 0.5

    def test_beta_inf(self):
        """Test with beta=inf."""
        rho = jnp.ones((10, 10)) * 0.6

        result = smoothed_projection(rho, beta=jnp.inf, eta=0.5, resolution=10.0)

        # Should be close to 1 (since rho > eta)
        assert jnp.mean(result) > 0.5

    def test_resolution_effect(self):
        """Test that resolution affects smoothing."""
        x = jnp.linspace(0, 1, 20)
        y = jnp.linspace(0, 1, 20)
        X, _ = jnp.meshgrid(x, y)
        rho = X

        result_low_res = smoothed_projection(rho, beta=10.0, eta=0.5, resolution=5.0)
        result_high_res = smoothed_projection(rho, beta=10.0, eta=0.5, resolution=50.0)

        # Both should produce valid output
        assert result_low_res.shape == (20, 20)
        assert result_high_res.shape == (20, 20)


class TestTanhProjectionClass:
    """Tests for TanhProjection class."""

    @pytest.fixture
    def two_materials(self):
        """Two materials fixture."""
        return {
            "Air": Material(permittivity=1.0),
            "Silicon": Material(permittivity=11.7),
        }

    @pytest.fixture
    def dummy_config(self):
        """Minimal simulation config."""
        return SimulationConfig(
            time=100e-15,
            resolution=500e-9,
            backend="cpu",
        )

    def test_call_with_beta(self, two_materials, dummy_config):
        """Test __call__ with beta parameter."""
        transform = TanhProjection(projection_midpoint=0.5)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )

        params = {"params": jnp.ones((4, 4, 4)) * 0.5}

        result = transform(params, beta=5.0)

        assert "params" in result
        assert result["params"].shape == (4, 4, 4)

    def test_call_missing_beta_raises_error(self, two_materials, dummy_config):
        """Test that missing beta raises error."""
        transform = TanhProjection(projection_midpoint=0.5)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )

        params = {"params": jnp.ones((4, 4, 4)) * 0.5}

        with pytest.raises(Exception, match="beta parameter"):
            transform(params)  # Missing beta

    def test_custom_projection_midpoint(self, two_materials, dummy_config):
        """Test with custom projection midpoint."""
        transform = TanhProjection(projection_midpoint=0.3)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )

        # Values around 0.3 should be near the midpoint
        params = {"params": jnp.ones((4, 4, 4)) * 0.3}

        result = transform(params, beta=5.0)

        assert result["params"].shape == (4, 4, 4)

    def test_multiple_keys(self, two_materials, dummy_config):
        """Test with multiple parameter keys."""
        transform = TanhProjection(projection_midpoint=0.5)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params1": (4, 4, 4), "params2": (4, 4, 4)},
        )

        params = {
            "params1": jnp.ones((4, 4, 4)) * 0.3,
            "params2": jnp.ones((4, 4, 4)) * 0.7,
        }

        result = transform(params, beta=5.0)

        assert "params1" in result
        assert "params2" in result


class TestSubpixelSmoothedProjectionClass:
    """Tests for SubpixelSmoothedProjection class."""

    @pytest.fixture
    def two_materials(self):
        """Two materials fixture."""
        return {
            "Air": Material(permittivity=1.0),
            "Silicon": Material(permittivity=11.7),
        }

    @pytest.fixture
    def dummy_config(self):
        """Minimal simulation config."""
        return SimulationConfig(
            time=100e-15,
            resolution=500e-9,
            backend="cpu",
        )

    def test_call_2d_array(self, two_materials, dummy_config):
        """Test __call__ with 2D array (one axis has size 1)."""
        transform = SubpixelSmoothedProjection(projection_midpoint=0.5)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(8, 8, 1),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (8, 8, 1)},
        )

        params = {"params": jnp.ones((8, 8, 1)) * 0.5}

        result = transform(params, beta=5.0)

        assert "params" in result
        assert result["params"].shape == (8, 8, 1)

    def test_call_missing_beta_raises_error(self, two_materials, dummy_config):
        """Test that missing beta raises error."""
        transform = SubpixelSmoothedProjection(projection_midpoint=0.5)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(8, 8, 1),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (8, 8, 1)},
        )

        params = {"params": jnp.ones((8, 8, 1)) * 0.5}

        with pytest.raises(Exception, match="beta parameter"):
            transform(params)  # Missing beta

    def test_requires_equal_voxel_sizes(self, two_materials, dummy_config):
        """Test that unequal voxel sizes raise error."""
        transform = SubpixelSmoothedProjection(projection_midpoint=0.5)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(8, 8, 1),
            single_voxel_size=(1e-6, 2e-6, 1e-6),  # Different x and y sizes
            output_shape={"params": (8, 8, 1)},
        )

        params = {"params": jnp.ones((8, 8, 1)) * 0.5}

        with pytest.raises(Exception, match="voxel size to be equal"):
            transform(params, beta=5.0)

    def test_different_vertical_axis(self, two_materials, dummy_config):
        """Test with vertical axis in different position."""
        transform = SubpixelSmoothedProjection(projection_midpoint=0.5)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(1, 8, 8),  # Vertical axis at position 0
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (1, 8, 8)},
        )

        params = {"params": jnp.ones((1, 8, 8)) * 0.5}

        result = transform(params, beta=5.0)

        assert result["params"].shape == (1, 8, 8)
