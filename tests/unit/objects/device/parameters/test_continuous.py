"""Tests for continuous parameter transformations."""

import jax.numpy as jnp
import pytest

from fdtdx.materials import Material
from fdtdx.objects.device.parameters.continuous import (
    GaussianSmoothing2D,
    StandardToCustomRange,
    StandardToInversePermittivityRange,
    StandardToPlusOneMinusOneRange,
)


class TestStandardToInversePermittivityRange:
    """Test StandardToInversePermittivityRange transformation."""

    def test_isotropic_materials(self):
        """Test transformation with isotropic materials."""
        # Create isotropic materials with different permittivities
        materials = {
            "air": Material(permittivity=1.0),
            "glass": Material(permittivity=2.25),  # eps=2.25, inv_eps=0.444
            "silicon": Material(permittivity=12.25),  # eps=12.25, inv_eps=0.0816
        }

        transform = StandardToInversePermittivityRange()
        transform = transform.aset("_materials", materials)

        # Test with a parameter value of 0.5
        params = {"test_param": jnp.array([[[0.5]]])}
        result = transform(params)

        # For isotropic case: min_inv_perm = 0.0816, max_inv_perm = 1.0
        # mapped = 0.5 * (1.0 - 0.0816) + 0.0816 = 0.5408
        expected = 0.5 * (1.0 - 1 / 12.25) + 1 / 12.25
        assert jnp.allclose(result["test_param"], expected)

    def test_isotropic_materials_boundary_values(self):
        """Test boundary values (0 and 1) with isotropic materials."""
        materials = {
            "air": Material(permittivity=1.0),
            "glass": Material(permittivity=4.0),
        }

        transform = StandardToInversePermittivityRange()
        transform = transform.aset("_materials", materials)

        # Test with value 0 -> should map to min_inv_perm
        params_0 = {"test": jnp.array([[[0.0]]])}
        result_0 = transform(params_0)
        assert jnp.allclose(result_0["test"], 1 / 4.0)  # min inv_perm

        # Test with value 1 -> should map to max_inv_perm
        params_1 = {"test": jnp.array([[[1.0]]])}
        result_1 = transform(params_1)
        assert jnp.allclose(result_1["test"], 1.0)  # max inv_perm

    def test_anisotropic_materials(self):
        """Test transformation with anisotropic materials."""
        # Create anisotropic materials
        materials = {
            "iso": Material(permittivity=1.0),
            "aniso": Material(permittivity=(4.0, 9.0, 16.0)),
        }

        transform = StandardToInversePermittivityRange()
        transform = transform.aset("_materials", materials)

        # Test with a 3D spatial parameter
        params = {"test_param": jnp.ones((2, 3, 4)) * 0.5}
        result = transform(params)

        # Output should have shape (3, 2, 3, 4) for anisotropic
        assert result["test_param"].shape == (3, 2, 3, 4)

        # For axis 0: min=1/4, max=1 -> 0.5*(1-0.25)+0.25 = 0.625
        # For axis 1: min=1/9, max=1 -> 0.5*(1-1/9)+1/9 = 0.556
        # For axis 2: min=1/16, max=1 -> 0.5*(1-1/16)+1/16 = 0.531
        assert jnp.allclose(result["test_param"][0], 0.5 * (1.0 - 0.25) + 0.25)
        assert jnp.allclose(result["test_param"][1], 0.5 * (1.0 - 1 / 9) + 1 / 9)
        assert jnp.allclose(result["test_param"][2], 0.5 * (1.0 - 1 / 16) + 1 / 16)

    def test_anisotropic_materials_multiple_params(self):
        """Test transformation with multiple parameters and anisotropic materials."""
        materials = {
            "mat1": Material(permittivity=(2.0, 3.0, 4.0)),
            "mat2": Material(permittivity=(1.0, 1.5, 2.0)),
        }

        transform = StandardToInversePermittivityRange()
        transform = transform.aset("_materials", materials)

        params = {
            "param1": jnp.zeros((3, 3, 3)),
            "param2": jnp.ones((3, 3, 3)),
        }
        result = transform(params)

        # Both should be transformed
        assert "param1" in result
        assert "param2" in result
        assert result["param1"].shape == (3, 3, 3, 3)
        assert result["param2"].shape == (3, 3, 3, 3)

        # param1 (value=0) should map to min inv_perm for each axis
        # param2 (value=1) should map to max inv_perm for each axis
        # Axis 0: min=1/2 (from mat1), max=1 (from mat2)
        assert jnp.allclose(result["param1"][0], 1 / 2.0)
        assert jnp.allclose(result["param2"][0], 1.0)

        # Axis 1: min=1/3 (from mat1), max=2/3 (from mat2)
        assert jnp.allclose(result["param1"][1], 1 / 3.0)
        assert jnp.allclose(result["param2"][1], 2 / 3.0)

        # Axis 2: min=1/4 (from mat1), max=1/2 (from mat2)
        assert jnp.allclose(result["param1"][2], 1 / 4.0)
        assert jnp.allclose(result["param2"][2], 1 / 2.0)

    def test_get_input_shape_impl(self):
        """Test _get_input_shape_impl returns output shape."""
        transform = StandardToInversePermittivityRange()
        output_shape = {"param": (3, 10, 10, 10)}
        input_shape = transform._get_input_shape_impl(output_shape)
        assert input_shape == output_shape

    def test_get_output_type_impl(self):
        """Test _get_output_type_impl returns same type as input."""
        from fdtdx.typing import ParameterType

        transform = StandardToInversePermittivityRange()
        input_type = {"param": ParameterType.CONTINUOUS}
        output_type = transform._get_output_type_impl(input_type)
        assert output_type == input_type


class TestStandardToCustomRange:
    """Test StandardToCustomRange transformation."""

    def test_default_range(self):
        """Test with default range [0, 1]."""
        transform = StandardToCustomRange()
        params = {"test": jnp.array([0.0, 0.5, 1.0])}
        result = transform(params)

        # Default range is [0, 1], so values should be unchanged
        assert jnp.allclose(result["test"], params["test"])

    def test_custom_range(self):
        """Test with custom range [2, 5]."""
        transform = StandardToCustomRange(min_value=2.0, max_value=5.0)
        params = {"test": jnp.array([0.0, 0.5, 1.0])}
        result = transform(params)

        # 0 -> 2, 0.5 -> 3.5, 1 -> 5
        expected = jnp.array([2.0, 3.5, 5.0])
        assert jnp.allclose(result["test"], expected)

    def test_negative_range(self):
        """Test with range including negative values [-10, -5]."""
        transform = StandardToCustomRange(min_value=-10.0, max_value=-5.0)
        params = {"test": jnp.array([0.0, 0.25, 0.5, 1.0])}
        result = transform(params)

        # 0 -> -10, 0.25 -> -8.75, 0.5 -> -7.5, 1 -> -5
        expected = jnp.array([-10.0, -8.75, -7.5, -5.0])
        assert jnp.allclose(result["test"], expected)

    def test_multidimensional_array(self):
        """Test with multidimensional arrays."""
        transform = StandardToCustomRange(min_value=0.0, max_value=10.0)
        params = {"test": jnp.ones((3, 4, 5)) * 0.3}
        result = transform(params)

        # 0.3 * 10 + 0 = 3.0
        assert result["test"].shape == (3, 4, 5)
        assert jnp.allclose(result["test"], 3.0)

    def test_multiple_parameters(self):
        """Test with multiple parameters."""
        transform = StandardToCustomRange(min_value=1.0, max_value=2.0)
        params = {
            "param1": jnp.array([0.0]),
            "param2": jnp.array([1.0]),
        }
        result = transform(params)

        assert jnp.allclose(result["param1"], 1.0)
        assert jnp.allclose(result["param2"], 2.0)


class TestStandardToPlusOneMinusOneRange:
    """Test StandardToPlusOneMinusOneRange transformation."""

    def test_default_range(self):
        """Test that range is [-1, 1]."""
        transform = StandardToPlusOneMinusOneRange()
        params = {"test": jnp.array([0.0, 0.5, 1.0])}
        result = transform(params)

        # 0 -> -1, 0.5 -> 0, 1 -> 1
        expected = jnp.array([-1.0, 0.0, 1.0])
        assert jnp.allclose(result["test"], expected)

    def test_symmetric_values(self):
        """Test that transformation creates symmetric values."""
        transform = StandardToPlusOneMinusOneRange()
        params = {"test": jnp.array([0.25, 0.75])}
        result = transform(params)

        # 0.25 -> -0.5, 0.75 -> 0.5
        assert jnp.allclose(result["test"], jnp.array([-0.5, 0.5]))

    def test_multidimensional(self):
        """Test with multidimensional arrays."""
        transform = StandardToPlusOneMinusOneRange()
        params = {"test": jnp.ones((2, 3)) * 0.5}
        result = transform(params)

        assert result["test"].shape == (2, 3)
        assert jnp.allclose(result["test"], 0.0)


class TestGaussianSmoothing2D:
    """Test GaussianSmoothing2D transformation."""

    def test_basic_smoothing(self):
        """Test basic Gaussian smoothing operation."""
        transform = GaussianSmoothing2D(std_discrete=1)

        # Create a simple 2D array with a spike in the middle
        # Shape must be (nx, 1, ny) or similar to be squeezable to 2D
        arr = jnp.zeros((5, 1, 5))
        arr = arr.at[2, 0, 2].set(1.0)

        params = {"test": arr}
        result = transform(params)

        # After smoothing, the spike should be spread out
        assert result["test"].shape == arr.shape
        # Center should still be highest but lower than 1
        assert result["test"][2, 0, 2] < 1.0
        # Neighbors should have non-zero values
        assert result["test"][1, 0, 2] > 0.0
        assert result["test"][2, 0, 1] > 0.0

    def test_smoothing_preserves_sum_approximately(self):
        """Test that smoothing approximately preserves the sum."""
        transform = GaussianSmoothing2D(std_discrete=2)

        arr = jnp.zeros((10, 1, 10))
        arr = arr.at[5, 0, 5].set(10.0)

        params = {"test": arr}
        result = transform(params)

        # Sum should be approximately preserved (Gaussian kernel is normalized)
        original_sum = jnp.sum(arr)
        smoothed_sum = jnp.sum(result["test"])
        assert jnp.isclose(original_sum, smoothed_sum, rtol=0.05)

    def test_different_std_values(self):
        """Test with different standard deviation values."""
        arr = jnp.zeros((10, 1, 10))
        arr = arr.at[5, 0, 5].set(1.0)

        transform_small = GaussianSmoothing2D(std_discrete=1)
        transform_large = GaussianSmoothing2D(std_discrete=3)

        result_small = transform_small({"test": arr})
        result_large = transform_large({"test": arr})

        # Larger std should spread the values more (lower peak, more spread)
        assert result_large["test"][5, 0, 5] < result_small["test"][5, 0, 5]
        # Check that far-away point has more contribution with larger std
        assert result_large["test"][8, 0, 8] > result_small["test"][8, 0, 8]

    def test_invalid_shape_raises_error(self):
        """Test that non-2D squeezed arrays raise ValueError."""
        transform = GaussianSmoothing2D(std_discrete=1)

        # 3D array that can't be squeezed to 2D (no axis with size 1)
        arr_3d = jnp.ones((3, 3, 3))
        params = {"test": arr_3d}

        # Should raise ValueError when trying to find vertical axis or squeeze
        with pytest.raises(ValueError):
            transform(params)

    def test_multiple_parameters(self):
        """Test smoothing multiple parameters."""
        transform = GaussianSmoothing2D(std_discrete=1)

        params = {
            "param1": jnp.ones((5, 1, 5)),
            "param2": jnp.ones((7, 1, 7)),
        }

        result = transform(params)

        assert "param1" in result
        assert "param2" in result
        assert result["param1"].shape == (5, 1, 5)
        assert result["param2"].shape == (7, 1, 7)

    def test_edge_padding(self):
        """Test that edge padding works correctly."""
        transform = GaussianSmoothing2D(std_discrete=1)

        # Create array with edge values
        arr = jnp.zeros((5, 1, 5))
        arr = arr.at[0, 0, 0].set(1.0)  # Corner value

        params = {"test": arr}
        result = transform(params)

        # Edge padding should prevent artifacts at boundaries
        assert jnp.all(jnp.isfinite(result["test"]))
        assert result["test"].shape == arr.shape

    def test_kernel_size_calculation(self):
        """Test that kernel size is calculated correctly."""
        transform = GaussianSmoothing2D(std_discrete=3)

        # Kernel size should be 6 * std_discrete + 1 = 19
        kernel = transform._create_gaussian_kernel(19, 3.0)

        # Kernel should be square
        assert kernel.shape[0] == kernel.shape[1]
        # Kernel should sum to approximately 1
        assert jnp.isclose(jnp.sum(kernel), 1.0)
        # Kernel should be symmetric
        assert jnp.allclose(kernel, kernel.T)

    def test_uniform_input_stays_uniform(self):
        """Test that smoothing a uniform field keeps it uniform."""
        transform = GaussianSmoothing2D(std_discrete=2)

        arr = jnp.ones((8, 1, 8)) * 5.0
        params = {"test": arr}
        result = transform(params)

        # Uniform input should remain approximately uniform
        assert jnp.allclose(result["test"], 5.0, rtol=0.01)