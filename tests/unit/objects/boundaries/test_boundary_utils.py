"""Tests for objects/boundaries/utils.py - boundary utility functions."""

import jax.numpy as jnp
import pytest

from fdtdx.objects.boundaries.utils import (
    alpha_from_direction_axis,
    axis_direction_from_kind,
    compute_extent,
    compute_extent_boundary,
    kappa_from_direction_axis,
    sigma_fn,
    standard_max_sigma_E_fn,
    standard_max_sigma_H_fn,
    standard_min_sigma_E_fn,
    standard_min_sigma_H_fn,
    standard_sigma_from_direction_axis,
)


class TestComputeExtent:
    """Tests for compute_extent function."""

    def test_min_x(self):
        """Test extent for min_x boundary."""
        slices = compute_extent("min_x", thickness=10)
        assert slices[0] == slice(None, 10)
        assert slices[1] == slice(None)
        assert slices[2] == slice(None)

    def test_max_x(self):
        """Test extent for max_x boundary."""
        slices = compute_extent("max_x", thickness=10)
        assert slices[0] == slice(-10, None)
        assert slices[1] == slice(None)
        assert slices[2] == slice(None)

    def test_min_y(self):
        """Test extent for min_y boundary."""
        slices = compute_extent("min_y", thickness=5)
        assert slices[0] == slice(None)
        assert slices[1] == slice(None, 5)
        assert slices[2] == slice(None)

    def test_max_y(self):
        """Test extent for max_y boundary."""
        slices = compute_extent("max_y", thickness=5)
        assert slices[0] == slice(None)
        assert slices[1] == slice(-5, None)
        assert slices[2] == slice(None)

    def test_min_z(self):
        """Test extent for min_z boundary."""
        slices = compute_extent("min_z", thickness=8)
        assert slices[0] == slice(None)
        assert slices[1] == slice(None)
        assert slices[2] == slice(None, 8)

    def test_max_z(self):
        """Test extent for max_z boundary."""
        slices = compute_extent("max_z", thickness=8)
        assert slices[0] == slice(None)
        assert slices[1] == slice(None)
        assert slices[2] == slice(-8, None)

    def test_invalid_kind_raises(self):
        """Test that invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="Invalid kind"):
            compute_extent("invalid", thickness=10)


class TestComputeExtentBoundary:
    """Tests for compute_extent_boundary function."""

    def test_min_x(self):
        """Test extent boundary for min_x."""
        slices = compute_extent_boundary("min_x", thickness=10)
        assert len(slices) == 3
        assert slices[0] == slice(11, 12)

    def test_max_x(self):
        """Test extent boundary for max_x."""
        slices = compute_extent_boundary("max_x", thickness=10)
        assert len(slices) == 3
        assert slices[0] == slice(-12, -11)

    def test_min_y(self):
        """Test extent boundary for min_y."""
        slices = compute_extent_boundary("min_y", thickness=5)
        assert len(slices) == 3
        assert slices[1] == slice(6, 7)

    def test_max_y(self):
        """Test extent boundary for max_y."""
        slices = compute_extent_boundary("max_y", thickness=5)
        assert len(slices) == 3
        assert slices[1] == slice(-7, -6)

    def test_min_z(self):
        """Test extent boundary for min_z."""
        slices = compute_extent_boundary("min_z", thickness=8)
        assert len(slices) == 3
        assert slices[2] == slice(9, 10)

    def test_max_z(self):
        """Test extent boundary for max_z."""
        slices = compute_extent_boundary("max_z", thickness=8)
        assert len(slices) == 3
        assert slices[2] == slice(-10, -9)

    def test_invalid_kind_raises(self):
        """Test that invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="Invalid kind"):
            compute_extent_boundary("invalid", thickness=10)


class TestSigmaFn:
    """Tests for sigma_fn function."""

    def test_returns_array(self):
        """Test that sigma_fn returns an array."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = sigma_fn(x, thickness=10)
        assert result.shape == x.shape

    def test_cubic_profile(self):
        """Test that profile is cubic."""
        x = jnp.array([0.0, 1.0, 2.0])
        result = sigma_fn(x, thickness=10)
        # At x=0, result should be 0
        assert jnp.isclose(result[0], 0.0)
        # Result should increase with x^3
        assert result[1] < result[2]

    def test_zero_at_origin(self):
        """Test that sigma is zero at origin."""
        x = jnp.array([0.0])
        result = sigma_fn(x, thickness=5)
        assert jnp.isclose(result[0], 0.0)


class TestStandardSigmaFunctions:
    """Tests for standard sigma profile functions."""

    def test_min_sigma_E_shape(self):
        """Test min_sigma_E shape."""
        thickness = 10
        result = standard_min_sigma_E_fn(thickness, dtype=jnp.float32)
        assert result.shape == (thickness,)

    def test_min_sigma_H_shape(self):
        """Test min_sigma_H shape."""
        thickness = 10
        result = standard_min_sigma_H_fn(thickness, dtype=jnp.float32)
        assert result.shape == (thickness - 1,)

    def test_max_sigma_E_shape(self):
        """Test max_sigma_E shape."""
        thickness = 10
        result = standard_max_sigma_E_fn(thickness, dtype=jnp.float32)
        assert result.shape == (thickness,)

    def test_max_sigma_H_shape(self):
        """Test max_sigma_H shape."""
        thickness = 10
        result = standard_max_sigma_H_fn(thickness, dtype=jnp.float32)
        assert result.shape == (thickness - 1,)

    def test_dtype_preserved(self):
        """Test that dtype is preserved."""
        # Use float32 as JAX x64 may not be enabled
        result = standard_min_sigma_E_fn(10, dtype=jnp.float32)
        assert result.dtype == jnp.float32


class TestStandardSigmaFromDirectionAxis:
    """Tests for standard_sigma_from_direction_axis function."""

    def test_minus_direction_axis_0(self):
        """Test minus direction on x axis."""
        sigma_E, sigma_H = standard_sigma_from_direction_axis(thickness=10, direction="-", axis=0, dtype=jnp.float32)
        assert sigma_E.shape == (3, 10, 1, 1)
        assert sigma_H.shape == (3, 10, 1, 1)

    def test_plus_direction_axis_0(self):
        """Test plus direction on x axis."""
        sigma_E, sigma_H = standard_sigma_from_direction_axis(thickness=10, direction="+", axis=0, dtype=jnp.float32)
        assert sigma_E.shape == (3, 10, 1, 1)
        assert sigma_H.shape == (3, 10, 1, 1)

    def test_minus_direction_axis_1(self):
        """Test minus direction on y axis."""
        sigma_E, sigma_H = standard_sigma_from_direction_axis(thickness=8, direction="-", axis=1, dtype=jnp.float32)
        assert sigma_E.shape == (3, 1, 8, 1)
        assert sigma_H.shape == (3, 1, 8, 1)

    def test_plus_direction_axis_1(self):
        """Test plus direction on y axis."""
        sigma_E, sigma_H = standard_sigma_from_direction_axis(thickness=8, direction="+", axis=1, dtype=jnp.float32)
        assert sigma_E.shape == (3, 1, 8, 1)
        assert sigma_H.shape == (3, 1, 8, 1)

    def test_minus_direction_axis_2(self):
        """Test minus direction on z axis."""
        sigma_E, sigma_H = standard_sigma_from_direction_axis(thickness=6, direction="-", axis=2, dtype=jnp.float32)
        assert sigma_E.shape == (3, 1, 1, 6)
        assert sigma_H.shape == (3, 1, 1, 6)

    def test_plus_direction_axis_2(self):
        """Test plus direction on z axis."""
        sigma_E, sigma_H = standard_sigma_from_direction_axis(thickness=6, direction="+", axis=2, dtype=jnp.float32)
        assert sigma_E.shape == (3, 1, 1, 6)
        assert sigma_H.shape == (3, 1, 1, 6)

    def test_invalid_direction_raises(self):
        """Test that invalid direction raises exception."""
        with pytest.raises(Exception, match="Invalid Direction"):
            standard_sigma_from_direction_axis(thickness=10, direction="x", axis=0, dtype=jnp.float32)

    def test_invalid_axis_raises(self):
        """Test that invalid axis raises exception."""
        with pytest.raises(Exception, match="Invalid axis"):
            standard_sigma_from_direction_axis(thickness=10, direction="+", axis=3, dtype=jnp.float32)


class TestKappaFromDirectionAxis:
    """Tests for kappa_from_direction_axis function."""

    def test_minus_direction_axis_0(self):
        """Test minus direction on x axis."""
        kappa = kappa_from_direction_axis(
            kappa_start=1.0, kappa_end=2.0, thickness=10, direction="-", axis=0, dtype=jnp.float32
        )
        assert kappa.shape == (1, 10, 1, 1)
        # For minus direction, should go from end to start
        assert jnp.isclose(kappa[0, 0, 0, 0], 2.0)
        assert jnp.isclose(kappa[0, -1, 0, 0], 1.0)

    def test_plus_direction_axis_0(self):
        """Test plus direction on x axis."""
        kappa = kappa_from_direction_axis(
            kappa_start=1.0, kappa_end=2.0, thickness=10, direction="+", axis=0, dtype=jnp.float32
        )
        assert kappa.shape == (1, 10, 1, 1)
        # For plus direction, should go from start to end
        assert jnp.isclose(kappa[0, 0, 0, 0], 1.0)
        assert jnp.isclose(kappa[0, -1, 0, 0], 2.0)

    def test_axis_1(self):
        """Test y axis."""
        kappa = kappa_from_direction_axis(
            kappa_start=1.0, kappa_end=2.0, thickness=8, direction="+", axis=1, dtype=jnp.float32
        )
        assert kappa.shape == (1, 1, 8, 1)

    def test_axis_2(self):
        """Test z axis."""
        kappa = kappa_from_direction_axis(
            kappa_start=1.0, kappa_end=2.0, thickness=6, direction="+", axis=2, dtype=jnp.float32
        )
        assert kappa.shape == (1, 1, 1, 6)

    def test_invalid_direction_raises(self):
        """Test that invalid direction raises exception."""
        with pytest.raises(Exception, match="Invalid direction"):
            kappa_from_direction_axis(
                kappa_start=1.0, kappa_end=2.0, thickness=10, direction="x", axis=0, dtype=jnp.float32
            )

    def test_invalid_axis_raises(self):
        """Test that invalid axis raises exception."""
        with pytest.raises(Exception, match="Invalid axis"):
            kappa_from_direction_axis(
                kappa_start=1.0, kappa_end=2.0, thickness=10, direction="+", axis=3, dtype=jnp.float32
            )


class TestAlphaFromDirectionAxis:
    """Tests for alpha_from_direction_axis function."""

    def test_minus_direction_axis_0(self):
        """Test minus direction on x axis."""
        alpha = alpha_from_direction_axis(
            alpha_start=0.0, alpha_end=1.0, thickness=10, direction="-", axis=0, dtype=jnp.float32
        )
        assert alpha.shape == (1, 10, 1, 1)

    def test_plus_direction_axis_0(self):
        """Test plus direction on x axis."""
        alpha = alpha_from_direction_axis(
            alpha_start=0.0, alpha_end=1.0, thickness=10, direction="+", axis=0, dtype=jnp.float32
        )
        assert alpha.shape == (1, 10, 1, 1)

    def test_axis_1(self):
        """Test y axis."""
        alpha = alpha_from_direction_axis(
            alpha_start=0.0, alpha_end=1.0, thickness=8, direction="+", axis=1, dtype=jnp.float32
        )
        assert alpha.shape == (1, 1, 8, 1)

    def test_axis_2(self):
        """Test z axis."""
        alpha = alpha_from_direction_axis(
            alpha_start=0.0, alpha_end=1.0, thickness=6, direction="+", axis=2, dtype=jnp.float32
        )
        assert alpha.shape == (1, 1, 1, 6)

    def test_invalid_direction_raises(self):
        """Test that invalid direction raises exception."""
        with pytest.raises(Exception, match="Invalid direction"):
            alpha_from_direction_axis(
                alpha_start=0.0, alpha_end=1.0, thickness=10, direction="x", axis=0, dtype=jnp.float32
            )

    def test_invalid_axis_raises(self):
        """Test that invalid axis raises exception."""
        with pytest.raises(Exception, match="Invalid axis"):
            alpha_from_direction_axis(
                alpha_start=0.0, alpha_end=1.0, thickness=10, direction="+", axis=3, dtype=jnp.float32
            )


class TestAxisDirectionFromKind:
    """Tests for axis_direction_from_kind function."""

    def test_min_x(self):
        """Test min_x parsing."""
        axis, direction = axis_direction_from_kind("min_x")
        assert axis == 0
        assert direction == "-"

    def test_max_x(self):
        """Test max_x parsing."""
        axis, direction = axis_direction_from_kind("max_x")
        assert axis == 0
        assert direction == "+"

    def test_min_y(self):
        """Test min_y parsing."""
        axis, direction = axis_direction_from_kind("min_y")
        assert axis == 1
        assert direction == "-"

    def test_max_y(self):
        """Test max_y parsing."""
        axis, direction = axis_direction_from_kind("max_y")
        assert axis == 1
        assert direction == "+"

    def test_min_z(self):
        """Test min_z parsing."""
        axis, direction = axis_direction_from_kind("min_z")
        assert axis == 2
        assert direction == "-"

    def test_max_z(self):
        """Test max_z parsing."""
        axis, direction = axis_direction_from_kind("max_z")
        assert axis == 2
        assert direction == "+"

    def test_invalid_axis_raises(self):
        """Test that invalid axis raises exception."""
        with pytest.raises(Exception, match="Invalid kind"):
            axis_direction_from_kind("min_w")

    def test_invalid_direction_raises(self):
        """Test that invalid direction raises exception."""
        with pytest.raises(Exception, match="Invalid kind"):
            axis_direction_from_kind("mid_x")
