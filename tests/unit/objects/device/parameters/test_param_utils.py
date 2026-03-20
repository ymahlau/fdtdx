"""Tests for objects/device/parameters/utils.py - device parameter utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.objects.device.parameters.utils import (
    compute_allowed_indices,
    compute_allowed_indices_without_holes,
    compute_allowed_indices_without_holes_single_polymer_columns,
    nearest_index,
)


class TestComputeAllowedIndices:
    """Tests for compute_allowed_indices function."""

    def test_delegates_to_single_polymer(self):
        """Test that single_polymer_columns=True uses correct function."""
        result = compute_allowed_indices(
            num_layers=2,
            indices=[0, 1],
            fill_holes_with_index=[],
            single_polymer_columns=True,
        )
        assert result.shape[1] == 2  # num_layers columns

    def test_delegates_to_without_holes(self):
        """Test that single_polymer_columns=False uses correct function."""
        result = compute_allowed_indices(
            num_layers=2,
            indices=[0, 1],
            fill_holes_with_index=[0],  # Need at least one fill index
            single_polymer_columns=False,
        )
        assert result.shape[1] == 2  # num_layers columns


class TestComputeAllowedIndicesSinglePolymerColumns:
    """Tests for compute_allowed_indices_without_holes_single_polymer_columns."""

    def test_no_fill_indices(self):
        """Test with no fill indices - returns all permutations."""
        result = compute_allowed_indices_without_holes_single_polymer_columns(
            num_layers=2,
            indices=[0, 1],
            fill_holes_with_index=[],
        )

        # Should have all 2^2 = 4 permutations
        assert result.shape == (4, 2)

    def test_with_fill_indices(self):
        """Test with fill indices."""
        result = compute_allowed_indices_without_holes_single_polymer_columns(
            num_layers=2,
            indices=[0, 1, 2],
            fill_holes_with_index=[0],
        )

        # Result should have valid permutations
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_single_layer(self):
        """Test with single layer."""
        result = compute_allowed_indices_without_holes_single_polymer_columns(
            num_layers=1,
            indices=[0, 1],
            fill_holes_with_index=[],
        )

        assert result.shape == (2, 1)

    def test_three_layers_no_fill(self):
        """Test with three layers and no fill."""
        result = compute_allowed_indices_without_holes_single_polymer_columns(
            num_layers=3,
            indices=[0, 1],
            fill_holes_with_index=[],
        )

        # Should have all 2^3 = 8 permutations
        assert result.shape == (8, 3)


class TestComputeAllowedIndicesWithoutHoles:
    """Tests for compute_allowed_indices_without_holes."""

    def test_basic_two_layer(self):
        """Test basic two-layer case."""
        result = compute_allowed_indices_without_holes(
            num_layers=2,
            indices=[0, 1],
            fill_holes_with_index=[0],
        )

        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_returns_unique_permutations(self):
        """Test that returned permutations are unique."""
        result = compute_allowed_indices_without_holes(
            num_layers=2,
            indices=[0, 1, 2],
            fill_holes_with_index=[0],
        )

        # Check uniqueness
        result_np = np.array(result)
        unique_rows = np.unique(result_np, axis=0)
        assert len(unique_rows) == len(result_np)


class TestNearestIndex:
    """Tests for nearest_index function."""

    def test_with_3d_values_and_allowed_indices_axis0(self):
        """Test with 3D values and allowed_indices on axis 0."""
        # Values close to 0.0 → nearest index is 0 (allowed_values[0] = 0.0)
        values = jnp.ones((4, 4, 4)) * 0.1
        # allowed_values shape: (num_allowed, size_along_axis=4)
        allowed_values = jnp.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        allowed_indices = jnp.array([[0, 1], [1, 0], [0, 1], [1, 0]])

        indices = nearest_index(
            values,
            allowed_values,
            axis=0,
            allowed_indices=allowed_indices,
        )

        assert indices is not None
        # The default "permittivity_differences_plus_average_permittivity" metric
        # uses jnp.diff along axis 0, reducing that dimension from 4 to 3.
        assert indices.shape == (3, 4, 4)
        # With values ≈ 0.1, every position should select the material with value 0.0
        assert jnp.all(indices == 0)

    def test_with_3d_values_selects_nearest(self):
        """Euclidean metric returns shape (4,4,4) — no diff-based dimension reduction."""
        values = jnp.ones((4, 4, 4)) * 0.9
        allowed_values = jnp.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        allowed_indices = jnp.array([[0, 1], [1, 0], [0, 1], [1, 0]])

        indices = nearest_index(
            values,
            allowed_values,
            axis=0,
            allowed_indices=allowed_indices,
            distance_metric="euclidean",
        )

        assert indices is not None
        # Euclidean metric does not use jnp.diff so the axis-0 dimension is preserved.
        assert indices.shape == (4, 4, 4)

    def test_euclidean_distance_metric(self):
        """Test with euclidean distance metric - values near 0 select index 0."""
        values = jnp.ones((4, 4, 4)) * 0.1
        allowed_values = jnp.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        allowed_indices = jnp.array([[0, 1], [1, 0], [0, 1], [1, 0]])

        indices = nearest_index(
            values,
            allowed_values,
            axis=0,
            allowed_indices=allowed_indices,
            distance_metric="euclidean",
        )

        assert indices is not None
        assert indices.shape == (4, 4, 4)
        assert jnp.all(indices == 0)

    def test_error_no_axis_with_allowed_indices(self):
        """Test that missing axis with allowed_indices raises error."""
        values = jnp.ones((4, 4, 4)) * 0.5
        allowed_values = jnp.array([[0.0], [1.0]])
        allowed_indices = jnp.array([[0, 1]])

        with pytest.raises(Exception, match="Need axis"):
            nearest_index(values, allowed_values, allowed_indices=allowed_indices)

    def test_error_invalid_shape_with_allowed_indices(self):
        """Test that invalid shape raises error."""
        values = jnp.ones((4, 4)) * 0.5  # 2D instead of 3D
        allowed_values = jnp.array([[0.0], [1.0]])
        allowed_indices = jnp.array([[0, 1]])

        with pytest.raises(Exception, match="Invalid array shape"):
            nearest_index(values, allowed_values, axis=0, allowed_indices=allowed_indices)

    def test_error_invalid_axis(self):
        """Test that invalid axis raises error."""
        values = jnp.ones((4, 4, 4)) * 0.5
        allowed_values = jnp.array([[0.0], [1.0]])
        allowed_indices = jnp.array([[0, 1], [1, 0], [0, 1], [1, 0]])

        with pytest.raises(Exception, match="Invalid axis"):
            nearest_index(values, allowed_values, axis=3, allowed_indices=allowed_indices)

    def test_error_unknown_distance_metric(self):
        """Test that unknown distance metric raises error."""
        values = jnp.ones((4, 4, 4)) * 0.5
        allowed_values = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        allowed_indices = jnp.array([[0, 1], [1, 0], [0, 1], [1, 0]])

        with pytest.raises(ValueError, match="Unknown distance metric"):
            nearest_index(
                values,
                allowed_values,
                axis=0,
                allowed_indices=allowed_indices,
                distance_metric="invalid_metric",
            )
