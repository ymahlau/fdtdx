import tempfile
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from fdtdx.utils.plot_field_slice import plot_field_slice, plot_field_slice_component


class TestPlotFieldSliceComponent:
    """Tests for plot_field_slice_component function."""

    def test_valid_2d_field(self):
        """Test plotting a valid 2D field component."""
        field = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        fig, ax = plt.subplots()

        # Should not raise any exception
        plot_field_slice_component(field, "Ex", ax, plot_legend=True)
        plt.close(fig)

    def test_valid_2d_field_no_legend(self):
        """Test plotting without colorbar legend."""
        field = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        fig, ax = plt.subplots()

        plot_field_slice_component(field, "Hy", ax, plot_legend=False)
        plt.close(fig)

    def test_negative_values(self):
        """Test that negative values are handled correctly."""
        field = jnp.array([[-1.0, 2.0], [3.0, -4.0]])
        fig, ax = plt.subplots()

        plot_field_slice_component(field, "Ez", ax)
        plt.close(fig)

    def test_zero_field(self):
        """Test plotting a field with all zeros."""
        field = jnp.zeros((5, 5))
        fig, ax = plt.subplots()

        plot_field_slice_component(field, "Ex", ax)
        plt.close(fig)

    def test_non_2d_field_raises_error(self):
        """Test that non-2D field raises ValueError."""
        field_1d = jnp.array([1.0, 2.0, 3.0])
        fig, ax = plt.subplots()

        with pytest.raises(ValueError, match="Field must be 2D"):
            plot_field_slice_component(field_1d, "Ex", ax)
        plt.close(fig)

    def test_3d_field_raises_error(self):
        """Test that 3D field raises ValueError."""
        field_3d = jnp.ones((3, 4, 5))
        fig, ax = plt.subplots()

        with pytest.raises(ValueError, match="Field must be 2D"):
            plot_field_slice_component(field_3d, "Ex", ax)
        plt.close(fig)

    def test_nan_values_raise_error(self):
        """Test that NaN values raise ValueError."""
        field = jnp.array([[1.0, jnp.nan], [3.0, 4.0]])
        fig, ax = plt.subplots()

        with pytest.raises(ValueError, match="contains NaN values"):
            plot_field_slice_component(field, "Ex", ax)
        plt.close(fig)

    def test_inf_values_raise_error(self):
        """Test that infinite values raise ValueError."""
        field = jnp.array([[1.0, jnp.inf], [3.0, 4.0]])
        fig, ax = plt.subplots()

        with pytest.raises(ValueError, match="contains infinite values"):
            plot_field_slice_component(field, "Ex", ax)
        plt.close(fig)

    def test_neg_inf_values_raise_error(self):
        """Test that negative infinite values raise ValueError."""
        field = jnp.array([[1.0, -jnp.inf], [3.0, 4.0]])
        fig, ax = plt.subplots()

        with pytest.raises(ValueError, match="contains infinite values"):
            plot_field_slice_component(field, "Ex", ax)
        plt.close(fig)


class TestPlotFieldSlice:
    """Tests for plot_field_slice function."""

    def test_valid_3d_input(self):
        """Test with valid 3D input (3, w, h)."""
        E = jnp.ones((3, 10, 10))
        H = jnp.ones((3, 10, 10)) * 0.5

        fig = plot_field_slice(E, H)
        assert fig is not None
        plt.close(fig)

    def test_valid_3d_input_with_filename(self):
        """Test saving to file with 3D input."""
        E = jnp.ones((3, 10, 10))
        H = jnp.ones((3, 10, 10)) * 0.5

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            plot_field_slice(E, H, filename=tmp_path)
            assert tmp_path.exists()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_valid_3d_input_no_legend(self):
        """Test plotting without legends."""
        E = jnp.ones((3, 10, 10))
        H = jnp.ones((3, 10, 10)) * 0.5

        fig = plot_field_slice(E, H, plot_legend=False)
        assert fig is not None
        plt.close(fig)

    def test_valid_3d_input_with_axes(self):
        """Test with pre-created axes."""
        E = jnp.ones((3, 10, 10))
        H = jnp.ones((3, 10, 10)) * 0.5

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        result_fig = plot_field_slice(E, H, axs=axs)
        assert result_fig is fig  # Should return the figure that was passed in
        plt.close(fig)

    def test_valid_4d_input_z_slice(self):
        """Test with 4D input where z dimension is 1."""
        E = jnp.ones((3, 10, 10, 1))
        H = jnp.ones((3, 10, 10, 1)) * 0.5

        fig = plot_field_slice(E, H)
        assert fig is not None
        plt.close(fig)

    def test_valid_4d_input_y_slice(self):
        """Test with 4D input where y dimension is 1."""
        E = jnp.ones((3, 10, 1, 10))
        H = jnp.ones((3, 10, 1, 10)) * 0.5

        fig = plot_field_slice(E, H)
        assert fig is not None
        plt.close(fig)

    def test_valid_4d_input_x_slice(self):
        """Test with 4D input where x dimension is 1."""
        E = jnp.ones((3, 1, 10, 10))
        H = jnp.ones((3, 1, 10, 10)) * 0.5

        fig = plot_field_slice(E, H)
        assert fig is not None
        plt.close(fig)

    def test_complex_field_values(self):
        """Test with realistic field patterns."""
        x = jnp.linspace(-1, 1, 20)
        y = jnp.linspace(-1, 1, 20)
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        E = jnp.stack([jnp.sin(X) * jnp.cos(Y), jnp.cos(X) * jnp.sin(Y), jnp.exp(-(X**2 + Y**2))])

        H = jnp.stack([jnp.cos(X) * jnp.cos(Y), -jnp.sin(X) * jnp.sin(Y), jnp.exp(-(X**2 + Y**2)) * 0.5])

        fig = plot_field_slice(E, H)
        assert fig is not None
        plt.close(fig)

    def test_e_wrong_first_dimension(self):
        """Test that E field with wrong first dimension raises error."""
        E = jnp.ones((2, 10, 10))  # Should be 3
        H = jnp.ones((3, 10, 10))

        with pytest.raises(ValueError, match="E field first dimension must be 3"):
            plot_field_slice(E, H)

    def test_h_wrong_first_dimension(self):
        """Test that H field with wrong first dimension raises error."""
        E = jnp.ones((3, 10, 10))
        H = jnp.ones((2, 10, 10))  # Should be 3

        with pytest.raises(ValueError, match="H field first dimension must be 3"):
            plot_field_slice(E, H)

    def test_e_wrong_ndim_2d(self):
        """Test that 2D E field raises error."""
        E = jnp.ones((3, 10))
        H = jnp.ones((3, 10, 10))

        with pytest.raises(ValueError, match="E field must be 3D.*or 4D"):
            plot_field_slice(E, H)

    def test_e_wrong_ndim_5d(self):
        """Test that 5D E field raises error."""
        E = jnp.ones((3, 10, 10, 10, 10))
        H = jnp.ones((3, 10, 10))

        with pytest.raises(ValueError, match="E field must be 3D.*or 4D"):
            plot_field_slice(E, H)

    def test_h_wrong_ndim_2d(self):
        """Test that 2D H field raises error."""
        E = jnp.ones((3, 10, 10))
        H = jnp.ones((3, 10))

        with pytest.raises(ValueError, match="H field must be 3D.*or 4D"):
            plot_field_slice(E, H)

    def test_h_wrong_ndim_5d(self):
        """Test that 5D H field raises error."""
        E = jnp.ones((3, 10, 10))
        H = jnp.ones((3, 10, 10, 10, 10))

        with pytest.raises(ValueError, match="H field must be 3D.*or 4D"):
            plot_field_slice(E, H)

    def test_shape_mismatch(self):
        """Test that mismatched E and H shapes raise error."""
        E = jnp.ones((3, 10, 10))
        H = jnp.ones((3, 15, 15))

        with pytest.raises(ValueError, match="E and H fields must have same shape"):
            plot_field_slice(E, H)

    def test_4d_no_singleton_dimension(self):
        """Test that 4D input with no singleton dimension raises error."""
        E = jnp.ones((3, 10, 10, 10))
        H = jnp.ones((3, 10, 10, 10))

        with pytest.raises(ValueError, match="exactly one of nx, ny, nz must be 1"):
            plot_field_slice(E, H)

    def test_4d_multiple_singleton_dimensions(self):
        """Test that 4D input with multiple singleton dimensions raises error."""
        E = jnp.ones((3, 1, 1, 10))
        H = jnp.ones((3, 1, 1, 10))

        with pytest.raises(ValueError, match="exactly one of nx, ny, nz must be 1"):
            plot_field_slice(E, H)

    def test_4d_all_singleton_dimensions(self):
        """Test that 4D input with all singleton dimensions raises error."""
        E = jnp.ones((3, 1, 1, 1))
        H = jnp.ones((3, 1, 1, 1))

        with pytest.raises(ValueError, match="exactly one of nx, ny, nz must be 1"):
            plot_field_slice(E, H)

    def test_e_contains_nan(self):
        """Test that E field with NaN raises error."""
        E = jnp.ones((3, 10, 10))
        E = E.at[0, 5, 5].set(jnp.nan)
        H = jnp.ones((3, 10, 10))

        with pytest.raises(ValueError, match="E field contains NaN values"):
            plot_field_slice(E, H)

    def test_e_contains_inf(self):
        """Test that E field with inf raises error."""
        E = jnp.ones((3, 10, 10))
        E = E.at[1, 3, 3].set(jnp.inf)
        H = jnp.ones((3, 10, 10))

        with pytest.raises(ValueError, match="E field contains infinite values"):
            plot_field_slice(E, H)

    def test_e_contains_neg_inf(self):
        """Test that E field with -inf raises error."""
        E = jnp.ones((3, 10, 10))
        E = E.at[2, 7, 7].set(-jnp.inf)
        H = jnp.ones((3, 10, 10))

        with pytest.raises(ValueError, match="E field contains infinite values"):
            plot_field_slice(E, H)

    def test_h_contains_nan(self):
        """Test that H field with NaN raises error."""
        E = jnp.ones((3, 10, 10))
        H = jnp.ones((3, 10, 10))
        H = H.at[0, 5, 5].set(jnp.nan)

        with pytest.raises(ValueError, match="H field contains NaN values"):
            plot_field_slice(E, H)

    def test_h_contains_inf(self):
        """Test that H field with inf raises error."""
        E = jnp.ones((3, 10, 10))
        H = jnp.ones((3, 10, 10))
        H = H.at[1, 3, 3].set(jnp.inf)

        with pytest.raises(ValueError, match="H field contains infinite values"):
            plot_field_slice(E, H)

    def test_h_contains_neg_inf(self):
        """Test that H field with -inf raises error."""
        E = jnp.ones((3, 10, 10))
        H = jnp.ones((3, 10, 10))
        H = H.at[2, 7, 7].set(-jnp.inf)

        with pytest.raises(ValueError, match="H field contains infinite values"):
            plot_field_slice(E, H)

    def test_large_field_arrays(self):
        """Test with larger field arrays."""
        E = jnp.ones((3, 100, 100))
        H = jnp.ones((3, 100, 100)) * 0.5

        fig = plot_field_slice(E, H)
        assert fig is not None
        plt.close(fig)

    def test_small_field_arrays(self):
        """Test with minimal size arrays."""
        E = jnp.ones((3, 2, 2))
        H = jnp.ones((3, 2, 2))

        fig = plot_field_slice(E, H)
        assert fig is not None
        plt.close(fig)

    def test_rectangular_field(self):
        """Test with non-square field arrays."""
        E = jnp.ones((3, 20, 30))
        H = jnp.ones((3, 20, 30)) * 0.5

        fig = plot_field_slice(E, H)
        assert fig is not None
        plt.close(fig)

    def test_4d_rectangular_x_slice(self):
        """Test 4D rectangular array with x slice."""
        E = jnp.ones((3, 1, 20, 30))
        H = jnp.ones((3, 1, 20, 30)) * 0.5

        fig = plot_field_slice(E, H)
        assert fig is not None
        plt.close(fig)

    def test_4d_rectangular_y_slice(self):
        """Test 4D rectangular array with y slice."""
        E = jnp.ones((3, 20, 1, 30))
        H = jnp.ones((3, 20, 1, 30)) * 0.5

        fig = plot_field_slice(E, H)
        assert fig is not None
        plt.close(fig)

    def test_4d_rectangular_z_slice(self):
        """Test 4D rectangular array with z slice."""
        E = jnp.ones((3, 20, 30, 1))
        H = jnp.ones((3, 20, 30, 1)) * 0.5

        fig = plot_field_slice(E, H)
        assert fig is not None
        plt.close(fig)

    def test_all_components_visualized(self):
        """Test that all six components are plotted."""
        E = jnp.ones((3, 10, 10))
        H = jnp.ones((3, 10, 10)) * 0.5

        fig, axs = plt.subplots(2, 3)
        plot_field_slice(E, H, axs=axs)

        # Check that all axes have been used (have images)
        for i in range(2):
            for j in range(3):
                assert len(axs[i, j].images) > 0

        plt.close(fig)

    def test_colorbar_generated_component(self):
        """Test that colorbar is generated for component plot."""
        field = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        fig, ax = plt.subplots()

        # Get initial number of axes (should be 1)
        initial_axes_count = len(fig.axes)

        plot_field_slice_component(field, "Ex", ax, plot_legend=True)

        # After adding colorbar, should have 2 axes (main + colorbar)
        final_axes_count = len(fig.axes)
        assert final_axes_count == initial_axes_count + 1, "Colorbar axis should be added"

        plt.close(fig)

    def test_no_colorbar_when_disabled_component(self):
        """Test that no colorbar is generated when plot_legend=False."""
        field = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        fig, ax = plt.subplots()

        # Get initial number of axes (should be 1)
        initial_axes_count = len(fig.axes)

        plot_field_slice_component(field, "Ex", ax, plot_legend=False)

        # Should still have only 1 axis (no colorbar added)
        final_axes_count = len(fig.axes)
        assert final_axes_count == initial_axes_count, "No colorbar axis should be added"

        plt.close(fig)

    def test_colorbars_generated_full_plot(self):
        """Test that colorbars are generated for all components in full plot."""
        E = jnp.ones((3, 10, 10))
        H = jnp.ones((3, 10, 10)) * 0.5

        fig = plot_field_slice(E, H, plot_legend=True)

        # Should have 6 main axes + 6 colorbar axes = 12 total
        assert len(fig.axes) == 12, f"Expected 12 axes (6 plots + 6 colorbars), got {len(fig.axes)}"

        plt.close(fig)

    def test_no_colorbars_when_disabled_full_plot(self):
        """Test that no colorbars are generated when plot_legend=False."""
        E = jnp.ones((3, 10, 10))
        H = jnp.ones((3, 10, 10)) * 0.5

        fig = plot_field_slice(E, H, plot_legend=False)

        # Should have only 6 main axes (no colorbars)
        assert len(fig.axes) == 6, f"Expected 6 axes (6 plots, no colorbars), got {len(fig.axes)}"

        plt.close(fig)

    def test_colorbar_labels(self):
        """Test that colorbar has correct label."""
        field = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        fig, ax = plt.subplots()

        plot_field_slice_component(field, "Ex", ax, plot_legend=True)

        # Get the colorbar axis (should be the second axis)
        assert len(fig.axes) == 2
        cbar_ax = fig.axes[1]

        # Check that colorbar has a label
        cbar_label = cbar_ax.get_ylabel()
        assert cbar_label == "Field value", f"Expected 'Field value', got '{cbar_label}'"

        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
