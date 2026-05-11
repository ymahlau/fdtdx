"""Unit tests for objects/sources/mode.py.

Tests for ModePlaneSource class.
These are unit tests that use mocks where possible to avoid full simulation.
"""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.sources.mode import ModePlaneSource


@pytest.fixture
def mode_source():
    """Create a basic ModePlaneSource for testing."""
    return ModePlaneSource(
        name="TestModeSource",
        partial_grid_shape=(None, None, 1),
        wave_character=WaveCharacter(wavelength=1.55e-6),
        direction="-",
        mode_index=0,
    )


@pytest.fixture
def te_mode_source():
    """Create a TE-filtered ModePlaneSource."""
    return ModePlaneSource(
        name="TEModeSource",
        partial_grid_shape=(None, None, 1),
        wave_character=WaveCharacter(wavelength=1.55e-6),
        direction="-",
        mode_index=0,
        filter_pol="te",
    )


@pytest.fixture
def tm_mode_source():
    """Create a TM-filtered ModePlaneSource."""
    return ModePlaneSource(
        name="TMModeSource",
        partial_grid_shape=(None, None, 1),
        wave_character=WaveCharacter(wavelength=1.55e-6),
        direction="-",
        mode_index=0,
        filter_pol="tm",
    )


@pytest.fixture
def micro_config():
    """Minimal simulation config for mode source testing."""
    return SimulationConfig(
        time=100e-15,
        resolution=100e-9,
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=None,
    )


@pytest.fixture
def jax_key():
    """JAX random key for tests."""
    return jax.random.PRNGKey(42)


class TestModePlaneSourceBasic:
    """Tests for basic ModePlaneSource functionality."""

    def test_initialization_default(self):
        """Test default initialization."""
        source = ModePlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        assert source.mode_index == 0
        assert source.filter_pol is None
        assert source.direction == "-"

    def test_initialization_with_mode_index(self):
        """Test initialization with custom mode index."""
        source = ModePlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="+",
            mode_index=2,
        )
        assert source.mode_index == 2

    def test_initialization_with_filter_pol_te(self, te_mode_source):
        """Test initialization with TE polarization filter."""
        assert te_mode_source.filter_pol == "te"

    def test_initialization_with_filter_pol_tm(self, tm_mode_source):
        """Test initialization with TM polarization filter."""
        assert tm_mode_source.filter_pol == "tm"

    def test_direction_positive(self):
        """Test source with positive direction."""
        source = ModePlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="+",
        )
        assert source.direction == "+"

    def test_direction_negative(self):
        """Test source with negative direction."""
        source = ModePlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        assert source.direction == "-"


class TestModePlaneSourceApply:
    """Tests for the apply method and angle restrictions."""

    def test_apply_with_azimuth_angle_raises(self, micro_config, jax_key):
        """Test that non-zero azimuth angle raises NotImplementedError."""
        source = ModePlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            azimuth_angle=10.0,
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)

        with pytest.raises(NotImplementedError):
            placed.apply(jax_key, inv_perm, 1.0)

    def test_apply_with_elevation_angle_raises(self, micro_config, jax_key):
        """Test that non-zero elevation angle raises NotImplementedError."""
        source = ModePlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            elevation_angle=5.0,
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)

        with pytest.raises(NotImplementedError):
            placed.apply(jax_key, inv_perm, 1.0)

    def test_apply_with_max_angle_random_offset_raises(self, micro_config, jax_key):
        """Test that non-zero max_angle_random_offset raises NotImplementedError."""
        source = ModePlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            max_angle_random_offset=5.0,
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)

        with pytest.raises(NotImplementedError):
            placed.apply(jax_key, inv_perm, 1.0)

    def test_apply_with_max_vertical_offset_raises(self, micro_config, jax_key):
        """Test that non-zero max_vertical_offset raises NotImplementedError."""
        source = ModePlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            max_vertical_offset=1e-6,
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)

        with pytest.raises(NotImplementedError):
            placed.apply(jax_key, inv_perm, 1.0)

    def test_apply_with_max_horizontal_offset_raises(self, micro_config, jax_key):
        """Test that non-zero max_horizontal_offset raises NotImplementedError."""
        source = ModePlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            max_horizontal_offset=1e-6,
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)

        with pytest.raises(NotImplementedError):
            placed.apply(jax_key, inv_perm, 1.0)

    @patch("fdtdx.objects.sources.mode.compute_mode")
    def test_apply_stores_inv_permittivity(self, mock_compute_mode, micro_config, jax_key):
        """Test that apply() stores inv_permittivity slice."""
        mock_E = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64)
        mock_H = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64)
        mock_neff = jnp.array(1.5 + 0j, dtype=jnp.complex64)
        mock_compute_mode.return_value = (mock_E, mock_H, mock_neff)

        source = ModePlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32) * 0.5

        applied = placed.apply(jax_key, inv_perm, 1.0)

        # Check that _inv_permittivity is set
        assert applied._inv_permittivity is not None
        assert applied._inv_permittivity.shape == (1, 8, 8, 1)

    @patch("fdtdx.objects.sources.mode.compute_mode")
    def test_apply_with_anisotropic_permeability(self, mock_compute_mode, micro_config, jax_key):
        """Test apply() with anisotropic permeability array."""
        mock_E = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64)
        mock_H = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64)
        mock_neff = jnp.array(1.5 + 0j, dtype=jnp.complex64)
        mock_compute_mode.return_value = (mock_E, mock_H, mock_neff)

        source = ModePlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        inv_perm_mu = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)

        applied = placed.apply(jax_key, inv_perm, inv_perm_mu)

        assert applied._inv_permeability is not None
        assert applied._inv_permeability.shape == (3, 8, 8, 1)


class TestModePlaneSourcePlot:
    """Tests for the plot method."""

    def test_plot_without_apply_raises(self, mode_source, micro_config, jax_key, tmp_path):
        """Test that plot() raises if apply() wasn't called first."""
        placed = mode_source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        save_path = tmp_path / "mode_plot.png"

        with pytest.raises(Exception):
            placed.plot(save_path)

    @patch("fdtdx.objects.sources.mode.plt")
    @patch("fdtdx.objects.sources.mode.compute_mode")
    def test_plot_calls_matplotlib(self, mock_compute_mode, mock_plt, mode_source, micro_config, jax_key, tmp_path):
        """Test that plot() calls matplotlib functions correctly."""
        mock_E = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64)
        mock_H = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64)
        mock_neff = jnp.array(1.5 + 0j, dtype=jnp.complex64)
        mock_compute_mode.return_value = (mock_E, mock_H, mock_neff)

        mock_fig = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_plt.imshow.return_value = MagicMock()

        placed = mode_source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        applied = placed.apply(jax_key, inv_perm, 1.0)

        save_path = tmp_path / "mode_plot.png"
        applied.plot(str(save_path))

        mock_plt.clf.assert_called()
        mock_plt.figure.assert_called()
        mock_plt.imshow.assert_called()
        mock_plt.savefig.assert_called()
        mock_plt.close.assert_called()


class TestModePlaneSourceGetEHVariation:
    """Tests for get_EH_variation method with mocks."""

    @patch("fdtdx.objects.sources.mode.compute_mode")
    @patch("fdtdx.objects.sources.mode.calculate_time_offset_yee")
    def test_get_EH_variation_returns_correct_shapes(self, mock_time_offset, mock_compute_mode, micro_config, jax_key):
        """Test that get_EH_variation returns arrays with correct shapes."""
        mock_E = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64) * 2.0
        mock_H = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64) * 0.5
        mock_neff = jnp.array(1.5 + 0j, dtype=jnp.complex64)
        mock_compute_mode.return_value = (mock_E, mock_H, mock_neff)

        mock_time_offset.return_value = (
            jnp.zeros((3, 8, 8, 1)),
            jnp.zeros((3, 8, 8, 1)),
        )

        source = ModePlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )

        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        applied = placed.apply(jax_key, inv_perm, 1.0)
        E, H, time_E, time_H = applied._E, applied._H, applied._time_offset_E, applied._time_offset_H

        assert E.shape == (3, 8, 8, 1)
        assert H.shape == (3, 8, 8, 1)
        assert time_E.shape == (3, 8, 8, 1)
        assert time_H.shape == (3, 8, 8, 1)

    @patch("fdtdx.objects.sources.mode.compute_mode")
    @patch("fdtdx.objects.sources.mode.calculate_time_offset_yee")
    def test_get_EH_variation_calls_compute_mode(self, mock_time_offset, mock_compute_mode, micro_config, jax_key):
        """Test that get_EH_variation calls compute_mode with correct args."""
        mock_E = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64)
        mock_H = jnp.ones((3, 8, 8, 1), dtype=jnp.complex64)
        mock_neff = jnp.array(1.5 + 0j, dtype=jnp.complex64)
        mock_compute_mode.return_value = (mock_E, mock_H, mock_neff)

        mock_time_offset.return_value = (
            jnp.zeros((3, 8, 8, 1)),
            jnp.zeros((3, 8, 8, 1)),
        )

        source = ModePlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            mode_index=1,
            filter_pol="te",
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )

        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        placed.apply(jax_key, inv_perm, 1.0)

        # Verify compute_mode was called
        mock_compute_mode.assert_called_once()
        call_kwargs = mock_compute_mode.call_args.kwargs
        assert call_kwargs["mode_index"] == 1
        assert call_kwargs["filter_pol"] == "te"
        assert call_kwargs["direction"] == "-"


class TestModePlaneSourceProperties:
    """Tests for inherited properties from TFSFPlaneSource."""

    def test_propagation_axis(self, mode_source, micro_config, jax_key):
        """Test propagation axis detection based on grid shape."""
        placed = mode_source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        assert placed.propagation_axis == 2

    def test_horizontal_axis(self, mode_source, micro_config, jax_key):
        """Test horizontal axis calculation."""
        placed = mode_source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        assert placed.horizontal_axis == 0

    def test_vertical_axis(self, mode_source, micro_config, jax_key):
        """Test vertical axis calculation."""
        placed = mode_source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        assert placed.vertical_axis == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
