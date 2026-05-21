"""Tests for objects/device/parameters/discrete.py - discrete parameter transformations."""

import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.misc import PaddingConfig
from fdtdx.materials import Material
from fdtdx.objects.device.parameters.discrete import (
    BOTTOM_Z_PADDING_CONFIG,
    BOTTOM_Z_PADDING_CONFIG_REPEAT,
    BinaryMedianFilterModule,
    ConnectHolesAndStructures,
    RemoveFloatingMaterial,
)
from fdtdx.typing import ParameterType


@pytest.fixture
def two_materials():
    """Two materials fixture with air and silicon."""
    return {
        "Air": Material(permittivity=1.0),
        "Silicon": Material(permittivity=11.7),
    }


@pytest.fixture
def three_materials():
    """Three materials fixture."""
    return {
        "Air": Material(permittivity=1.0),
        "SiO2": Material(permittivity=2.25),
        "Silicon": Material(permittivity=11.7),
    }


@pytest.fixture
def dummy_config():
    """Minimal simulation config."""
    return SimulationConfig(
        time=100e-15,
        resolution=500e-9,
        backend="cpu",
    )


class TestRemoveFloatingMaterial:
    """Tests for RemoveFloatingMaterial transformation."""

    def test_removes_disconnected_region(self, two_materials, dummy_config):
        """Test that floating (disconnected) material is removed."""
        transform = RemoveFloatingMaterial()
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(8, 8, 8),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (8, 8, 8)},
        )
        transform = transform.init_type({"params": ParameterType.BINARY})

        # Create array with floating region (not connected to bottom)
        # 0 = Air (background), 1 = Silicon
        params_arr = jnp.zeros((8, 8, 8), dtype=jnp.int32)
        # Add connected region at bottom (z=0-2)
        params_arr = params_arr.at[2:6, 2:6, 0:3].set(1)
        # Add floating region at top (z=5-7) - should be removed
        params_arr = params_arr.at[2:6, 2:6, 5:8].set(1)

        params = {"params": params_arr}
        result = transform(params)

        # The floating region should be removed (set to 0)
        assert result["params"].shape == (8, 8, 8)
        # Bottom region should remain
        assert jnp.any(result["params"][:, :, 0:3] == 1)
        # Floating region (z=5-7) should have been removed
        assert jnp.all(result["params"][:, :, 5:8] == 0)

    def test_keeps_connected_region(self, two_materials, dummy_config):
        """Test that connected material is preserved."""
        transform = RemoveFloatingMaterial()
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(8, 8, 8),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (8, 8, 8)},
        )
        transform = transform.init_type({"params": ParameterType.BINARY})

        # Create fully connected structure from bottom to top
        params_arr = jnp.zeros((8, 8, 8), dtype=jnp.int32)
        params_arr = params_arr.at[3:5, 3:5, :].set(1)  # Pillar from bottom to top

        params = {"params": params_arr}
        result = transform(params)

        # The pillar should remain
        assert jnp.all(result["params"][3:5, 3:5, :] == 1)

    def test_explicit_background_material(self, two_materials, dummy_config):
        """Test with explicitly specified background material."""
        transform = RemoveFloatingMaterial(background_material="Air")
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(8, 8, 8),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (8, 8, 8)},
        )
        transform = transform.init_type({"params": ParameterType.BINARY})

        params_arr = jnp.ones((8, 8, 8), dtype=jnp.int32)  # All silicon
        params_arr = params_arr.at[:, :, 0].set(0)  # Air at bottom

        params = {"params": params_arr}
        result = transform(params)

        assert result["params"].shape == (8, 8, 8)

    def test_uniform_material_preserved(self, two_materials, dummy_config):
        """Test uniform material (all connected to bottom) is preserved."""
        transform = RemoveFloatingMaterial()
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )
        transform = transform.init_type({"params": ParameterType.BINARY})

        # All silicon (connected)
        params_arr = jnp.ones((4, 4, 4), dtype=jnp.int32)
        params = {"params": params_arr}
        result = transform(params)

        # All silicon should remain
        assert jnp.all(result["params"] == 1)

    def test_all_air_preserved(self, two_materials, dummy_config):
        """Test all-air array is preserved."""
        transform = RemoveFloatingMaterial()
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )
        transform = transform.init_type({"params": ParameterType.BINARY})

        params_arr = jnp.zeros((4, 4, 4), dtype=jnp.int32)  # All air
        params = {"params": params_arr}
        result = transform(params)

        assert jnp.all(result["params"] == 0)


class TestConnectHolesAndStructures:
    """Tests for ConnectHolesAndStructures transformation."""

    def test_connects_floating_region(self, two_materials, dummy_config):
        """Test that floating regions get connected."""
        transform = ConnectHolesAndStructures()
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(8, 8, 8),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (8, 8, 8)},
        )
        transform = transform.init_type({"params": ParameterType.BINARY})

        # Create array with connected and floating regions
        params_arr = jnp.zeros((8, 8, 8), dtype=jnp.int32)
        # Connected region at bottom
        params_arr = params_arr.at[2:6, 2:6, 0:2].set(1)
        # Floating region (not connected initially)
        params_arr = params_arr.at[2:6, 2:6, 4:6].set(1)

        params = {"params": params_arr}
        result = transform(params)

        assert result["params"].shape == (8, 8, 8)

    def test_requires_fill_material_for_three_materials(self, three_materials, dummy_config):
        """Test that fill_material is required for 3+ materials."""
        transform = ConnectHolesAndStructures()  # No fill_material
        transform = transform.init_module(
            config=dummy_config,
            materials=three_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )
        transform = transform.init_type({"params": ParameterType.DISCRETE})

        params = {"params": jnp.zeros((4, 4, 4), dtype=jnp.int32)}

        with pytest.raises(Exception, match="fill_material"):
            transform(params)

    def test_with_fill_material(self, three_materials, dummy_config):
        """Test with explicit fill_material for 3+ materials."""
        transform = ConnectHolesAndStructures(fill_material="SiO2")
        transform = transform.init_module(
            config=dummy_config,
            materials=three_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )
        transform = transform.init_type({"params": ParameterType.DISCRETE})

        params = {"params": jnp.zeros((4, 4, 4), dtype=jnp.int32)}
        result = transform(params)

        assert result["params"].shape == (4, 4, 4)

    def test_explicit_background_material(self, two_materials, dummy_config):
        """Test with explicitly specified background material."""
        transform = ConnectHolesAndStructures(background_material="Air")
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )
        transform = transform.init_type({"params": ParameterType.BINARY})

        params = {"params": jnp.zeros((4, 4, 4), dtype=jnp.int32)}
        result = transform(params)

        assert result["params"].shape == (4, 4, 4)

    def test_uniform_air_preserved(self, two_materials, dummy_config):
        """Test uniform air array is preserved."""
        transform = ConnectHolesAndStructures()
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )
        transform = transform.init_type({"params": ParameterType.BINARY})

        params_arr = jnp.zeros((4, 4, 4), dtype=jnp.int32)  # All air
        params = {"params": params_arr}
        result = transform(params)

        assert jnp.all(result["params"] == 0)


class TestBinaryMedianFilterModule:
    """Tests for BinaryMedianFilterModule transformation."""

    def test_basic_filtering(self, two_materials, dummy_config):
        """Test basic median filtering."""
        padding_cfg = PaddingConfig(
            widths=(1, 1, 1, 1, 1, 1),
            modes=("edge", "edge", "edge", "edge", "edge", "edge"),
        )
        transform = BinaryMedianFilterModule(
            padding_cfg=padding_cfg,
            kernel_sizes=(3, 3, 3),
        )
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(8, 8, 8),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (8, 8, 8)},
        )
        transform = transform.init_type({"params": ParameterType.BINARY})

        # Create binary array
        params_arr = jnp.zeros((8, 8, 8), dtype=jnp.float32)
        params_arr = params_arr.at[2:6, 2:6, 2:6].set(1.0)

        params = {"params": params_arr}
        result = transform(params)

        assert result["params"].shape == (8, 8, 8)
        assert jnp.all((result["params"] == 0) | (result["params"] == 1))

    def test_removes_isolated_pixels(self, two_materials, dummy_config):
        """Test that isolated pixels are removed by median filter."""
        padding_cfg = PaddingConfig(
            widths=(1, 1, 1, 1, 1, 1),
            modes=("edge", "edge", "edge", "edge", "edge", "edge"),
        )
        transform = BinaryMedianFilterModule(
            padding_cfg=padding_cfg,
            kernel_sizes=(3, 3, 3),
        )
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(8, 8, 8),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (8, 8, 8)},
        )
        transform = transform.init_type({"params": ParameterType.BINARY})

        # Create array with isolated pixel in center
        params_arr = jnp.zeros((8, 8, 8), dtype=jnp.float32)
        params_arr = params_arr.at[4, 4, 4].set(1.0)  # Isolated pixel

        params = {"params": params_arr}
        result = transform(params)

        # Isolated pixel should be removed by median filter
        assert result["params"][4, 4, 4] == 0.0

    def test_multiple_repeats(self, two_materials, dummy_config):
        """Test multiple filter repetitions."""
        padding_cfg = PaddingConfig(
            widths=(1, 1, 1, 1, 1, 1),
            modes=("edge", "edge", "edge", "edge", "edge", "edge"),
        )
        transform = BinaryMedianFilterModule(
            padding_cfg=padding_cfg,
            kernel_sizes=(3, 3, 3),
            num_repeats=3,
        )
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(8, 8, 8),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (8, 8, 8)},
        )
        transform = transform.init_type({"params": ParameterType.BINARY})

        params_arr = jnp.zeros((8, 8, 8), dtype=jnp.float32)
        params_arr = params_arr.at[2:6, 2:6, 2:6].set(1.0)

        params = {"params": params_arr}
        result = transform(params)

        assert result["params"].shape == (8, 8, 8)

    def test_different_kernel_sizes(self, two_materials, dummy_config):
        """Test with different kernel sizes per axis."""
        padding_cfg = PaddingConfig(
            widths=(2, 2, 1, 1, 1, 1),
            modes=("edge", "edge", "edge", "edge", "edge", "edge"),
        )
        transform = BinaryMedianFilterModule(
            padding_cfg=padding_cfg,
            kernel_sizes=(5, 3, 3),  # Different sizes
        )
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(10, 10, 10),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (10, 10, 10)},
        )
        transform = transform.init_type({"params": ParameterType.BINARY})

        params_arr = jnp.zeros((10, 10, 10), dtype=jnp.float32)
        params_arr = params_arr.at[3:7, 3:7, 3:7].set(1.0)

        params = {"params": params_arr}
        result = transform(params)

        assert result["params"].shape == (10, 10, 10)

    def test_preserves_uniform_array(self, two_materials, dummy_config):
        """Test that uniform arrays are preserved."""
        padding_cfg = PaddingConfig(
            widths=(1, 1, 1, 1, 1, 1),
            modes=("edge", "edge", "edge", "edge", "edge", "edge"),
        )
        transform = BinaryMedianFilterModule(
            padding_cfg=padding_cfg,
            kernel_sizes=(3, 3, 3),
        )
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(6, 6, 6),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (6, 6, 6)},
        )
        transform = transform.init_type({"params": ParameterType.BINARY})

        # All ones
        params_arr = jnp.ones((6, 6, 6), dtype=jnp.float32)
        params = {"params": params_arr}
        result = transform(params)

        # Should still be all ones
        assert jnp.all(result["params"] == 1.0)


class TestPaddingConfigs:
    """Tests for pre-defined padding configurations."""

    def test_bottom_z_padding_config_repeat_structure(self):
        """Test BOTTOM_Z_PADDING_CONFIG_REPEAT has expected structure."""
        cfg = BOTTOM_Z_PADDING_CONFIG_REPEAT
        assert cfg.widths == (20,)
        assert len(cfg.modes) == 6
        assert cfg.values == (1,)

    def test_bottom_z_padding_config_structure(self):
        """Test BOTTOM_Z_PADDING_CONFIG has expected structure."""
        cfg = BOTTOM_Z_PADDING_CONFIG
        assert cfg.widths == (10,)
        assert len(cfg.modes) == 6
        assert cfg.values == (1, 0, 1, 1, 1, 0)


class TestInputTypeValidation:
    """Tests for input type validation in discrete transforms."""

    def test_remove_floating_expects_binary_or_discrete(self, two_materials, dummy_config):
        """Test RemoveFloatingMaterial accepts BINARY and DISCRETE types."""
        transform = RemoveFloatingMaterial()
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )

        # Should accept BINARY
        transform_binary = transform.init_type({"params": ParameterType.BINARY})
        assert transform_binary._input_type["params"] == ParameterType.BINARY

        # Should accept DISCRETE
        transform_discrete = transform.init_type({"params": ParameterType.DISCRETE})
        assert transform_discrete._input_type["params"] == ParameterType.DISCRETE

    def test_binary_median_expects_binary(self, two_materials, dummy_config):
        """Test BinaryMedianFilterModule expects BINARY type only."""
        padding_cfg = PaddingConfig(
            widths=(1, 1, 1, 1, 1, 1),
            modes=("edge", "edge", "edge", "edge", "edge", "edge"),
        )
        transform = BinaryMedianFilterModule(
            padding_cfg=padding_cfg,
            kernel_sizes=(3, 3, 3),
        )
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )

        # Should accept BINARY
        transform_binary = transform.init_type({"params": ParameterType.BINARY})
        assert transform_binary._input_type["params"] == ParameterType.BINARY

        # Should reject LATENT
        with pytest.raises(Exception):
            transform.init_type({"params": ParameterType.LATENT})
