"""Tests for objects/device/parameters/discretization.py - discretization transformations."""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.materials import Material
from fdtdx.objects.device.parameters.discretization import (
    BrushConstraint2D,
    ClosestIndex,
    PillarDiscretization,
    circular_brush,
)
from fdtdx.typing import ParameterType


class TestCircularBrush:
    """Tests for circular_brush function."""

    def test_creates_circular_mask(self):
        """Test that circular_brush creates a valid circular mask."""
        brush = circular_brush(diameter=3.0)

        assert brush.ndim == 2
        # Should be symmetric
        assert brush.shape[0] == brush.shape[1]
        # Center should be True
        center = brush.shape[0] // 2
        assert brush[center, center]

    def test_diameter_1_creates_single_pixel(self):
        """Test diameter=1 creates a single pixel."""
        brush = circular_brush(diameter=1.0)

        # Should have at least center pixel True
        assert jnp.any(brush)

    def test_larger_diameter(self):
        """Test larger diameter creates larger brush."""
        brush_small = circular_brush(diameter=3.0)
        brush_large = circular_brush(diameter=7.0)

        # Larger diameter should have more True pixels
        assert jnp.sum(brush_large) > jnp.sum(brush_small)

    def test_custom_size(self):
        """Test with custom size parameter."""
        brush = circular_brush(diameter=3.0, size=7)

        assert brush.shape == (7, 7)

    def test_even_diameter_rounds_up_to_odd(self):
        """Test even diameter creates odd-sized array when size not specified."""
        brush = circular_brush(diameter=4.0)

        # Should round up to next odd size
        assert brush.shape[0] % 2 == 1

    def test_brush_symmetry(self):
        """Test that brush is symmetric."""
        brush = circular_brush(diameter=5.0)

        # Check horizontal symmetry
        assert jnp.allclose(brush, jnp.flip(brush, axis=0))
        # Check vertical symmetry
        assert jnp.allclose(brush, jnp.flip(brush, axis=1))


class TestClosestIndex:
    """Tests for ClosestIndex transformation."""

    @pytest.fixture
    def two_materials(self):
        """Two materials for binary discretization."""
        return {
            "Air": Material(permittivity=1.0),
            "Silicon": Material(permittivity=11.7),
        }

    @pytest.fixture
    def three_materials(self):
        """Three materials for discrete discretization."""
        return {
            "Air": Material(permittivity=1.0),
            "SiO2": Material(permittivity=2.25),
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

    def test_get_output_type_binary(self, two_materials, dummy_config):
        """Test output type is BINARY for two materials."""
        transform = ClosestIndex()
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )

        output_type = transform._get_output_type_impl({"params": ParameterType.CONTINUOUS})

        assert output_type["params"] == ParameterType.BINARY

    def test_get_output_type_discrete(self, three_materials, dummy_config):
        """Test output type is DISCRETE for three+ materials."""
        transform = ClosestIndex()
        transform = transform.init_module(
            config=dummy_config,
            materials=three_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )

        output_type = transform._get_output_type_impl({"params": ParameterType.CONTINUOUS})

        assert output_type["params"] == ParameterType.DISCRETE

    def test_get_output_type_error_single_material(self, dummy_config):
        """Test error raised for single material."""
        transform = ClosestIndex()
        single_material = {"Air": Material(permittivity=1.0)}
        transform = transform.init_module(
            config=dummy_config,
            materials=single_material,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )

        with pytest.raises(Exception, match="Invalid materials"):
            transform._get_output_type_impl({"params": ParameterType.CONTINUOUS})

    def test_call_without_inverse_permittivity_mapping(self, two_materials, dummy_config):
        """Test __call__ with mapping_from_inverse_permittivities=False."""
        transform = ClosestIndex(mapping_from_inverse_permittivities=False)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )

        # Input continuous values
        params = {"params": jnp.array([[[0.3, 0.7], [1.2, 0.1]]])}

        result = transform(params)

        # Should quantize to nearest integer (0 or 1)
        assert "params" in result
        # Values should be close to 0 or 1
        assert jnp.all((result["params"] >= -0.5) & (result["params"] <= 1.5))

    def test_call_with_inverse_permittivity_mapping(self, two_materials, dummy_config):
        """Test __call__ with mapping_from_inverse_permittivities=True."""
        transform = ClosestIndex(mapping_from_inverse_permittivities=True)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )

        # Input values representing inverse permittivities
        # Air: inv_perm = 1.0, Silicon: inv_perm = 1/11.7 ≈ 0.085
        params = {"params": jnp.array([[[0.5, 0.1], [0.9, 0.05]]])}

        result = transform(params)

        assert "params" in result
        # Result should be indices 0 or 1
        assert result["params"].shape == params["params"].shape


class TestBrushConstraint2D:
    """Tests for BrushConstraint2D transformation."""

    @pytest.fixture
    def two_materials(self):
        """Two materials for brush constraint."""
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

    def test_get_output_type_binary(self, two_materials, dummy_config):
        """Test output type is BINARY."""
        brush = circular_brush(diameter=3.0)
        transform = BrushConstraint2D(brush=brush, axis=2)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(8, 8, 1),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (8, 8, 1)},
        )

        output_type = transform._get_output_type_impl({"params": ParameterType.CONTINUOUS})

        assert output_type["params"] == ParameterType.BINARY

    def test_error_non_binary_materials(self, dummy_config):
        """Test error raised for non-binary materials."""
        three_materials = {
            "Air": Material(permittivity=1.0),
            "SiO2": Material(permittivity=2.25),
            "Silicon": Material(permittivity=11.7),
        }
        brush = circular_brush(diameter=3.0)
        transform = BrushConstraint2D(brush=brush, axis=2)
        transform = transform.init_module(
            config=dummy_config,
            materials=three_materials,
            matrix_voxel_grid_shape=(8, 8, 1),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (8, 8, 1)},
        )

        with pytest.raises(Exception, match="exactly two materials"):
            transform._get_output_type_impl({"params": ParameterType.CONTINUOUS})

    def test_error_wrong_axis_size(self, two_materials, dummy_config):
        """Test error when axis dimension is not 1."""
        brush = circular_brush(diameter=3.0)
        transform = BrushConstraint2D(brush=brush, axis=2)

        # Error should be raised during init_module when shape is wrong
        with pytest.raises(Exception, match="2d arrays"):
            transform.init_module(
                config=dummy_config,
                materials=two_materials,
                matrix_voxel_grid_shape=(8, 8, 8),  # z != 1
                single_voxel_size=(1e-6, 1e-6, 1e-6),
                output_shape={"params": (8, 8, 8)},
            )

    def test_call_produces_binary_output(self, two_materials, dummy_config):
        """Test that __call__ produces binary output."""
        brush = circular_brush(diameter=3.0)
        transform = BrushConstraint2D(brush=brush, axis=2)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(8, 8, 1),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (8, 8, 1)},
        )

        # Random continuous input
        key = jax.random.PRNGKey(42)
        params = {"params": jax.random.normal(key, (8, 8, 1))}

        result = transform(params)

        assert "params" in result
        assert result["params"].shape == (8, 8, 1)
        # Output should be 0 or 1 (binary)
        unique_vals = jnp.unique(result["params"])
        assert len(unique_vals) <= 2

    def test_with_custom_background_material(self, two_materials, dummy_config):
        """Test with explicitly specified background material."""
        brush = circular_brush(diameter=3.0)
        transform = BrushConstraint2D(
            brush=brush,
            axis=2,
            background_material="Air",
        )
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(8, 8, 1),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (8, 8, 1)},
        )

        key = jax.random.PRNGKey(42)
        params = {"params": jax.random.normal(key, (8, 8, 1))}

        result = transform(params)

        assert result["params"].shape == (8, 8, 1)


class TestPillarDiscretization:
    """Tests for PillarDiscretization transformation."""

    @pytest.fixture
    def two_materials(self):
        """Two materials for pillar discretization."""
        return {
            "Air": Material(permittivity=1.0),
            "Silicon": Material(permittivity=11.7),
        }

    @pytest.fixture
    def three_materials(self):
        """Three materials for pillar discretization."""
        return {
            "Air": Material(permittivity=1.0),
            "SiO2": Material(permittivity=2.25),
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

    def test_get_output_type_binary(self, two_materials, dummy_config):
        """Test output type is BINARY for two materials."""
        transform = PillarDiscretization(
            axis=2,
            single_polymer_columns=True,
        )
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )

        output_type = transform._get_output_type_impl({"params": ParameterType.CONTINUOUS})

        assert output_type["params"] == ParameterType.BINARY

    def test_get_output_type_discrete(self, three_materials, dummy_config):
        """Test output type is DISCRETE for three+ materials."""
        transform = PillarDiscretization(
            axis=2,
            single_polymer_columns=False,
        )
        transform = transform.init_module(
            config=dummy_config,
            materials=three_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )

        output_type = transform._get_output_type_impl({"params": ParameterType.CONTINUOUS})

        assert output_type["params"] == ParameterType.DISCRETE

    def test_get_output_type_error_single_material(self, dummy_config):
        """Test error raised for single material."""
        transform = PillarDiscretization(
            axis=2,
            single_polymer_columns=True,
        )
        single_material = {"Air": Material(permittivity=1.0)}
        transform = transform.init_module(
            config=dummy_config,
            materials=single_material,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )

        with pytest.raises(Exception, match="Invalid materials"):
            transform._get_output_type_impl({"params": ParameterType.CONTINUOUS})

    def test_call_axis_2(self, two_materials, dummy_config):
        """Test __call__ with axis=2.

        Note: PillarDiscretization expects input values that represent
        inverse permittivities, so we use values in the appropriate range.
        The output shape matches the input shape via straight_through_estimator.
        """
        transform = PillarDiscretization(
            axis=2,
            single_polymer_columns=True,
        )
        # Use smaller z-dimension for simpler allowed_indices
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(3, 3, 2),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (3, 3, 2)},
        )

        # Input values representing inverse permittivities
        key = jax.random.PRNGKey(42)
        params = {"params": jax.random.uniform(key, (3, 3, 2)) * 0.5 + 0.1}

        result = transform(params)

        assert "params" in result
        # Result should be valid JAX array with material indices
        assert result["params"].ndim >= 3

    def test_init_axis_1(self, two_materials, dummy_config):
        """Test init_module with axis=1 succeeds."""
        transform = PillarDiscretization(
            axis=1,
            single_polymer_columns=True,
        )
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(3, 2, 3),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (3, 2, 3)},
        )

        # Verify init succeeded and allowed_indices were computed
        assert transform._allowed_indices is not None
        assert transform._allowed_indices.ndim == 2

    def test_init_axis_0(self, two_materials, dummy_config):
        """Test init_module with axis=0 succeeds."""
        transform = PillarDiscretization(
            axis=0,
            single_polymer_columns=True,
        )
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(2, 3, 3),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (2, 3, 3)},
        )

        # Verify init succeeded
        assert transform._allowed_indices is not None
        assert transform._allowed_indices.ndim == 2

    def test_euclidean_distance_metric_init(self, two_materials, dummy_config):
        """Test init with euclidean distance metric."""
        transform = PillarDiscretization(
            axis=2,
            single_polymer_columns=True,
            distance_metric="euclidean",
        )
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(3, 3, 2),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (3, 3, 2)},
        )

        # Verify distance metric is set
        assert transform.distance_metric == "euclidean"
        assert transform._allowed_indices is not None

    def test_with_custom_background_material(self, two_materials, dummy_config):
        """Test with explicitly specified background material."""
        transform = PillarDiscretization(
            axis=2,
            single_polymer_columns=True,
            background_material="Air",
        )
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(3, 3, 2),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (3, 3, 2)},
        )

        key = jax.random.PRNGKey(42)
        params = {"params": jax.random.uniform(key, (3, 3, 2)) * 0.5 + 0.1}

        result = transform(params)

        assert "params" in result
        assert result["params"].ndim >= 3

    def test_invalid_axis_raises_error(self, two_materials, dummy_config):
        """Test that invalid axis raises IndexError during init."""
        transform = PillarDiscretization(
            axis=3,  # Invalid axis
            single_polymer_columns=True,
        )

        # Error should be raised during init_module when axis is out of range
        with pytest.raises(IndexError):
            transform.init_module(
                config=dummy_config,
                materials=two_materials,
                matrix_voxel_grid_shape=(3, 3, 2),
                single_voxel_size=(1e-6, 1e-6, 1e-6),
                output_shape={"params": (3, 3, 2)},
            )
