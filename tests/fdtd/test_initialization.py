from unittest.mock import MagicMock, Mock, patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer

# Import the functions to test
from fdtdx.fdtd.initialization import (
    _init_arrays,
    _init_params,
    _resolve_object_constraints,
    apply_params,
    place_objects,
)
from fdtdx.objects.device.parameters.transform import ParameterType
from fdtdx.objects.object import (
    GridCoordinateConstraint,
    PositionConstraint,
    RealCoordinateConstraint,
    SimulationObject,
)
from fdtdx.objects.static_material.static import UniformMaterialObject


class TestInitialization:
    @pytest.fixture
    def mock_config(self):
        """Create a mock SimulationConfig for testing."""
        config = Mock(spec=SimulationConfig)
        config.dtype = jnp.float32
        config.backend = "cpu"
        config.resolution = 1.0
        config.time_steps_total = 100
        config.gradient_config = None
        return config

    @pytest.fixture
    def mock_volume(self):
        """Create a mock volume object for testing."""
        volume = Mock(spec=SimulationObject)
        volume.name = "volume"
        volume.grid_shape = (10, 10, 10)
        volume.partial_grid_shape = (10, 10, 10)
        volume.partial_real_shape = (10.0, 10.0, 10.0)
        volume.placement_order = 0
        return volume

    @pytest.fixture
    def mock_object(self):
        """Create a mock simulation object for testing."""
        obj = Mock(spec=SimulationObject)
        obj.name = "test_object"
        obj.grid_shape = (5, 5, 5)
        obj.partial_grid_shape = (5, 5, 5)
        obj.partial_real_shape = (5.0, 5.0, 5.0)
        obj.placement_order = 1
        return obj

    @pytest.fixture
    def mock_objects(self):
        """Create a mock ObjectContainer for testing."""
        objects = Mock(spec=ObjectContainer)
        objects.devices = []
        return objects

    @pytest.fixture
    def mock_uniform_material_object(self):
        """Create a mock UniformMaterialObject for testing."""
        material = Mock()
        material.permittivity = 2.0
        material.permeability = 1.0
        material.electric_conductivity = 0.0
        material.magnetic_conductivity = 0.0

        obj = Mock(spec=UniformMaterialObject)
        obj.name = "material_object"
        obj.grid_shape = (5, 5, 5)
        obj.grid_slice = (slice(2, 7), slice(2, 7), slice(2, 7))
        obj.material = material
        obj.placement_order = 1
        return obj

    @pytest.fixture
    def random_key(self):
        """Create a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    def test_place_objects(self, mock_volume, mock_config, random_key):
        """Test the place_objects function."""
        # Create a simple position constraint
        constraint = Mock(spec=PositionConstraint)
        constraint.object = mock_volume
        constraint.other_object = None

        # Mock the internal functions
        with (
            patch("fdtdx.fdtd.initialization._resolve_object_constraints") as mock_resolve,
            patch("fdtdx.fdtd.initialization._init_params") as mock_init_params,
            patch("fdtdx.fdtd.initialization._init_arrays") as mock_init_arrays,
        ):
            # Set up mock returns
            mock_resolve.return_value = {mock_volume: ((0, 10), (0, 10), (0, 10))}
            mock_init_params.return_value = {}

            mock_volume.place_on_grid.return_value = mock_volume

            # Mock array container and config
            mock_arrays = Mock(spec=ArrayContainer)
            mock_updated_config = Mock(spec=SimulationConfig)
            mock_info = {}
            mock_init_arrays.return_value = (mock_arrays, mock_updated_config, mock_info)

            # Call the function
            objects, arrays, params, config, info = place_objects(
                volume=mock_volume, config=mock_config, constraints=[constraint], key=random_key
            )

            # Verify the function was called with correct parameters
            mock_resolve.assert_called_once_with(volume=mock_volume, constraints=[constraint], config=mock_config)

            mock_volume.place_on_grid.assert_called_once()
            mock_init_params.assert_called_once()
            mock_init_arrays.assert_called_once()

            # Verify the return values
            assert isinstance(objects, ObjectContainer)
            assert arrays == mock_arrays
            assert params == {}
            assert config == mock_updated_config
            assert info == mock_info

    def test_apply_params(self, mock_config, random_key):
        """Test the apply_params function."""
        # Create a proper mock for arrays with JAX array operations
        # Create real JAX arrays for the ArrayContainer
        inv_permittivities = jnp.ones((10, 10, 10), dtype=jnp.float32)

        # Create a real ArrayContainer with the JAX arrays
        arrays = ArrayContainer(
            E=jnp.zeros((3, 10, 10, 10), dtype=jnp.float32),
            H=jnp.zeros((3, 10, 10, 10), dtype=jnp.float32),
            inv_permittivities=inv_permittivities,
            inv_permeabilities=1.0,  # scalar since all objects are non-magnetic
            boundary_states={},
            detector_states={},
            recording_state=None,
            electric_conductivity=None,
            magnetic_conductivity=None,
        )

        mock_objects = Mock(spec=ObjectContainer)

        # Create a mock parameter container that supports __getitem__
        mock_params = MagicMock()

        # Create mock device
        mock_device = Mock()
        mock_device.name = "test_device"
        mock_device.output_type = ParameterType.CONTINUOUS
        mock_device.materials = ["material1", "material2"]
        mock_device.grid_slice = (slice(2, 7), slice(2, 7), slice(2, 7))

        # Set up mocks
        mock_objects.devices = [mock_device]
        mock_objects.object_list = [Mock(), Mock()]
        mock_objects.volume_idx = 0

        # Mock the device call to return a proper array
        # The device is called as: device(params[device.name], expand_to_sim_grid=True, **transform_kwargs)
        mock_device.return_value = jnp.ones((5, 5, 5), dtype=jnp.float32) * 0.5
        mock_params.__getitem__.return_value = jnp.array(0.5)

        # Mock the compute_allowed_permittivities function
        with patch("fdtdx.fdtd.initialization.compute_allowed_permittivities") as mock_compute_permittivities:
            mock_compute_permittivities.return_value = [1.0, 2.0]

            # Call the function
            arrays, objects, info = apply_params(
                arrays=arrays, objects=mock_objects, params=mock_params, key=random_key
            )

            # Verify the function was called with correct parameters
            mock_compute_permittivities.assert_called_once_with(["material1", "material2"])

            # Verify the return values
            assert arrays == arrays
            assert isinstance(objects, ObjectContainer)
            assert isinstance(info, dict)

    def test_init_arrays(self, mock_config, mock_uniform_material_object):
        """Test the _init_arrays function."""
        # Create mock object container
        mock_objects = Mock(spec=ObjectContainer)
        mock_objects.volume.grid_shape = (10, 10, 10)
        mock_objects.all_objects_non_magnetic = True
        mock_objects.all_objects_non_electrically_conductive = True
        mock_objects.all_objects_non_magnetically_conductive = True
        mock_objects.static_material_objects = [mock_uniform_material_object]
        mock_objects.detectors = []
        mock_objects.boundary_objects = []
        mock_objects.pml_objects = []

        # Mock the create_named_sharded_matrix function
        with patch("fdtdx.fdtd.initialization.create_named_sharded_matrix") as mock_create_matrix:
            mock_create_matrix.return_value = jnp.zeros((10, 10, 10), dtype=mock_config.dtype)

            # Call the function
            arrays, config, info = _init_arrays(objects=mock_objects, config=mock_config)

            # Verify the function was called with correct parameters
            assert mock_create_matrix.call_count >= 3  # Called for E, H, and inv_permittivities

            # Verify the return values
            assert isinstance(arrays, ArrayContainer)
            assert config == mock_config
            assert isinstance(info, dict)

    def test_init_params(self, mock_objects, random_key):
        """Test the _init_params function."""
        # Create mock device
        mock_device = Mock()
        mock_device.name = "test_device"
        mock_device.init_params.return_value = {"param1": jnp.array(0.5)}

        # Set up mock objects
        mock_objects.devices = [mock_device]

        # Call the function
        params = _init_params(objects=mock_objects, key=random_key)

        # Verify the function was called with correct parameters
        mock_device.init_params.assert_called_once()

        # Verify the return values - check if it's a dict instead of ParameterContainer
        assert isinstance(params, dict)
        assert "test_device" in params
        assert "param1" in params["test_device"]

    def test_resolve_object_constraints(self, mock_volume, mock_config):
        """Test the _resolve_object_constraints function."""
        # Create a grid coordinate constraint
        constraint = Mock(spec=GridCoordinateConstraint)
        constraint.object = mock_volume
        constraint.axes = [0, 1, 2]
        constraint.coordinates = [0, 0, 0]
        constraint.sides = ["-", "-", "-"]

        # Call the function
        result = _resolve_object_constraints(volume=mock_volume, constraints=[constraint], config=mock_config)

        # Verify the return values
        assert isinstance(result, dict)
        assert mock_volume in result
        assert result[mock_volume] == ((0, 10), (0, 10), (0, 10))

    def test_resolve_object_constraints_with_real_coordinate(self, mock_volume, mock_config):
        """Test the _resolve_object_constraints function with RealCoordinateConstraint."""
        # Create a second object for the constraint
        mock_object = Mock(spec=SimulationObject)
        mock_object.name = "test_object"
        mock_object.partial_grid_shape = (None, None, None)
        mock_object.partial_real_shape = (None, None, None)

        # Create a real coordinate constraint for the second object
        constraint = Mock(spec=RealCoordinateConstraint)
        constraint.object = mock_object
        constraint.axes = [0]
        constraint.coordinates = [5.0]  # 5.0 units
        constraint.sides = ["-"]

        # Call the function
        result = _resolve_object_constraints(volume=mock_volume, constraints=[constraint], config=mock_config)

        # Verify the return values
        assert isinstance(result, dict)
        assert mock_volume in result
        assert mock_object in result
        # Should convert 5.0 units to 5 grid cells (with resolution=1.0)
        assert result[mock_object][0][0] == 5
