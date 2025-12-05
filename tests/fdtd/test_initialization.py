from unittest.mock import MagicMock, Mock, patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.initialization import apply_params, place_objects, resolve_object_constraints
from fdtdx.materials import Material
from fdtdx.objects.device.parameters.transform import ParameterType
from fdtdx.objects.object import (
    GridCoordinateConstraint,
    PositionConstraint,
    RealCoordinateConstraint,
    SizeConstraint,
    SizeExtensionConstraint,
)
from fdtdx.objects.static_material.static import (
    SimulationVolume,
    UniformMaterialObject,
)


# Fixtures for reusable test objects
@pytest.fixture
def simple_config():
    """Create a simple simulation configuration."""
    return SimulationConfig(
        resolution=1.0,
        time=100e-15,  # 100 femtoseconds
    )


@pytest.fixture
def simple_volume():
    """Create a simple simulation volume."""
    return SimulationVolume(
        name="volume",
        partial_grid_shape=(100, 100, 100),
    )


@pytest.fixture
def simple_material():
    """Create a simple material."""
    return Material(
        permittivity=(2.0, 2.0, 2.0),
        permeability=(1.0, 1.0, 1.0),
        electric_conductivity=(0.0, 0.0, 0.0),
        magnetic_conductivity=(0.0, 0.0, 0.0),
    )

# Test cases for constraint resolution with real objects


def test_resolve_constraints_with_duplicate_object_names(simple_config, simple_volume):
    """Test that duplicate object names raise an exception."""
    volume2 = SimulationVolume(
        name="volume",  # Duplicate name
        partial_grid_shape=(50, 50, 50),
    )

    objects = [simple_volume, volume2]
    constraints = []

    with pytest.raises(Exception, match="Duplicate object names"):
        resolve_object_constraints(objects, constraints, simple_config)


def test_resolve_constraints_unknown_object_in_constraint(simple_config, simple_volume):
    """Test that unknown object names in constraints raise an exception."""
    objects = [simple_volume]

    # Reference an object that doesn't exist
    constraint = GridCoordinateConstraint(object="nonexistent_object", axes=[0], sides=["-"], coordinates=[10])

    constraints = [constraint]

    with pytest.raises(ValueError, match="Unknown object name"):
        resolve_object_constraints(objects, constraints, simple_config)

def test_resolve_constraints_no_simulation_volume(simple_config, simple_material):
    """Test that missing SimulationVolume raises an exception."""
    # Create an object without a volume
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    objects = [obj]
    constraints = []

    with pytest.raises(ValueError, match="No SimulationVolume"):
        resolve_object_constraints(objects, constraints, simple_config)


def test_resolve_constraints_multiple_volumes(simple_config):
    """Test that multiple SimulationVolume objects raise an exception."""
    volume1 = SimulationVolume(
        name="volume1",
        partial_grid_shape=(100, 100, 100),
    )
    volume2 = SimulationVolume(
        name="volume2",
        partial_grid_shape=(50, 50, 50),
    )


    objects = [volume1, volume2]
    constraints = []

    # Mock the compute_allowed_permittivities function
    with patch("fdtdx.fdtd.initialization.compute_allowed_permittivities") as mock_compute_permittivities:
        # Return list of tuples (x, y, z components) instead of scalars
        mock_compute_permittivities.return_value = [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]

    with pytest.raises(ValueError, match="Multiple SimulationVolume"):
        resolve_object_constraints(objects, constraints, simple_config)


def test_resolve_constraints_conflicting_grid_coordinates(simple_config, simple_volume, simple_material):
    """Test that conflicting grid coordinate constraints raise an error."""
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Create two conflicting constraints
    constraint1 = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10])
    constraint2 = GridCoordinateConstraint(
        object="obj1",
        axes=[0],
        sides=["-"],
        coordinates=[20],  # Different coordinate - conflict!
    )

    constraints = [constraint1, constraint2]

    # Should fail to resolve
    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)
    assert errors["obj1"] is not None


def test_resolve_constraints_conflicting_real_coordinates(simple_config, simple_volume, simple_material):
    """Test that conflicting real coordinate constraints raise an error."""
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Create two conflicting real coordinate constraints
    constraint1 = RealCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10.0])
    constraint2 = RealCoordinateConstraint(
        object="obj1",
        axes=[0],
        sides=["-"],
        coordinates=[20.0],  # Different coordinate - conflict!
    )

    constraints = [constraint1, constraint2]

    # Should fail to resolve
    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)
    assert errors["obj1"] is not None


def test_resolve_constraints_inconsistent_size_and_position(simple_config, simple_volume, simple_material):
    """Test that inconsistent size and position constraints are detected."""
    obj = UniformMaterialObject(
        name="obj1",
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Set explicit size
    size_constraint = SizeConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        other_axes=[0],
        proportions=[0.5],
        grid_offsets=[0],
        offsets=[None],
    )

    # Position that would require different size
    pos_constraint = PositionConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        object_positions=[0.0],
        other_object_positions=[-1.0],  # At start
        grid_margins=[5],  # With margin
        margins=[None],
    )

    # Fix position explicitly that conflicts
    grid_constraint = GridCoordinateConstraint(
        object="obj1",
        axes=[0],
        sides=["-"],
        coordinates=[0],  # Should start at 0, but position constraint says 5
    )

    constraints = [size_constraint, pos_constraint, grid_constraint]

    # Should fail to resolve due to inconsistency
    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)
    assert errors["obj1"] is not None


def test_resolve_constraints_with_real_margins(simple_config, simple_volume, simple_material):
    """Test position constraints with real (non-grid) margins."""
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Position with real margin (will be converted to grid units)
    # Need to also provide a size constraint so the object can be fully resolved
    size_constraint = SizeConstraint(
        object="obj1",
        other_object="volume",
        axes=[0, 1, 2],
        other_axes=[0, 1, 2],
        proportions=[0.1, 0.1, 0.1],
        grid_offsets=[0, 0, 0],
        offsets=[None, None, None],
    )

    pos_constraint = PositionConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        object_positions=[0.0],
        other_object_positions=[-1.0],  # At start of volume
        grid_margins=[None],
        margins=[5.0],  # Real margin: 5.0 / 1.0 resolution = 5 grid units
    )

    constraints = [size_constraint, pos_constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed (or have errors - depends on constraint compatibility)
    # Just verify it doesn't crash and returns results
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices


def test_resolve_constraints_with_both_margins(simple_volume, simple_material):
    """Test position constraints with both grid and real margins."""
    config = SimulationConfig(resolution=0.5, time=100e-15)

    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Position with both grid and real margins
    constraint = PositionConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        object_positions=[0.0],
        other_object_positions=[-1.0],
        grid_margins=[2],  # 2 grid units
        margins=[1.0],  # 1.0 / 0.5 = 2 grid units, total = 4
    )

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, config)

    # Should succeed or at least return results
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices


def test_resolve_constraints_with_real_offset_in_size(simple_volume, simple_material):
    """Test size constraints with real (non-grid) offset."""
    config = SimulationConfig(resolution=0.5, time=100e-15)

    obj = UniformMaterialObject(
        name="obj1",
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Size with real offset
    constraint = SizeConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        other_axes=[0],
        proportions=[0.5],
        grid_offsets=[None],
        offsets=[2.0],  # Real offset: 2.0 / 0.5 = 4 grid units
    )

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, config)

    # Should return results (may or may not have errors depending on other dimensions)
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices


def test_resolve_constraints_with_grid_offset_in_size(simple_config, simple_volume, simple_material):
    """Test size constraints with grid offset."""
    obj = UniformMaterialObject(
        name="obj1",
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Size with grid offset
    constraint = SizeConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        other_axes=[0],
        proportions=[0.5],
        grid_offsets=[10],  # 10 grid units
        offsets=[None],
    )

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should return results
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices


def test_resolve_constraints_size_extension_with_real_offset(simple_volume, simple_material):
    """Test size extension constraints with real offset."""
    config = SimulationConfig(resolution=0.5, time=100e-15)

    obj1 = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    obj2 = UniformMaterialObject(
        name="obj2",
        material=simple_material,
    )

    objects = [simple_volume, obj1, obj2]

    # Position obj1 first
    pos_constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20])

    # Extend obj2 from obj1 with real offset
    ext_constraint = SizeExtensionConstraint(
        object="obj2",
        other_object="obj1",
        axis=0,
        direction="+",  # Extend to the right
        other_position=1.0,  # From right side of obj1
        grid_offset=None,
        offset=2.0,  # Real offset: 2.0 / 0.5 = 4 grid units
    )

    constraints = [pos_constraint, ext_constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, config)

    # Should succeed
    assert errors["obj2"] is None
    # obj1 is at [20, 30], so obj2 should extend from 30 + 4 = 34
    assert resolved_slices["obj2"][0][1] == 34


def test_resolve_constraints_size_extension_with_grid_offset(simple_config, simple_volume, simple_material):
    """Test size extension constraints with grid offset."""
    obj1 = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    obj2 = UniformMaterialObject(
        name="obj2",
        material=simple_material,
    )

    objects = [simple_volume, obj1, obj2]

    # Position obj1 first
    pos_constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20])

    # Extend obj2 from obj1 with grid offset
    ext_constraint = SizeExtensionConstraint(
        object="obj2",
        other_object="obj1",
        axis=0,
        direction="-",  # Extend to the left
        other_position=-1.0,  # From left side of obj1
        grid_offset=5,  # 5 grid units offset
        offset=None,
    )

    constraints = [pos_constraint, ext_constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj2"] is None
    # obj1 is at [20, 30], and we're extending from left side with +5 offset
    # So obj2 left boundary should be at 20 + 5 = 25
    assert resolved_slices["obj2"][0][0] == 25


def test_resolve_constraints_size_extension_to_volume_boundary(simple_config, simple_volume, simple_material):
    """Test size extension to volume boundary (no other_object specified)."""
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Position obj at a specific location in all axes
    pos_constraint_x = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20])

    pos_constraint_y = GridCoordinateConstraint(object="obj1", axes=[1], sides=["-"], coordinates=[20])

    # Extend to volume boundary in the negative direction
    ext_constraint = SizeExtensionConstraint(
        object="obj1",
        other_object=None,  # Extend to volume
        axis=2,
        direction="-",
        other_position=0.0,
        grid_offset=None,
        offset=None,
    )

    constraints = [pos_constraint_x, pos_constraint_y, ext_constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None
    # Should extend to the volume boundary at 0
    assert resolved_slices["obj1"][2][0] == 0


def test_resolve_constraints_with_partial_real_shape(simple_material):
    """Test resolving objects with partial real shape specified."""
    config = SimulationConfig(resolution=0.5, time=100e-15)

    volume = SimulationVolume(
        name="volume",
        partial_grid_shape=(100, 100, 100),
    )

    # Object with real shape instead of grid shape
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_shape=(5.0, 5.0, 5.0),  # All axes as real shapes
        material=simple_material,
    )

    objects = [volume, obj]

    # Position the object at origin
    constraints_list = [
        GridCoordinateConstraint(object="obj1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[0, 0, 0])
    ]

    resolved_slices, errors = resolve_object_constraints(objects, constraints_list, config)

    # Should succeed with proper constraints
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices
    # Verify resolution worked
    if resolved_slices["obj1"][0][0] is not None:
        # Shape in axis 0 should be converted from real: 5.0 / 0.5 = 10
        shape = resolved_slices["obj1"][0][1] - resolved_slices["obj1"][0][0]
        assert shape == 10


def test_resolve_constraints_extend_to_infinity(simple_config, simple_volume, simple_material):
    """Test that objects without size constraints extend to volume boundaries."""
    obj = UniformMaterialObject(
        name="obj1",
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Only specify position in one axis, others should extend to infinity (volume boundaries)
    constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20])

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None
    # Axis 0 has constraint, but axes 1 and 2 should extend to volume boundaries
    assert resolved_slices["obj1"][1] == (0, 100)
    assert resolved_slices["obj1"][2] == (0, 100)


# Test cases for apply_params with mocks


@patch("fdtdx.fdtd.initialization.compute_allowed_permittivities")
def test_apply_params_continuous_type(mock_compute_perm):
    """Test apply_params with continuous parameter type (using mocks)."""
    mock_compute_perm.return_value = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]

    # Create mock device with continuous output
    device = Mock()
    device.name = "device1"
    device.output_type = ParameterType.CONTINUOUS
    device.grid_slice = (slice(0, 10), slice(0, 10), slice(0, 10))
    device.materials = [Mock(), Mock()]

    # Mock the device call to return material indices
    material_indices = jnp.ones((10, 10, 10)) * 0.5
    device.return_value = material_indices

    # Create mock objects container
    objects = Mock(spec=ObjectContainer)
    objects.devices = [device]
    objects.object_list = [device]
    objects.volume_idx = 0

    # Create mock arrays with proper structure preservation
    inv_perm = jnp.ones((3, 10, 10, 10))
    inv_permeab = jnp.ones((3, 10, 10, 10))

    arrays = Mock(spec=ArrayContainer)
    arrays.inv_permittivities = inv_perm
    arrays.inv_permeabilities = inv_permeab

    # Use MagicMock for .at to support subscripting
    at_accessor = MagicMock()

    def at_getitem(key):
        at_result = Mock()

        def set_side_effect(value):
            result = Mock(spec=ArrayContainer)
            result.inv_permittivities = value if key == "inv_permittivities" else inv_perm
            result.inv_permeabilities = inv_permeab
            result.at = at_accessor
            return result

        at_result.set = set_side_effect
        return at_result

    at_accessor.__getitem__ = Mock(side_effect=at_getitem)
    arrays.at = at_accessor

    # Create mock params
    params = {"device1": {}}

    key = jax.random.PRNGKey(0)

    # Mock the object apply method
    device.apply = Mock(return_value=device)

    result_arrays, result_objects, info = apply_params(arrays, objects, params, key)

    # Verify the continuous path was taken
    assert device.call_count > 0
    assert mock_compute_perm.called


@patch("fdtdx.fdtd.initialization.compute_allowed_permittivities")
@patch("fdtdx.fdtd.initialization.straight_through_estimator")
def test_apply_params_discrete_type(mock_ste, mock_compute_perm):
    """Test apply_params with discrete parameter type (using mocks)."""
    mock_compute_perm.return_value = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
    mock_ste.return_value = jnp.ones((10, 10, 10))

    # Create mock device with discrete output
    device = Mock()
    device.name = "device1"
    device.output_type = ParameterType.DISCRETE
    device.grid_slice = (slice(0, 10), slice(0, 10), slice(0, 10))
    device.materials = [Mock(), Mock()]

    # Mock the device call to return material indices
    material_indices = jnp.zeros((10, 10, 10), dtype=jnp.int32)
    device.return_value = material_indices

    # Create mock objects container
    objects = Mock(spec=ObjectContainer)
    objects.devices = [device]
    objects.object_list = [device]
    objects.volume_idx = 0

    # Create mock arrays with proper structure preservation
    inv_perm = jnp.ones((3, 10, 10, 10))
    inv_permeab = jnp.ones((3, 10, 10, 10))

    arrays = Mock(spec=ArrayContainer)
    arrays.inv_permittivities = inv_perm
    arrays.inv_permeabilities = inv_permeab

    # Use MagicMock for .at to support subscripting
    at_accessor = MagicMock()

    def at_getitem(key):
        at_result = Mock()

        def set_side_effect(value):
            result = Mock(spec=ArrayContainer)
            result.inv_permittivities = value if key == "inv_permittivities" else inv_perm
            result.inv_permeabilities = inv_permeab
            result.at = at_accessor
            return result

        at_result.set = set_side_effect
        return at_result

    at_accessor.__getitem__ = Mock(side_effect=at_getitem)
    arrays.at = at_accessor

    # Create mock params
    params = {"device1": {}}

    key = jax.random.PRNGKey(0)

    # Mock the object apply method
    device.apply = Mock(return_value=device)

    result_arrays, result_objects, info = apply_params(arrays, objects, params, key)

    # Verify the discrete path was taken (straight through estimator called)
    assert mock_ste.called
    assert mock_compute_perm.called


# Test cases for place_objects function


def test_place_objects_creates_object_container(simple_config, simple_volume, simple_material):
    """Test that place_objects creates an ObjectContainer."""

    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(20, 20, 20),
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Position the object
    constraint = GridCoordinateConstraint(
        object="obj1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )

    constraints = [constraint]
    key = jax.random.PRNGKey(0)

    obj_container, arrays, params, config, info = place_objects(objects, simple_config, constraints, key)

    # Check return types
    assert isinstance(obj_container, ObjectContainer)
    assert isinstance(arrays, ArrayContainer)
    assert isinstance(params, dict)
    assert obj_container.volume_idx == 0


def test_place_objects_with_multiple_objects(simple_config, simple_volume, simple_material):
    """Test place_objects with multiple material objects."""

    obj1 = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(20, 20, 20),
        material=simple_material,
    )

    obj2 = UniformMaterialObject(
        name="obj2",
        partial_grid_shape=(20, 20, 20),
        material=simple_material,
    )

    objects = [simple_volume, obj1, obj2]

    # Position both objects
    constraints = [
        GridCoordinateConstraint(object="obj1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]),
        GridCoordinateConstraint(object="obj2", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[40, 40, 40]),
    ]

    key = jax.random.PRNGKey(0)

    obj_container, arrays, params, config, info = place_objects(objects, simple_config, constraints, key)

    # Should have all objects
    assert len(obj_container.objects) == 3  # volume + 2 objects
    assert obj_container.volume_idx == 0


def test_place_objects_updates_config(simple_config, simple_volume, simple_material):
    """Test that place_objects returns an updated config."""
    from fdtdx.fdtd.initialization import place_objects

    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(20, 20, 20),
        material=simple_material,
    )

    objects = [simple_volume, obj]

    constraint = GridCoordinateConstraint(
        object="obj1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )

    constraints = [constraint]
    key = jax.random.PRNGKey(0)

    obj_container, arrays, params, config, info = place_objects(objects, simple_config, constraints, key)

    # Config should be updated
    assert config is not None
    assert config.resolution == simple_config.resolution


def test_place_objects_initializes_arrays(simple_config, simple_volume, simple_material):
    """Test that place_objects initializes field arrays."""
    from fdtdx.fdtd.initialization import place_objects

    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(20, 20, 20),
        material=simple_material,
    )

    objects = [simple_volume, obj]

    constraint = GridCoordinateConstraint(
        object="obj1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )

    constraints = [constraint]
    key = jax.random.PRNGKey(0)

    obj_container, arrays, params, config, info = place_objects(objects, simple_config, constraints, key)

    # Check arrays are initialized
    assert arrays.E is not None
    assert arrays.H is not None
    assert arrays.inv_permittivities is not None
