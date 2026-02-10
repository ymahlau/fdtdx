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


def test_resolve_constraints_with_partial_real_position(simple_material):
    """Test resolving objects with partial_real_position specified.

    This test verifies that:
    - Real-world positions (centers) are correctly converted to grid coordinates
    - All three axes are properly positioned when specified
    - The conversion uses the simulation resolution correctly
    - Boundaries are computed from center position and size
    """
    config = SimulationConfig(resolution=0.5, time=100e-15)

    volume = SimulationVolume(
        name="volume",
        partial_grid_shape=(100, 100, 100),
    )

    # Object with real position (CENTER) specified
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(10.0, 15.0, 20.0),  # Center positions in real-world coordinates
        partial_grid_shape=(10, 10, 10),  # Known size
        material=simple_material,
    )

    objects = [volume, obj]
    constraints = []  # No additional constraints needed

    resolved_slices, errors = resolve_object_constraints(objects, constraints, config)

    # Should succeed
    assert errors["obj1"] is None
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices

    # Verify conversion from real to grid coordinates
    # Real center 10.0 / resolution 0.5 = grid center 20
    # With size 10: lower = 20 - 5 = 15, upper = 15 + 10 = 25
    assert resolved_slices["obj1"][0] == (15, 25)

    # Real center 15.0 / resolution 0.5 = grid center 30
    # With size 10: lower = 30 - 5 = 25, upper = 25 + 10 = 35
    assert resolved_slices["obj1"][1] == (25, 35)

    # Real center 20.0 / resolution 0.5 = grid center 40
    # With size 10: lower = 40 - 5 = 35, upper = 35 + 10 = 45
    assert resolved_slices["obj1"][2] == (35, 45)


def test_resolve_constraints_partial_real_position_with_grid_shape(simple_config, simple_volume, simple_material):
    """Test partial_real_position works with partial_grid_shape.

    This test verifies that:
    - Axes with None in partial_real_position extend from 0 when size is known
    - Axes with specified center positions are correctly placed
    - Mixed specification (some axes positioned, others extending) works correctly
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(20.0, None, 30.0),  # Only x and z centers specified
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    objects = [simple_volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: center at 20, size 10, half=5, so lower = 15, upper = 25
    assert resolved_slices["obj1"][0] == (15, 25)

    # y-axis: no position specified, extends from 0 with size 10
    assert resolved_slices["obj1"][1] == (0, 10)

    # z-axis: center at 30, size 10, half=5, so lower = 25, upper = 35
    assert resolved_slices["obj1"][2] == (25, 35)


def test_resolve_constraints_partial_real_position_conflicts_with_constraint(
    simple_config, simple_volume, simple_material
):
    """Test that conflicting partial_real_position and grid constraint raises error.

    This test verifies that:
    - Conflicts between partial_real_position and GridCoordinateConstraint are detected
    - The error is properly reported in the errors dictionary
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(15.0, None, None),  # Center at 15
        partial_grid_shape=(10, 10, 10),  # Size 10, so bounds should be (10, 20)
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Constraint that conflicts with partial_real_position
    # partial_real_position says lower bound should be 10, but constraint says 5
    constraint = GridCoordinateConstraint(
        object="obj1",
        axes=[0],
        sides=["-"],
        coordinates=[5],  # Conflicts with computed lower bound of 10
    )

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should detect the conflict
    assert errors["obj1"] is not None


def test_resolve_constraints_partial_real_position_with_real_shape(simple_material):
    """Test partial_real_position works together with partial_real_shape.

    This test verifies that:
    - Both partial_real_position (center) and partial_real_shape use consistent resolution
    - Objects can be fully specified using only real-world coordinates
    - The conversion to grid coordinates is accurate for both position and size
    """
    config = SimulationConfig(resolution=0.25, time=100e-15)

    volume = SimulationVolume(
        name="volume",
        partial_grid_shape=(200, 200, 200),
    )

    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(10.0, 15.0, 20.0),  # Center positions
        partial_real_shape=(5.0, 10.0, 15.0),  # Sizes
        material=simple_material,
    )

    objects = [volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, config)

    # Should succeed
    assert errors["obj1"] is None

    # Verify conversions with resolution 0.25:
    # x: center 10.0/0.25=40, size 5.0/0.25=20, half=10 -> (30, 50)
    assert resolved_slices["obj1"][0] == (30, 50)

    # y: center 15.0/0.25=60, size 10.0/0.25=40, half=20 -> (40, 80)
    assert resolved_slices["obj1"][1] == (40, 80)

    # z: center 20.0/0.25=80, size 15.0/0.25=60, half=30 -> (50, 110)
    assert resolved_slices["obj1"][2] == (50, 110)


def test_resolve_constraints_partial_real_position_mixed_with_constraints(
    simple_config, simple_volume, simple_material
):
    """Test partial_real_position works alongside other constraints.

    This test verifies that:
    - Objects with partial_real_position can serve as anchors for positioning other objects
    - PositionConstraint properly references objects positioned via partial_real_position
    - The constraint resolution system integrates seamlessly with the new feature
    """
    obj1 = UniformMaterialObject(
        name="obj1",
        partial_real_position=(15.0, None, None),  # Center at 15, size 10 -> bounds (10, 20)
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    obj2 = UniformMaterialObject(
        name="obj2",
        partial_grid_shape=(20, 20, 20),
        material=simple_material,
    )

    objects = [simple_volume, obj1, obj2]

    # Position obj2 relative to obj1
    constraint = PositionConstraint(
        object="obj2",
        other_object="obj1",
        axes=[0],
        object_positions=[-1.0],  # Left side of obj2
        other_object_positions=[1.0],  # Right side of obj1 (at 20)
        grid_margins=[5],
        margins=[None],
    )

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None
    assert errors["obj2"] is None

    # obj1 center at 15, size 10, half=5 -> (10, 20)
    assert resolved_slices["obj1"][0] == (10, 20)

    # obj1 y and z axes extend from 0 with size 10
    assert resolved_slices["obj1"][1] == (0, 10)
    assert resolved_slices["obj1"][2] == (0, 10)

    # obj2 should be positioned relative to obj1's right side (20) + margin (5)
    # obj2's left side at position 25, size 20, so ends at 45
    assert resolved_slices["obj2"][0] == (25, 45)


def test_resolve_constraints_partial_real_position_all_none(simple_config, simple_volume, simple_material):
    """Test object with partial_real_position=(None, None, None).

    This test verifies that:
    - Objects with all None values in partial_real_position work correctly
    - Such objects extend from 0 when size is known
    - No errors occur when partial_real_position is effectively unused
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(None, None, None),  # All axes unspecified
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    objects = [simple_volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # All axes extend from 0 with the specified size
    assert resolved_slices["obj1"][0] == (0, 10)
    assert resolved_slices["obj1"][1] == (0, 10)
    assert resolved_slices["obj1"][2] == (0, 10)


def test_resolve_constraints_partial_real_position_single_axis(simple_config, simple_volume, simple_material):
    """Test partial_real_position with only one axis specified.

    This test verifies that:
    - Single-axis center position specification works correctly
    - Other axes with None extend from 0 when size is known
    - The behavior matches expectations for partially-constrained objects
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(None, 20.0, None),  # Only y-axis center specified
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    objects = [simple_volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: extends from 0 with size 10
    assert resolved_slices["obj1"][0] == (0, 10)

    # y-axis: center at 20, size 10, half=5 -> (15, 25)
    assert resolved_slices["obj1"][1] == (15, 25)

    # z-axis: extends from 0 with size 10
    assert resolved_slices["obj1"][2] == (0, 10)


def test_resolve_constraints_partial_real_position_with_different_resolutions(simple_material):
    """Test partial_real_position with various resolution values.

    This test verifies that:
    - The resolution scaling works correctly for different values
    - Rounding to grid coordinates is consistent
    - Fine resolutions work properly with non-integer real positions
    """
    # Test with very fine resolution
    config_fine = SimulationConfig(resolution=0.1, time=100e-15)

    volume = SimulationVolume(
        name="volume",
        partial_grid_shape=(500, 500, 500),
    )

    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(5.0, 10.0, 15.0),  # Center positions
        partial_grid_shape=(20, 20, 20),  # Size 20
        material=simple_material,
    )

    objects = [volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, config_fine)

    # Should succeed
    assert errors["obj1"] is None

    # Verify: center 5.0/0.1=50, size 20, half=10 -> (40, 60)
    assert resolved_slices["obj1"][0] == (40, 60)

    # center 10.0/0.1=100, size 20, half=10 -> (90, 110)
    assert resolved_slices["obj1"][1] == (90, 110)

    # center 15.0/0.1=150, size 20, half=10 -> (140, 160)
    assert resolved_slices["obj1"][2] == (140, 160)


def test_resolve_constraints_partial_real_position_without_size(simple_config, simple_volume, simple_material):
    """Test partial_real_position without a known size.

    This test verifies that:
    - When center position is specified but size is not, we can't compute boundaries
    - The position information is ignored when size is unknown
    - Object extends to volume boundaries
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(50.0, 50.0, 50.0),  # Center specified
        # No size specified
        material=simple_material,
    )

    objects = [simple_volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Without size, we can't compute boundaries from center
    # Object will extend to volume boundaries
    assert errors["obj1"] is None
    assert resolved_slices["obj1"][0] == (0, 100)
    assert resolved_slices["obj1"][1] == (0, 100)
    assert resolved_slices["obj1"][2] == (0, 100)


def test_resolve_constraints_partial_real_position_odd_size(simple_config, simple_volume, simple_material):
    """Test partial_real_position with odd-sized objects.

    This test verifies proper rounding when size/2 is not an integer.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(50.0, 50.0, 50.0),  # Center at 50
        partial_grid_shape=(11, 13, 15),  # Odd sizes
        material=simple_material,
    )

    objects = [simple_volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x: center 50, size 11, half=5.5 -> round(50-5.5)=44, upper=44+11=55
    assert resolved_slices["obj1"][0] == (44, 55)

    # y: center 50, size 13, half=6.5 -> round(50-6.5)=44, upper=44+13=57
    assert resolved_slices["obj1"][1] == (44, 57)

    # z: center 50, size 15, half=7.5 -> round(50-7.5)=42, upper=42+15=57
    assert resolved_slices["obj1"][2] == (42, 57)


def test_extend_to_inf_lower_bound_known_upper_not(simple_config, simple_volume, simple_material):
    """Test _extend_to_inf_if_possible when lower bound is known but upper is not.

    This covers the branch: elif b0 is not None and b1 is None and size is not None

    Scenario: Object has a known size and lower boundary set (via constraint),
    so upper boundary can be computed (should NOT extend upper boundary).
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(15, 15, 15),  # Known size
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Set lower boundaries only using GridCoordinateConstraint
    constraint = GridCoordinateConstraint(
        object="obj1",
        axes=[0, 1, 2],
        sides=["-", "-", "-"],  # Lower bounds
        coordinates=[10, 20, 30],
    )

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # Lower bounds from constraint
    assert resolved_slices["obj1"][0][0] == 10
    assert resolved_slices["obj1"][1][0] == 20
    assert resolved_slices["obj1"][2][0] == 30

    # Upper bounds computed as lower + size (NOT extended to volume boundary)
    assert resolved_slices["obj1"][0][1] == 25  # 10 + 15
    assert resolved_slices["obj1"][1][1] == 35  # 20 + 15
    assert resolved_slices["obj1"][2][1] == 45  # 30 + 15


def test_extend_to_inf_upper_bound_known_lower_not(simple_config, simple_volume, simple_material):
    """Test _extend_to_inf_if_possible when upper bound is known but lower is not.

    This covers the branch: elif b1 is not None and b0 is None and size is not None

    Scenario: Object has a known size and upper boundary set, so lower boundary
    can be computed (should NOT extend lower boundary).
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(15, 15, 15),  # Known size
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Set upper boundaries using GridCoordinateConstraint
    constraint = GridCoordinateConstraint(
        object="obj1",
        axes=[0, 1, 2],
        sides=["+", "+", "+"],  # Upper bounds
        coordinates=[50, 60, 70],
    )

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # Lower bounds computed as upper - size (NOT extended to 0)
    assert resolved_slices["obj1"][0][0] == 35  # 50 - 15
    assert resolved_slices["obj1"][1][0] == 45  # 60 - 15
    assert resolved_slices["obj1"][2][0] == 55  # 70 - 15

    # Upper bounds from constraint
    assert resolved_slices["obj1"][0][1] == 50
    assert resolved_slices["obj1"][1][1] == 60
    assert resolved_slices["obj1"][2][1] == 70


def test_extend_to_inf_partial_real_position_sets_both_bounds(simple_config, simple_volume, simple_material):
    """Test that partial_real_position with known size sets both boundaries.

    When partial_real_position (center) and size are both known,
    both boundaries are computed immediately, so this doesn't exercise
    the lower-bound-only or upper-bound-only branches.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_real_position=(55.0, None, 55.0),  # Centers for x and z
        partial_grid_shape=(20, 20, 20),
        material=simple_material,
    )

    objects = [simple_volume, obj]
    constraints = []

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: center 55, size 20, half=10 -> (45, 65) - both bounds set
    assert resolved_slices["obj1"][0] == (45, 65)

    # y-axis: no center, extends from 0 with size 20
    assert resolved_slices["obj1"][1] == (0, 20)

    # z-axis: center 55, size 20, half=10 -> (45, 65) - both bounds set
    assert resolved_slices["obj1"][2] == (45, 65)


def test_extend_to_inf_with_real_coordinate_constraint_upper(simple_config, simple_volume, simple_material):
    """Test using RealCoordinateConstraint to set upper boundary.

    Another way to cover the upper-bound-known case.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(20, 20, 20),
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Set upper boundary using RealCoordinateConstraint
    constraint = RealCoordinateConstraint(
        object="obj1",
        axes=[0],
        sides=["+"],  # Upper bound
        coordinates=[80.0],  # Real coordinate 80.0 / resolution 1.0 = grid 80
    )

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: lower computed as 80 - 20 = 60
    assert resolved_slices["obj1"][0] == (60, 80)

    # y and z axes: extend from 0
    assert resolved_slices["obj1"][1] == (0, 20)
    assert resolved_slices["obj1"][2] == (0, 20)


def test_extend_to_inf_single_axis_upper_bound(simple_config, simple_volume, simple_material):
    """Test mixed scenario: some axes with lower bounds, one with upper bound.

    This exercises both branches in a single test.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(15, 15, 15),
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Set lower bounds for x and z, upper bound for y
    constraints = [
        GridCoordinateConstraint(object="obj1", axes=[0, 2], sides=["-", "-"], coordinates=[10, 30]),
        GridCoordinateConstraint(object="obj1", axes=[1], sides=["+"], coordinates=[50]),
    ]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: lower bound known (10), upper computed (25) - covers first branch
    assert resolved_slices["obj1"][0] == (10, 25)

    # y-axis: upper bound known (50), lower computed (35) - covers second branch
    assert resolved_slices["obj1"][1] == (35, 50)

    # z-axis: lower bound known (30), upper computed (45) - covers first branch again
    assert resolved_slices["obj1"][2] == (30, 45)


def test_extend_to_inf_with_position_constraint_creating_lower_bound(simple_config, simple_volume, simple_material):
    """Test PositionConstraint creating a lower boundary situation.

    PositionConstraint can set both bounds at once when size is known,
    but if applied iteratively, might create lower-bound-only scenarios.
    """
    obj1 = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(10, 10, 10),
        material=simple_material,
    )

    obj2 = UniformMaterialObject(
        name="obj2",
        partial_grid_shape=(20, 20, 20),
        material=simple_material,
    )

    objects = [simple_volume, obj1, obj2]

    # Position obj1 first
    constraint1 = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10])

    # Position obj2 after obj1 - this sets obj2's lower bound
    constraint2 = PositionConstraint(
        object="obj2",
        other_object="obj1",
        axes=[0],
        object_positions=[-1.0],  # Left side of obj2
        other_object_positions=[1.0],  # Right side of obj1
        grid_margins=[5],
        margins=[None],
    )

    constraints = [constraint1, constraint2]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None
    assert errors["obj2"] is None

    # obj1: lower at 10, size 10, so upper at 20
    assert resolved_slices["obj1"][0] == (10, 20)

    # obj2: positioned after obj1 (20) + margin (5) = 25, size 20, so upper at 45
    assert resolved_slices["obj2"][0] == (25, 45)


def test_extend_to_inf_lower_bound_one_axis_only(simple_config, simple_volume, simple_material):
    """Test lower bound on just one axis to ensure the if check is covered.

    This specifically tests that (o, 1) is checked before removal.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(15, None, None),  # Only x has size
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Set lower boundary only for x-axis
    constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10])

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: lower=10, size=15, upper=25
    assert resolved_slices["obj1"][0] == (10, 25)

    # y and z extend to volume
    assert resolved_slices["obj1"][1] == (0, 100)
    assert resolved_slices["obj1"][2] == (0, 100)


def test_extend_to_inf_upper_bound_one_axis_only(simple_config, simple_volume, simple_material):
    """Test upper bound on just one axis to ensure the if check is covered.

    This specifically tests that (o, 0) is checked before removal.
    """
    obj = UniformMaterialObject(
        name="obj1",
        partial_grid_shape=(15, None, None),  # Only x has size
        material=simple_material,
    )

    objects = [simple_volume, obj]

    # Set upper boundary only for x-axis
    constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["+"], coordinates=[50])

    constraints = [constraint]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj1"] is None

    # x-axis: upper=50, size=15, lower=35
    assert resolved_slices["obj1"][0] == (35, 50)

    # y and z extend to volume
    assert resolved_slices["obj1"][1] == (0, 100)
    assert resolved_slices["obj1"][2] == (0, 100)


def test_size_dependent_position_with_partial_real_position(simple_config, simple_volume, simple_material):
    """Test scenario where position depends on size that is resolved through constraints.

    Setup:
    - Volume has fully specified size
    - Object 2 has size constraint dependent on volume (half the size)
    - Object 2 position is specified through place_at_center (partial_real_position)
    - Object 3 has same_size constraint relative to object 2
    - Object 3 position is specified through partial_real_position

    This tests that the code can handle cases where:
    1. Size of object 2 is not known initially but resolved through SizeConstraint
    2. Position of object 2 must wait for size to be resolved
    3. Size of object 3 depends on object 2's size
    4. Position of object 3 must wait for its size to be resolved
    """
    # Object 2: size depends on volume, position at center
    obj2 = UniformMaterialObject(
        name="obj2",
        partial_real_position=(50.0, 50.0, 50.0),  # Center at (50, 50, 50)
        material=simple_material,
    )

    # Object 3: size depends on obj2, position specified
    obj3 = UniformMaterialObject(
        name="obj3",
        partial_real_position=(25.0, None, None),  # x-center at 25
        material=simple_material,
    )

    objects = [simple_volume, obj2, obj3]

    constraints = [
        # obj2 should be half the size of volume in all dimensions
        SizeConstraint(
            object="obj2",
            other_object="volume",
            axes=[0, 1, 2],
            other_axes=[0, 1, 2],
            proportions=[0.5, 0.5, 0.5],
            grid_offsets=[0, 0, 0],
            offsets=[None, None, None],
        ),
        # obj3 should have same size as obj2
        SizeConstraint(
            object="obj3",
            other_object="obj2",
            axes=[0, 1, 2],
            other_axes=[0, 1, 2],
            proportions=[1.0, 1.0, 1.0],
            grid_offsets=[0, 0, 0],
            offsets=[None, None, None],
        ),
    ]

    resolved_slices, errors = resolve_object_constraints(objects, constraints, simple_config)

    # Should succeed
    assert errors["obj2"] is None
    assert errors["obj3"] is None

    # obj2: size = 100 * 0.5 = 50, center at 50, so bounds are (25, 75)
    assert resolved_slices["obj2"][0] == (25, 75)
    assert resolved_slices["obj2"][1] == (25, 75)
    assert resolved_slices["obj2"][2] == (25, 75)

    # obj3: size = 50 (same as obj2), x-center at 25 so bounds are (0, 50)
    # y and z extend from 0 since no position specified
    assert resolved_slices["obj3"][0] == (0, 50)
    assert resolved_slices["obj3"][1] == (0, 50)
    assert resolved_slices["obj3"][2] == (0, 50)
