from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ObjectContainer
from fdtdx.fdtd.initialization import (
    _apply_constraints_iteratively,
    _apply_grid_coordinate_constraint,
    _apply_position_constraint,
    _apply_real_coordinate_constraint,
    _apply_size_constraint,
    _apply_size_extension_constraint,
    _check_objects_names_from_constraints,
    _extend_to_inf_if_possible,
    _handle_unresolved_objects,
    _init_arrays,
    _resolve_static_shapes,
    _resolve_volume_name,
    _update_grid_shapes_from_slices,
    _update_grid_slices_from_shapes,
    place_objects,
    resolve_object_constraints,
)
from fdtdx.objects.object import (
    GridCoordinateConstraint,
    PositionConstraint,
    RealCoordinateConstraint,
    SimulationObject,
    SizeConstraint,
    SizeExtensionConstraint,
)
from fdtdx.objects.static_material.static import (
    SimulationVolume,
    StaticMultiMaterialObject,
    UniformMaterialObject,
)


# Test for duplicate object names
def test_resolve_object_constraints_duplicate_names():
    """Test that duplicate object names raise an exception."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    obj1 = Mock(spec=SimulationVolume)
    obj1.name = "volume"
    obj2 = Mock(spec=UniformMaterialObject)
    obj2.name = "volume"  # Duplicate name

    objects = [obj1, obj2]
    constraints = []

    with pytest.raises(Exception, match="Duplicate object names detected"):
        resolve_object_constraints(objects, constraints, config)


# Test for invalid object types
def test_resolve_object_constraints_invalid_objects():
    """Test that invalid object types raise an exception."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    obj1 = Mock(spec=SimulationVolume)
    obj1.name = "volume"

    # Create an object without proper SimulationObject type
    invalid_obj = Mock()
    invalid_obj.name = "invalid"

    objects = [obj1, invalid_obj]
    constraints = []

    with pytest.raises((ValueError, TypeError)):
        resolve_object_constraints(objects, constraints, config)


# Test for unknown object name in constraints
def test_check_objects_names_unknown_object():
    """Test that unknown object names in constraints raise an exception."""
    constraint = Mock(spec=PositionConstraint)
    constraint.object = "unknown_object"
    constraint.other_object = None

    constraints = [constraint]
    object_names = ["volume", "obj1"]

    with pytest.raises(ValueError, match="Unknown object name in constraint"):
        _check_objects_names_from_constraints(constraints, object_names)


# Test for unknown object name in other_object field
def test_check_objects_names_unknown_other_object():
    """Test that unknown object names in other_object field raise an exception."""
    constraint = Mock(spec=PositionConstraint)
    constraint.object = "obj1"
    constraint.other_object = "unknown_other"

    constraints = [constraint]
    object_names = ["volume", "obj1"]

    with pytest.raises(ValueError, match="Unknown object name in constraint"):
        _check_objects_names_from_constraints(constraints, object_names)


# Test for no SimulationVolume
def test_resolve_volume_name_no_volume():
    """Test that missing SimulationVolume raises an exception."""
    obj1 = Mock(spec=UniformMaterialObject)
    obj1.name = "obj1"

    object_map = {"obj1": obj1}

    with pytest.raises(ValueError, match="No SimulationVolume object found"):
        _resolve_volume_name(object_map)


# Test for multiple SimulationVolumes
def test_resolve_volume_name_multiple_volumes():
    """Test that multiple SimulationVolume objects raise an exception."""
    vol1 = Mock(spec=SimulationVolume)
    vol1.name = "volume1"
    vol2 = Mock(spec=SimulationVolume)
    vol2.name = "volume2"

    object_map = {"volume1": vol1, "volume2": vol2}

    with pytest.raises(ValueError, match="Multiple SimulationVolume objects found"):
        _resolve_volume_name(object_map)


# Test for inconsistent grid shape in _update_grid_slices_from_shapes
def test_update_grid_slices_inconsistent_shape():
    """Test inconsistent grid shape detection."""
    obj = Mock()
    obj.name = "obj1"
    object_map = {"obj1": obj}

    shape_dict = {"obj1": [10, None, None]}
    slice_dict = {"obj1": [[0, 5], [None, None], [None, None]]}  # 5-0=5, but shape is 10
    errors = {"obj1": None}

    resolved, slice_dict, errors = _update_grid_slices_from_shapes(object_map, shape_dict, slice_dict, errors)

    assert errors["obj1"] is not None
    assert "Inconsistent grid shape" in errors["obj1"]


# Test for inconsistent grid shape in _update_grid_shapes_from_slices
def test_update_grid_shapes_inconsistent_shape():
    """Test inconsistent grid shape detection when updating from slices."""
    obj = Mock()
    obj.name = "obj1"
    object_map = {"obj1": obj}

    shape_dict = {"obj1": [10, None, None]}
    slice_dict = {"obj1": [[0, 5], [None, None], [None, None]]}  # 5-0=5, but shape is 10
    errors = {"obj1": None}

    resolved, shape_dict, errors = _update_grid_shapes_from_slices(object_map, shape_dict, slice_dict, errors)

    assert errors["obj1"] is not None
    assert "Inconsistent grid shape" in errors["obj1"]


# Test for inconsistent grid coordinates in GridCoordinateConstraint
def test_apply_grid_coordinate_constraint_inconsistent():
    """Test that inconsistent grid coordinates raise an exception."""
    obj = Mock()
    obj.name = "obj1"
    object_map = {"obj1": obj}

    constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10])

    slice_dict = {"obj1": [[5, None], [None, None], [None, None]]}  # Already set to 5

    with pytest.raises(Exception, match="Inconsistent grid coordinates"):
        _apply_grid_coordinate_constraint(constraint, object_map, slice_dict)


# Test for inconsistent grid coordinates in RealCoordinateConstraint
def test_apply_real_coordinate_constraint_inconsistent():
    """Test that inconsistent real coordinates raise an exception."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    obj = Mock()
    obj.name = "obj1"
    object_map = {"obj1": obj}

    constraint = RealCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10.0])

    slice_dict = {"obj1": [[5, None], [None, None], [None, None]]}  # Already set to 5

    with pytest.raises(Exception, match="Inconsistent grid coordinates"):
        _apply_real_coordinate_constraint(constraint, object_map, slice_dict, config)


# Test for inconsistent position at lower bound
def test_apply_position_constraint_inconsistent_lower():
    """Test that inconsistent position at lower bound raises an exception."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    obj = Mock()
    obj.name = "obj1"
    other = Mock()
    other.name = "other"
    object_map = {"obj1": obj, "other": other}

    constraint = PositionConstraint(
        object="obj1",
        other_object="other",
        axes=[0],
        object_positions=[0.0],
        other_object_positions=[0.0],
        grid_margins=[0],
        margins=[0.0],
    )

    shape_dict = {"obj1": [10, None, None], "other": [20, None, None]}
    # Set lower bound to 5, but the constraint should place it at 5 (midpoint of other at 10)
    # So this won't conflict. Let's make it conflict by setting it to a different value
    slice_dict = {
        "obj1": [[15, None], [None, None], [None, None]],  # Already set to 15 (will conflict)
        "other": [[0, 20], [None, None], [None, None]],
    }

    with pytest.raises(Exception, match="Inconsistent grid shape.*lower bound"):
        _apply_position_constraint(constraint, object_map, config, shape_dict, slice_dict)


# Test for inconsistent position at upper bound
def test_apply_position_constraint_inconsistent_upper():
    """Test that inconsistent position at upper bound raises an exception."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    obj = Mock()
    obj.name = "obj1"
    other = Mock()
    other.name = "other"
    object_map = {"obj1": obj, "other": other}

    constraint = PositionConstraint(
        object="obj1",
        other_object="other",
        axes=[0],
        object_positions=[0.0],
        other_object_positions=[0.0],
        grid_margins=[0],
        margins=[0.0],
    )

    shape_dict = {"obj1": [10, None, None], "other": [20, None, None]}
    slice_dict = {
        "obj1": [[None, 20], [None, None], [None, None]],  # Already set to 20
        "other": [[0, 20], [None, None], [None, None]],
    }

    with pytest.raises(Exception, match="Inconsistent grid shape.*lower bound"):
        _apply_position_constraint(constraint, object_map, config, shape_dict, slice_dict)


# Test for inconsistent size constraint
def test_apply_size_constraint_inconsistent():
    """Test that inconsistent size constraint raises an exception."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    obj = Mock()
    obj.name = "obj1"
    other = Mock()
    other.name = "other"
    object_map = {"obj1": obj, "other": other}

    constraint = SizeConstraint(
        object="obj1",
        other_object="other",
        axes=[0],
        other_axes=[0],
        proportions=[1.0],
        grid_offsets=[0],
        offsets=[0.0],
    )

    shape_dict = {
        "obj1": [15, None, None],  # Already set to 15
        "other": [10, None, None],  # Would calculate to 10
    }

    with pytest.raises(Exception, match="Inconsistent grid shape"):
        _apply_size_constraint(constraint, object_map, config, shape_dict)


# Test for inconsistent size extension constraint
def test_apply_size_extension_constraint_inconsistent():
    """Test that inconsistent size extension constraint raises an exception."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    obj = Mock()
    obj.name = "obj1"
    other = Mock()
    other.name = "other"
    object_map = {"obj1": obj, "other": other}

    constraint = SizeExtensionConstraint(
        object="obj1", other_object="other", axis=0, direction="-", other_position=0.0, grid_offset=0, offset=0.0
    )

    slice_dict = {
        "obj1": [[5, None], [None, None], [None, None]],  # Already set to 5
        "other": [[10, 20], [None, None], [None, None]],  # Would calculate to 10
    }

    with pytest.raises(Exception, match="Inconsistent grid shape"):
        _apply_size_extension_constraint(constraint, object_map, config, slice_dict, "volume")


# Test for size extension constraint extending to volume boundary
def test_apply_size_extension_constraint_to_volume():
    """Test size extension constraint that extends to volume boundary."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    obj = Mock()
    obj.name = "obj1"
    volume = Mock()
    volume.name = "volume"
    object_map = {"obj1": obj, "volume": volume}

    constraint = SizeExtensionConstraint(
        object="obj1",
        other_object=None,  # Extends to volume
        axis=0,
        direction="+",
        other_position=0.0,
        grid_offset=None,
        offset=None,
    )

    slice_dict = {"obj1": [[None, None], [None, None], [None, None]], "volume": [[0, 100], [0, 100], [0, 100]]}

    resolved, slice_dict = _apply_size_extension_constraint(constraint, object_map, config, slice_dict, "volume")

    assert resolved is True
    assert slice_dict["obj1"][0][1] == 100


# Test for exception when volume not specified (should never happen)
def test_apply_size_extension_constraint_volume_not_specified():
    """Test exception when volume boundary is not specified."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    obj = Mock()
    obj.name = "obj1"
    volume = Mock()
    volume.name = "volume"
    object_map = {"obj1": obj, "volume": volume}

    constraint = SizeExtensionConstraint(
        object="obj1", other_object=None, axis=0, direction="-", other_position=0.0, grid_offset=None, offset=None
    )

    slice_dict = {
        "obj1": [[None, None], [None, None], [None, None]],
        "volume": [[None, None], [None, None], [None, None]],  # Not specified
    }

    with pytest.raises(Exception, match="Simulation volume not specified"):
        _apply_size_extension_constraint(constraint, object_map, config, slice_dict, "volume")


# Test for unknown constraint type
def test_apply_constraints_unknown_constraint_type():
    """Test that unknown constraint types are handled properly."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    volume = Mock(spec=SimulationVolume)
    volume.name = "volume"
    volume.partial_grid_shape = [100, 100, 100]
    volume.partial_real_shape = [None, None, None]

    objects = [volume]

    # Create an unknown constraint type
    unknown_constraint = Mock()
    unknown_constraint.object = "volume"

    constraints = [unknown_constraint]

    _, errors = _apply_constraints_iteratively(objects, constraints, config, max_iter=10)

    # Should have an error for the volume object
    assert errors["volume"] is not None
    assert "Unknown constraint type" in errors["volume"]


# Test _init_arrays with magnetic objects
def test_init_arrays_with_magnetic_objects():
    """Test _init_arrays when objects have magnetic properties."""
    config = Mock(spec=SimulationConfig)
    config.dtype = jnp.float32
    config.backend = "cpu"
    config.resolution = 1.0
    config.gradient_config = None

    # Create volume
    volume = Mock()
    volume.grid_shape = (10, 10, 10)

    # Create object with magnetic properties
    obj = Mock(spec=UniformMaterialObject)
    obj.placement_order = 1
    obj.grid_slice = (slice(0, 5), slice(0, 5), slice(0, 5))
    obj.material = Mock()
    obj.material.permittivity = 2.0
    obj.material.permeability = 1.5  # Magnetic
    obj.material.electric_conductivity = 0.0
    obj.material.magnetic_conductivity = 0.0

    # Create container
    objects = Mock(spec=ObjectContainer)
    objects.volume = volume
    objects.all_objects_non_magnetic = False  # Has magnetic objects
    objects.all_objects_non_electrically_conductive = True
    objects.all_objects_non_magnetically_conductive = True
    objects.static_material_objects = [obj]
    objects.detectors = []
    objects.boundary_objects = []
    objects.pml_objects = []

    with patch("fdtdx.fdtd.initialization.create_named_sharded_matrix") as mock_create:
        mock_create.return_value = jnp.zeros((10, 10, 10))

        arrays, updated_config, info = _init_arrays(objects, config)

        # Verify that inv_permeabilities is an array (not scalar)
        # The mock will be called multiple times for different arrays
        assert mock_create.call_count >= 3  # E, H, inv_permittivities, inv_permeabilities


# Test _init_arrays with electrically conductive objects
def test_init_arrays_with_electric_conductivity():
    """Test _init_arrays when objects have electric conductivity."""
    config = Mock(spec=SimulationConfig)
    config.dtype = jnp.float32
    config.backend = "cpu"
    config.resolution = 1.0
    config.gradient_config = None

    # Create volume
    volume = Mock()
    volume.grid_shape = (10, 10, 10)

    # Create object with electric conductivity
    obj = Mock(spec=UniformMaterialObject)
    obj.placement_order = 1
    obj.grid_slice = (slice(0, 5), slice(0, 5), slice(0, 5))
    obj.material = Mock()
    obj.material.permittivity = 2.0
    obj.material.permeability = 1.0
    obj.material.electric_conductivity = 0.5  # Has conductivity
    obj.material.magnetic_conductivity = 0.0

    # Create container
    objects = Mock(spec=ObjectContainer)
    objects.volume = volume
    objects.all_objects_non_magnetic = True
    objects.all_objects_non_electrically_conductive = False  # Has conductive objects
    objects.all_objects_non_magnetically_conductive = True
    objects.static_material_objects = [obj]
    objects.detectors = []
    objects.boundary_objects = []
    objects.pml_objects = []

    with patch("fdtdx.fdtd.initialization.create_named_sharded_matrix") as mock_create:
        mock_create.return_value = jnp.zeros((10, 10, 10))

        arrays, updated_config, info = _init_arrays(objects, config)

        # Should create electric_conductivity array
        assert mock_create.call_count >= 4  # E, H, inv_permittivities, electric_conductivity


# Test _init_arrays with magnetically conductive objects
def test_init_arrays_with_magnetic_conductivity():
    """Test _init_arrays when objects have magnetic conductivity."""
    config = Mock(spec=SimulationConfig)
    config.dtype = jnp.float32
    config.backend = "cpu"
    config.resolution = 1.0
    config.gradient_config = None

    # Create volume
    volume = Mock()
    volume.grid_shape = (10, 10, 10)

    # Create object with magnetic conductivity
    obj = Mock(spec=UniformMaterialObject)
    obj.placement_order = 1
    obj.grid_slice = (slice(0, 5), slice(0, 5), slice(0, 5))
    obj.material = Mock()
    obj.material.permittivity = 2.0
    obj.material.permeability = 1.5
    obj.material.electric_conductivity = 0.0
    obj.material.magnetic_conductivity = 0.3  # Has magnetic conductivity

    # Create container
    objects = Mock(spec=ObjectContainer)
    objects.volume = volume
    objects.all_objects_non_magnetic = False
    objects.all_objects_non_electrically_conductive = True
    objects.all_objects_non_magnetically_conductive = False  # Has magnetically conductive
    objects.static_material_objects = [obj]
    objects.detectors = []
    objects.boundary_objects = []
    objects.pml_objects = []

    with patch("fdtdx.fdtd.initialization.create_named_sharded_matrix") as mock_create:
        mock_create.return_value = jnp.zeros((10, 10, 10))

        arrays, updated_config, info = _init_arrays(objects, config)

        # Should create both inv_permeabilities and magnetic_conductivity
        assert mock_create.call_count >= 5


# Test _init_arrays with StaticMultiMaterialObject
@patch("fdtdx.fdtd.initialization.compute_allowed_permittivities")
@patch("fdtdx.fdtd.initialization.compute_allowed_permeabilities")
@patch("fdtdx.fdtd.initialization.compute_allowed_electric_conductivities")
@patch("fdtdx.fdtd.initialization.compute_allowed_magnetic_conductivities")
def test_init_arrays_with_multi_material_object(mock_mag_cond, mock_elec_cond, mock_permeab, mock_perm):
    """Test _init_arrays with StaticMultiMaterialObject."""
    mock_perm.return_value = [2.0, 4.0]
    mock_permeab.return_value = [1.0, 1.5]
    mock_elec_cond.return_value = [0.0, 0.5]
    mock_mag_cond.return_value = [0.0, 0.3]

    config = Mock(spec=SimulationConfig)
    config.dtype = jnp.float32
    config.backend = "cpu"
    config.resolution = 1.0
    config.gradient_config = None

    # Create volume
    volume = Mock()
    volume.grid_shape = (10, 10, 10)

    # Create multi-material object
    obj = Mock(spec=StaticMultiMaterialObject)
    obj.placement_order = 1
    obj.grid_slice = (slice(0, 5), slice(0, 5), slice(0, 5))
    obj.materials = [Mock(), Mock()]
    obj.get_material_mapping = Mock(return_value=jnp.zeros((5, 5, 5), dtype=jnp.int32))
    obj.get_voxel_mask_for_shape = Mock(return_value=jnp.ones((5, 5, 5)))

    # Create container
    objects = Mock(spec=ObjectContainer)
    objects.volume = volume
    objects.all_objects_non_magnetic = False
    objects.all_objects_non_electrically_conductive = False
    objects.all_objects_non_magnetically_conductive = False
    objects.static_material_objects = [obj]
    objects.detectors = []
    objects.boundary_objects = []
    objects.pml_objects = []

    with patch("fdtdx.fdtd.initialization.create_named_sharded_matrix") as mock_create:
        mock_create.return_value = jnp.zeros((10, 10, 10))

        arrays, updated_config, info = _init_arrays(objects, config)

        # Verify the methods were called
        assert obj.get_material_mapping.called
        assert obj.get_voxel_mask_for_shape.called


# Test _init_arrays with unknown object type
def test_init_arrays_with_unknown_object_type():
    """Test _init_arrays raises exception for unknown object type."""
    config = Mock(spec=SimulationConfig)
    config.dtype = jnp.float32
    config.backend = "cpu"
    config.resolution = 1.0
    config.gradient_config = None

    # Create volume
    volume = Mock()
    volume.grid_shape = (10, 10, 10)

    # Create unknown object type
    obj = Mock()
    obj.placement_order = 1

    # Create container
    objects = Mock(spec=ObjectContainer)
    objects.volume = volume
    objects.all_objects_non_magnetic = True
    objects.all_objects_non_electrically_conductive = True
    objects.all_objects_non_magnetically_conductive = True
    objects.static_material_objects = [obj]
    objects.detectors = []
    objects.boundary_objects = []
    objects.pml_objects = []

    with patch("fdtdx.fdtd.initialization.create_named_sharded_matrix") as mock_create:
        mock_create.return_value = jnp.zeros((10, 10, 10))

        with pytest.raises(Exception, match="Unknown object type"):
            _init_arrays(objects, config)


# Test _init_arrays with gradient recorder
def test_init_arrays_with_gradient_recorder():
    """Test _init_arrays when gradient recorder is configured."""
    config = Mock(spec=SimulationConfig)
    config.dtype = jnp.float32
    config.backend = "cpu"
    config.resolution = 1.0
    config.time_steps_total = 100

    # Create gradient config with recorder
    recorder = Mock()
    recorder.init_state = Mock(return_value=(recorder, {"state": "data"}))

    grad_config = Mock()
    grad_config.recorder = recorder
    grad_config.aset = Mock(return_value=grad_config)

    config.gradient_config = grad_config
    config.aset = Mock(return_value=config)

    # Create volume
    vol = Mock()
    vol.grid_shape = (10, 10, 10)

    # Create PML object
    pml = Mock()
    pml.name = "pml"
    pml.interface_grid_shape = Mock(return_value=(8, 8, 8))

    # Create container
    objects = Mock(spec=ObjectContainer)
    objects.volume = vol
    objects.all_objects_non_magnetic = True
    objects.all_objects_non_electrically_conductive = True
    objects.all_objects_non_magnetically_conductive = True
    objects.static_material_objects = []
    objects.detectors = []
    objects.boundary_objects = []
    objects.pml_objects = [pml]

    with patch("fdtdx.fdtd.initialization.create_named_sharded_matrix") as mock_create:
        mock_create.return_value = jnp.zeros((10, 10, 10))

        arrays, updated_config, info = _init_arrays(objects, config)

        # Verify recorder was initialized
        assert recorder.init_state.called
        assert arrays.recording_state is not None


# Test for real margin in position constraint
def test_apply_position_constraint_with_real_margin():
    """Test position constraint with real (non-grid) margin."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 0.5

    obj = Mock()
    obj.name = "obj1"
    other = Mock()
    other.name = "other"
    object_map = {"obj1": obj, "other": other}

    constraint = PositionConstraint(
        object="obj1",
        other_object="other",
        axes=[0],
        object_positions=[0.0],
        other_object_positions=[0.0],
        grid_margins=[None],
        margins=[2.0],  # Real margin, not grid margin
    )

    shape_dict = {"obj1": [10, None, None], "other": [20, None, None]}
    slice_dict = {"obj1": [[None, None], [None, None], [None, None]], "other": [[0, 20], [None, None], [None, None]]}

    resolved, slice_dict = _apply_position_constraint(constraint, object_map, config, shape_dict, slice_dict)

    assert resolved is True
    # Should calculate position with real margin converted to grid units


# Test for grid offset in size constraint
def test_apply_size_constraint_with_grid_offset():
    """Test size constraint with grid offset."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    obj = Mock()
    obj.name = "obj1"
    other = Mock()
    other.name = "other"
    object_map = {"obj1": obj, "other": other}

    constraint = SizeConstraint(
        object="obj1",
        other_object="other",
        axes=[0],
        other_axes=[0],
        proportions=[1.0],
        grid_offsets=[5],  # Grid offset
        offsets=[None],
    )

    shape_dict = {"obj1": [None, None, None], "other": [10, None, None]}

    resolved, shape_dict = _apply_size_constraint(constraint, object_map, config, shape_dict)

    assert resolved is True
    assert shape_dict["obj1"][0] == 15  # 10 * 1.0 + 5


# Test for real offset in size constraint
def test_apply_size_constraint_with_real_offset():
    """Test size constraint with real (non-grid) offset."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 0.5

    obj = Mock()
    obj.name = "obj1"
    other = Mock()
    other.name = "other"
    object_map = {"obj1": obj, "other": other}

    constraint = SizeConstraint(
        object="obj1",
        other_object="other",
        axes=[0],
        other_axes=[0],
        proportions=[1.0],
        grid_offsets=[None],
        offsets=[2.0],  # Real offset
    )

    shape_dict = {"obj1": [None, None, None], "other": [10, None, None]}

    resolved, shape_dict = _apply_size_constraint(constraint, object_map, config, shape_dict)

    assert resolved is True
    assert shape_dict["obj1"][0] == 14  # 10 * 1.0 + 2.0/0.5


# Test for grid offset in size extension constraint
def test_apply_size_extension_constraint_with_grid_offset():
    """Test size extension constraint with grid offset."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    obj = Mock()
    obj.name = "obj1"
    other = Mock()
    other.name = "other"
    object_map = {"obj1": obj, "other": other}

    constraint = SizeExtensionConstraint(
        object="obj1",
        other_object="other",
        axis=0,
        direction="-",
        other_position=0.0,
        grid_offset=5,  # Grid offset
        offset=None,
    )

    slice_dict = {"obj1": [[None, None], [None, None], [None, None]], "other": [[10, 20], [None, None], [None, None]]}

    resolved, slice_dict = _apply_size_extension_constraint(constraint, object_map, config, slice_dict, "volume")

    assert resolved is True
    assert slice_dict["obj1"][0][0] == 20  # 15 + 5


# Test for real offset in size extension constraint
def test_apply_size_extension_constraint_with_real_offset():
    """Test size extension constraint with real offset."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 0.5

    obj = Mock()
    obj.name = "obj1"
    other = Mock()
    other.name = "other"
    object_map = {"obj1": obj, "other": other}

    constraint = SizeExtensionConstraint(
        object="obj1",
        other_object="other",
        axis=0,
        direction="-",
        other_position=0.0,
        grid_offset=None,
        offset=2.0,  # Real offset
    )

    slice_dict = {"obj1": [[None, None], [None, None], [None, None]], "other": [[10, 20], [None, None], [None, None]]}

    resolved, slice_dict = _apply_size_extension_constraint(constraint, object_map, config, slice_dict, "volume")

    assert resolved is True
    assert slice_dict["obj1"][0][0] == 19  # 15 + 2.0/0.5


# Test for size extension returning early when other object not resolved
def test_apply_size_extension_constraint_other_not_resolved():
    """Test size extension constraint returns early when other object not resolved."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    obj = Mock()
    obj.name = "obj1"
    other = Mock()
    other.name = "other"
    object_map = {"obj1": obj, "other": other}

    constraint = SizeExtensionConstraint(
        object="obj1", other_object="other", axis=0, direction="-", other_position=0.0, grid_offset=None, offset=None
    )

    slice_dict = {
        "obj1": [[None, None], [None, None], [None, None]],
        "other": [[None, None], [None, None], [None, None]],  # Not resolved
    }

    resolved, slice_dict = _apply_size_extension_constraint(constraint, object_map, config, slice_dict, "volume")

    assert resolved is False


# Test for extend to infinity with size constraint
def test_extend_to_inf_with_size_constraint():
    """Test that objects with size constraints are not extended to infinity."""
    volume = Mock()
    volume.name = "volume"
    obj = Mock()
    obj.name = "obj1"
    object_map = {"volume": volume, "obj1": obj}

    constraint = SizeConstraint(
        object="obj1",
        other_object="volume",
        axes=[0],
        other_axes=[0],
        proportions=[1.0],
        grid_offsets=[0],
        offsets=[None],
    )

    constraints = [constraint]
    shape_dict = {"volume": [100, 100, 100], "obj1": [None, None, None]}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[None, None], [None, None], [None, None]]}

    resolved, slice_dict = _extend_to_inf_if_possible(constraints, object_map, slice_dict, shape_dict, "volume")

    # Should not extend obj1 in axis 0 because it has a size constraint
    # But should extend in other axes
    assert slice_dict["obj1"][0][0] is None  # Not extended (has size constraint)
    assert slice_dict["obj1"][1][0] == 0  # Extended to lower bound
    assert slice_dict["obj1"][1][1] == 100  # Extended to upper bound


# Test for extend to infinity with size extension constraint
def test_extend_to_inf_with_size_extension_constraint():
    """Test that objects with size extension constraints are not extended."""
    volume = Mock()
    volume.name = "volume"
    obj = Mock()
    obj.name = "obj1"
    object_map = {"volume": volume, "obj1": obj}

    constraint = SizeExtensionConstraint(
        object="obj1", other_object="volume", axis=0, direction="-", other_position=0.0, grid_offset=None, offset=None
    )

    constraints = [constraint]
    shape_dict = {"volume": [100, 100, 100], "obj1": [None, None, None]}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[None, None], [None, None], [None, None]]}

    resolved, slice_dict = _extend_to_inf_if_possible(constraints, object_map, slice_dict, shape_dict, "volume")

    # Should not extend obj1 at lower bound of axis 0 (has extension constraint)
    assert slice_dict["obj1"][0][0] is None
    # But should extend upper bound
    assert slice_dict["obj1"][0][1] == 100


# Test for extend to infinity with already known shape
def test_extend_to_inf_with_known_shape():
    """Test that objects with known shapes are not extended."""
    volume = Mock()
    volume.name = "volume"
    obj = Mock()
    obj.name = "obj1"
    object_map = {"volume": volume, "obj1": obj}

    constraints = []
    shape_dict = {"volume": [100, 100, 100], "obj1": [50, None, None]}  # Known shape in axis 0
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[None, None], [None, None], [None, None]]}

    resolved, slice_dict = _extend_to_inf_if_possible(constraints, object_map, slice_dict, shape_dict, "volume")

    # Should not extend obj1 in axis 0 (has known shape)
    assert slice_dict["obj1"][0][0] is None
    assert slice_dict["obj1"][0][1] is None
    # But should extend in other axes
    assert slice_dict["obj1"][1][0] == 0
    assert slice_dict["obj1"][1][1] == 100


# Test for extend to infinity with already resolved slice
def test_extend_to_inf_with_resolved_slice():
    """Test that already resolved slices are not changed."""
    volume = Mock()
    volume.name = "volume"
    obj = Mock()
    obj.name = "obj1"
    object_map = {"volume": volume, "obj1": obj}

    constraints = []
    shape_dict = {"volume": [100, 100, 100], "obj1": [None, None, None]}
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[10, None], [None, None], [None, None]],  # Lower bound already set
    }

    resolved, slice_dict = _extend_to_inf_if_possible(constraints, object_map, slice_dict, shape_dict, "volume")

    # Should not change the already set lower bound
    assert slice_dict["obj1"][0][0] == 10
    # Should extend upper bound
    assert slice_dict["obj1"][0][1] == 100


# Test handle unresolved objects
def test_handle_unresolved_objects():
    """Test error reporting for unresolved objects."""
    obj1 = Mock()
    obj1.name = "obj1"
    obj2 = Mock()
    obj2.name = "obj2"
    object_map = {"obj1": obj1, "obj2": obj2}

    slice_dict = {
        "obj1": [[0, 10], [0, 10], [None, None]],  # Not fully resolved
        "obj2": [[0, 10], [0, 10], [0, 10]],  # Fully resolved
    }

    errors = {"obj1": None, "obj2": None}

    errors = _handle_unresolved_objects(object_map, slice_dict, errors)

    assert errors["obj1"] is not None
    assert "Could not resolve" in errors["obj1"]
    assert errors["obj2"] is None  # No error for resolved object


# Test place_objects with constraint errors
def test_place_objects_with_constraint_errors():
    """Test that place_objects raises ValueError when constraints fail."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    volume = Mock(spec=SimulationVolume)
    volume.name = "volume"
    volume.partial_grid_shape = [100, 100, 100]
    volume.partial_real_shape = [None, None, None]

    # Create a proper mock that won't fail the isinstance check
    obj = Mock(spec=SimulationObject)
    obj.name = "obj1"
    obj.partial_grid_shape = [None, None, None]
    obj.partial_real_shape = [None, None, None]

    object_list = [volume, obj]

    # Create an inconsistent constraint
    constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10])

    # Add another constraint that conflicts
    constraint2 = GridCoordinateConstraint(
        object="obj1",
        axes=[0],
        sides=["-"],
        coordinates=[20],  # Different coordinate
    )

    constraints = [constraint, constraint2]
    key = jax.random.PRNGKey(0)

    with pytest.raises(ValueError, match="Failed to resolve object constraints"):
        place_objects(object_list, config, constraints, key)


# Test position constraint with both grid and real margins
def test_position_constraint_with_both_margins():
    """Test position constraint with both grid and real margins."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 0.5

    obj = Mock()
    obj.name = "obj1"
    other = Mock()
    other.name = "other"
    object_map = {"obj1": obj, "other": other}

    constraint = PositionConstraint(
        object="obj1",
        other_object="other",
        axes=[0],
        object_positions=[0.0],
        other_object_positions=[0.0],
        grid_margins=[2],  # Grid margin
        margins=[1.0],  # Real margin
    )

    shape_dict = {"obj1": [10, None, None], "other": [20, None, None]}
    slice_dict = {"obj1": [[None, None], [None, None], [None, None]], "other": [[0, 20], [None, None], [None, None]]}

    resolved, slice_dict = _apply_position_constraint(constraint, object_map, config, shape_dict, slice_dict)

    assert resolved is True
    # Should add both grid margin (2) and real margin (1.0/0.5 = 2) = 4 total


# Test size constraint with both grid and real offsets
def test_size_constraint_with_both_offsets():
    """Test size constraint with both grid and real offsets."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 0.5

    obj = Mock()
    obj.name = "obj1"
    other = Mock()
    other.name = "other"
    object_map = {"obj1": obj, "other": other}

    constraint = SizeConstraint(
        object="obj1",
        other_object="other",
        axes=[0],
        other_axes=[0],
        proportions=[1.0],
        grid_offsets=[3],  # Grid offset
        offsets=[2.0],  # Real offset
    )

    shape_dict = {"obj1": [None, None, None], "other": [10, None, None]}

    resolved, shape_dict = _apply_size_constraint(constraint, object_map, config, shape_dict)

    assert resolved is True
    assert shape_dict["obj1"][0] == 17  # 10 * 1.0 + 3 + 2.0/0.5


# Test size extension with both grid and real offsets
def test_size_extension_with_both_offsets():
    """Test size extension constraint with both grid and real offsets."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 0.5

    obj = Mock()
    obj.name = "obj1"
    other = Mock()
    other.name = "other"
    object_map = {"obj1": obj, "other": other}

    constraint = SizeExtensionConstraint(
        object="obj1",
        other_object="other",
        axis=0,
        direction="-",
        other_position=0.0,
        grid_offset=3,  # Grid offset
        offset=2.0,  # Real offset
    )

    slice_dict = {"obj1": [[None, None], [None, None], [None, None]], "other": [[10, 20], [None, None], [None, None]]}

    resolved, slice_dict = _apply_size_extension_constraint(constraint, object_map, config, slice_dict, "volume")

    assert resolved is True
    assert slice_dict["obj1"][0][0] == 22  # 15 + 3 + 2.0/0.5


# Test update grid slices with only lower bound known
def test_update_grid_slices_only_lower_bound():
    """Test updating grid slices when only lower bound is known."""
    obj = Mock()
    obj.name = "obj1"
    object_map = {"obj1": obj}

    shape_dict = {"obj1": [10, None, None]}
    slice_dict = {"obj1": [[5, None], [None, None], [None, None]]}
    errors = {"obj1": None}

    resolved, slice_dict, errors = _update_grid_slices_from_shapes(object_map, shape_dict, slice_dict, errors)

    assert resolved is True
    assert slice_dict["obj1"][0][1] == 15  # 5 + 10


# Test update grid slices with only upper bound known
def test_update_grid_slices_only_upper_bound():
    """Test updating grid slices when only upper bound is known."""
    obj = Mock()
    obj.name = "obj1"
    object_map = {"obj1": obj}

    shape_dict = {"obj1": [10, None, None]}
    slice_dict = {"obj1": [[None, 20], [None, None], [None, None]]}
    errors = {"obj1": None}

    resolved, slice_dict, errors = _update_grid_slices_from_shapes(object_map, shape_dict, slice_dict, errors)

    assert resolved is True
    assert slice_dict["obj1"][0][0] == 10  # 20 - 10


# Test resolve static shapes with partial real shape
def test_resolve_static_shapes_with_real_shape():
    """Test resolving static shapes when object has partial real shape."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 0.5

    obj = Mock()
    obj.name = "obj1"
    obj.partial_grid_shape = [None, None, None]
    obj.partial_real_shape = [5.0, None, None]  # Real shape specified

    object_map = {"obj1": obj}
    shape_dict = {"obj1": [None, None, None]}

    shape_dict = _resolve_static_shapes(object_map, shape_dict, config)

    assert shape_dict["obj1"][0] == 10  # 5.0 / 0.5


# Test resolve static shapes with partial grid shape
def test_resolve_static_shapes_with_grid_shape():
    """Test resolving static shapes when object has partial grid shape."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 0.5

    obj = Mock()
    obj.name = "obj1"
    obj.partial_grid_shape = [20, None, None]  # Grid shape specified
    obj.partial_real_shape = [None, None, None]

    object_map = {"obj1": obj}
    shape_dict = {"obj1": [None, None, None]}

    shape_dict = _resolve_static_shapes(object_map, shape_dict, config)

    assert shape_dict["obj1"][0] == 20


# Test apply position constraint when other object not resolved
def test_position_constraint_other_not_resolved():
    """Test position constraint returns False when other object not resolved."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    obj = Mock()
    obj.name = "obj1"
    other = Mock()
    other.name = "other"
    object_map = {"obj1": obj, "other": other}

    constraint = PositionConstraint(
        object="obj1",
        other_object="other",
        axes=[0],
        object_positions=[0.0],
        other_object_positions=[0.0],
        grid_margins=[0],
        margins=[0.0],
    )

    shape_dict = {"obj1": [10, None, None], "other": [20, None, None]}
    slice_dict = {
        "obj1": [[None, None], [None, None], [None, None]],
        "other": [[None, None], [None, None], [None, None]],  # Not resolved
    }

    resolved, slice_dict = _apply_position_constraint(constraint, object_map, config, shape_dict, slice_dict)

    assert resolved is False


# Test apply position constraint when object size not known
def test_position_constraint_object_size_unknown():
    """Test position constraint returns False when object size not known."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    obj = Mock()
    obj.name = "obj1"
    other = Mock()
    other.name = "other"
    object_map = {"obj1": obj, "other": other}

    constraint = PositionConstraint(
        object="obj1",
        other_object="other",
        axes=[0],
        object_positions=[0.0],
        other_object_positions=[0.0],
        grid_margins=[0],
        margins=[0.0],
    )

    shape_dict = {"obj1": [None, None, None], "other": [20, None, None]}  # obj1 size unknown
    slice_dict = {"obj1": [[None, None], [None, None], [None, None]], "other": [[0, 20], [None, None], [None, None]]}

    resolved, slice_dict = _apply_position_constraint(constraint, object_map, config, shape_dict, slice_dict)

    assert resolved is False


# Test apply size constraint when other shape unknown
def test_size_constraint_other_shape_unknown():
    """Test size constraint returns False when other object shape unknown."""
    config = Mock(spec=SimulationConfig)
    config.resolution = 1.0

    obj = Mock()
    obj.name = "obj1"
    other = Mock()
    other.name = "other"
    object_map = {"obj1": obj, "other": other}

    constraint = SizeConstraint(
        object="obj1",
        other_object="other",
        axes=[0],
        other_axes=[0],
        proportions=[1.0],
        grid_offsets=[0],
        offsets=[None],
    )

    shape_dict = {
        "obj1": [None, None, None],
        "other": [None, None, None],  # Other shape unknown
    }

    resolved, shape_dict = _apply_size_constraint(constraint, object_map, config, shape_dict)

    assert resolved is False
