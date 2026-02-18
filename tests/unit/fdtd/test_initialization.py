from unittest.mock import MagicMock, Mock, patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.initialization import (
    _apply_grid_coordinate_constraint,
    _apply_position_constraint,
    _apply_real_coordinate_constraint,
    _apply_size_constraint,
    _apply_size_extension_constraint,
    _check_objects_names_from_constraints,
    _handle_unresolved_objects,
    _init_arrays,
    _resolve_static_shapes,
    _resolve_volume_name,
    _update_grid_shapes_from_slices,
    _update_grid_slices_from_shapes,
    apply_params,
    resolve_object_constraints,
)
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_config():
    return SimulationConfig(resolution=1.0, time=100e-15)


@pytest.fixture
def simple_volume():
    return SimulationVolume(name="volume", partial_grid_shape=(100, 100, 100))


@pytest.fixture
def simple_material():
    return Material(
        permittivity=(2.0, 2.0, 2.0),
        permeability=(1.0, 1.0, 1.0),
        electric_conductivity=(0.0, 0.0, 0.0),
        magnetic_conductivity=(0.0, 0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# resolve_object_constraints – error path tests
# ---------------------------------------------------------------------------


def test_resolve_constraints_with_duplicate_object_names(simple_config, simple_volume):
    volume2 = SimulationVolume(name="volume", partial_grid_shape=(50, 50, 50))
    with pytest.raises(Exception, match="Duplicate object names"):
        resolve_object_constraints([simple_volume, volume2], [], simple_config)


def test_resolve_constraints_unknown_object_in_constraint(simple_config, simple_volume):
    constraint = GridCoordinateConstraint(object="nonexistent_object", axes=[0], sides=["-"], coordinates=[10])
    with pytest.raises(ValueError, match="Unknown object name"):
        resolve_object_constraints([simple_volume], [constraint], simple_config)


def test_resolve_constraints_no_simulation_volume(simple_config, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    with pytest.raises(ValueError, match="No SimulationVolume"):
        resolve_object_constraints([obj], [], simple_config)


def test_resolve_constraints_multiple_volumes(simple_config):
    volume1 = SimulationVolume(name="volume1", partial_grid_shape=(100, 100, 100))
    volume2 = SimulationVolume(name="volume2", partial_grid_shape=(50, 50, 50))
    with pytest.raises(ValueError, match="Multiple SimulationVolume"):
        resolve_object_constraints([volume1, volume2], [], simple_config)


def test_resolve_constraints_conflicting_grid_coordinates(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    c1 = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10])
    c2 = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20])
    resolved_slices, errors = resolve_object_constraints([simple_volume, obj], [c1, c2], simple_config)
    assert errors["obj1"] is not None


def test_resolve_constraints_conflicting_real_coordinates(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    c1 = RealCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10.0])
    c2 = RealCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20.0])
    resolved_slices, errors = resolve_object_constraints([simple_volume, obj], [c1, c2], simple_config)
    assert errors["obj1"] is not None


def test_resolve_constraints_inconsistent_size_and_position(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    size_constraint = SizeConstraint(
        object="obj1", other_object="volume", axes=[0], other_axes=[0],
        proportions=[0.5], grid_offsets=[0], offsets=[None],
    )
    pos_constraint = PositionConstraint(
        object="obj1", other_object="volume", axes=[0],
        object_positions=[0.0], other_object_positions=[-1.0],
        grid_margins=[5], margins=[None],
    )
    grid_constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[0])
    resolved_slices, errors = resolve_object_constraints(
        [simple_volume, obj], [size_constraint, pos_constraint, grid_constraint], simple_config
    )
    assert errors["obj1"] is not None


def test_resolve_constraints_with_real_margins(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    size_constraint = SizeConstraint(
        object="obj1", other_object="volume", axes=[0, 1, 2], other_axes=[0, 1, 2],
        proportions=[0.1, 0.1, 0.1], grid_offsets=[0, 0, 0], offsets=[None, None, None],
    )
    pos_constraint = PositionConstraint(
        object="obj1", other_object="volume", axes=[0],
        object_positions=[0.0], other_object_positions=[-1.0],
        grid_margins=[None], margins=[5.0],
    )
    resolved_slices, errors = resolve_object_constraints(
        [simple_volume, obj], [size_constraint, pos_constraint], simple_config
    )
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices


def test_resolve_constraints_with_both_margins(simple_volume, simple_material):
    config = SimulationConfig(resolution=0.5, time=100e-15)
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    constraint = PositionConstraint(
        object="obj1", other_object="volume", axes=[0],
        object_positions=[0.0], other_object_positions=[-1.0],
        grid_margins=[2], margins=[1.0],
    )
    resolved_slices, errors = resolve_object_constraints([simple_volume, obj], [constraint], config)
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices


def test_resolve_constraints_with_real_offset_in_size(simple_volume, simple_material):
    config = SimulationConfig(resolution=0.5, time=100e-15)
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    constraint = SizeConstraint(
        object="obj1", other_object="volume", axes=[0], other_axes=[0],
        proportions=[0.5], grid_offsets=[None], offsets=[2.0],
    )
    resolved_slices, errors = resolve_object_constraints([simple_volume, obj], [constraint], config)
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices


def test_resolve_constraints_with_grid_offset_in_size(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    constraint = SizeConstraint(
        object="obj1", other_object="volume", axes=[0], other_axes=[0],
        proportions=[0.5], grid_offsets=[10], offsets=[None],
    )
    resolved_slices, errors = resolve_object_constraints([simple_volume, obj], [constraint], simple_config)
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices


def test_resolve_constraints_size_extension_with_real_offset(simple_volume, simple_material):
    config = SimulationConfig(resolution=0.5, time=100e-15)
    obj1 = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj2 = UniformMaterialObject(name="obj2", material=simple_material)
    pos_constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20])
    ext_constraint = SizeExtensionConstraint(
        object="obj2", other_object="obj1", axis=0, direction="+",
        other_position=1.0, grid_offset=None, offset=2.0,
    )
    resolved_slices, errors = resolve_object_constraints(
        [simple_volume, obj1, obj2], [pos_constraint, ext_constraint], config
    )
    assert errors["obj2"] is None
    assert resolved_slices["obj2"][0][1] == 34


def test_resolve_constraints_size_extension_with_grid_offset(simple_config, simple_volume, simple_material):
    obj1 = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj2 = UniformMaterialObject(name="obj2", material=simple_material)
    pos_constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20])
    ext_constraint = SizeExtensionConstraint(
        object="obj2", other_object="obj1", axis=0, direction="-",
        other_position=-1.0, grid_offset=5, offset=None,
    )
    resolved_slices, errors = resolve_object_constraints(
        [simple_volume, obj1, obj2], [pos_constraint, ext_constraint], simple_config
    )
    assert errors["obj2"] is None
    assert resolved_slices["obj2"][0][0] == 25


def test_resolve_constraints_size_extension_to_volume_boundary(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    constraints = [
        GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20]),
        GridCoordinateConstraint(object="obj1", axes=[1], sides=["-"], coordinates=[20]),
        SizeExtensionConstraint(
            object="obj1", other_object=None, axis=2, direction="-",
            other_position=0.0, grid_offset=None, offset=None,
        ),
    ]
    resolved_slices, errors = resolve_object_constraints([simple_volume, obj], constraints, simple_config)
    assert errors["obj1"] is None
    assert resolved_slices["obj1"][2][0] == 0


def test_resolve_constraints_with_partial_real_shape(simple_material):
    config = SimulationConfig(resolution=0.5, time=100e-15)
    volume = SimulationVolume(name="volume", partial_grid_shape=(100, 100, 100))
    obj = UniformMaterialObject(name="obj1", partial_real_shape=(5.0, 5.0, 5.0), material=simple_material)
    constraints = [
        GridCoordinateConstraint(object="obj1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[0, 0, 0])
    ]
    resolved_slices, errors = resolve_object_constraints([volume, obj], constraints, config)
    assert isinstance(resolved_slices, dict)
    assert "obj1" in resolved_slices
    if resolved_slices["obj1"][0][0] is not None:
        shape = resolved_slices["obj1"][0][1] - resolved_slices["obj1"][0][0]
        assert shape == 10


def test_resolve_constraints_extend_to_infinity(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    constraint = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20])
    resolved_slices, errors = resolve_object_constraints([simple_volume, obj], [constraint], simple_config)
    assert errors["obj1"] is None
    assert resolved_slices["obj1"][1] == (0, 100)
    assert resolved_slices["obj1"][2] == (0, 100)


# ---------------------------------------------------------------------------
# _resolve_volume_name – direct unit tests
# ---------------------------------------------------------------------------


def test_resolve_volume_name_success(simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {simple_volume.name: simple_volume, obj.name: obj}
    assert _resolve_volume_name(obj_map) == "volume"


def test_resolve_volume_name_no_volume(simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    with pytest.raises(ValueError, match="No SimulationVolume"):
        _resolve_volume_name({"obj1": obj})


def test_resolve_volume_name_multiple_volumes():
    v1 = SimulationVolume(name="v1", partial_grid_shape=(10, 10, 10))
    v2 = SimulationVolume(name="v2", partial_grid_shape=(10, 10, 10))
    with pytest.raises(ValueError, match="Multiple SimulationVolume"):
        _resolve_volume_name({"v1": v1, "v2": v2})


# ---------------------------------------------------------------------------
# _check_objects_names_from_constraints – direct unit tests
# ---------------------------------------------------------------------------


def test_check_objects_names_no_constraints():
    result = _check_objects_names_from_constraints([], ["volume", "obj1"])
    assert result == []


def test_check_objects_names_valid_names():
    c = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10])
    result = _check_objects_names_from_constraints([c], ["obj1", "volume"])
    assert "obj1" in result


def test_check_objects_names_unknown_name():
    c = GridCoordinateConstraint(object="missing", axes=[0], sides=["-"], coordinates=[10])
    with pytest.raises(ValueError, match="Unknown object name"):
        _check_objects_names_from_constraints([c], ["obj1", "volume"])


def test_check_objects_names_other_object_unknown():
    c = SizeConstraint(
        object="obj1", other_object="missing", axes=[0], other_axes=[0],
        proportions=[1.0], grid_offsets=[0], offsets=[None],
    )
    with pytest.raises(ValueError, match="Unknown object name"):
        _check_objects_names_from_constraints([c], ["obj1", "volume"])


# ---------------------------------------------------------------------------
# _resolve_static_shapes – direct unit tests
# ---------------------------------------------------------------------------


def test_resolve_static_shapes_grid_shape(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 20, 30), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    shape_dict = {"volume": [None, None, None], "obj1": [None, None, None]}
    result = _resolve_static_shapes(obj_map, shape_dict, simple_config)
    assert result["obj1"] == [10, 20, 30]


def test_resolve_static_shapes_real_shape(simple_material):
    config = SimulationConfig(resolution=0.5, time=100e-15)
    volume = SimulationVolume(name="volume", partial_grid_shape=(100, 100, 100))
    obj = UniformMaterialObject(name="obj1", partial_real_shape=(5.0, 10.0, 15.0), material=simple_material)
    obj_map = {"volume": volume, "obj1": obj}
    shape_dict = {"volume": [None, None, None], "obj1": [None, None, None]}
    result = _resolve_static_shapes(obj_map, shape_dict, config)
    # 5.0 / 0.5 = 10, 10.0 / 0.5 = 20, 15.0 / 0.5 = 30
    assert result["obj1"] == [10, 20, 30]


def test_resolve_static_shapes_no_shape(simple_config, simple_material):
    volume = SimulationVolume(name="volume", partial_grid_shape=(100, 100, 100))
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    obj_map = {"volume": volume, "obj1": obj}
    shape_dict = {"volume": [None, None, None], "obj1": [None, None, None]}
    result = _resolve_static_shapes(obj_map, shape_dict, simple_config)
    # volume has grid shape, obj1 has nothing
    assert result["volume"] == [100, 100, 100]
    assert result["obj1"] == [None, None, None]


# ---------------------------------------------------------------------------
# _apply_grid_coordinate_constraint – direct unit tests
# ---------------------------------------------------------------------------


def test_apply_grid_coordinate_constraint_sets_lower(simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[None, None], [None, None], [None, None]]}
    c = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[15])
    resolved, new_slices = _apply_grid_coordinate_constraint(c, obj_map, slice_dict)
    assert resolved is True
    assert new_slices["obj1"][0][0] == 15


def test_apply_grid_coordinate_constraint_sets_upper(simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[None, None], [None, None], [None, None]]}
    c = GridCoordinateConstraint(object="obj1", axes=[0], sides=["+"], coordinates=[50])
    resolved, new_slices = _apply_grid_coordinate_constraint(c, obj_map, slice_dict)
    assert resolved is True
    assert new_slices["obj1"][0][1] == 50


def test_apply_grid_coordinate_constraint_consistent_no_change(simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[15, None], [None, None], [None, None]]}
    c = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[15])
    resolved, new_slices = _apply_grid_coordinate_constraint(c, obj_map, slice_dict)
    assert resolved is False


def test_apply_grid_coordinate_constraint_conflicting_raises(simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[15, None], [None, None], [None, None]]}
    c = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20])
    with pytest.raises(Exception, match="Inconsistent"):
        _apply_grid_coordinate_constraint(c, obj_map, slice_dict)


# ---------------------------------------------------------------------------
# _apply_real_coordinate_constraint – direct unit tests
# ---------------------------------------------------------------------------


def test_apply_real_coordinate_constraint_converts_to_grid(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[None, None], [None, None], [None, None]]}
    # With resolution=1.0, coordinate 15.0 -> grid 15
    c = RealCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[15.0])
    resolved, new_slices = _apply_real_coordinate_constraint(c, obj_map, slice_dict, simple_config)
    assert resolved is True
    assert new_slices["obj1"][0][0] == 15


def test_apply_real_coordinate_constraint_sub_resolution(simple_material):
    config = SimulationConfig(resolution=0.5, time=100e-15)
    volume = SimulationVolume(name="volume", partial_grid_shape=(100, 100, 100))
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": volume, "obj1": obj}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[None, None], [None, None], [None, None]]}
    # resolution=0.5, coordinate 5.0 -> grid 10
    c = RealCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[5.0])
    resolved, new_slices = _apply_real_coordinate_constraint(c, obj_map, slice_dict, config)
    assert resolved is True
    assert new_slices["obj1"][0][0] == 10


def test_apply_real_coordinate_constraint_conflicting_raises(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {"volume": [[0, 100], [0, 100], [0, 100]], "obj1": [[15, None], [None, None], [None, None]]}
    c = RealCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20.0])
    with pytest.raises(Exception, match="Inconsistent"):
        _apply_real_coordinate_constraint(c, obj_map, slice_dict, simple_config)


# ---------------------------------------------------------------------------
# _apply_size_extension_constraint – error path
# ---------------------------------------------------------------------------


def test_apply_size_extension_constraint_other_not_placed_yet(simple_config, simple_volume, simple_material):
    obj1 = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj2 = UniformMaterialObject(name="obj2", material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj1, "obj2": obj2}
    # obj1 not yet placed (None boundaries)
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[None, None], [None, None], [None, None]],
        "obj2": [[None, None], [None, None], [None, None]],
    }
    c = SizeExtensionConstraint(
        object="obj2", other_object="obj1", axis=0, direction="+",
        other_position=1.0, grid_offset=None, offset=None,
    )
    resolved, _ = _apply_size_extension_constraint(c, obj_map, simple_config, slice_dict, "volume")
    assert resolved is False


# ---------------------------------------------------------------------------
# _handle_unresolved_objects – direct unit tests
# ---------------------------------------------------------------------------


def test_handle_unresolved_objects_marks_error(simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[None, None], [None, None], [None, None]],
    }
    errors = {"volume": None, "obj1": None}
    result = _handle_unresolved_objects(obj_map, slice_dict, errors)
    assert result["obj1"] is not None
    assert result["volume"] is None


def test_handle_unresolved_objects_no_errors_when_resolved(simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[5, 15], [5, 15], [5, 15]],
    }
    errors = {"volume": None, "obj1": None}
    result = _handle_unresolved_objects(obj_map, slice_dict, errors)
    assert result["obj1"] is None
    assert result["volume"] is None


# ---------------------------------------------------------------------------
# apply_params – unit tests (mocked)
# ---------------------------------------------------------------------------


def _make_arrays_mock(shape=(10, 10, 10)):
    inv_perm = jnp.ones((3, *shape))
    inv_permeab = jnp.ones((3, *shape))
    arrays = Mock(spec=ArrayContainer)
    arrays.inv_permittivities = inv_perm
    arrays.inv_permeabilities = inv_permeab
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
    return arrays


@patch("fdtdx.fdtd.initialization.compute_allowed_permittivities")
def test_apply_params_continuous_type(mock_compute_perm):
    mock_compute_perm.return_value = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
    device = Mock()
    device.name = "device1"
    device.output_type = ParameterType.CONTINUOUS
    device.grid_slice = (slice(0, 10), slice(0, 10), slice(0, 10))
    device.materials = [Mock(), Mock()]
    material_indices = jnp.ones((10, 10, 10)) * 0.5
    device.return_value = material_indices
    objects = Mock(spec=ObjectContainer)
    objects.devices = [device]
    objects.object_list = [device]
    objects.volume_idx = 0
    arrays = _make_arrays_mock()
    params = {"device1": {}}
    key = jax.random.PRNGKey(0)
    device.apply = Mock(return_value=device)
    result_arrays, result_objects, info = apply_params(arrays, objects, params, key)
    assert device.call_count > 0
    assert mock_compute_perm.called


@patch("fdtdx.fdtd.initialization.compute_allowed_permittivities")
@patch("fdtdx.fdtd.initialization.straight_through_estimator")
def test_apply_params_discrete_type(mock_ste, mock_compute_perm):
    mock_compute_perm.return_value = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
    mock_ste.return_value = jnp.ones((10, 10, 10))
    device = Mock()
    device.name = "device1"
    device.output_type = ParameterType.DISCRETE
    device.grid_slice = (slice(0, 10), slice(0, 10), slice(0, 10))
    device.materials = [Mock(), Mock()]
    material_indices = jnp.zeros((10, 10, 10), dtype=jnp.int32)
    device.return_value = material_indices
    objects = Mock(spec=ObjectContainer)
    objects.devices = [device]
    objects.object_list = [device]
    objects.volume_idx = 0
    arrays = _make_arrays_mock()
    params = {"device1": {}}
    key = jax.random.PRNGKey(0)
    device.apply = Mock(return_value=device)
    result_arrays, result_objects, info = apply_params(arrays, objects, params, key)
    assert mock_ste.called
    assert mock_compute_perm.called


def test_apply_params_no_devices():
    """apply_params with no devices should still update sources."""
    objects = Mock(spec=ObjectContainer)
    objects.devices = []
    mock_obj = Mock()
    mock_obj.apply = Mock(return_value=mock_obj)
    objects.object_list = [mock_obj]
    objects.volume_idx = 0
    arrays = _make_arrays_mock()
    key = jax.random.PRNGKey(0)
    result_arrays, result_objects, info = apply_params(arrays, objects, {}, key)
    assert mock_obj.apply.called
    assert isinstance(info, dict)


@patch("fdtdx.fdtd.initialization.compute_allowed_permittivities")
def test_apply_params_isotropic_components(mock_compute_perm):
    """Test apply_params with isotropic (1-component) permittivity."""
    mock_compute_perm.return_value = [[2.0], [4.0]]
    device = Mock()
    device.name = "device1"
    device.output_type = ParameterType.CONTINUOUS
    device.grid_slice = (slice(0, 5), slice(0, 5), slice(0, 5))
    device.materials = [Mock(), Mock()]
    material_indices = jnp.ones((5, 5, 5)) * 0.5
    device.return_value = material_indices

    # 1-component (isotropic) arrays
    inv_perm = jnp.ones((1, 10, 10, 10))
    inv_permeab = jnp.ones((1, 10, 10, 10))
    arrays = Mock(spec=ArrayContainer)
    arrays.inv_permittivities = inv_perm
    arrays.inv_permeabilities = inv_permeab
    at_accessor = MagicMock()

    def at_getitem(key):
        at_result = Mock()
        at_result.set = lambda v: Mock(
            spec=ArrayContainer,
            inv_permittivities=v,
            inv_permeabilities=inv_permeab,
            at=at_accessor,
        )
        return at_result

    at_accessor.__getitem__ = Mock(side_effect=at_getitem)
    arrays.at = at_accessor

    objects = Mock(spec=ObjectContainer)
    objects.devices = [device]
    mock_obj = Mock()
    mock_obj.apply = Mock(return_value=mock_obj)
    objects.object_list = [mock_obj]
    objects.volume_idx = 0
    device.apply = Mock(return_value=device)
    result_arrays, result_objects, info = apply_params(arrays, objects, {"device1": {}}, jax.random.PRNGKey(0))
    assert mock_compute_perm.called


@patch("fdtdx.fdtd.initialization.compute_allowed_permittivities")
def test_apply_params_fully_anisotropic_continuous(mock_compute_perm):
    """Test apply_params with fully anisotropic (9-component) permittivity."""
    identity_flat = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    two_x_flat = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]
    mock_compute_perm.return_value = [identity_flat, two_x_flat]

    device = Mock()
    device.name = "device1"
    device.output_type = ParameterType.CONTINUOUS
    device.grid_slice = (slice(0, 5), slice(0, 5), slice(0, 5))
    device.materials = [Mock(), Mock()]
    material_indices = jnp.ones((5, 5, 5)) * 0.5
    device.return_value = material_indices

    # 9-component (fully anisotropic) arrays
    inv_perm = jnp.ones((9, 10, 10, 10))
    inv_permeab = jnp.ones((9, 10, 10, 10))
    arrays = Mock(spec=ArrayContainer)
    arrays.inv_permittivities = inv_perm
    arrays.inv_permeabilities = inv_permeab
    at_accessor = MagicMock()

    def at_getitem(key):
        at_result = Mock()
        at_result.set = lambda v: Mock(
            spec=ArrayContainer,
            inv_permittivities=v,
            inv_permeabilities=inv_permeab,
            at=at_accessor,
        )
        return at_result

    at_accessor.__getitem__ = Mock(side_effect=at_getitem)
    arrays.at = at_accessor

    objects = Mock(spec=ObjectContainer)
    objects.devices = [device]
    mock_obj = Mock()
    mock_obj.apply = Mock(return_value=mock_obj)
    objects.object_list = [mock_obj]
    objects.volume_idx = 0
    device.apply = Mock(return_value=device)
    result_arrays, result_objects, info = apply_params(arrays, objects, {"device1": {}}, jax.random.PRNGKey(0))
    assert mock_compute_perm.called


@patch("fdtdx.fdtd.initialization.compute_allowed_permittivities")
@patch("fdtdx.fdtd.initialization.straight_through_estimator")
def test_apply_params_fully_anisotropic_discrete(mock_ste, mock_compute_perm):
    """Test apply_params discrete path with fully anisotropic (9-component) permittivity."""
    identity_flat = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    two_x_flat = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]
    mock_compute_perm.return_value = [identity_flat, two_x_flat]
    mock_ste.return_value = jnp.ones((9, 5, 5, 5))

    device = Mock()
    device.name = "device1"
    device.output_type = ParameterType.DISCRETE
    device.grid_slice = (slice(0, 5), slice(0, 5), slice(0, 5))
    device.materials = [Mock(), Mock()]
    material_indices = jnp.zeros((5, 5, 5), dtype=jnp.int32)
    device.return_value = material_indices

    inv_perm = jnp.ones((9, 10, 10, 10))
    inv_permeab = jnp.ones((9, 10, 10, 10))
    arrays = Mock(spec=ArrayContainer)
    arrays.inv_permittivities = inv_perm
    arrays.inv_permeabilities = inv_permeab
    at_accessor = MagicMock()

    def at_getitem(key):
        at_result = Mock()
        at_result.set = lambda v: Mock(
            spec=ArrayContainer,
            inv_permittivities=v,
            inv_permeabilities=inv_permeab,
            at=at_accessor,
        )
        return at_result

    at_accessor.__getitem__ = Mock(side_effect=at_getitem)
    arrays.at = at_accessor

    objects = Mock(spec=ObjectContainer)
    objects.devices = [device]
    mock_obj = Mock()
    mock_obj.apply = Mock(return_value=mock_obj)
    objects.object_list = [mock_obj]
    objects.volume_idx = 0
    device.apply = Mock(return_value=device)
    result_arrays, result_objects, info = apply_params(arrays, objects, {"device1": {}}, jax.random.PRNGKey(0))
    assert mock_ste.called
    assert mock_compute_perm.called


# ---------------------------------------------------------------------------
# _apply_position_constraint – edge cases
# ---------------------------------------------------------------------------


def test_apply_position_constraint_other_not_placed(simple_config, simple_volume, simple_material):
    """Position constraint skipped when other_object position unknown."""
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {
        "volume": [[None, None], [None, None], [None, None]],
        "obj1": [[None, None], [None, None], [None, None]],
    }
    shape_dict = {"volume": [100, 100, 100], "obj1": [10, 10, 10]}
    c = PositionConstraint(
        object="obj1", other_object="volume", axes=[0],
        object_positions=[0.0], other_object_positions=[-1.0],
        grid_margins=[0], margins=[None],
    )
    resolved, _ = _apply_position_constraint(c, obj_map, simple_config, shape_dict, slice_dict)
    assert resolved is False


def test_apply_position_constraint_object_size_unknown(simple_config, simple_volume, simple_material):
    """Position constraint skipped when object size unknown."""
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[None, None], [None, None], [None, None]],
    }
    shape_dict = {"volume": [100, 100, 100], "obj1": [None, None, None]}
    c = PositionConstraint(
        object="obj1", other_object="volume", axes=[0],
        object_positions=[0.0], other_object_positions=[-1.0],
        grid_margins=[0], margins=[None],
    )
    resolved, _ = _apply_position_constraint(c, obj_map, simple_config, shape_dict, slice_dict)
    assert resolved is False


# ---------------------------------------------------------------------------
# _apply_size_constraint – edge cases
# ---------------------------------------------------------------------------


def test_apply_size_constraint_other_shape_unknown(simple_config, simple_volume, simple_material):
    """Size constraint skipped when other object shape is unknown."""
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    shape_dict = {"volume": [None, None, None], "obj1": [None, None, None]}
    c = SizeConstraint(
        object="obj1", other_object="volume", axes=[0], other_axes=[0],
        proportions=[0.5], grid_offsets=[0], offsets=[None],
    )
    resolved, _ = _apply_size_constraint(c, obj_map, simple_config, shape_dict)
    assert resolved is False


def test_apply_size_constraint_conflicting_shape_raises(simple_config, simple_volume, simple_material):
    """Size constraint raises when computed shape conflicts with existing one."""
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    # Set obj1 shape to 60 in axis 0, but constraint would compute 50
    shape_dict = {"volume": [100, 100, 100], "obj1": [60, None, None]}
    c = SizeConstraint(
        object="obj1", other_object="volume", axes=[0], other_axes=[0],
        proportions=[0.5], grid_offsets=[0], offsets=[None],
    )
    with pytest.raises(Exception):
        _apply_size_constraint(c, obj_map, simple_config, shape_dict)


# ---------------------------------------------------------------------------
# _update_grid_slices_from_shapes – direct unit tests
# ---------------------------------------------------------------------------


def test_update_grid_slices_from_shapes_b0_known(simple_volume, simple_material):
    """When b0 is known and shape known, b1 should be set."""
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    shape_dict = {"volume": [100, 100, 100], "obj1": [10, 10, 10]}
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[5, None], [None, None], [None, None]],
    }
    errors = {"volume": None, "obj1": None}
    resolved, new_slices, new_errors = _update_grid_slices_from_shapes(obj_map, shape_dict, slice_dict, errors)
    assert resolved is True
    assert new_slices["obj1"][0][1] == 15  # b0(5) + shape(10)


def test_update_grid_slices_from_shapes_b1_known(simple_volume, simple_material):
    """When b1 is known and shape known, b0 should be set."""
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    shape_dict = {"volume": [100, 100, 100], "obj1": [10, 10, 10]}
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[None, 30], [None, None], [None, None]],
    }
    errors = {"volume": None, "obj1": None}
    resolved, new_slices, new_errors = _update_grid_slices_from_shapes(obj_map, shape_dict, slice_dict, errors)
    assert resolved is True
    assert new_slices["obj1"][0][0] == 20  # b1(30) - shape(10)


def test_update_grid_slices_from_shapes_inconsistent_shape(simple_volume, simple_material):
    """When both bounds known but shape mismatches, error is set."""
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    shape_dict = {"volume": [100, 100, 100], "obj1": [10, 10, 10]}
    # b1 - b0 = 20, but shape = 10 → inconsistency
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[5, 25], [None, None], [None, None]],
    }
    errors = {"volume": None, "obj1": None}
    resolved, new_slices, new_errors = _update_grid_slices_from_shapes(obj_map, shape_dict, slice_dict, errors)
    assert new_errors["obj1"] is not None


# ---------------------------------------------------------------------------
# _update_grid_shapes_from_slices – direct unit tests
# ---------------------------------------------------------------------------


def test_update_grid_shapes_from_slices_infers_shape(simple_volume, simple_material):
    """When both bounds known and shape unknown, shape is inferred."""
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    shape_dict = {"volume": [100, 100, 100], "obj1": [None, None, None]}
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[10, 30], [0, 100], [0, 100]],
    }
    errors = {"volume": None, "obj1": None}
    resolved, new_shapes, new_errors = _update_grid_shapes_from_slices(obj_map, shape_dict, slice_dict, errors)
    assert resolved is True
    assert new_shapes["obj1"][0] == 20


def test_update_grid_shapes_from_slices_inconsistent(simple_volume, simple_material):
    """When bounds imply a different shape, error is set."""
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    # shape says 10 but bounds say 20
    shape_dict = {"volume": [100, 100, 100], "obj1": [10, None, None]}
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[5, 25], [None, None], [None, None]],
    }
    errors = {"volume": None, "obj1": None}
    resolved, new_shapes, new_errors = _update_grid_shapes_from_slices(obj_map, shape_dict, slice_dict, errors)
    assert new_errors["obj1"] is not None


# ---------------------------------------------------------------------------
# _apply_position_constraint – conflict paths
# ---------------------------------------------------------------------------


def test_apply_position_constraint_conflicting_b0_raises(simple_config, simple_volume, simple_material):
    """Position constraint raises when computed b0 conflicts with existing b0."""
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    # Place obj1 lower bound at 5 (b0=5), but constraint would compute b0=0
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[5, None], [None, None], [None, None]],
    }
    shape_dict = {"volume": [100, 100, 100], "obj1": [10, 10, 10]}
    # Constraint: obj1 left edge at volume left edge (b0=0 for obj1)
    c = PositionConstraint(
        object="obj1", other_object="volume", axes=[0],
        object_positions=[-1.0], other_object_positions=[-1.0],
        grid_margins=[0], margins=[None],
    )
    with pytest.raises(Exception, match="Inconsistent"):
        _apply_position_constraint(c, obj_map, simple_config, shape_dict, slice_dict)


def test_apply_position_constraint_conflicting_b1_raises(simple_config, simple_volume, simple_material):
    """Position constraint raises when computed b1 conflicts with existing b1."""
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    # Place obj1: b0=None but b1=50, and constraint would compute b0=0→b1=10
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[None, 50], [None, None], [None, None]],
    }
    shape_dict = {"volume": [100, 100, 100], "obj1": [10, 10, 10]}
    # Constraint: obj1 left edge at volume left edge → b0=0, b1=10
    c = PositionConstraint(
        object="obj1", other_object="volume", axes=[0],
        object_positions=[-1.0], other_object_positions=[-1.0],
        grid_margins=[0], margins=[None],
    )
    with pytest.raises(Exception, match="Inconsistent"):
        _apply_position_constraint(c, obj_map, simple_config, shape_dict, slice_dict)


# ---------------------------------------------------------------------------
# resolve_object_constraints – unknown constraint type via iterative solver
# ---------------------------------------------------------------------------


def test_apply_size_extension_constraint_conflicting_value_raises(simple_config, simple_volume, simple_material):
    """Size extension raises when computed anchor conflicts with existing bound."""
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    # Extend obj1 to volume lower boundary (0), but obj1's lower bound is already set to 20
    slice_dict = {
        "volume": [[0, 100], [0, 100], [0, 100]],
        "obj1": [[20, None], [None, None], [None, None]],
    }
    c = SizeExtensionConstraint(
        object="obj1", other_object=None, axis=0, direction="-",
        other_position=0.0, grid_offset=None, offset=None,
    )
    with pytest.raises(Exception, match="Inconsistent grid shape"):
        _apply_size_extension_constraint(c, obj_map, simple_config, slice_dict, "volume")


def test_apply_size_extension_constraint_volume_upper_bound_none_raises(simple_config, simple_volume, simple_material):
    """Raises when volume's upper bound is None (should never happen in normal flow)."""
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    obj_map = {"volume": simple_volume, "obj1": obj}
    # Volume upper bound is None (abnormal state)
    slice_dict = {
        "volume": [[0, None], [0, None], [0, None]],
        "obj1": [[None, None], [None, None], [None, None]],
    }
    c = SizeExtensionConstraint(
        object="obj1", other_object=None, axis=0, direction="+",
        other_position=0.0, grid_offset=None, offset=None,
    )
    with pytest.raises(Exception, match="This should never happen"):
        _apply_size_extension_constraint(c, obj_map, simple_config, slice_dict, "volume")


# ---------------------------------------------------------------------------
# resolve_object_constraints – unknown constraint type via iterative solver
# ---------------------------------------------------------------------------


def test_resolve_constraints_unknown_constraint_type_sets_error(simple_config, simple_volume, simple_material):
    """An unrecognised constraint type is caught and stored as an error."""

    class FakeConstraint:
        object = "obj1"
        other_object = None

    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(10, 10, 10), material=simple_material)
    fake = FakeConstraint()
    # _check_objects_names_from_constraints inspects getattr(c, "object") and getattr(c, "other_object")
    resolved_slices, errors = resolve_object_constraints([simple_volume, obj], [fake], simple_config)
    # The unknown type raises inside the loop, exception is caught → error stored
    assert errors["obj1"] is not None


# ---------------------------------------------------------------------------
# _init_arrays – unknown static material object type
# ---------------------------------------------------------------------------


@patch("fdtdx.fdtd.initialization.create_named_sharded_matrix")
def test_init_arrays_unknown_static_material_type_raises(mock_create_matrix):
    """_init_arrays should raise Exception for an unknown static material object type.

    Covers line 530 (else: raise Exception("Unknown object type")).
    The static_material_objects property normally only returns UniformMaterialObject
    or StaticMultiMaterialObject instances. This test calls _init_arrays directly
    with a mocked ObjectContainer to exercise the defensive else branch.
    """
    mock_create_matrix.side_effect = lambda shape, **kwargs: jnp.zeros(shape)

    class FakeStaticObj:
        placement_order = 0

    fake_obj = FakeStaticObj()

    config = Mock(spec=SimulationConfig)
    config.dtype = jnp.float32
    config.resolution = 1.0
    config.backend = "cpu"
    config.gradient_config = None

    objects = Mock(spec=ObjectContainer)
    objects.volume = Mock()
    objects.volume.grid_shape = (2, 2, 2)
    objects.all_objects_isotropic_permittivity = True
    objects.all_objects_isotropic_permeability = True
    objects.all_objects_isotropic_electric_conductivity = True
    objects.all_objects_isotropic_magnetic_conductivity = True
    objects.all_objects_diagonally_anisotropic_permittivity = True
    objects.all_objects_diagonally_anisotropic_permeability = True
    objects.all_objects_diagonally_anisotropic_electric_conductivity = True
    objects.all_objects_diagonally_anisotropic_magnetic_conductivity = True
    objects.all_objects_non_magnetic = True
    objects.all_objects_non_electrically_conductive = True
    objects.all_objects_non_magnetically_conductive = True
    objects.static_material_objects = [fake_obj]
    objects.detectors = []
    objects.boundary_objects = []

    with pytest.raises(Exception, match="Unknown object type"):
        _init_arrays(objects, config)
