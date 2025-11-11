import pytest
from unittest.mock import MagicMock, patch
import jax.numpy as jnp

import fdtdx.fdtd.initialization as initialization


# ---------------------------------------------------------------------
# Fixtures and Mocks
# ---------------------------------------------------------------------
@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.resolution = 1.0
    cfg.dtype = "float32"
    cfg.backend = "cpu"
    cfg.gradient_config = None
    cfg.time_steps_total = 100
    cfg.grid_shape = (10, 10, 10)
    return cfg


@pytest.fixture
def mock_object():
    obj = MagicMock()
    obj.name = "obj1"
    obj.partial_grid_shape = [None, None, None]
    obj.partial_real_shape = [None, None, None]
    return obj



def test_collect_objects_from_constraints_valid_names():
    mock_constraint = MagicMock()
    mock_constraint.object_name = "a"
    mock_constraint.other_name = "b"

    result = initialization._collect_objects_from_constraints(
        constraints=[mock_constraint],
        object_map={"a": MagicMock(), "b": MagicMock()}
    )
    assert "a" in result and "b" in result


def test_collect_objects_from_constraints_raises_for_unknown_name():
    mock_constraint = MagicMock()
    mock_constraint.object_name = "missing"
    mock_constraint.other_name = None
    with pytest.raises(ValueError):
        initialization._collect_objects_from_constraints([mock_constraint], {"x": MagicMock()})


# ---------------------------------------------------------------------
# Helper constraint applications
# ---------------------------------------------------------------------
def test_apply_position_constraint_updates_slice():
    obj, other = MagicMock(), MagicMock()
    shape_dict = {obj: [5, 5, 5], other: [5, 5, 5]}
    slice_dict = {obj: [[None, None], [None, None], [None, None]], other: [[0, 5], [0, 5], [0, 5]]}

    c = MagicMock()
    c.axis = 0
    c.offset = 1

    changed = initialization._apply_position_constraint(c, obj, other, shape_dict, slice_dict)
    assert changed
    assert slice_dict[obj][0][0] is not None


def test_apply_size_constraint_computes_correct_ratio():
    obj, other = MagicMock(), MagicMock()
    shape_dict = {obj: [None, None, None], other: [10, 10, 10]}
    c = MagicMock()
    c.axis = 1
    c.ratio = 0.5

    changed = initialization._apply_size_constraint(c, obj, other, shape_dict)
    assert changed
    assert shape_dict[obj][1] == 5


def test_apply_size_extension_constraint_adds_extension():
    obj, other = MagicMock(), MagicMock()
    shape_dict = {obj: [None, None, None], other: [10, 10, 10]}
    c = MagicMock()
    c.axis = 2
    c.extension = 3

    changed = initialization._apply_size_extension_constraint(c, obj, other, shape_dict)
    assert changed
    assert shape_dict[obj][2] == 13


def test_apply_grid_coordinate_constraint_sets_correct_indices(mock_config):
    obj = MagicMock()
    shape_dict = {obj: [10, 10, 10]}
    slice_dict = {obj: [[None, None], [None, None], [None, None]]}

    c = MagicMock()
    c.axis = 0
    c.center = 0.0

    changed = initialization._apply_grid_coordinate_constraint(c, obj, shape_dict, slice_dict, config=mock_config)
    assert changed
    start, end = slice_dict[obj][0]
    assert isinstance(start, int)
    assert isinstance(end, int)


# ---------------------------------------------------------------------
# place_objects and internal init functions
# ---------------------------------------------------------------------
@patch("fdtdx.fdtd.initialization.resolve_object_constraints", return_value=({}, {}))
def test_place_objects_detects_duplicate_names(mock_resolve, mock_config):
    obj1 = MagicMock(name="Object1")
    obj1.name = "duplicate"
    obj2 = MagicMock(name="Object2")
    obj2.name = "duplicate"

    with pytest.raises(Exception) as excinfo:
        initialization.place_objects([obj1, obj2], mock_config, [], jnp.array([0]))
    assert "Duplicate object names" in str(excinfo.value)


@patch("fdtdx.fdtd.initialization.resolve_object_constraints", return_value=({}, {"obj1": "error"}))
def test_place_objects_raises_on_constraint_error(mock_resolve, mock_config):
    obj = MagicMock()
    obj.name = "obj1"
    with pytest.raises(Exception):
        initialization.place_objects([obj], mock_config, [], jnp.array([0]))


# ---------------------------------------------------------------------
# _init_params basic tests
# ---------------------------------------------------------------------
def test_init_params_creates_entries():
    obj = MagicMock()
    obj.name = "device1"
    obj.init_params.return_value = {"param": 1}
    container = MagicMock()
    container.devices = [obj]

    with patch("jax.random.split", side_effect=lambda k: (k, k)):
        result = initialization._init_params(container, jnp.array([0]))
    assert "device1" in result
    assert isinstance(result["device1"], dict)
