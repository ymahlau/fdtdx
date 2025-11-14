import pytest

import fdtdx.fdtd.initialization as init
from fdtdx.config import SimulationConfig
from fdtdx.materials import Material
from fdtdx.objects.static_material.static import SimulationVolume, UniformMaterialObject


# ---------------------------------------------------------------------
# Fixtures and Mocks
# ---------------------------------------------------------------------
@pytest.fixture
def mock_config():
    cfg = SimulationConfig(
        time=100e-12,
        resolution=100e-9,
    )
    return cfg


@pytest.fixture
def mock_volume():
    return SimulationVolume(partial_real_shape=(5e-6, 5e-6, 5e-6))


@pytest.fixture
def mock_object():
    obj = UniformMaterialObject(
        partial_real_shape=(1e-6, 1e-6, 1e-6),
        material=Material(),
    )
    return obj


@pytest.fixture
def mock_object2():
    obj = UniformMaterialObject(
        partial_real_shape=(1e-6, 1e-6, 1e-6),
        material=Material(),
    )
    return obj


def test_same_size_constraint(mock_volume, mock_config):
    obj = UniformMaterialObject(
        material=Material(),
    )
    slice_dict, errors = init.resolve_object_constraints(
        objects=[obj, mock_volume],
        constraints=[obj.same_size(mock_volume), obj.same_position(mock_volume)],
        config=mock_config,
    )
    assert all([v is None for v in errors.values()])
    assert slice_dict[obj.name] == slice_dict[mock_volume.name]


def test_same_size_and_position_constraint(mock_volume, mock_config):
    obj = UniformMaterialObject(
        material=Material(),
    )
    slice_dict, errors = init.resolve_object_constraints(
        objects=[obj, mock_volume],
        constraints=[*obj.same_position_and_size(mock_volume)],
        config=mock_config,
    )
    assert all([v is None for v in errors.values()])
    assert slice_dict[obj.name] == slice_dict[mock_volume.name]


def test_place_at_center_constraint(mock_volume, mock_config):
    obj = UniformMaterialObject(
        material=Material(),
    )
    slice_dict, errors = init.resolve_object_constraints(
        objects=[obj, mock_volume],
        constraints=[obj.same_size(mock_volume), obj.place_at_center(mock_volume)],
        config=mock_config,
    )
    assert all([v is None for v in errors.values()])
    assert slice_dict[obj.name] == slice_dict[mock_volume.name]


def test_place_face_to_face_neg(mock_volume, mock_config, mock_object, mock_object2):
    slice_dict, errors = init.resolve_object_constraints(
        objects=[mock_volume, mock_object, mock_object2],
        constraints=[
            mock_object.place_at_center(mock_volume),
            mock_object2.same_position(mock_object, axes=(0, 1)),
            mock_object2.face_to_face_negative_direction(mock_object, axes=2),
        ],
        config=mock_config,
    )
    assert all([v is None for v in errors.values()])
    assert slice_dict[mock_object.name][2][0] == slice_dict[mock_object2.name][2][1]


def test_place_face_to_face_pos(mock_volume, mock_config, mock_object, mock_object2):
    slice_dict, errors = init.resolve_object_constraints(
        objects=[mock_volume, mock_object, mock_object2],
        constraints=[
            mock_object.place_at_center(mock_volume),
            mock_object2.same_position(mock_object, axes=(0, 1)),
            mock_object2.face_to_face_positive_direction(mock_object, axes=2),
        ],
        config=mock_config,
    )
    assert all([v is None for v in errors.values()])
    assert slice_dict[mock_object.name][2][1] == slice_dict[mock_object2.name][2][0]


def test_unspecified_position(mock_volume, mock_object, mock_config):
    _, errors = init.resolve_object_constraints(
        objects=[mock_object, mock_volume],
        constraints=[],
        config=mock_config,
    )
    assert errors[mock_object.name] is not None


def test_infinity_extension(mock_volume, mock_config):
    obj = UniformMaterialObject(
        material=Material(),
    )
    _, errors = init.resolve_object_constraints(
        objects=[obj, mock_volume],
        constraints=[],
        config=mock_config,
    )
    assert all([v is None for v in errors.values()])
