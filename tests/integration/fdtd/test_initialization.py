"""Integration tests for fdtdx.fdtd.initialization – place_objects and _init_arrays."""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.initialization import place_objects
from fdtdx.interfaces.recorder import Recorder
from fdtdx.materials import Material
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.device.device import Device
from fdtdx.objects.object import GridCoordinateConstraint
from fdtdx.objects.static_material.sphere import Sphere
from fdtdx.objects.static_material.static import SimulationVolume, UniformMaterialObject

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_config():
    return SimulationConfig(resolution=1.0, time=100e-15, backend="cpu")


@pytest.fixture
def simple_volume():
    return SimulationVolume(name="volume", partial_grid_shape=(50, 50, 50))


@pytest.fixture
def simple_material():
    return Material(
        permittivity=(2.0, 2.0, 2.0),
        permeability=(1.0, 1.0, 1.0),
        electric_conductivity=(0.0, 0.0, 0.0),
        magnetic_conductivity=(0.0, 0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# Basic place_objects tests
# ---------------------------------------------------------------------------


def test_place_objects_creates_object_container(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(20, 20, 20), material=simple_material)
    constraint = GridCoordinateConstraint(
        object="obj1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, info = place_objects([simple_volume, obj], simple_config, [constraint], key)
    assert isinstance(obj_container, ObjectContainer)
    assert isinstance(arrays, ArrayContainer)
    assert isinstance(params, dict)
    assert obj_container.volume_idx == 0


def test_place_objects_with_multiple_objects(simple_config, simple_volume, simple_material):
    obj1 = UniformMaterialObject(name="obj1", partial_grid_shape=(20, 20, 20), material=simple_material)
    obj2 = UniformMaterialObject(name="obj2", partial_grid_shape=(20, 20, 20), material=simple_material)
    constraints = [
        GridCoordinateConstraint(object="obj1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[5, 5, 5]),
        GridCoordinateConstraint(object="obj2", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[30, 30, 30]),
    ]
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, info = place_objects(
        [simple_volume, obj1, obj2], simple_config, constraints, key
    )
    assert len(obj_container.objects) == 3
    assert obj_container.volume_idx == 0


def test_place_objects_updates_config(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(20, 20, 20), material=simple_material)
    constraint = GridCoordinateConstraint(
        object="obj1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, info = place_objects([simple_volume, obj], simple_config, [constraint], key)
    assert config is not None
    assert config.resolution == simple_config.resolution


def test_place_objects_initializes_arrays(simple_config, simple_volume, simple_material):
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(20, 20, 20), material=simple_material)
    constraint = GridCoordinateConstraint(
        object="obj1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[10, 10, 10]
    )
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, info = place_objects([simple_volume, obj], simple_config, [constraint], key)
    assert arrays.E is not None
    assert arrays.H is not None
    assert arrays.inv_permittivities is not None
    # simple_material has permittivity=2.0 → inv_perm ≈ 0.5 in the material region.
    # Verify at least one voxel was actually updated (differs from vacuum value 1.0).
    assert jnp.any(arrays.inv_permittivities < 0.9), (
        "Material with permittivity=2.0 not reflected in inv_permittivities"
    )


def test_place_objects_raises_on_unresolvable_constraint(simple_config, simple_volume, simple_material):
    """place_objects should raise ValueError when constraints can't be resolved."""
    obj = UniformMaterialObject(name="obj1", material=simple_material)
    c1 = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[10])
    c2 = GridCoordinateConstraint(object="obj1", axes=[0], sides=["-"], coordinates=[20])
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="Failed to resolve object constraints"):
        place_objects([simple_volume, obj], simple_config, [c1, c2], key)


# ---------------------------------------------------------------------------
# Anisotropic material tests – covers component-count logic and update paths
# ---------------------------------------------------------------------------


def test_diagonally_anisotropic_material(simple_config, simple_volume):
    """Diagonally anisotropic material triggers 3-component arrays for all properties.

    Covers lines: 301-302 (perm), 308-309 (permeab), 315-316 (elec cond), 322-323 (mag cond),
                  386-390 (perm update), 399, 405-409 (permeab update),
                  418-428 (elec cond update), 439-448 (mag cond update).
    """
    # Material with diagonally anisotropic (but not isotropic) values for all properties
    mat = Material(
        permittivity=(2.0, 2.5, 3.0),  # diag-aniso
        permeability=(1.5, 1.0, 2.0),  # diag-aniso, magnetic
        electric_conductivity=(0.1, 0.2, 0.3),  # diag-aniso, conductive
        magnetic_conductivity=(0.1, 0.2, 0.3),  # diag-aniso, mag-conductive
    )
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(20, 20, 20), material=mat)
    constraint = GridCoordinateConstraint(object="obj1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[5, 5, 5])
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, info = place_objects([simple_volume, obj], simple_config, [constraint], key)
    # 3-component inv_permittivities (diagonally anisotropic)
    assert arrays.inv_permittivities.shape[0] == 3
    # 3-component inv_permeabilities (diagonally anisotropic, magnetic)
    assert isinstance(arrays.inv_permeabilities, jax.Array)
    assert arrays.inv_permeabilities.shape[0] == 3
    # electric and magnetic conductivity arrays created
    assert arrays.electric_conductivity is not None
    assert arrays.magnetic_conductivity is not None


def test_fully_anisotropic_material(simple_config, simple_volume):
    """Fully anisotropic material (off-diagonal) triggers 9-component arrays.

    Covers lines: 303-304 (perm), 310-311 (permeab), 317-318 (elec cond), 324-325 (mag cond),
                  391-397 (perm update), 410-416 (permeab update),
                  429-431 (elec cond update), 450-452 (mag cond update).
    """
    mat = Material(
        permittivity=(2.0, 0.1, 0.0, 0.1, 2.5, 0.0, 0.0, 0.0, 3.0),  # off-diagonal
        permeability=(1.5, 0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 2.0),  # off-diagonal, magnetic
        electric_conductivity=(0.1, 0.01, 0.0, 0.01, 0.2, 0.0, 0.0, 0.0, 0.3),  # off-diagonal
        magnetic_conductivity=(0.1, 0.01, 0.0, 0.01, 0.2, 0.0, 0.0, 0.0, 0.3),  # off-diagonal
    )
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(20, 20, 20), material=mat)
    constraint = GridCoordinateConstraint(object="obj1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[5, 5, 5])
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, info = place_objects([simple_volume, obj], simple_config, [constraint], key)
    # 9-component inv_permittivities (fully anisotropic)
    assert arrays.inv_permittivities.shape[0] == 9
    assert isinstance(arrays.inv_permeabilities, jax.Array)
    assert arrays.inv_permeabilities.shape[0] == 9
    assert arrays.electric_conductivity is not None
    assert arrays.magnetic_conductivity is not None


# ---------------------------------------------------------------------------
# StaticMultiMaterialObject test
# ---------------------------------------------------------------------------


def test_static_multi_material_object_sphere(simple_config, simple_volume):
    """Sphere (StaticMultiMaterialObject subclass) is correctly processed in _init_arrays.

    Covers lines: 460-530 (StaticMultiMaterialObject update path).
    """
    materials = {
        "background": Material(permittivity=1.0),
        "sphere_mat": Material(permittivity=3.0),
    }
    sphere = Sphere(
        name="sphere1",
        partial_grid_shape=(20, 20, 20),
        materials=materials,
        material_name="sphere_mat",
        radius=5.0,  # 5 grid units at resolution=1.0
    )
    constraint = GridCoordinateConstraint(
        object="sphere1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[15, 15, 15]
    )
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, info = place_objects(
        [simple_volume, sphere], simple_config, [constraint], key
    )
    assert isinstance(obj_container, ObjectContainer)
    assert arrays.inv_permittivities is not None
    # sphere is in static_material_objects
    assert any(o.name == "sphere1" for o in obj_container.objects)


# ---------------------------------------------------------------------------
# PML boundary test
# ---------------------------------------------------------------------------


def test_pml_boundary_modifies_arrays(simple_config, simple_volume, simple_material):
    """PML boundary triggers boundary array modification in _init_arrays.

    Covers lines: 539-553 (boundary modify_arrays call).
    """
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(20, 20, 20), material=simple_material)
    pml = PerfectlyMatchedLayer(
        name="pml_xmin",
        axis=0,
        direction="-",
        partial_grid_shape=(10, None, None),
    )
    constraints = [
        GridCoordinateConstraint(object="obj1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[15, 15, 15]),
        GridCoordinateConstraint(object="pml_xmin", axes=[0], sides=["-"], coordinates=[0]),
    ]
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, info = place_objects(
        [simple_volume, obj, pml], simple_config, constraints, key
    )
    assert isinstance(obj_container, ObjectContainer)
    # PML is in boundary_objects
    assert len(obj_container.boundary_objects) == 1
    assert obj_container.boundary_objects[0].name == "pml_xmin"
    # Arrays were modified by PML
    assert arrays.alpha is not None
    assert arrays.kappa is not None
    assert arrays.sigma is not None


# ---------------------------------------------------------------------------
# Gradient config / recording state test
# ---------------------------------------------------------------------------


def test_recording_state_with_gradient_config(simple_volume, simple_material):
    """GradientConfig with Recorder triggers recording state initialization in _init_arrays.

    Covers lines: 558-574 (recording state initialization path).
    """
    recorder = Recorder(modules=[])
    gradient_config = GradientConfig(recorder=recorder)
    config = SimulationConfig(
        resolution=1.0,
        time=100e-15,
        backend="cpu",
        gradient_config=gradient_config,
    )
    obj = UniformMaterialObject(name="obj1", partial_grid_shape=(20, 20, 20), material=simple_material)
    constraint = GridCoordinateConstraint(object="obj1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[5, 5, 5])
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, updated_config, info = place_objects([simple_volume, obj], config, [constraint], key)
    assert updated_config.gradient_config is not None
    assert updated_config.gradient_config.recorder is not None
    # The recording state should be initialized (not None) when a Recorder is present.
    assert arrays.recording_state is not None


# ---------------------------------------------------------------------------
# Device test – _init_params
# ---------------------------------------------------------------------------


def test_device_init_params(simple_config, simple_volume):
    """Device in object list triggers _init_params loop body.

    Covers lines: 609-611 (_init_params device initialization).
    """
    materials = {
        "mat1": Material(permittivity=1.0),
        "mat2": Material(permittivity=2.0),
    }
    device = Device(
        name="device1",
        partial_grid_shape=(20, 20, 20),
        partial_voxel_grid_shape=(4, 4, 4),
        materials=materials,
        param_transforms=[],  # empty → output_type=CONTINUOUS, needs exactly 2 materials ✓
    )
    constraint = GridCoordinateConstraint(
        object="device1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[5, 5, 5]
    )
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, info = place_objects(
        [simple_volume, device], simple_config, [constraint], key
    )
    # params should contain an entry for the device
    assert "device1" in params
    # The params should be a JAX array
    assert isinstance(params["device1"], jax.Array)
