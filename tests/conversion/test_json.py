from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from fdtdx.colors import PINK
from fdtdx.config import SimulationConfig
from fdtdx.conversion.json import JsonSetup, export_json, export_json_str, import_from_json
from fdtdx.core.switch import OnOffSwitch
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, ParameterContainer
from fdtdx.fdtd.initialization import apply_params, place_objects
from fdtdx.fdtd.wrapper import run_fdtd
from fdtdx.materials import Material
from fdtdx.objects.boundaries.initialization import BoundaryConfig, boundary_objects_from_config
from fdtdx.objects.detectors.energy import EnergyDetector
from fdtdx.objects.sources.linear_polarization import UniformPlaneSource
from fdtdx.objects.static_material.static import SimulationVolume, UniformMaterialObject


@pytest.fixture
def setup_simulation_inputs():
    """Set up a basic simulation inputs for testing."""
    seed = 42
    key = jax.random.PRNGKey(seed=seed)

    object_list = []
    constraints = []

    # Simulation config
    config = SimulationConfig(time=100e-15, resolution=100e-9, dtype=jnp.float32, courant_factor=0.99)

    # Volume
    volume = SimulationVolume(
        partial_real_shape=(12.0e-6, 12e-6, 12e-6),
        material=Material(
            permittivity=1.0,
            permeability=1.0,
        ),
    )
    object_list.append(volume)

    # Boundaries
    bound_cfg = BoundaryConfig.from_uniform_bound(thickness=10, boundary_type="pml")
    bound_dict, c_list = boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    object_list.extend(bound_dict.values())

    # Source
    source = UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(10e-6, 10e-6, None),
        fixed_E_polarization_vector=(1, 0, 0),
        wave_character=WaveCharacter(wavelength=1.550e-6),
        direction="-",
    )
    object_list.append(source)

    constraints.extend(
        [
            source.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(0, 0, 1),
                other_positions=(0, 0, 1),
                margins=(0, 0, -1.5e-6),
            ),
        ]
    )

    # Cube
    cube = UniformMaterialObject(
        partial_real_shape=(3e-6, 3e-6, 3e-6),
        material=Material(permittivity=2.5, permeability=1.7),
        name="Cube",
        color=PINK,
    )
    object_list.append(cube)
    constraints.append(cube.place_at_center(volume))

    # Detector
    det = EnergyDetector(
        name="Detector",
        as_slices=True,
        switch=OnOffSwitch(interval=3),
        exact_interpolation=True,
        num_video_workers=8,
    )

    object_list.append(det)
    constraints.extend(det.same_position_and_size(volume))

    return {
        "seed": seed,
        "key": key,
        "config": config,
        "volume": volume,
        "source": source,
        "object": cube,
        "detector": det,
        "object_list": object_list,
        "constraints": constraints,
    }


@pytest.fixture
def jsonsetup_path(request) -> Path:
    test_dir = Path(str(request.fspath)).parent
    return test_dir / "setup_test.json"


def test_detector_json(setup_simulation_inputs):
    """test JSON serialization and deserialization of the detector."""
    det = setup_simulation_inputs["detector"]
    d = export_json(det)
    assert d["name"] == "Detector"
    s = export_json_str(det)
    rec = import_from_json(s)
    assert rec.name == "Detector"


def test_config_json(setup_simulation_inputs):
    """test JSON serialization and deserialization of the simulation config."""
    conf = setup_simulation_inputs["config"]
    c = export_json(conf)
    assert c["time"] == 100e-15
    s = export_json_str(conf)
    rec = import_from_json(s)
    assert rec.time == 100e-15
    assert rec.resolution == 100e-9


def test_volume_json(setup_simulation_inputs):
    """test JSON serialization and deserialization of the volume."""
    vol = setup_simulation_inputs["volume"]
    c = export_json(vol)
    assert c["partial_real_shape"]["__value__"] == [12.0e-6, 12e-6, 12e-6]
    s = export_json_str(vol)
    rec = import_from_json(s)
    assert rec.partial_real_shape == (12.0e-6, 12e-6, 12e-6)

    expected_eps = Material(permittivity=1.0).permittivity
    expected_mu = Material(permeability=1.0).permeability
    assert rec.material.permittivity == expected_eps
    assert rec.material.permeability == expected_mu


def test_source_json(setup_simulation_inputs):
    """test JSON serialization and deserialization of the source."""
    sor = setup_simulation_inputs["source"]
    c = export_json(sor)
    assert c["partial_real_shape"]["__value__"] == [10e-6, 10e-6, None]
    assert c["fixed_E_polarization_vector"]["__value__"] == [1, 0, 0]
    s = export_json_str(sor)
    rec = import_from_json(s)
    assert rec.partial_real_shape == (10e-6, 10e-6, None)
    assert rec.wave_character.wavelength == 1.550e-6


def test_object_json(setup_simulation_inputs):
    """test JSON serialization and deserialization of the object."""
    obj = setup_simulation_inputs["object"]
    c = export_json(obj)
    assert c["partial_real_shape"]["__value__"] == [3e-6, 3e-6, 3e-6]
    assert c["name"] == "Cube"
    s = export_json_str(obj)
    rec = import_from_json(s)
    assert rec.partial_real_shape == (3e-6, 3e-6, 3e-6)
    assert rec.name == "Cube"
    assert rec.color == PINK


def test_constraints_json(setup_simulation_inputs):
    """test JSON serialization and deserialization of the constraints."""
    cond = setup_simulation_inputs["constraints"]
    items = export_json(cond)["__value__"]

    keys = {(it["__name__"], it["__value__"].get("object"), it["__value__"].get("other_object")) for it in items}

    cube_other = next(other for (k, obj, other) in keys if k == "PositionConstraint" and obj == "Cube")
    assert isinstance(cube_other, str) and cube_other.startswith("Object_")

    assert ("PositionConstraint", "Detector", cube_other) in keys
    assert ("SizeConstraint", "Detector", cube_other) in keys


def test_object_container_json():
    """test JSON serialization and deserialization of the object container."""
    obj_list = [
        EnergyDetector(),
        UniformPlaneSource(wave_character=WaveCharacter(wavelength=1e-6), direction="-"),
        SimulationVolume(),
    ]
    container = ObjectContainer(object_list=obj_list, volume_idx=2)
    s = export_json_str(container)
    rec = import_from_json(s)
    assert isinstance(rec.object_list, list)
    assert rec.object_list[1].wave_character.wavelength == 1e-6
    assert rec.object_list[1].direction == "-"


def test_run_simulation_with_imported_json(setup_simulation_inputs):
    """Ensures that a simulation can be executed from exported JSON settings."""
    key = setup_simulation_inputs["key"]
    key, subkey = jax.random.split(key)
    config = setup_simulation_inputs["config"]
    object_list = setup_simulation_inputs["object_list"]
    constraints = setup_simulation_inputs["constraints"]

    export_data = {
        "SimulationConfig": config,  # class
        "object_list": object_list,  # list
        "constraints": constraints,  # list
    }
    json_export = export_json(export_data)
    assert set(json_export.keys()) == {"__module__", "__name__", "SimulationConfig", "object_list", "constraints"}
    json_export_str = export_json_str(export_data)
    setting = import_from_json(json_export_str)

    objects, arrays, params, config, info = place_objects(
        object_list=setting["object_list"],
        config=setting["SimulationConfig"],
        constraints=setting["constraints"],
        key=subkey,
    )

    def sim_fn(
        params: ParameterContainer,
        arrays: ArrayContainer,
        key: jax.Array,
    ):
        # Apply parameters to objects and arrays
        arrays, new_objects, _ = apply_params(arrays, objects, params, key)

        # Run FDTD simulation (forward)
        final_state = run_fdtd(
            arrays=arrays,
            objects=new_objects,
            config=config,
            key=key,
        )
        _, arrays = final_state

        return arrays

    jitted_loss = jax.jit(sim_fn).lower(params, arrays, key).compile()
    new_arrays = jitted_loss(params, arrays, subkey)
    assert new_arrays is not None


def test_jsonsetup_dumps_and_loads(setup_simulation_inputs):
    "test json dumps and loads from JsonSetup"
    config = setup_simulation_inputs["config"]
    object_list = setup_simulation_inputs["object_list"]
    constraints = setup_simulation_inputs["constraints"]

    setup = JsonSetup(
        config=config,
        object_list=list(object_list),
        constraints=list(constraints),
        meta={"seed": setup_simulation_inputs["seed"], "test": "other meta data"},
    )

    s = setup.dumps()
    setup2 = JsonSetup.loads(s)

    assert isinstance(setup2, JsonSetup)
    assert isinstance(setup2.object_list, list)
    assert isinstance(setup2.constraints, list)
    assert setup2.meta == {"seed": 42, "test": "other meta data"}


def test_run_simulation_with_jsonsetup(jsonsetup_path: Path):
    """Ensures that a simulation can be executed by loaded json setup."""
    setup2 = JsonSetup.load_json(jsonsetup_path)
    assert setup2 is not None
    assert setup2.meta is not None
    seed = setup2.meta.get("seed")
    assert seed is not None
    key = jax.random.PRNGKey(seed)
    assert seed is not None
    key, subkey = jax.random.split(key)

    objects, arrays, params, cfg2, _ = place_objects(
        object_list=setup2.object_list,
        config=setup2.config,
        constraints=setup2.constraints,
        key=subkey,
    )

    def sim_fn(params: ParameterContainer, arrays: ArrayContainer, key: jax.Array):
        # Apply parameters to objects and arrays
        arrays, new_objects, _ = apply_params(arrays, objects, params, key)

        # Run FDTD simulation (forward)
        final_state = run_fdtd(arrays=arrays, objects=new_objects, config=cfg2, key=key)
        _, arrays = final_state
        return arrays

    compiled = jax.jit(sim_fn).lower(params, arrays, key).compile()
    out = compiled(params, arrays, subkey)
    assert out is not None
