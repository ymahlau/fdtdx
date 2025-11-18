import jax
import numpy as np

import fdtdx


def test_placement():
    config = fdtdx.SimulationConfig(
        resolution=100e-9,
        time=10e-15,
    )

    constraints, objects = [], []
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(2e-6, 2e-6, 2e-6),
    )
    objects.append(volume)

    # boundaries
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=4)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    # detectors
    d_energy = fdtdx.EnergyDetector(partial_real_shape=(1e-6, 1e-6, 1e-6))
    constraints.append(d_energy.same_position(volume))
    objects.append(d_energy)

    d_poynting = fdtdx.PoyntingFluxDetector(
        direction="-",
        partial_real_shape=(1e-6, 1e-6, None),
        partial_grid_shape=(None, None, 1),
    )
    constraints.append(d_poynting.same_position(volume))
    objects.append(d_poynting)

    d_field = fdtdx.FieldDetector(partial_real_shape=(1e-6, 1e-6, 1e-6))
    constraints.append(d_field.same_position(volume))
    objects.append(d_field)

    d_mode = fdtdx.ModeOverlapDetector(
        direction="-",
        wave_characters=(fdtdx.WaveCharacter(wavelength=1e-6),),
        partial_real_shape=(1e-6, 1e-6, None),
        partial_grid_shape=(None, None, 1),
    )
    constraints.append(d_mode.same_position(volume))
    objects.append(d_mode)

    d_phasor = fdtdx.PhasorDetector(
        wave_characters=(fdtdx.WaveCharacter(wavelength=1e-6),),
        partial_real_shape=(1e-6, 1e-6, 1e-6),
    )
    constraints.append(d_phasor.same_position(volume))
    objects.append(d_phasor)

    # device
    materials = {
        "air": fdtdx.Material(),
        "polymer": fdtdx.Material(permittivity=2.25),
    }
    device = fdtdx.Device(
        partial_real_shape=(1e-6, 1e-6, 1e-6),
        partial_voxel_real_shape=(0.2e-6, 0.2e-6, 0.2e-6),
        materials=materials,
        param_transforms=[],
    )
    constraints.append(device.same_position(volume))
    objects.append(device)

    # sources
    plane_source = fdtdx.UniformPlaneSource(
        partial_real_shape=(1e-6, 1e-6, None),
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1e-6),
        direction="-",
    )
    constraints.append(plane_source.same_position(volume))
    objects.append(plane_source)

    gauss_source = fdtdx.GaussianPlaneSource(
        radius=1e-6,
        partial_real_shape=(1e-6, 1e-6, None),
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1e-6),
        direction="-",
    )
    constraints.append(gauss_source.same_position(volume))
    objects.append(gauss_source)

    mode_source = fdtdx.ModePlaneSource(
        partial_real_shape=(1e-6, 1e-6, None),
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1e-6),
        direction="-",
    )
    constraints.append(mode_source.same_position(volume))
    objects.append(mode_source)

    # uniform material box
    box = fdtdx.UniformMaterialObject(
        partial_real_shape=(1e-6, 1e-6, 1e-6),
        material=materials["polymer"],
    )
    constraints.append(box.same_position(volume))
    objects.append(box)

    # multi material objects
    polygon = fdtdx.ExtrudedPolygon(
        partial_real_shape=(1e-6, 1e-6, 1e-6),
        materials=materials,
        axis=2,
        vertices=np.asarray([[0, 0], [0, 200e-9], [200e-9, 200e-9], [200e-9, 100e-9], [0, 0]]),
        material_name="polymer",
    )
    constraints.append(polygon.same_position(volume))
    objects.append(polygon)

    sphere = fdtdx.Sphere(
        partial_real_shape=(1e-6, 1e-6, 1e-6),
        materials=materials,
        radius=300e-9,
        material_name="polymer",
    )
    constraints.append(sphere.same_position(volume))
    objects.append(sphere)

    cylinder = fdtdx.Cylinder(
        axis=2,
        partial_real_shape=(1e-6, 1e-6, 1e-6),
        materials=materials,
        radius=300e-9,
        material_name="polymer",
    )
    constraints.append(cylinder.same_position(volume))
    objects.append(cylinder)

    # place the objects
    key = jax.random.PRNGKey(42)
    object_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
