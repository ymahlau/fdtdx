import jax
import numpy as np

import fdtdx


def test_placement():
    config = fdtdx.SimulationConfig(
        resolution=100e-9,
        time=10e-15,
    )

    constraints = []
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(2e-6, 2e-6, 2e-6),
    )

    # boundaries
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=4)
    _, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)

    # detectors
    d_energy = fdtdx.EnergyDetector(partial_real_shape=(1e-6, 1e-6, 1e-6))
    constraints.append(d_energy.place_at_center(volume))

    d_poynting = fdtdx.PoyntingFluxDetector(
        direction="-",
        partial_real_shape=(1e-6, 1e-6, None),
        partial_grid_shape=(None, None, 1),
    )
    constraints.append(d_poynting.place_at_center(volume))

    d_field = fdtdx.FieldDetector(partial_real_shape=(1e-6, 1e-6, 1e-6))
    constraints.append(d_field.place_at_center(volume))

    d_mode = fdtdx.ModeOverlapDetector(
        direction="-",
        wave_characters=(fdtdx.WaveCharacter(wavelength=1e-6),),
        partial_real_shape=(1e-6, 1e-6, None),
        partial_grid_shape=(None, None, 1),
    )
    constraints.append(d_mode.place_at_center(volume))

    d_phasor = fdtdx.PhasorDetector(
        wave_characters=(fdtdx.WaveCharacter(wavelength=1e-6),),
        partial_real_shape=(1e-6, 1e-6, 1e-6),
    )
    constraints.append(d_phasor.place_at_center(volume))

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
    constraints.append(device.place_at_center(volume))

    # sources
    plane_source = fdtdx.UniformPlaneSource(
        partial_real_shape=(1e-6, 1e-6, None),
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1e-6),
        direction="-",
    )
    constraints.append(plane_source.place_at_center(volume))

    gauss_source = fdtdx.GaussianPlaneSource(
        radius=1e-6,
        partial_real_shape=(1e-6, 1e-6, None),
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1e-6),
        direction="-",
    )
    constraints.append(gauss_source.place_at_center(volume))

    mode_source = fdtdx.ModePlaneSource(
        partial_real_shape=(1e-6, 1e-6, None),
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1e-6),
        direction="-",
    )
    constraints.append(mode_source.place_at_center(volume))

    # uniform material box
    box = fdtdx.UniformMaterialObject(
        partial_real_shape=(1e-6, 1e-6, 1e-6),
        material=materials["polymer"],
    )
    constraints.append(box.place_at_center(volume))

    # multi material objects
    polygon = fdtdx.ExtrudedPolygon(
        partial_real_shape=(1e-6, 1e-6, 1e-6),
        materials=materials,
        axis=2,
        vertices=np.asarray([[0, 0], [0, 200e-9], [200e-9, 200e-9], [200e-9, 100e-9], [0, 0]]),
        material_name="polymer",
    )
    constraints.append(polygon.place_at_center(volume))

    sphere = fdtdx.Sphere(
        partial_real_shape=(1e-6, 1e-6, 1e-6),
        materials=materials,
        radius=300e-9,
        material_name="polymer",
    )
    constraints.append(sphere.place_at_center(volume))

    cylinder = fdtdx.Cylinder(
        axis=2,
        partial_real_shape=(1e-6, 1e-6, 1e-6),
        materials=materials,
        radius=300e-9,
        material_name="polymer",
    )
    constraints.append(cylinder.place_at_center(volume))

    # place the objects
    key = jax.random.PRNGKey(42)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        volume=volume,
        config=config,
        constraints=constraints,
        key=key,
    )


if __name__ == "__main__":
    test_placement()
