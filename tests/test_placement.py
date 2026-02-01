import jax
import numpy as np

import fdtdx
from fdtdx.units import m, s


def test_placement():
    config = fdtdx.SimulationConfig(
        resolution=100e-9 * m,
        time=10e-15 * s,
    )

    constraints = []
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(2e-6 * m, 2e-6 * m, 2e-6 * m),
    )

    # boundaries
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=4)
    _, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)

    # detectors
    d_energy = fdtdx.EnergyDetector(partial_real_shape=(1e-6 * m, 1e-6 * m, 1e-6 * m))
    constraints.append(d_energy.same_position(volume))

    d_poynting = fdtdx.PoyntingFluxDetector(
        direction="-",
        partial_real_shape=(1e-6 * m, 1e-6 * m, None),
        partial_grid_shape=(None, None, 1),
    )
    constraints.append(d_poynting.same_position(volume))

    d_field = fdtdx.FieldDetector(partial_real_shape=(1e-6 * m, 1e-6 * m, 1e-6 * m))
    constraints.append(d_field.same_position(volume))

    d_mode = fdtdx.ModeOverlapDetector(
        direction="-",
        wave_characters=(fdtdx.WaveCharacter(wavelength=1e-6 * m),),
        partial_real_shape=(1e-6 * m, 1e-6 * m, None),
        partial_grid_shape=(None, None, 1),
    )
    constraints.append(d_mode.same_position(volume))

    d_phasor = fdtdx.PhasorDetector(
        wave_characters=(fdtdx.WaveCharacter(wavelength=1e-6 * m),),
        partial_real_shape=(1e-6 * m, 1e-6 * m, 1e-6 * m),
    )
    constraints.append(d_phasor.same_position(volume))

    # device
    materials = {
        "air": fdtdx.Material(),
        "polymer": fdtdx.Material(permittivity=2.25),
    }
    device = fdtdx.Device(
        partial_real_shape=(1e-6 * m, 1e-6 * m, 1e-6 * m),
        partial_voxel_real_shape=(0.2e-6 * m, 0.2e-6 * m, 0.2e-6 * m),
        materials=materials,
        param_transforms=[],
    )
    constraints.append(device.same_position(volume))

    # sources
    plane_source = fdtdx.UniformPlaneSource(
        partial_real_shape=(1e-6 * m, 1e-6 * m, None),
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1e-6 * m),
        direction="-",
    )
    constraints.append(plane_source.same_position(volume))

    gauss_source = fdtdx.GaussianPlaneSource(
        radius=1e-6 * m,
        partial_real_shape=(1e-6 * m, 1e-6 * m, None),
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1e-6 * m),
        direction="-",
    )
    constraints.append(gauss_source.same_position(volume))

    mode_source = fdtdx.ModePlaneSource(
        partial_real_shape=(1e-6 * m, 1e-6 * m, None),
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1e-6 * m),
        direction="-",
    )
    constraints.append(mode_source.same_position(volume))

    # uniform material box
    box = fdtdx.UniformMaterialObject(
        partial_real_shape=(1e-6 * m, 1e-6 * m, 1e-6 * m),
        material=materials["polymer"],
    )
    constraints.append(box.same_position(volume))

    # multi material objects
    polygon = fdtdx.ExtrudedPolygon(
        partial_real_shape=(1e-6 * m, 1e-6 * m, 1e-6 * m),
        materials=materials,
        axis=2,
        vertices=m * np.asarray([[0, 0], [0, 200e-9], [200e-9, 200e-9], [200e-9, 100e-9], [0, 0]]),
        material_name="polymer",
    )
    constraints.append(polygon.same_position(volume))

    sphere = fdtdx.Sphere(
        partial_real_shape=(1e-6 * m, 1e-6 * m, 1e-6 * m),
        materials=materials,
        radius=300e-9,
        material_name="polymer",
    )
    constraints.append(sphere.same_position(volume))

    cylinder = fdtdx.Cylinder(
        axis=2,
        partial_real_shape=(1e-6 * m, 1e-6 * m, 1e-6 * m),
        materials=materials,
        radius=300e-9 * m,
        material_name="polymer",
    )
    constraints.append(cylinder.same_position(volume))

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
