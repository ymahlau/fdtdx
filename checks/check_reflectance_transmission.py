import math
import time

import jax
import jax.numpy as jnp
import optax
import pytreeclass as tc
from loguru import logger

from fdtdx.core.switch import OnOffSwitch
from fdtdx.fdtd import (
    ArrayContainer,
    ParameterContainer,
    apply_params,
    place_objects,
    reversible_fdtd
)
from fdtdx.config import  SimulationConfig
from fdtdx import constants
from fdtdx.materials import Material
from fdtdx.objects import  SimulationVolume
from fdtdx.objects.boundaries import BoundaryConfig, boundary_objects_from_config
from fdtdx.objects.detectors import EnergyDetector
from fdtdx.objects.detectors.poynting_flux import PoyntingFluxDetector
from fdtdx.objects.sources import SimplePlaneSource
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.utils import Logger, plot_setup


def main():
    exp_logger = Logger(
        experiment_name="reflectance_transmission",
    )
    key = jax.random.PRNGKey(seed=42)

    wavelength = 1.55e-6
    angle_degree = 15
    angle_radians = angle_degree * math.pi / 180
    size_xy = wavelength / math.sin(angle_radians)
    resolution = size_xy / 100
    logger.info(f"{angle_degree=}")
    logger.info(f"{size_xy=}")
    logger.info(f"{resolution=}")
    
    period = constants.wavelength_to_period(wavelength)

    config = SimulationConfig(
        time=100e-15,
        resolution=resolution,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    period_steps = round(period / config.time_step_duration)
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    constraints = []

    volume = SimulationVolume(
        partial_real_shape=(size_xy, size_xy, 2 * size_xy),
        material=Material(  # Background material
            permittivity=1.0,
        )
    )

    bound_cfg = BoundaryConfig(
        boundary_type_maxx="periodic",
        boundary_type_maxy="periodic",
        boundary_type_minx="periodic",
        boundary_type_miny="periodic",
    )
    bound_dict, c_list = boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)

    source = SimplePlaneSource(
        partial_grid_shape=(None, None, 1),
        fixed_E_polarization_vector=(1, 0, 0),
        wave_character=WaveCharacter(wavelength=1.550e-6),
        direction="-",
        azimuth_angle=angle_degree,
        switch=OnOffSwitch(start_after_periods=2, on_for_periods=2, period=period),
    )
    constraints.extend(
        [
            source.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(0, 0, 1),
                other_positions=(0, 0, 1),
                grid_margins=(0, 0, -15),
            ),
        ]
    )
    
    flux_below_source = PoyntingFluxDetector(
        name="flux",
        partial_grid_shape=(1, 1, 1),
        direction="-",
        switch=OnOffSwitch(
            start_after_periods=10,
            period=period,
        ),
        exact_interpolation=True,
        keep_all_components=True,
        fixed_propagation_axis=2,
    )
    constraints.extend(
        [
            flux_below_source.place_relative_to(
                source,
                axes=(0, 1, 2),
                own_positions=(0, 0, 1),
                other_positions=(0, 0, 1),
                grid_margins=(0, 0, -5),
            ),
        ]
    )

    video_energy_detector = EnergyDetector(
        name="Energy Video",
        as_slices=True,
        switch=OnOffSwitch(interval=3),
        exact_interpolation=True,
    )
    constraints.extend(video_energy_detector.same_position_and_size(volume))

    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = place_objects(
        volume=volume,
        config=config,
        constraints=constraints,
        key=subkey,
    )
    logger.info(tc.tree_summary(arrays, depth=1))
    print(tc.tree_diagram(config, depth=4))

    exp_logger.savefig(
        exp_logger.cwd,
        "setup.png",
        plot_setup(
            config=config,
            objects=objects,
            exclude_object_list=[
                # video_energy_detector,
            ],
        ),
    )

    apply_params(arrays, objects, params, key)

    def sim_fn(
        params: ParameterContainer,
        arrays: ArrayContainer,
        key: jax.Array,
    ):
        arrays, new_objects, info = apply_params(arrays, objects, params, key)

        final_state = reversible_fdtd(
            arrays=arrays,
            objects=new_objects,
            config=config,
            key=key,
        )
        _, arrays = final_state

        new_info = {
            **info,
        }
        return arrays, new_info

    compile_start_time = time.time()
    jit_task_id = exp_logger.progress.add_task("JIT", total=None)
    jitted_loss = jax.jit(sim_fn, donate_argnames=["arrays"]).lower(params, arrays, key).compile()
    compile_delta_time = time.time() - compile_start_time
    exp_logger.progress.update(jit_task_id, total=1, completed=1, refresh=True)

    run_start_time = time.time()
    key, subkey = jax.random.split(key)
    arrays, info = jitted_loss(params, arrays, subkey)

    runtime_delta = time.time() - run_start_time
    info["runtime"] = runtime_delta
    info["compile time"] = compile_delta_time

    logger.info(f"{compile_delta_time=}")
    logger.info(f"{runtime_delta=}")

    exp_logger.log_detectors(iter_idx=0, objects=objects, detector_states=arrays.detector_states)
    
    # compute poynting flux value
    pf = arrays.detector_states[flux_below_source.name]["poynting_flux"].mean(axis=0)
    pf_norm = pf / optax.global_norm(pf)
    prop_vec = jnp.asarray([0, 0, 1], dtype=jnp.float32)
    measured_prop_angle = jnp.arccos(jnp.dot(prop_vec, pf_norm)) * 180 / jnp.pi
    difference = angle_degree - measured_prop_angle
    logger.info(f"{measured_prop_angle=} Degree, {difference=}")
    


if __name__ == "__main__":
    main()


