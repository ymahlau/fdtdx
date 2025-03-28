import time

import jax
import jax.numpy as jnp
import pytreeclass as tc
from loguru import logger

from fdtdx.fdtd import (
    reversible_fdtd,
    ArrayContainer,
    ParameterContainer,
    apply_params,
    place_objects,
)
from fdtdx.config import SimulationConfig
from fdtdx import constants
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects import SimulationVolume
from fdtdx.objects.boundaries import BoundaryConfig, boundary_objects_from_config
from fdtdx.objects.detectors import EnergyDetector
from fdtdx.objects.sources import GaussianPlaneSource
from fdtdx.utils import Logger, plot_setup


def main():
    exp_logger = Logger(
        experiment_name="line_detector_time_series_test",
    )
    key = jax.random.PRNGKey(seed=42)

    # Simple simulation setup
    wavelength = 1.55e-6
    period = constants.wavelength_to_period(wavelength)

    config = SimulationConfig(
        time=150e-15,  # Longer simulation to capture time evolution
        resolution=100e-9,
    )

    period_steps = round(period / config.time_step_duration)
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")

    constraints = []
    volume = SimulationVolume(
        partial_real_shape=(12e-6, 12e-6, 12e-6),
    )

    # Setup boundaries
    bound_cfg = BoundaryConfig.from_uniform_bound(thickness=20, boundary_type="pml")
    bound_dict, c_list = boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)

    # Create a source that will generate a pattern in the middle of the volume
    source = GaussianPlaneSource(
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(10e-6, 10e-6, None),
        fixed_E_polarization_vector=(1, 0, 0),
        wave_character=WaveCharacter(wavelength=wavelength),
        radius=4e-6,
        std=1 / 3,
        direction="-",
    )
    constraints.extend(
        [
            source.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(0, 0, 1),
                other_positions=(0, 0, 1),
                grid_margins=(0, 0, -bound_cfg.thickness_grid_maxz - 5)
            ),
        ]
    )

    # Create line detectors along each axis that record at multiple time steps
    line_detector_x = EnergyDetector(
        name="1D Line X Time Series",
        partial_grid_shape=(None, 1, 1),  # Line along X-axis
        partial_real_shape=(None, 0.1e-6, 0.1e-6),
        interval=3,  # Record every 3 time steps
        exact_interpolation=True,
        plot=True,
    )
    
    line_detector_y = EnergyDetector(
        color=(0, 0, 1),
        name="1D Line Y Time Series",
        partial_grid_shape=(1, None, 1),  # Line along Y-axis
        partial_real_shape=(0.1e-6, None, 0.1e-6),
        interval=3,  # Record every 3 time steps
        exact_interpolation=True,
        plot=True,
    )
    
    line_detector_z = EnergyDetector(
        color=(1, 0, 0),
        name="1D Line Z Time Series",
        partial_grid_shape=(1, 1, None),  # Line along Z-axis
        partial_real_shape=(0.1e-6, 0.1e-6, None),
        interval=3,  # Record every 3 time steps
        exact_interpolation=True,
        plot=True,
    )

    # Also add a full volume detector for reference
    video_energy_detector = EnergyDetector(
        color=(0, 43/255, 22/255),
        name="Energy Video",
        as_slices=True,
        interval=10,
        exact_interpolation=True,
    )
    constraints.extend(video_energy_detector.same_position_and_size(volume))
    
    # Place the line detectors to cross at the center of the volume
    constraints.extend(
        [
            line_detector_x.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(0, 0, 0),
                other_positions=(0, 0, 0),
            ),
            line_detector_y.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(0, 0, 0),
                other_positions=(0, 0, 0),
            ),
            line_detector_z.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(0, 0, 0),
                other_positions=(0, 0, 0),
            ),
        ]
    )

    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = place_objects(
        volume=volume,
        config=config,
        constraints=constraints,
        key=subkey,
    )
    logger.info(tc.tree_summary(arrays, depth=1))
    print(tc.tree_diagram(config, depth=4))

    # Show setup visualization
    exp_logger.savefig(
        exp_logger.cwd,
        "setup.png",
        plot_setup(
            config=config,
            objects=objects,
        ),
    )

    # Run simulation
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

    logger.info("Compiling simulation function...")
    compile_start_time = time.time()
    jit_task_id = exp_logger.progress.add_task("JIT", total=None)
    jitted_sim = jax.jit(sim_fn, donate_argnames=["arrays"]).lower(params, arrays, key).compile()
    compile_delta_time = time.time() - compile_start_time
    exp_logger.progress.update(jit_task_id, total=1, completed=1, refresh=True)

    logger.info("Running simulation...")
    run_start_time = time.time()
    key, subkey = jax.random.split(key)
    arrays, info = jitted_sim(params, arrays, subkey)

    runtime_delta = time.time() - run_start_time
    info["runtime"] = runtime_delta
    info["compile time"] = compile_delta_time

    logger.info(f"{compile_delta_time=}")
    logger.info(f"{runtime_delta=}")

    # Generate and save detector visualizations
    logger.info("Generating detector visualizations...")
    exp_logger.log_detectors(iter_idx=0, objects=objects, detector_states=arrays.detector_states)
    exp_logger.write(info)

    logger.info("Complete! Check the waterfall plots in the output directory.")


if __name__ == "__main__":
    main()
