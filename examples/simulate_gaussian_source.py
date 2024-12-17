import time

import jax
import jax.numpy as jnp
import pytreeclass as tc
from loguru import logger

from fdtdx.core.config import GradientConfig, SimulationConfig
from fdtdx.core.physics import constants
from fdtdx.fdtd.backward import full_backward
from fdtdx.fdtd.fdtd import checkpointed_fdtd
from fdtdx.interfaces.modules import DtypeConversion
from fdtdx.interfaces.recorder import Recorder
from fdtdx.objects.boundaries.initialization import BoundaryConfig, pml_objects_from_config
from fdtdx.objects.container import ArrayContainer, ParameterContainer
from fdtdx.objects.detectors.energy import EnergyDetector
from fdtdx.objects.initialization import apply_params, place_objects
from fdtdx.objects.material import SimulationVolume
from fdtdx.objects.sources.plane_source import GaussianPlaneSource
from fdtdx.shared.logger import Logger
from fdtdx.shared.plot_setup import plot_setup


def main():
    exp_logger = Logger(
        experiment_name="simulate_source",
    )
    key = jax.random.PRNGKey(seed=42)

    wavelength = 1.55e-6
    period = constants.wavelength_to_period(wavelength)

    config = SimulationConfig(
        time=100e-15,
        resolution=100e-9,
        dtype=jnp.float32,
        courant_factor=0.9,
    )

    period_steps = round(period / config.time_step_duration)
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    gradient_config = GradientConfig(
        recorder=Recorder(
            modules=[
                DtypeConversion(dtype=jnp.bfloat16),
            ]
        )
    )
    config = config.aset("gradient_config", gradient_config)

    constraints = []

    volume = SimulationVolume(
        partial_real_shape=(12.0e-6, 12e-6, 12e-6),
    )

    bound_cfg = BoundaryConfig.from_uniform_bound(thickness=10)
    bound_dict, c_list = pml_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)

    source = GaussianPlaneSource(
        # partial_grid_shape=(None, None, 1),
        # partial_real_shape=(10e-6, 10e-6, None),
        # fixed_E_polarization_vector=(1, 0, 0),
        # partial_grid_shape=(1, None, None),
        # partial_real_shape=(None, 10e-6, 10e-6),
        # fixed_E_polarization_vector=(0, 1, 0),
        partial_grid_shape=(None, 1, None),
        partial_real_shape=(10e-6, None, 10e-6),
        fixed_E_polarization_vector=(1, 0, 0),
        wavelength=1.550e-6,
        radius=4e-6,
        std=1 / 3,
        direction="-",
        # azimuth_angle=20.0,
        elevation_angle=-20.0,
    )

    constraints.extend(
        [
            source.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(0, 0, 0),
                other_positions=(0, 0, 0),
            ),
        ]
    )

    video_energy_detector = EnergyDetector(
        name="Energy Video",
        as_slices=True,
        interval=3,
        exact_interpolation=True,
        plot_interpolation="nearest",
    )
    constraints.extend(video_energy_detector.same_position_and_size(volume))

    backwards_video_energy_detector = EnergyDetector(
        name="Backwards Energy Video",
        as_slices=True,
        interval=3,
        inverse=True,
        exact_interpolation=True,
    )
    constraints.extend(backwards_video_energy_detector.same_position_and_size(volume))

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
                backwards_video_energy_detector,
                video_energy_detector,
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

        final_state = checkpointed_fdtd(
            arrays=arrays,
            objects=new_objects,
            config=config,
            key=key,
        )
        _, arrays = final_state

        _, arrays = full_backward(
            state=final_state,
            objects=new_objects,
            config=config,
            key=key,
            record_detectors=True,
            reset_fields=True,
        )

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

    # videos
    exp_logger.log_detectors(iter_idx=0, objects=objects, detector_states=arrays.detector_states)


if __name__ == "__main__":
    main()
