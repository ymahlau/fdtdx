# Import required libraries and modules
import time

import jax
import jax.numpy as jnp
import pytreeclass as tc
from loguru import logger
import fdtdx
import fdtdx.units as fu

def main():
    # Initialize experiment logger for saving outputs and tracking progress
    exp_logger = fdtdx.Logger(
        experiment_name="simulate_source",
    )
    # Create a JAX random key for reproducibility and stochastic operations
    key = jax.random.PRNGKey(seed=42)

    # Set simulation wavelength and calculate corresponding period
    wavelength = 1.55e-6 * fu.m
    period = fdtdx.constants.wavelength_to_period(wavelength)

    # Define simulation configuration (duration, resolution, data type, etc.)
    config = fdtdx.SimulationConfig(
        time=100e-15 * fu.s,
        resolution=100e-9 * fu.m,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    # Calculate number of time steps per period and log simulation info
    exact_period_steps = (period / config.time_step_duration).float_materialise()
    period_steps = round(exact_period_steps)
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    # Set up gradient recording for simulation (for autodiff/backprop)
    gradient_config = fdtdx.GradientConfig(
        recorder=fdtdx.Recorder(
            modules=[
                fdtdx.DtypeConversion(dtype=jnp.bfloat16),
            ]
        )
    )
    config = config.aset("gradient_config", gradient_config)

    # List to hold placement constraints for simulation objects
    constraints = []

    # Define the simulation volume and background material
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(12.0e-6 * fu.m, 12e-6 * fu.m, 12e-6 * fu.m),
        material=fdtdx.Material(  # Background material
            permittivity=2.0,
        )
    )

    # Choose boundary type: periodic or PML (absorbing)
    periodic = True
    if periodic:
        bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic")
    else:
        bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=10, boundary_type="pml")
    # Add boundary objects and constraints to the simulation
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)

    # Define and place the Gaussian source in the simulation volume
    source = fdtdx.GaussianPlaneSource(
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(10e-6 * fu.m, 10e-6 * fu.m, None),
        fixed_E_polarization_vector=(1, 0, 0),
        wave_character=fdtdx.WaveCharacter(wavelength=1.550e-6 * fu.m),
        radius=4e-6 * fu.m,
        std=1 / 3,
        direction="-",
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

    # Add energy detector for video output (forward simulation)
    video_energy_detector = fdtdx.EnergyDetector(
        name="Energy Video",
        as_slices=True,
        switch=fdtdx.OnOffSwitch(interval=3),
        exact_interpolation=True,
        num_video_workers=8,
    )
    constraints.extend(video_energy_detector.same_position_and_size(volume))

    # Add energy detector for video output (backward simulation)
    backwards_video_energy_detector = fdtdx.EnergyDetector(
        name="Backwards Energy Video",
        as_slices=True,
        switch=fdtdx.OnOffSwitch(interval=3),
        inverse=True,
        exact_interpolation=True,
        num_video_workers=8, 
    )
    constraints.extend(backwards_video_energy_detector.same_position_and_size(volume))

    # Place all objects in the simulation volume according to constraints
    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        volume=volume,
        config=config,
        constraints=constraints,
        key=subkey,
    )
    # Log summary of arrays and print configuration diagram
    logger.info(tc.tree_summary(arrays, depth=1))
    print(tc.tree_diagram(config, depth=4))

    # Save initial setup figure for experiment documentation
    exp_logger.savefig(
        exp_logger.cwd,
        "setup.png",
        fdtdx.plot_setup(
            config=config,
            objects=objects,
            exclude_object_list=[
                backwards_video_energy_detector,
                video_energy_detector,
            ],
        ),
    )

    # Define simulation function (forward and backward propagation)
    def sim_fn(
        params: fdtdx.ParameterContainer,
        arrays: fdtdx.ArrayContainer,
        key: jax.Array,
    ):
        # Apply parameters to objects and arrays
        arrays, new_objects, info = fdtdx.apply_params(arrays, objects, params, key)

        # Run FDTD simulation (forward)
        final_state = fdtdx.run_fdtd(
            arrays=arrays,
            objects=new_objects,
            config=config,
            key=key,
        )
        _, arrays = final_state

        # Run backward simulation (for gradients or adjoint fields)
        _, arrays = fdtdx.full_backward(
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

    # Compile simulation function with JAX JIT for fast execution
    compile_start_time = time.time()
    jit_task_id = exp_logger.progress.add_task("JIT", total=None)
    jitted_loss = jax.jit(sim_fn, donate_argnames=["arrays"]).lower(params, arrays, key).compile()
    compile_delta_time = time.time() - compile_start_time
    exp_logger.progress.update(jit_task_id, total=1, completed=1, refresh=True)

    # Run the simulation and measure runtime
    run_start_time = time.time()
    key, subkey = jax.random.split(key)
    arrays, info = jitted_loss(params, arrays, subkey)

    runtime_delta = time.time() - run_start_time
    info["runtime"] = runtime_delta
    info["compile time"] = compile_delta_time

    # Log compile and runtime info
    logger.info(f"{compile_delta_time=}")
    logger.info(f"{runtime_delta=}")

    # Save detector outputs (e.g., video frames) for visualization
    exp_logger.log_detectors(iter_idx=0, objects=objects, detector_states=arrays.detector_states)


# Entry point: run main function
if __name__ == "__main__":
    main()
