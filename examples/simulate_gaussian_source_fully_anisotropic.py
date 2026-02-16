"""
This script is an example of a simple simulation with a diagonally or fully anisotropic material.
In this simulation, a Gaussian source is placed at the top of the simulation volume, and an anisotropic material is placed at the bottom.
PML boundaries are placed on the top and bottom of the simulation volume, and periodic boundaries are placed on the sides.
The source is X polarized and Ey in the time domain is recorded across the entire simulation volume.
If the material is diagonally anisotropic, no Ey signal is observed.
If the material is fully anisotropic, an Ey signal is observed.
"""

# Import required libraries and modules
import time

import jax
import jax.numpy as jnp
import pytreeclass as tc
from loguru import logger
import fdtdx

polarization = 'X' #'Y'
fully_anisotropic = True

def main():
    # Initialize experiment logger for saving outputs and tracking progress
    exp_logger = fdtdx.Logger(
        experiment_name="simulate_source_fully_anisotropic",
    )
    # Create a JAX random key for reproducibility and stochastic operations
    key = jax.random.PRNGKey(seed=42)

    # Set simulation wavelength and calculate corresponding period
    wavelength = 1.55e-6
    period = fdtdx.constants.wavelength_to_period(wavelength)

    # Define simulation configuration (duration, resolution, data type, etc.)
    config = fdtdx.SimulationConfig(
        time=100e-15,
        resolution=100e-9,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    # Calculate number of time steps per period and log simulation info
    period_steps = round(period / config.time_step_duration)
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    # Set up gradient recording for simulation (for autodiff/backprop)
    gradient_config = fdtdx.GradientConfig(
        recorder=fdtdx.Recorder(
            modules=[
            ]
        )
    )
    config = config.aset("gradient_config", gradient_config)

    # List to hold placement constraints for simulation objects
    constraints, object_list = [], []

    # Define the simulation volume and background material
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(12e-6, 12e-6, 12e-6),
        material=fdtdx.Material(  # Background material
            permittivity=1.0,
            permeability=1.0,
        )
    )
    object_list.append(volume)

    # Choose boundary type
    bound_cfg = fdtdx.BoundaryConfig(
        boundary_type_minx="periodic",
        boundary_type_maxx="periodic",
        boundary_type_miny="periodic",
        boundary_type_maxy="periodic",
        boundary_type_minz="pml",
        thickness_grid_minz=10,
        boundary_type_maxz="pml",
        thickness_grid_maxz=10,
    )
    # Add boundary objects and constraints to the simulation
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    object_list.extend(list(bound_dict.values()))
    
    boundary_slab = fdtdx.UniformMaterialObject(
        partial_grid_shape=(None, None, 20),  # Slab thickness ~2 microns (2x grid points due to finer resolution)
        partial_real_shape=(12e-6, 12e-6, 6e-6),
        material=fdtdx.Material(
            permittivity=((2.5, 1.5, 0.0), (1.5, 2.5, 0.0), (0.0, 0.0, 1.0)) if fully_anisotropic else ((4.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
    )
    # Place boundary slab at the center of the volume
    constraints.extend(
        [
            boundary_slab.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(0, 0, 0),
                other_positions=(0, 0, 0),
                margins=(0, 0, -3e-6),
            ),
        ]
    )
    object_list.append(boundary_slab)
    
    # Define and place the uniform plane source in the simulation volume
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        fixed_E_polarization_vector=(1, 0, 0) if polarization == 'X' else (0, 1, 0),
        wave_character=fdtdx.WaveCharacter(wavelength=1.550e-6),
        direction="-",
    )
    constraints.extend(
        [
            source.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(0, 0, 0),
                other_positions=(0, 0, 0),
                margins=(0, 0, 4e-6),
            ),
        ]
    )
    object_list.append(source)

    # Add energy detector for video output (forward simulation)
    video_energy_detector = fdtdx.FieldDetector(
        name="Electric Field Video",
        components=("Ey",),
        switch=fdtdx.OnOffSwitch(interval=3), 
        exact_interpolation=True,
        num_video_workers=8,
    )
    constraints.extend(video_energy_detector.same_position_and_size(volume))
    object_list.append(video_energy_detector)

    # Place all objects in the simulation volume according to constraints
    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list,
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