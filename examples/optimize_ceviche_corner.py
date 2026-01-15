"""
This scripts exemplifies how fdtdx can be used for inverse design. The component that is supposed to be optimized here
is a corner component, connecting two silicon waveguides with as low loss as possible. This is one of the examples from
the ceviche challenges, which can be found here: https://github.com/google/ceviche-challenges

The scripts first sets up the simulation scene with sources, detectors, waveguides etc. Then a loss function is defined,
which takes the component parameters as input, runs a simulation and computes the figure of merit. Since it is standard
in machine learning to minimize, the figure of merit is negated and returned as a loss value.
Using gradient descend, the design parameters are continously updated in a loop.
"""

# Imports: standard libraries and required packages
import sys
import time

import chex
import jax
import jax.numpy as jnp
import optax
import pytreeclass as tc
from loguru import logger
import fdtdx

def main(
    seed: int,
    evaluation: bool,
    backward: bool,
):
    # Log the random seed for reproducibility
    logger.info(f"{seed=}")

    # Initialize experiment logger for tracking results and saving outputs
    exp_logger = fdtdx.Logger(
        experiment_name="ceviche_corner",
        name=None,
    )
    # Create a JAX random key for stochastic operations
    key = jax.random.PRNGKey(seed=seed)

    # Set simulation wavelength and compute corresponding period
    wavelength = 1.55e-6
    period = fdtdx.constants.wavelength_to_period(wavelength)

    # Define simulation configuration (time, resolution, data type, etc.)
    config = fdtdx.SimulationConfig(
        time=50e-15,
        resolution=20e-9,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    # Calculate number of time steps per period and total time steps
    period_steps = round(period / config.time_step_duration)
    all_time_steps = list(range(config.time_steps_total))
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    # If not in evaluation mode or if backward simulation is requested, set up gradient recording
    if not evaluation or backward:
        gradient_config = fdtdx.GradientConfig(
            recorder=fdtdx.Recorder(
                modules=[
                    fdtdx.LinearReconstructEveryK(k=5),
                    fdtdx.DtypeConversion(dtype=jnp.float8_e4m3fnuz),
                ]
            )
        )
        config = config.aset("gradient_config", gradient_config)

    # List to hold placement constraints for simulation objects
    placement_constraints, object_list = [], []

    # Define the simulation volume (physical size)
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(2.7e-6, 2.7e-6, 1.5e-6),
    )
    object_list.append(volume)

    # Add boundary objects and constraints to the simulation
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=10)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    placement_constraints.extend(c_list)
    object_list.extend(list(bound_dict.values()))

    # Add substrate object (base layer) and its placement constraint
    substrate = fdtdx.UniformMaterialObject(
        partial_real_shape=(None, None, 0.5e-6),
        material=fdtdx.Material(permittivity=fdtdx.constants.relative_permittivity_silica),
        color=fdtdx.colors.XKCD_ORANGE,
    )
    placement_constraints.append(
        substrate.place_relative_to(
            volume,
            axes=2,
            own_positions=-1,
            other_positions=-1,
        )
    )
    object_list.append(substrate)

    # Define device geometry and material configuration
    height = 400e-9
    width = 400e-9
    material_config = {
        "Air": fdtdx.Material(permittivity=fdtdx.constants.relative_permittivity_air),
        "Silicon": fdtdx.Material(permittivity=fdtdx.constants.relative_permittivity_silicon),
    }
    voxel_size = 20e-9
    device = fdtdx.Device(
        name="Device",
        partial_real_shape=(1.6e-6, 1.6e-6, height),
        materials=material_config,
        param_transforms=[
            # Optional symmetry and smoothing transforms
            # DiagonalSymmetry2D(min_min_to_max_max=False),
            fdtdx.GaussianSmoothing2D(std_discrete=3),
            fdtdx.SubpixelSmoothedProjection(),
            # TanhProjection(),
        ],
        partial_voxel_real_shape=(voxel_size, voxel_size, height),
    )
    # Place device relative to substrate with specified margins
    placement_constraints.append(
        device.place_relative_to(
            substrate,
            axes=(0, 1, 2),
            own_positions=(1, -1, -1),
            other_positions=(1, -1, 1),
            grid_margins=(-bound_cfg.thickness_grid_maxx, bound_cfg.thickness_grid_miny, 0),
            margins=(-0.2e-6, 0.2e-6, 0),
        )
    )
    object_list.append(device)

    # Add input waveguide and its placement constraints
    waveguide_in = fdtdx.UniformMaterialObject(
        partial_real_shape=(None, width, height),
        material=material_config["Silicon"],
        color=fdtdx.colors.XKCD_LIGHT_BLUE,
    )
    placement_constraints.extend(
        [
            waveguide_in.place_at_center(
                device,
                axes=1,
            ),
            waveguide_in.extend_to(
                device,
                axis=0,
                direction="+",
            ),
            waveguide_in.place_above(substrate),
        ]
    )
    object_list.append(waveguide_in)

    # Add source object and its placement constraint
    source = fdtdx.ModePlaneSource(
        partial_grid_shape=(1, None, None),
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength),
        direction="+",
    )
    placement_constraints.extend(
        [
            source.place_relative_to(
                waveguide_in,
                axes=(0,),
                other_positions=(-1,),
                own_positions=(1,),
                grid_margins=(bound_cfg.thickness_grid_minx + 4,),
            )
        ]
    )
    object_list.append(source)

    # Add output waveguide and its placement constraints
    waveguide_out = fdtdx.UniformMaterialObject(
        partial_real_shape=(width, None, height),
        material=material_config["Silicon"],
        color=fdtdx.colors.XKCD_LIGHT_BLUE,
    )
    placement_constraints.extend(
        [
            waveguide_out.place_at_center(
                device,
                axes=0,
            ),
            waveguide_out.extend_to(
                device,
                axis=1,
                direction="-",
            ),
            waveguide_out.place_above(substrate),
        ]
    )
    object_list.append(waveguide_out)
    
    # Add input flux detector and its placement constraint
    flux_in_detector = fdtdx.PoyntingFluxDetector(
        name="in flux",
        partial_grid_shape=(1, None, None),
        direction="+",
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=all_time_steps[7*period_steps:8*period_steps]),
    )
    placement_constraints.append(
        flux_in_detector.place_relative_to(
            source,
            axes=0,
            own_positions=1,
            other_positions=1,
            grid_margins=2,
        )
    )
    object_list.append(flux_in_detector)
    
    # Add output flux detector and its placement constraint
    flux_out_detector = fdtdx.PoyntingFluxDetector(
        name="out flux",
        partial_grid_shape=(None, 1, None),
        direction="+",
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=all_time_steps[-period_steps :]),
    )
    placement_constraints.append(
        flux_out_detector.place_relative_to(
            waveguide_out,
            axes=1,
            own_positions=1,
            other_positions=1,
            grid_margins=-bound_cfg.thickness_grid_maxy - 5,
        )
    )
    object_list.append(flux_out_detector)
    
    # Add energy detector for the last simulation step
    energy_last_step = fdtdx.EnergyDetector(
        name="energy_last_step",
        as_slices=True,
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=[-1]),
    )
    placement_constraints.extend([*energy_last_step.same_position_and_size(volume)])
    object_list.append(energy_last_step)

    # List of objects to exclude from some outputs (e.g., video)
    exclude_object_list: list[fdtdx.SimulationObject] = [energy_last_step]
    if evaluation:
        # Add video detector for visualization during evaluation
        video_detector = fdtdx.EnergyDetector(
            name="video",
            as_slices=True,
            switch=fdtdx.OnOffSwitch(interval=10),
            exact_interpolation=True,
            num_video_workers=10,
        )
        placement_constraints.extend([*video_detector.same_position_and_size(volume)])
        exclude_object_list.append(video_detector)
        object_list.append(video_detector)
        if backward:
            # Add backward video detector if backward simulation is enabled
            backward_video_detector = fdtdx.EnergyDetector(
                name="backward_video",
                as_slices=True,
                inverse=True,
                switch=fdtdx.OnOffSwitch(interval=10),
                exact_interpolation=True,
                num_video_workers=10,
            )
            placement_constraints.extend([*backward_video_detector.same_position_and_size(volume)])
            exclude_object_list.append(backward_video_detector)
            object_list.append(backward_video_detector)

    # Place all objects in the simulation volume according to constraints
    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list,
        config=config,
        constraints=placement_constraints,
        key=subkey,
    )
    
    # Optionally load saved parameters for the device (commented out)
    # saved_params = jnp.load("...")
    # params[device.name] = saved_params
    start_idx = 0

    # Log summary of arrays and print configuration diagram
    logger.info(tc.tree_summary(arrays, depth=2))
    print(tc.tree_diagram(config, depth=4))

    # Set number of optimization epochs
    epochs = 501
    if not evaluation:
        # Define learning rate schedule for fine-tuning
        schedule_finetune: optax.Schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-5,
            peak_value=0.005,
            end_value=0.0005,
            warmup_steps=10,
            decay_steps=round(0.9 * epochs),
        )
        # Create optimizer with schedule and wrap with MultiSteps for gradient accumulation
        optimizer_finetune = optax.inject_hyperparams(optax.nadam)(learning_rate=schedule_finetune)
        optimizer_finetune = optax.MultiSteps(optimizer_finetune, every_k_schedule=1)
        # Initialize optimizer state with current parameters
        opt_state_finetune: optax.OptState = optimizer_finetune.init(params)
    
    # Custom schedule for beta parameter (used in param transforms)
    def custom_schedule(idx: chex.Numeric) -> chex.Numeric:
        beta_schedule = optax.linear_schedule(0.1, 50, epochs)
        return jax.lax.cond(
            idx < epochs - 2,
            lambda: beta_schedule(idx),
            lambda: jnp.inf
        )
    
    # Save initial setup figure for experiment documentation
    exp_logger.savefig(
        exp_logger.cwd,
        "setup.png",
        fdtdx.plot_setup(
            config=config,
            objects=objects,
            exclude_object_list=exclude_object_list,
        ),
    )

    # Log initial parameters and export figures/STL files
    changed_voxels = exp_logger.log_params(
        iter_idx=-1,
        params=params,
        objects=objects,
        export_stl=True,
        export_figure=True,
        beta=custom_schedule(start_idx),
    )
    
    # Apply initial parameters and plot source mode profile
    x, tmp, _ = fdtdx.apply_params(arrays, objects, params, key, beta=custom_schedule(start_idx))
    tmp.sources[0].plot(  # type: ignore
        exp_logger.cwd / "figures" / "mode.png"
    )

    # Define loss function for optimization/evaluation
    def loss_func(
        params: fdtdx.ParameterContainer,
        arrays: fdtdx.ArrayContainer,
        key: jax.Array,
        idx: int,
    ):
        # Apply parameters to objects and arrays
        arrays, new_objects, info = fdtdx.apply_params(arrays, objects, params, key, beta=custom_schedule(idx))

        # Run FDTD simulation
        final_state = fdtdx.run_fdtd(
            arrays=arrays,
            objects=new_objects,
            config=config,
            key=key,
        )

        # Extract updated arrays from simulation state
        _, arrays = final_state
        
        # Compute total output and input flux from detectors
        total_out_flux = arrays.detector_states[flux_out_detector.name]["poynting_flux"].sum()
        total_in_flux = arrays.detector_states[flux_in_detector.name]["poynting_flux"].sum()
        total_flux_ratio = total_out_flux / total_in_flux
        
        # Objective is the flux ratio (higher is better)
        objective = total_flux_ratio

        # Optionally run backward simulation for gradient calculation
        if evaluation and backward:
            _, arrays = fdtdx.full_backward(
                state=final_state,
                objects=new_objects,
                config=config,
                key=key,
                record_detectors=True,
                reset_fields=True,
            )

        # Collect info for logging
        new_info = {
            "total_out_flux": total_out_flux,
            "total_in_flux": total_in_flux,
            "total_flux_ratio": total_flux_ratio,
            "objective": objective,
            **info,
        }
        # Return negative objective (for minimization) and auxiliary info
        return -objective, (arrays, new_info) 

    # Compile loss function with JAX JIT for fast execution
    compile_start_time = time.time()
    print("Started Compilation...")
    jit_task_id = exp_logger.progress.add_task("JIT", total=None)
    idx_dummy_arr = jnp.asarray(start_idx, dtype=jnp.float32)
    if evaluation:
        # JIT loss function for evaluation mode
        jitted_loss = jax.jit(loss_func, donate_argnames=["arrays"]).lower(params, arrays, key, idx_dummy_arr).compile()
    else:
        # JIT loss and gradient function for optimization mode
        jitted_loss = (
            jax.jit(jax.value_and_grad(loss_func, has_aux=True), donate_argnames=["arrays"])
            .lower(params, arrays, key, idx_dummy_arr)
            .compile()
        )
    compile_delta_time = time.time() - compile_start_time
    exp_logger.progress.update(jit_task_id, total=1, completed=1, refresh=True)
    print(f"Finished Compilation in {compile_delta_time} seconds")

    # Add progress bar for optimization/evaluation loop
    optim_task_id = exp_logger.progress.add_task("Optimization", total=1 if evaluation else epochs)
    for epoch in range(start_idx, start_idx + 1 if evaluation else epochs):

        # Start timer for this epoch
        run_start_time = time.time()
        key, subkey = jax.random.split(key)
        idx_arr = jnp.asarray(epoch, dtype=jnp.float32)
        if evaluation:
            # Run evaluation step
            loss, (arrays, info) = jitted_loss(params, arrays, subkey, idx_arr)
        else:
            # Run optimization step and compute gradients
            (loss, (arrays, info)), grads = jitted_loss(params, arrays, subkey, idx_arr)

            # Update optimizer state and parameters
            updates, opt_state_finetune = optimizer_finetune.update(grads, opt_state_finetune, params) # type: ignore
            info["lr"] = opt_state_finetune.inner_opt_state.hyperparams["learning_rate"]
            params = optax.apply_updates(params, updates)
            # Clip parameters to [0, 1] range
            params = jax.tree_util.tree_map(lambda p: jnp.clip(p, 0, 1), params)
            # Log gradient and update norms
            info["grad_norm"] = optax.global_norm(grads)
            info["update_norm"] = optax.global_norm(updates)

        # Compute runtime for this epoch
        runtime_delta = time.time() - run_start_time
        info["runtime"] = runtime_delta
        # Compute attenuation metric for logging
        info["attenuation"] = 10 * jnp.log10(-loss)

        # Log compile and runtime info during evaluation
        if evaluation:
            logger.info(f"{compile_delta_time=}")
            logger.info(f"{runtime_delta=}")

        # Log parameters and export figures/STL files for this epoch
        changed_voxels = exp_logger.log_params(
            iter_idx=epoch,
            params=params,
            objects=objects,
            export_stl=True,
            export_figure=True,
            beta=custom_schedule(epoch)
        )
        info["changed_voxels"] = changed_voxels

        # Log detector outputs (e.g., for video)
        exp_logger.log_detectors(iter_idx=epoch, objects=objects, detector_states=arrays.detector_states)

        # Write all info to experiment log
        exp_logger.write(info)
        exp_logger.progress.update(optim_task_id, advance=1)


# Entry point: parse command line arguments and run main function
if __name__ == "__main__": 
    seed = 0
    evaluation = False
    backward = False
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        evaluation = False
    main(
        seed,
        evaluation=evaluation,
        backward=backward,
    )