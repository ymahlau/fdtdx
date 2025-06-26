import sys

# these options will make the code slower, but more memory efficient
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["NCCL_LL128_BUFFSIZE"] = "-2"
# os.environ["NCCL_LL_BUFFSIZE"] = "-2"
# os.environ["NCCL_PROTO"] = "SIMPLE,LL,LL128"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"
import time

import jax
import jax.numpy as jnp
import pytreeclass as tc
from loguru import logger
import fdtdx

from fdtdx.fdtd.fdtd import custom_fdtd_forward


# Design of Shen et al.: https://opg.optica.org/oe/fulltext.cfm?uri=oe-22-22-27175&id=303419
# Note that we are unable to reproduce the reported efficiency of their coupling device.
# Our simulation and commercial software only yields an efficiency of about 7% compared to their reported 50%.
REFERENCE_DESIGN_GRID = jnp.asarray(
    [
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=jnp.float32,
)[::-1].T.reshape(30, 30, 1)


def main(
    seed: int,
):
    evaluation: bool = True
    use_reference_grid: bool = True

    logger.info(f"{seed=}")
    logger.info(f"{evaluation=}")
    logger.info(f"{use_reference_grid=}")

    permittivity_silicon = 12.25
    permittivity_silica = 2.25

    exp_logger = fdtdx.Logger(
        experiment_name="vertical_silicon_coupler",
    )
    key = jax.random.PRNGKey(seed=seed)

    wavelength = 1.55e-6
    period = fdtdx.constants.wavelength_to_period(wavelength)

    config = fdtdx.SimulationConfig(
        time=200e-15,
        resolution=20e-9,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    period_steps = round(period / config.time_step_duration)
    all_time_steps = list(range(config.time_steps_total))
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    placement_constraints = []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(6e-6, 4e-6, 1.5e-6),
    )

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=10)
    _, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    placement_constraints.extend(c_list)

    dioxide_substrate = fdtdx.UniformMaterialObject(
        name="Silica",
        color=fdtdx.colors.LIGHT_BROWN,
        partial_real_shape=(None, None, 0.5e-6),
        material=fdtdx.Material(permittivity=permittivity_silica)
    )
    placement_constraints.append(
        dioxide_substrate.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, -1),
            other_positions=(0, 0, -1),
        )
    )

    material_config = {
        "Air": fdtdx.Material(permittivity=fdtdx.constants.relative_permittivity_air),
        "Silicon": fdtdx.Material(permittivity=permittivity_silicon),
    }
    
    device = fdtdx.Device(
        name="Device",
        partial_real_shape=(3e-6, 3e-6, 300e-9),
        materials=material_config,
        partial_voxel_real_shape=(100e-9, 100e-9, 300e-9),
        param_transforms=[fdtdx.ClosestIndex()],
    )
    placement_constraints.append(
        device.place_relative_to(
            dioxide_substrate,
            axes=(0, 1, 2),
            own_positions=(-1, 0, -1),
            other_positions=(-1, 0, 1),
            grid_margins=(bound_cfg.thickness_grid_minx, 0, 0),
            margins=(0.5e-6, 0, 0),
        )
    )

    source = fdtdx.UniformPlaneSource(
        partial_real_shape=(None, None, None),
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1.550e-6),
        direction="-",
        fixed_E_polarization_vector=(0, 1, 0),
    )
    placement_constraints.extend(
        [
            source.place_relative_to(
                volume,
                axes=(2,),
                own_positions=(1,),
                other_positions=(1,),
                grid_margins=(-3 - bound_cfg.thickness_grid_maxz,),
            ),
        ]
    )

    in_detector = fdtdx.PoyntingFluxDetector(
        name="Input Flux Detector",
        direction="-",
        partial_grid_shape=(None, None, 1),
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=all_time_steps[-2 * period_steps :]),
        exact_interpolation=False,
    )
    placement_constraints.extend(
        [
            in_detector.place_relative_to(
                device,
                axes=(0, 1),
                own_positions=(0, 0),
                other_positions=(0, 0),
            ),
            in_detector.place_relative_to(
                source, axes=(2,), own_positions=(1,), other_positions=(-1,), grid_margins=(-3,)
            ),
            in_detector.size_relative_to(device, axes=(0, 1)),
        ]
    )

    waveguide = fdtdx.UniformMaterialObject(
        partial_real_shape=(None, 0.4e-6, 0.3e-6),
        material=fdtdx.Material(permittivity=permittivity_silicon),
    )
    placement_constraints.extend(
        [
            waveguide.place_relative_to(
                device,
                axes=(1,),
                own_positions=(0,),
                other_positions=(0,),
            ),
            waveguide.extend_to(
                device,
                axis=0,
                direction="-",
            ),
            waveguide.place_above(dioxide_substrate),
        ]
    )

    waveguide_detector = fdtdx.PoyntingFluxDetector(
        name="Output Flux Detector",
        direction="+",
        partial_grid_shape=(1, None, None),
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=all_time_steps[-2 * period_steps :]),
        exact_interpolation=False,
    )
    placement_constraints.extend(
        [
            waveguide_detector.place_relative_to(
                waveguide,
                axes=(0, 1, 2),
                own_positions=(1, 0, 0),
                other_positions=(1, 0, 0),
                grid_margins=(-bound_cfg.thickness_grid_maxx - 3, 0, 0),
            ),
            waveguide_detector.size_relative_to(
                waveguide,
                axes=(1, 2),
                proportions=(1, 1),
            ),
        ]
    )

    energy_last_step = fdtdx.EnergyDetector(
        name="energy_last_step",
        as_slices=True,
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=[-1]),
    )
    placement_constraints.extend([*energy_last_step.same_position_and_size(volume)])
    
    exclude_object_list: list[fdtdx.SimulationObject] = [energy_last_step]
    
    if evaluation:
        video_energy_detector = fdtdx.EnergyDetector(
            name="Energy Video",
            as_slices=True,
            switch=fdtdx.OnOffSwitch(interval=10),
            exact_interpolation=True,
        )
        placement_constraints.extend(video_energy_detector.same_position_and_size(volume))
        exclude_object_list.append(video_energy_detector)

    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        volume=volume,
        config=config,
        constraints=placement_constraints,
        key=subkey,
    )

    if use_reference_grid:
        params[device.name] = REFERENCE_DESIGN_GRID

    logger.info(tc.tree_summary(arrays, depth=2))
    print(tc.tree_diagram(config, depth=4))

    exp_logger.savefig(
        exp_logger.cwd,
        "setup.png",
        fdtdx.plot_setup(
            config=config,
            objects=objects,
            exclude_object_list=exclude_object_list,
        ),
    )

    changed_voxels = exp_logger.log_params(
        iter_idx=-1,
        params=params,
        objects=objects,
        export_stl=True,
        export_figure=True,
    )

    arrays, new_objects, info = fdtdx.apply_params(arrays, objects, params, key)

    def loss_func(
        arrays: fdtdx.ArrayContainer,
        key: jax.Array,
    ):
        medium_time = config.time_steps_total - 3 * period_steps
        intermedium_state = custom_fdtd_forward(
            arrays=arrays,
            objects=new_objects,
            config=config,
            key=key,
            reset_container=False,
            record_detectors=False,
            start_time=0,
            end_time=medium_time,
        )

        final_state = custom_fdtd_forward(
            arrays=intermedium_state[1],
            objects=new_objects,
            config=config,
            key=key,
            reset_container=False,
            record_detectors=True,
            start_time=intermedium_state[0],
            end_time=config.time_steps_total,
        )

        _, arrays = final_state
        objective, objective_info = fdtdx.metric_efficiency(
            arrays.detector_states,
            in_names=(in_detector.name,),
            out_names=(waveguide_detector.name,),
            metric_name="poynting_flux",
        )

        new_info = {
            "objective": objective,
            **info,
            **objective_info,
        }
        return -objective, (arrays, new_info)

    epochs = 1

    compile_start_time = time.time()
    jit_task_id = exp_logger.progress.add_task("JIT", total=None)
    jitted_loss = jax.jit(loss_func, donate_argnames=["arrays"]).lower(arrays, key).compile()
    compile_delta_time = time.time() - compile_start_time
    exp_logger.progress.update(jit_task_id, total=1, completed=1, refresh=True)

    optim_task_id = exp_logger.progress.add_task("Optimization", total=epochs)
    run_start_time = time.time()
    key, subkey = jax.random.split(key)

    loss, (arrays, info) = jitted_loss(arrays, subkey)
    loss = loss.block_until_ready()
    runtime_delta = time.time() - run_start_time
    info["runtime"] = runtime_delta
    info["attenuation"] = 10 * jnp.log10(-loss)

    logger.info(f"{compile_delta_time=}")
    logger.info(f"{runtime_delta=}")

    # log parameters
    changed_voxels = exp_logger.log_params(
        iter_idx=1,
        params=params,
        objects=objects,
        export_stl=True,
        export_figure=True,
    )
    info["changed_voxels"] = changed_voxels

    # videos
    # exp_logger.log_detectors(iter_idx=1, objects=objects, detector_states=arrays.detector_states)

    exp_logger.write(info)
    exp_logger.progress.update(optim_task_id, advance=1)


if __name__ == "__main__":
    seed = 42
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    main(seed)