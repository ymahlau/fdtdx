import sys
import time

import jax
import jax.numpy as jnp
import optax
import pytreeclass as tc
from loguru import logger

from fdtdx.constraints.discrete import BOTTOM_Z_PADDING_CONFIG_REPEAT, BinaryMedianFilterModule, RemoveFloatingMaterial
from fdtdx.constraints.mapping import ConstraintMapping
from fdtdx.constraints.module import ClosestIndex, IndicesToInversePermittivities, StandardToInversePermittivityRange
from fdtdx.core.config import GradientConfig, SimulationConfig
from fdtdx.core.physics import constants
from fdtdx.core.physics.losses import metric_efficiency
from fdtdx.fdtd.backward import full_backward
from fdtdx.fdtd.fdtd import reversible_fdtd
from fdtdx.interfaces.modules import DtypeConversion
from fdtdx.interfaces.recorder import Recorder
from fdtdx.interfaces.time_filter import LinearReconstructEveryK
from fdtdx.objects.boundaries.initialization import BoundaryConfig, pml_objects_from_config
from fdtdx.objects.container import ArrayContainer, ParameterContainer
from fdtdx.objects.detectors.energy import EnergyDetector
from fdtdx.objects.detectors.poynting_flux import PoyntingFluxDetector
from fdtdx.objects.initialization import apply_params, place_objects
from fdtdx.objects.material import SimulationVolume, Substrate, WaveGuide
from fdtdx.objects.multi_material.device import Device
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.sources.plane_source import ModePlaneSource
from fdtdx.shared.logger import Logger
from fdtdx.shared.plot_setup import plot_setup


def main(
    seed: int,
    fixed_names: bool = False,
    evaluation: bool = True,
    backward: bool = True,
):
    logger.info(f"{seed=}")

    name = None
    if fixed_names:
        name = f"waveguide_stitching_{seed}"
    exp_logger = Logger(
        experiment_name="waveguide_stitching",
        name=name,
    )
    key = jax.random.PRNGKey(seed=seed)

    wavelength = 1.55e-6
    period = constants.wavelength_to_period(wavelength)

    config = SimulationConfig(
        time=300e-15,
        resolution=100e-9,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    period_steps = round(period / config.time_step_duration)
    all_time_steps = list(range(config.time_steps_total))
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    if not evaluation or backward:
        gradient_config = GradientConfig(
            recorder=Recorder(
                modules=[
                    LinearReconstructEveryK(2),
                    DtypeConversion(dtype=jnp.float16),
                ]
            )
        )
        config = config.aset("gradient_config", gradient_config)

    placement_constraints = []

    volume = SimulationVolume(
        partial_real_shape=(40e-6, 17e-6, 10e-6),
    )

    bound_cfg = BoundaryConfig.from_uniform_bound(thickness=10)
    _, c_list = pml_objects_from_config(bound_cfg, volume)
    placement_constraints.extend(c_list)

    substrate = Substrate(
        permittivity=constants.relative_permittivity_substrate,
        partial_real_shape=(None, None, 2e-6),
    )
    placement_constraints.append(
        substrate.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, -1),
            other_positions=(0, 0, -1),
        )
    )

    device_height = 3e-6
    device_x = 5e-6
    device_width = 10e-6
    waveguide_width = 1.5e-6
    waveguide_height = 1.5e-6
    mapping = ConstraintMapping(
        modules=[
            StandardToInversePermittivityRange(),
            ClosestIndex(),
            BinaryMedianFilterModule(
                padding_cfg=BOTTOM_Z_PADDING_CONFIG_REPEAT,
                kernel_sizes=(5, 5, 5),
                num_repeats=2,
            ),
            RemoveFloatingMaterial(),
            IndicesToInversePermittivities(),
        ],
    )
    device_voxel_real = (config.resolution, config.resolution, config.resolution)
    permittivity_config = {
        "Air": constants.relative_permittivity_air,
        "Polymer": constants.relative_permittivity_ma_N_1400_series,
    }
    device_scatter = Device(
        name="Device-Scatter",
        partial_real_shape=(device_x, device_width, device_height),
        permittivity_config=permittivity_config,
        constraint_mapping=mapping,
        partial_voxel_real_shape=device_voxel_real,
    )
    placement_constraints.append(
        device_scatter.place_relative_to(
            substrate,
            axes=(0, 1, 2),
            own_positions=(-1, 0, -1),
            other_positions=(-1, 0, 1),
            grid_margins=(bound_cfg.thickness_grid_minx, 0, 0),
            margins=(7e-6, 0, 0),
        )
    )

    waveguide_in = WaveGuide(
        partial_real_shape=(None, waveguide_width, waveguide_height),
        permittivity=permittivity_config["Polymer"],
    )
    placement_constraints.extend(
        [
            waveguide_in.place_at_center(
                device_scatter,
                axes=1,
            ),
            waveguide_in.place_above(substrate),
            waveguide_in.extend_to(
                device_scatter,
                axis=0,
                direction="+",
            ),
        ]
    )

    device_gather = Device(
        name="Device-Gather",
        partial_real_shape=(device_x, device_width, device_height),
        permittivity_config=permittivity_config,
        constraint_mapping=mapping,
        partial_voxel_real_shape=device_voxel_real,
        max_random_real_offsets=(2e-6, 2e-6, 0),
    )
    placement_constraints.append(
        device_gather.place_relative_to(
            device_scatter,
            axes=(0, 1, 2),
            own_positions=(-1, 0, 0),
            other_positions=(1, 0, 0),
            margins=(3e-6, 0, 0),
        )
    )

    waveguide_out = WaveGuide(
        partial_real_shape=(None, waveguide_width, waveguide_height),
        permittivity=permittivity_config["Polymer"],
    )
    placement_constraints.extend(
        [
            waveguide_out.place_at_center(
                device_gather,
                axes=1,
            ),
            waveguide_out.place_above(substrate),
            waveguide_out.extend_to(
                device_gather,
                axis=0,
                direction="-",
            ),
        ]
    )

    source = ModePlaneSource(
        partial_grid_shape=(1, None, None),
        wavelength=wavelength,
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

    detector_in = PoyntingFluxDetector(
        name="Input Detector",
        partial_grid_shape=(1, None, None),
        direction="+",
        time_steps=all_time_steps[3 * period_steps : 5 * period_steps],
    )
    placement_constraints.extend(
        [
            detector_in.place_relative_to(
                waveguide_in,
                axes=(1, 2),
                own_positions=(-0, 0),
                other_positions=(0, 0),
            ),
            detector_in.place_relative_to(
                source,
                axes=(0,),
                own_positions=(-1,),
                other_positions=(1,),
                grid_margins=(5,),
            ),
            detector_in.size_relative_to(
                waveguide_in,
                axes=(1, 2),
            ),
        ]
    )

    detector_out = PoyntingFluxDetector(
        name="Output Detector",
        partial_grid_shape=(1, None, None),
        direction="+",
        time_steps=all_time_steps[-2 * period_steps :],
    )
    placement_constraints.extend(
        [
            detector_out.place_relative_to(
                waveguide_out,
                axes=(0, 1, 2),
                own_positions=(1, 0, 0),
                other_positions=(1, 0, 0),
                grid_margins=(-bound_cfg.thickness_grid_maxx - 5, 0, 0),
            ),
            detector_out.size_relative_to(
                waveguide_out,
                axes=(1, 2),
            ),
        ]
    )

    energy_last_step = EnergyDetector(
        name="energy_last_step",
        as_slices=True,
        time_steps=[-1],
    )
    placement_constraints.extend([*energy_last_step.same_position_and_size(volume)])

    exclude_object_list: list[SimulationObject] = [energy_last_step]
    if evaluation:
        video_detector = EnergyDetector(
            name="video",
            as_slices=True,
            interval=10,
            exact_interpolation=True,
        )
        placement_constraints.extend([*video_detector.same_position_and_size(volume)])
        exclude_object_list.append(video_detector)
        if backward:
            backward_video_detector = EnergyDetector(
                name="backward_video",
                as_slices=True,
                inverse=True,
                interval=10,
                exact_interpolation=True,
            )
            placement_constraints.extend([*backward_video_detector.same_position_and_size(volume)])
            exclude_object_list.append(backward_video_detector)

    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = place_objects(
        volume=volume,
        config=config,
        constraints=placement_constraints,
        key=subkey,
    )

    logger.info(tc.tree_summary(arrays, depth=2))
    print(tc.tree_diagram(config, depth=4))

    exp_logger.savefig(
        exp_logger.cwd,
        "setup.png",
        plot_setup(
            config=config,
            objects=objects,
            exclude_object_list=exclude_object_list,
        ),
    )

    params["Device-Scatter"]["out"] = jnp.load(
        "outputs/nobackup/2024-12-04/waveguide_stitching/10-29-57.621927/device/params/params_252_Device-Scatter_out.npy"
    )
    params["Device-Gather"]["out"] = jnp.load(
        "outputs/nobackup/2024-12-04/waveguide_stitching/10-29-57.621927/device/params/params_252_Device-Gather_out.npy"
    )

    exp_logger.log_params(
        iter_idx=-1,
        params=params,
        objects=objects,
        export_stl=True,
    )

    def loss_func(
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
        objective, objective_info = metric_efficiency(
            arrays.detector_states,
            in_names=(detector_in.name,),
            out_names=(detector_out.name,),
            metric_name="poynting_flux",
        )

        if evaluation and backward:
            _, arrays = full_backward(
                state=final_state,
                objects=new_objects,
                config=config,
                key=key,
                record_detectors=True,
                reset_fields=True,
            )

        new_info = {
            "objective": objective,
            **info,
            **objective_info,
        }
        return -objective, (arrays, new_info)

    epochs = 501 if not evaluation else 1
    if not evaluation:
        schedule: optax.Schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0001,
            peak_value=0.01,
            end_value=0.001,
            warmup_steps=10,
            decay_steps=round(0.9 * epochs),
        )
        optimizer = optax.inject_hyperparams(optax.nadam)(learning_rate=schedule)

        optimizer = optax.MultiSteps(optimizer, every_k_schedule=1)
        opt_state: optax.OptState = optimizer.init(params)

    optim_task_id = exp_logger.progress.add_task("Optimization", total=epochs)

    for epoch in range(epochs):
        objects, arrays, _, _, _ = place_objects(
            volume=volume,
            config=config,
            constraints=placement_constraints,
            key=subkey,
        )

        exp_logger.savefig(
            exp_logger.cwd,
            f"setup_{epoch}.png",
            plot_setup(
                config=config,
                objects=objects,
                exclude_object_list=exclude_object_list,
            ),
        )

        compile_start_time = time.time()
        if epoch == 0:
            jit_task_id = exp_logger.progress.add_task("JIT", total=None)
        if evaluation:
            jitted_loss = jax.jit(loss_func, donate_argnames=["arrays"]).lower(params, arrays, key).compile()
        else:
            jitted_loss = (
                jax.jit(jax.value_and_grad(loss_func, has_aux=True), donate_argnames=["arrays"])
                .lower(params, arrays, key)
                .compile()
            )
        compile_delta_time = time.time() - compile_start_time
        if epoch == 0:
            exp_logger.progress.update(jit_task_id, total=1, completed=1, refresh=True)

        run_start_time = time.time()
        key, subkey = jax.random.split(key)
        if evaluation:
            loss, (arrays, info) = jitted_loss(params, arrays, subkey)
        else:
            (loss, (arrays, info)), grads = jitted_loss(params, arrays, subkey)

            # update
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = jax.tree_util.tree_map(lambda p: jnp.clip(p, 0, 1), params)
            info["grad_norm"] = optax.global_norm(grads)
            info["update_norm"] = optax.global_norm(updates)
            info["lr"] = opt_state.inner_opt_state.hyperparams["learning_rate"]
        runtime_delta = time.time() - run_start_time
        info["runtime"] = runtime_delta
        info["attenuation"] = 10 * jnp.log10(-loss)

        if evaluation:
            logger.info(f"{compile_delta_time=}")
            logger.info(f"{runtime_delta=}")

        # log parameters
        changed_voxels = exp_logger.log_params(
            iter_idx=epoch,
            params=params,
            objects=objects,
            export_stl=True,
            export_figure=True,
        )
        info["changed_voxels"] = changed_voxels

        # videos
        exp_logger.log_detectors(iter_idx=epoch, objects=objects, detector_states=arrays.detector_states)

        exp_logger.write(info)
        exp_logger.progress.update(optim_task_id, advance=1)


if __name__ == "__main__":
    seed = 42
    fixed_names: bool = False
    evaluation = False
    backward = True
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        fixed_names = False
        evaluation = False
    main(seed, evaluation=evaluation, backward=backward)
