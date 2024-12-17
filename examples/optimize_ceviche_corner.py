import sys
import time

import jax
import jax.numpy as jnp
import optax
import pytreeclass as tc
from loguru import logger

from fdtdx.constraints.discrete import BrushConstraint2D, circular_brush
from fdtdx.constraints.mapping import ConstraintMapping
from fdtdx.constraints.module import (
    ClosestIndex,
    ConstraintInterface,
    IndicesToInversePermittivities,
    StandardToInversePermittivityRange,
    StandardToPlusOneMinusOneRange,
)
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
    pretraining_ratio: float = 0.5,
):
    logger.info(f"{seed=}")

    name = None
    if fixed_names:
        formatted_ratio = f"{pretraining_ratio:.2f}".replace(".", "_")
        name = f"ceviche_corner_{seed}_{formatted_ratio}"
    exp_logger = Logger(
        experiment_name="ceviche_corner",
        name=name,
    )
    key = jax.random.PRNGKey(seed=seed)

    wavelength = 1.55e-6
    period = constants.wavelength_to_period(wavelength)

    config = SimulationConfig(
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

    if not evaluation or backward:
        gradient_config = GradientConfig(
            recorder=Recorder(
                modules=[
                    LinearReconstructEveryK(2),
                    DtypeConversion(dtype=jnp.float16),
                    # LinearReconstructEveryK(5),
                    # DtypeConversion(dtype=jnp.float8_e4m3fnuz),
                ]
            )
        )
        config = config.aset("gradient_config", gradient_config)

    placement_constraints = []

    volume = SimulationVolume(
        partial_real_shape=(4e-6, 4e-6, 1.5e-6),
    )

    bound_cfg = BoundaryConfig.from_uniform_bound(thickness=10)
    _, c_list = pml_objects_from_config(bound_cfg, volume)
    placement_constraints.extend(c_list)

    substrate = Substrate(
        partial_real_shape=(None, None, 0.6e-6),
        permittivity=constants.relative_permittivity_silica,
    )
    placement_constraints.append(
        substrate.place_relative_to(
            volume,
            axes=2,
            own_positions=-1,
            other_positions=-1,
        )
    )

    height = 400e-9
    permittivity_config = {
        "Air": constants.relative_permittivity_air,
        "Silicon": constants.relative_permittivity_silicon,
    }
    brush_diameter = round(100e-9 / config.resolution)
    modules_pretrain = [
        StandardToInversePermittivityRange(),
        ClosestIndex(),
        IndicesToInversePermittivities(),
    ]
    modules_finetune = [
        StandardToPlusOneMinusOneRange(),
        BrushConstraint2D(
            brush=circular_brush(diameter=brush_diameter),
            axis=2,
        ),
        IndicesToInversePermittivities(),
    ]
    device = Device(
        name="Device",
        partial_real_shape=(1.6e-6, 1.6e-6, height),
        permittivity_config=permittivity_config,
        constraint_mapping=ConstraintMapping(
            modules=modules_pretrain,
        ),
        partial_voxel_real_shape=(config.resolution, config.resolution, height),
    )
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

    waveguide_in = WaveGuide(
        partial_real_shape=(None, 0.4e-6, height),
        permittivity=permittivity_config["Silicon"],
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
                axes=(0, 1, 2),
                own_positions=(-1, 0, 0),
                other_positions=(-1, 0, 0),
                grid_margins=(bound_cfg.thickness_grid_minx, 0, 0),
                margins=(0.2e-6, 0, 0),
            ),
            detector_in.size_relative_to(
                waveguide_in,
                axes=(1, 2),
            ),
        ]
    )

    waveguide_out = WaveGuide(
        partial_real_shape=(0.4e-6, None, height),
        permittivity=permittivity_config["Silicon"],
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

    detector_out = PoyntingFluxDetector(
        name="Output Detector",
        partial_grid_shape=(None, 1, None),
        direction="+",
        time_steps=all_time_steps[-2 * period_steps :],
    )
    placement_constraints.extend(
        [
            detector_out.place_relative_to(
                waveguide_out,
                axes=(0, 1, 2),
                own_positions=(0, 1, 0),
                other_positions=(0, 1, 0),
                grid_margins=(0, -bound_cfg.thickness_grid_maxy - 5, 0),
            ),
            detector_out.size_relative_to(
                waveguide_out,
                axes=(0, 2),
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

    changed_voxels = exp_logger.log_params(
        iter_idx=-1,
        params=params,
        objects=objects,
        export_stl=True,
        export_figure=True,
    )

    _, tmp, _ = apply_params(arrays, objects, params, key)
    tmp.sources[0].plot(  # type: ignore
        exp_logger.cwd / "figures" / "mode.png"
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
    pretrain_epochs = round(epochs * pretraining_ratio)
    finetune_epochs = epochs - pretrain_epochs
    if not evaluation:
        if pretrain_epochs > 0:
            schedule_pretrain: optax.Schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0005,
                peak_value=0.005,
                end_value=0.0005,
                warmup_steps=10 if pretrain_epochs >= 10 else 1,
                decay_steps=round(0.9 * pretrain_epochs),
            )
            optimizer_pretrain = optax.inject_hyperparams(optax.nadam)(learning_rate=schedule_pretrain)

            optimizer_pretrain = optax.MultiSteps(optimizer_pretrain, every_k_schedule=1)
            opt_state_pretrain: optax.OptState = optimizer_pretrain.init(params)

        schedule_finetune: optax.Schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0005,
            peak_value=0.005,
            end_value=0.0005,
            warmup_steps=10,
            decay_steps=round(0.9 * finetune_epochs),
        )
        optimizer_finetune = optax.inject_hyperparams(optax.nadam)(learning_rate=schedule_finetune)

        optimizer_finetune = optax.MultiSteps(optimizer_finetune, every_k_schedule=1)
        opt_state_finetune: optax.OptState = optimizer_finetune.init(params)

    compile_start_time = time.time()
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
    exp_logger.progress.update(jit_task_id, total=1, completed=1, refresh=True)

    optim_task_id = exp_logger.progress.add_task("Optimization", total=epochs)
    for epoch in range(epochs):
        if epoch == pretrain_epochs:
            # recompile with fintuning constraints using minimum feature size
            new_constraints = ConstraintMapping(
                modules=modules_finetune,
            )
            new_constraints = new_constraints.init_modules(
                config=config,
                permittivity_config=permittivity_config,
                output_interface=ConstraintInterface(
                    shapes={
                        "out": objects.devices[0].matrix_voxel_grid_shape,
                    },
                    type="inv_permittivity",
                ),
            )
            new_object_list = [
                o if not isinstance(o, Device) else o.aset("constraint_mapping", new_constraints)
                for o in objects.objects
            ]
            objects = objects.aset("object_list", new_object_list)
            if evaluation:
                jitted_loss = jax.jit(loss_func, donate_argnames=["arrays"]).lower(params, arrays, key).compile()
            else:
                jitted_loss = (
                    jax.jit(jax.value_and_grad(loss_func, has_aux=True), donate_argnames=["arrays"])
                    .lower(params, arrays, key)
                    .compile()
                )

        run_start_time = time.time()
        key, subkey = jax.random.split(key)
        if evaluation:
            loss, (arrays, info) = jitted_loss(params, arrays, subkey)
        else:
            (loss, (arrays, info)), grads = jitted_loss(params, arrays, subkey)

            # update
            if epoch < pretrain_epochs:
                updates, opt_state_pretrain = optimizer_pretrain.update(grads, opt_state_pretrain, params)
                info["lr"] = opt_state_pretrain.inner_opt_state.hyperparams["learning_rate"]
            else:
                updates, opt_state_finetune = optimizer_finetune.update(grads, opt_state_finetune, params)
                info["lr"] = opt_state_finetune.inner_opt_state.hyperparams["learning_rate"]
            params = optax.apply_updates(params, updates)
            params = jax.tree_util.tree_map(lambda p: jnp.clip(p, 0, 1), params)
            info["grad_norm"] = optax.global_norm(grads)
            info["update_norm"] = optax.global_norm(updates)

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
    pretraining_ratio = 0.5
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        pretraining_ratio = float(sys.argv[2])
        fixed_names = True
        evaluation = False
    main(
        seed,
        evaluation=evaluation,
        fixed_names=fixed_names,
        backward=backward,
        pretraining_ratio=pretraining_ratio,
    )
