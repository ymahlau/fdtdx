import argparse
import itertools
import sys
import time
from typing import Literal

import jax
import jax.numpy as jnp
import jax.profiler
import optax
import pytreeclass as tc
from loguru import logger

from fdtdx.constraints.discrete import ConnectHolesAndStructures
from fdtdx.constraints.mapping import ConstraintMapping
from fdtdx.constraints.module import (
    ClosestIndex,
    ConstraintModule,
    IndicesToInversePermittivities,
    StandardToInversePermittivityRange,
)
from fdtdx.constraints.pillars import PillarMapping
from fdtdx.core.config import GradientConfig, SimulationConfig
from fdtdx.core.physics import constants
from fdtdx.core.physics.losses import metric_efficiency
from fdtdx.fdtd.backward import full_backward
from fdtdx.fdtd.fdtd import checkpointed_fdtd, reversible_fdtd
from fdtdx.interfaces.modules import DtypeConversion
from fdtdx.interfaces.recorder import Recorder
from fdtdx.objects.boundaries.initialization import BoundaryConfig, pml_objects_from_config
from fdtdx.objects.container import ArrayContainer, ParameterContainer
from fdtdx.objects.detectors.energy import EnergyDetector
from fdtdx.objects.detectors.poynting_flux import PoyntingFluxDetector
from fdtdx.objects.initialization import apply_params, place_objects
from fdtdx.objects.material import SimulationVolume, Substrate, WaveGuide
from fdtdx.objects.multi_material.device import Device
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.sources.plane_source import GaussianPlaneSource
from fdtdx.shared.logger import Logger
from fdtdx.shared.plot_setup import plot_setup


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Program description here", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--evaluation", type=str2bool, default=True, help="Whether to run in evaluation mode")

    parser.add_argument(
        "--dim",
        type=str,
        choices=["2d", "2_5d", "3d"],
        default="3d",
        help="Dimensionality of the output (2d, 2.5d, or 3d)",
    )

    parser.add_argument("--multi_material", type=str2bool, default=True, help="Whether to use multiple (2) polymers")

    return parser.parse_args()


def main(
    seed: int,
    evaluation: bool,
    multi_material: bool,
    dim: Literal["2d", "2_5d", "3d"],
    fixed_names: bool,
):
    logger.info(f"{seed=}")
    logger.info(f"{evaluation=}")
    logger.info(f"{multi_material=}")
    logger.info(f"{dim=}")

    name = None
    if fixed_names:
        name = f"coupler_{dim}_multi_{multi_material}_{seed}"
    exp_logger = Logger(
        experiment_name="vertical polymer coupler",
        name=name,
    )
    key = jax.random.PRNGKey(seed=seed)

    wavelength = 1.55e-6
    period = constants.wavelength_to_period(wavelength)

    config = SimulationConfig(
        time=250e-15,
        resolution=100e-9,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    period_steps = round(period / config.time_step_duration)
    all_time_steps = list(range(config.time_steps_total))
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    gradient_config = GradientConfig(
        recorder=Recorder(
            modules=[
                DtypeConversion(dtype=jnp.float16),
            ]
        )
    )
    config = config.aset("gradient_config", gradient_config)

    placement_constraints = []

    volume = SimulationVolume(
        partial_real_shape=(40e-6, 30e-6, 12e-6),
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

    if multi_material:
        permittivity_config = {
            "Air": constants.relative_permittivity_air,
            "SZ2080": constants.relative_permittivity_SZ_2080,
            "maN1400": constants.relative_permittivity_ma_N_1400_series,
        }
    else:
        permittivity_config = {
            "Air": constants.relative_permittivity_air,
            "maN1400": constants.relative_permittivity_ma_N_1400_series,
        }
    voxel_size = 0.5e-6
    device_z = 6.0e-6
    voxel_z = voxel_size if dim in ("2_5d", "3d") else device_z
    module_list: list[ConstraintModule] = [StandardToInversePermittivityRange()]
    if dim != "3d":
        module_list.append(
            PillarMapping(
                axis=2,
                single_polymer_columns=False,
            )
        )
    else:
        module_list.extend(
            [
                ClosestIndex(),
                ConnectHolesAndStructures(fill_material="SZ2080" if multi_material else "maN1400"),
                IndicesToInversePermittivities(),
            ]
        )

    device = Device(
        name="Device",
        permittivity_config=permittivity_config,
        partial_real_shape=(25.0e-6, 25.0e-6, device_z),
        constraint_mapping=ConstraintMapping(modules=module_list),
        partial_voxel_real_shape=(voxel_size, voxel_size, voxel_z),
    )
    placement_constraints.append(
        device.place_relative_to(
            substrate,
            axes=(0, 1, 2),
            own_positions=(-1, 0, -1),
            other_positions=(-1, 0, 1),
            grid_margins=(bound_cfg.thickness_grid_minx, 0, 0),
            margins=(0.5e-6, 0, 0),
        )
    )

    source = GaussianPlaneSource(
        partial_real_shape=(12.0e-6, 12.0e-6, None),
        partial_grid_shape=(None, None, 1),
        wavelength=1.550e-6,
        radius=4.0e-6,
        direction="-",
        fixed_E_polarization_vector=(1, 0, 0),
        max_angle_random_offset=2.0,
        std=1,
        max_vertical_offset=2.0e-6,
        max_horizontal_offset=2.0e-6,
    )
    placement_constraints.extend(
        [
            source.place_relative_to(
                device,
                axes=(0, 1),
                own_positions=(0, 0),
                other_positions=(0, 0),
                margins=(0, 0),
            ),
            source.place_relative_to(
                volume,
                axes=(2,),
                own_positions=(1,),
                other_positions=(1,),
                grid_margins=(-bound_cfg.thickness_grid_maxz - 3,),
            ),
        ]
    )

    in_detector = PoyntingFluxDetector(
        name="Input Flux",
        direction="-",
        partial_grid_shape=(None, None, 1),
        time_steps=all_time_steps[2 * period_steps : 4 * period_steps],
        exact_interpolation=True,
    )
    placement_constraints.extend(
        [
            in_detector.place_relative_to(
                source, axes=(0, 1, 2), own_positions=(0, 0, 1), other_positions=(0, 0, -1), grid_margins=(0, 0, -3)
            ),
            in_detector.same_size(
                source,
                axes=(0, 1),
            ),
        ]
    )

    waveguide = WaveGuide(
        partial_real_shape=(None, 1.0e-6, 1.0e-6),
        permittivity=permittivity_config["maN1400"],
    )
    placement_constraints.extend(
        [
            waveguide.place_at_center(
                device,
                axes=1,
            ),
            waveguide.extend_to(
                device,
                axis=0,
                direction="-",
            ),
            waveguide.place_above(substrate),
        ]
    )

    waveguide_detector = PoyntingFluxDetector(
        name="Output Flux",
        direction="+",
        partial_grid_shape=(1, None, None),
        time_steps=all_time_steps[-2 * period_steps :],
        exact_interpolation=True,
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

    energy_last_step = EnergyDetector(
        name="energy_last_step",
        as_slices=True,
        time_steps=[-1],
        exact_interpolation=True,
    )
    placement_constraints.extend([*energy_last_step.same_position_and_size(volume)])

    if evaluation:
        video_detector = EnergyDetector(
            name="video",
            as_slices=True,
            interval=10,
            exact_interpolation=True,
        )
        placement_constraints.extend([*video_detector.same_position_and_size(volume)])
        backward_video_detector = EnergyDetector(
            name="backward_video",
            as_slices=True,
            inverse=True,
            interval=10,
            exact_interpolation=True,
        )
        placement_constraints.extend([*backward_video_detector.same_position_and_size(volume)])

    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = place_objects(
        volume=volume,
        config=config,
        constraints=placement_constraints,
        key=subkey,
    )
    logger.info(tc.tree_summary(arrays, depth=2))
    print(tc.tree_diagram(config, depth=4))

    exclude_object_list: list[SimulationObject] = [energy_last_step]
    if evaluation:
        exclude_object_list.extend(
            [
                video_detector,
                backward_video_detector,
            ]
        )

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
        export_air_stl=True if dim == "3d" else False,
        export_figure=True,
    )

    def loss_func(
        params: ParameterContainer,
        arrays: ArrayContainer,
        key: jax.Array,
    ):
        arrays, new_objects, info = apply_params(arrays, objects, params, key)

        fdtd_fn = checkpointed_fdtd if evaluation else reversible_fdtd
        final_state = fdtd_fn(
            arrays=arrays,
            objects=new_objects,
            config=config,
            key=key,
        )

        _, arrays = final_state
        objective, objective_info = metric_efficiency(
            arrays.detector_states,
            in_names=(in_detector.name,),
            out_names=(waveguide_detector.name,),
            metric_name="poynting_flux",
        )

        if evaluation:
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
            init_value=1e-4,
            peak_value=0.05,
            end_value=1e-3,
            warmup_steps=10,
            decay_steps=round(0.9 * epochs),
        )
        optimizer = optax.inject_hyperparams(optax.nadam)(learning_rate=schedule)

        optimizer = optax.MultiSteps(optimizer, every_k_schedule=3)
        opt_state: optax.OptState = optimizer.init(params)

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
            export_air_stl=True if dim == "3d" else False,
            export_figure=True,
        )
        info["changed_voxels"] = changed_voxels

        # videos
        exp_logger.log_detectors(iter_idx=epoch, objects=objects, detector_states=arrays.detector_states)

        exp_logger.write(info)
        exp_logger.progress.update(optim_task_id, advance=1)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # integer for batch job
        options = [
            [False, True],  # multimaterial
            ["2d", "2_5d", "3d"],
            list(range(5)),  # seeds
        ]
        idx = int(sys.argv[1])
        prod = list(itertools.product(*options))
        multi, dim, seed = prod[idx]
        main(seed=seed, evaluation=False, multi_material=multi, dim=dim, fixed_names=True)
    else:
        args = parse_args()
        main(
            seed=args.seed,
            evaluation=args.evaluation,
            multi_material=args.multi_material,
            dim=args.dim,
            fixed_names=False,
        )
