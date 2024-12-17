import sys
import time

import jax
import jax.numpy as jnp
import jax.profiler
import optax
import pytreeclass as tc
from loguru import logger

from fdtdx.constraints.mapping import ConstraintMapping
from fdtdx.constraints.module import ClosestIndex, IndicesToInversePermittivities, StandardToInversePermittivityRange
from fdtdx.core.config import GradientConfig, SimulationConfig
from fdtdx.core.physics import constants
from fdtdx.core.physics.losses import metric_efficiency
from fdtdx.core.plotting import colors
from fdtdx.fdtd.backward import full_backward
from fdtdx.fdtd.fdtd import reversible_fdtd
from fdtdx.interfaces.recorder import Recorder
from fdtdx.objects.boundaries.initialization import BoundaryConfig, pml_objects_from_config
from fdtdx.objects.container import ArrayContainer, ParameterContainer
from fdtdx.objects.detectors.energy import EnergyDetector
from fdtdx.objects.detectors.poynting_flux import PoyntingFluxDetector
from fdtdx.objects.initialization import apply_params, place_objects
from fdtdx.objects.material import SimulationVolume, Substrate, WaveGuide
from fdtdx.objects.multi_material.device import Device
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.sources.plane_source import ConstantAmplitudePlaneSource
from fdtdx.shared.logger import Logger
from fdtdx.shared.plot_setup import plot_setup

# Design of Shen et al.: https://opg.optica.org/oe/fulltext.cfm?uri=oe-22-22-27175&id=303419
# Note that we are unable to reproduce the reported efficiency of their coupling device.
# Our simulation and commercial software only yields an efficiency of about 7% compared to their reported 50%.
REFERENCE_DESIGN_GRID = 1 - jnp.asarray(
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
    backward: bool = False  # only used when evaluation is true

    logger.info(f"{seed=}")
    logger.info(f"{evaluation=}")
    logger.info(f"{use_reference_grid=}")
    logger.info(f"{backward=}")

    permittivity_silicon = 12.25
    permittivity_silica = 2.25

    exp_logger = Logger(
        experiment_name="vertical_silicon_coupler",
    )
    key = jax.random.PRNGKey(seed=seed)

    wavelength = 1.55e-6
    period = constants.wavelength_to_period(wavelength)

    config = SimulationConfig(
        time=200e-15,
        resolution=25e-9,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    period_steps = round(period / config.time_step_duration)
    all_time_steps = list(range(config.time_steps_total))
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    if not evaluation or backward:
        gradient_config = GradientConfig(recorder=Recorder(modules=[]))
        config = config.aset("gradient_config", gradient_config)

    placement_constraints = []

    volume = SimulationVolume(
        partial_real_shape=(8e-6, 6e-6, 7e-6),
    )

    bound_cfg = BoundaryConfig.from_uniform_bound(thickness=10)
    _, c_list = pml_objects_from_config(bound_cfg, volume)
    placement_constraints.extend(c_list)

    waver_substrate = Substrate(
        name="Silicon",
        color=colors.TAN,
        partial_real_shape=(None, None, 1e-6),
        permittivity=permittivity_silicon,
    )
    placement_constraints.append(
        waver_substrate.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, -1),
            other_positions=(0, 0, -1),
        )
    )

    dioxide_substrate = Substrate(
        name="Silica",
        color=colors.LIGHT_BROWN,
        partial_real_shape=(None, None, 3e-6),
        permittivity=permittivity_silica,
    )
    placement_constraints.append(
        dioxide_substrate.place_relative_to(
            waver_substrate,
            axes=(0, 1, 2),
            own_positions=(0, 0, -1),
            other_positions=(0, 0, 1),
        )
    )

    permittivity_config = {
        "Air": constants.relative_permittivity_air,
        "Silicon": permittivity_silicon,
    }
    device = Device(
        name="Device",
        partial_real_shape=(3e-6, 3e-6, 300e-9),
        permittivity_config=permittivity_config,
        constraint_mapping=ConstraintMapping(
            modules=[
                StandardToInversePermittivityRange(),
                ClosestIndex(),
                IndicesToInversePermittivities(),
            ],
        ),
        partial_voxel_real_shape=(100e-9, 100e-9, 300e-9),
    )
    placement_constraints.append(
        device.place_relative_to(
            dioxide_substrate,
            axes=(0, 1, 2),
            own_positions=(-1, 0, -1),
            other_positions=(-1, 0, 1),
            grid_margins=(bound_cfg.thickness_grid_minx, 0, 0),
            margins=(1e-6, 0, 0),
        )
    )

    source = ConstantAmplitudePlaneSource(
        partial_real_shape=(None, None, None),
        partial_grid_shape=(None, None, 1),
        wavelength=1.550e-6,
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

    in_detector = PoyntingFluxDetector(
        name="Input Flux Detector",
        direction="-",
        partial_grid_shape=(None, None, 1),
        time_steps=all_time_steps[-2 * period_steps :],
        exact_interpolation=True,
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
            in_detector.size_relative_to(
                device,
                axes=(
                    0,
                    1,
                ),
            ),
        ]
    )

    waveguide = WaveGuide(
        partial_real_shape=(None, 0.4e-6, 0.3e-6),
        permittivity=permittivity_silicon,
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
            waveguide.place_above(dioxide_substrate),
        ]
    )

    waveguide_detector = PoyntingFluxDetector(
        name="Output Flux Detector",
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

    if use_reference_grid:
        params["Device"]["out"] = REFERENCE_DESIGN_GRID

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
            in_names=(in_detector.name,),
            out_names=(waveguide_detector.name,),
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
            init_value=0.005,
            peak_value=0.05,
            end_value=0.005,
            warmup_steps=10,
            decay_steps=round(0.9 * epochs),
        )
        optimizer = optax.inject_hyperparams(optax.nadam)(learning_rate=schedule)

        optimizer = optax.MultiSteps(optimizer, every_k_schedule=1)
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
            export_figure=True,
        )
        info["changed_voxels"] = changed_voxels

        # videos
        exp_logger.log_detectors(iter_idx=epoch, objects=objects, detector_states=arrays.detector_states)

        exp_logger.write(info)
        exp_logger.progress.update(optim_task_id, advance=1)


if __name__ == "__main__":
    seed = 42
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    main(seed)
