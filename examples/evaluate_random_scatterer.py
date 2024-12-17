import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytreeclass as tc
import seaborn as sns
from loguru import logger
from matplotlib.colors import LogNorm

from fdtdx.core.config import SimulationConfig
from fdtdx.core.physics import constants
from fdtdx.fdtd.fdtd import reversible_fdtd
from fdtdx.objects.boundaries.initialization import BoundaryConfig, pml_objects_from_config
from fdtdx.objects.container import ArrayContainer, ParameterContainer
from fdtdx.objects.detectors.energy import EnergyDetector
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.initialization import apply_params, place_objects
from fdtdx.objects.material import SimulationVolume
from fdtdx.objects.multi_material.random_scatterer import RandomScatterer
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.sources.plane_source import HardConstantAmplitudePlanceSource
from fdtdx.shared.logger import Logger
from fdtdx.shared.plot_setup import plot_setup


def main(seed: int = 42):
    logger.info(f"{seed=}")
    key = jax.random.PRNGKey(seed=seed)

    exp_logger = Logger(experiment_name="random_scatterer")
    rng = jax.random.PRNGKey(seed=seed)

    wavelength = 1.0e-6
    period = constants.wavelength_to_period(wavelength)

    config = SimulationConfig(
        time=100e-15,
        resolution=40e-9,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    period_steps = round(period / config.time_step_duration)
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    volume = SimulationVolume(
        partial_real_shape=(6.08e-6, 6.08e-6, 6.08e-6),
    )

    placement_constraints = []

    bound_cfg = BoundaryConfig.from_uniform_bound(thickness=12)
    bound_cfg = bound_cfg.aset("kappa_end_minx", 1)
    bound_cfg = bound_cfg.aset("kappa_end_maxx", 1)
    bound_cfg = bound_cfg.aset("kappa_end_miny", 1)
    bound_cfg = bound_cfg.aset("kappa_end_maxy", 1)

    _, c_list = pml_objects_from_config(bound_cfg, volume)
    placement_constraints.extend(c_list)

    source = HardConstantAmplitudePlanceSource(
        partial_real_shape=(None, None, None),
        partial_grid_shape=(None, None, 1),
        direction="+",
        wavelength=wavelength,
        fixed_E_polarization_vector=(1, 0, 0),
        fixed_H_polarization_vector=(0, 1, 0),
        end_after_periods=10.0,
        start_after_periods=0.0,
        phase_shift=jnp.pi,
    )

    placement_constraints.extend(
        [
            source.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(0, 0, -1),
                other_positions=(0, 0, -1),
                grid_margins=(0, 0, bound_cfg.thickness_grid_minz),
            ),
        ]
    )

    # path to a directory contianing only the dataset from the following paper:
    # https://radar.kit.edu/radar/en/folder/LMsUjehfktnKtTHK.fields_3d_exeyez
    # (paper: https://pubs.acs.org/doi/full/10.1021/acsphotonics.3c00156)
    path = "path/to/dataset"
    scatterer = RandomScatterer(
        dataset_path=path,
    )
    placement_constraints.extend(
        [
            scatterer.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(0, 0, 0),
                other_positions=(0, 0, 0),
            ),
            scatterer.size_relative_to(
                volume,
                axes=(0, 1, 2),
                proportions=(1, 1, 1),
                grid_offsets=(-24, -24, -24),
            ),
        ]
    )

    detector_phasor = PhasorDetector(
        name="Phasor",
        partial_real_shape=(None, None, None),
        wavelength=wavelength,
        components=["Ex", "Ey", "Ez"],
    )
    placement_constraints.extend(
        [
            detector_phasor.place_relative_to(
                scatterer,
                axes=(0, 1, 2),
                own_positions=(0, 0, 0),
                other_positions=(0, 0, 0),
            ),
            detector_phasor.same_size(scatterer),
        ]
    )

    energy_last_step = EnergyDetector(
        name="energy_last_step",
        as_slices=True,
        time_steps=[-1],
    )
    placement_constraints.extend([*energy_last_step.same_position_and_size(volume)])

    exclude_object_list: list[SimulationObject] = [energy_last_step]
    video_detector = EnergyDetector(
        name="video",
        as_slices=True,
        interval=10,
        exact_interpolation=True,
    )
    placement_constraints.extend([*video_detector.same_position_and_size(volume)])
    exclude_object_list.append(video_detector)

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

    def predict(
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

        prediction = jnp.real(arrays.detector_states["Phasor"]["phasor"])[0]
        ground_truth = new_objects[scatterer.name]._fields[::2]  # type: ignore
        design = new_objects[scatterer.name]._design  # type: ignore
        return prediction, ground_truth, design, arrays.detector_states

    def normalize_field(field):
        return (field - jnp.min(field)) / (jnp.max(field) - jnp.min(field))

    def plot_field_component_planes(field_component, component: str, design: jax.Array, norm=None):
        # Define the planes to plot
        planes = {
            "xy": field_component[int(field_component.shape[0] / 2), :, :],
            "xz": field_component[:, int(field_component.shape[1] / 2), :],
            "yz": field_component[:, :, int(field_component.shape[2] / 2)],
        }

        # Create a figure with subplots side by side
        fig, axs = plt.subplots(1, len(planes), figsize=(18, 6))  # Adjust size as needed

        for ax, (plane_name, plane_data) in zip(axs, planes.items()):
            im = ax.imshow(
                plane_data,
                cmap=sns.color_palette("vlag", as_cmap=True),
                norm=None if not norm else LogNorm(vmin=1e-2, vmax=1e0),
            )
            # Overlay scatterer.design on the respective plane
            design_plane = (
                design[int(field_component.shape[0] / 2), :, :]
                if plane_name == "xy"
                else design[:, int(field_component.shape[1] / 2), :]
                if plane_name == "xz"
                else design[:, :, int(field_component.shape[2] / 2)]
            )
            ax.contour(design_plane, levels=[0.5], colors="grey", linestyles="-", linewidths=3)
            fig.colorbar(im, ax=ax)
            ax.set_title(f"{component} component in {plane_name} plane")
            ax.set_xlabel("x" if "x" in plane_name else ("y" if "y" in plane_name else "z"))
            ax.set_ylabel("y" if "xy" in plane_name else ("z" if "yz" in plane_name else "x"))
        return fig

    def plot_absolute_error_planes(ground_truth, prediction, component, design):
        error = jnp.abs(ground_truth - prediction)
        return plot_field_component_planes(error, component, design=design, norm="log"), error.mean()

    jit_task_id = exp_logger.progress.add_task("JIT", total=None)
    jitted_predict = jax.jit(predict).lower(params, arrays, rng).compile()
    exp_logger.progress.update(jit_task_id, total=1, completed=1, refresh=True)

    def evaluate(epoch, key):
        error = 0
        components = ["Ex", "Ey", "Ez"]
        prediction, ground_truth, design, detector_states = jitted_predict(params, arrays, key)

        exp_logger.log_detectors(
            iter_idx=epoch,
            detector_states=detector_states,
            objects=objects,
        )

        for i, component in enumerate(components):
            normalized_ground_truth = normalize_field(ground_truth[i, ...])
            normalized_prediction = normalize_field(prediction[i, ...])
            exp_logger.savefig(
                exp_logger.cwd,
                f"{epoch}_meep_{component}_norm",
                plot_field_component_planes(normalized_ground_truth, component=component, design=design),
            )
            exp_logger.savefig(
                exp_logger.cwd,
                f"{epoch}_fdtdx_{component}_norm",
                plot_field_component_planes(normalized_prediction, component=component, design=design),
            )
            error_fig, component_error = plot_absolute_error_planes(
                normalize_field(ground_truth[i, ...]),
                normalize_field(prediction[i, ...]),
                component,
                design=design,
            )
            exp_logger.savefig(
                exp_logger.cwd,
                f"{epoch}_difference_{component}_norm",
                error_fig,
            )
            error += component_error
        error = error / len(components)
        return error

    epochs = 2
    evaluation_task_id = exp_logger.progress.add_task("Evaluation", start=True, total=epochs)
    for i in range(epochs):
        rng, key = jax.random.split(rng)
        error = evaluate(i, key)
        exp_logger.write({"error": error, "iteration": i})
        exp_logger.progress.update(evaluation_task_id, advance=1)
    exp_logger.progress.update(evaluation_task_id, completed=True)


if __name__ == "__main__":
    main()
