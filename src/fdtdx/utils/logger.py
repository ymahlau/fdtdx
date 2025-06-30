import atexit
import csv
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

from fdtdx.conversion.export import export_stl as export_stl_fn
from fdtdx.core.misc import cast_floating_to_numpy, get_background_material_name
from fdtdx.core.plotting.device_permittivity_index_utils import device_matrix_index_figure
from fdtdx.fdtd.container import ObjectContainer, ParameterContainer
from fdtdx.materials import compute_ordered_names
from fdtdx.objects.detectors.detector import DetectorState


def init_working_directory(experiment_name: str, wd_name: str | None) -> Path:
    """Initialize working directory for experiment outputs.

    Creates a timestamped directory structure for experiment outputs under outputs/nobackup/.
    Uses current date/time unless a specific working directory name is provided.

    Args:
        experiment_name (str): Name of the experiment
        wd_name (str | None): Optional specific name for the working directory. If None, uses timestamp.

    Returns:
        Path: Created working directory path
    """
    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    day, daytime = cur_time.split("_")
    new_cwd = Path().cwd() / "outputs" / "nobackup" / day / experiment_name / (daytime if wd_name is None else wd_name)
    new_cwd.mkdir(parents=True)
    return new_cwd


def _log_formatter(record: Any) -> str:
    """Log message formatter"""
    color_map = {
        "TRACE": "dim blue",
        "DEBUG": "cyan",
        "INFO": "bold",
        "SUCCESS": "bold green",
        "WARNING": "yellow",
        "ERROR": "bold red",
        "CRITICAL": "bold white on red",
    }
    lvl_color = color_map.get(record["level"].name, "cyan")
    loc = record["file"].path + ":" + str(record["line"])
    message = escape(record["message"])
    return (
        "[not bold green]{time:DD.MM.YYYY HH:mm:ss.SSS}[/not bold green] | "
        + f"{loc}"
        + f" - [{lvl_color}]{message}[/{lvl_color}]"
    )


def snapshot_python_files(snapshot_dir: Path):
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    # fdtdx
    root_dir = Path(__file__).parent.parent
    files = list(root_dir.rglob("*.py"))
    # scripts
    scripts_dir = Path(__file__).parent.parent.parent / "scripts"
    files = files + list(scripts_dir.rglob("*.py"))

    for python_file in files:
        relative_path = python_file.relative_to(root_dir.parent)
        destination = snapshot_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(python_file, destination)
    # make zip and delete directory
    shutil.make_archive(str(snapshot_dir.parent / "code"), "zip", snapshot_dir)
    shutil.rmtree(snapshot_dir)


class Logger:
    """Logger for managing experiment outputs and visualization.

    Handles experiment logging, metrics tracking, and visualization of simulation results.
    Creates a working directory structure, initializes logging, and provides methods for
    saving figures, metrics, and device parameters.

    Args:
        experiment_name (str): Name of the experiment. This is the naming of the parent directory where the experiment
            will be saved.
        name (str | None, optional): Optional specific name for the working directory. If None, uses timestamp.
    """

    def __init__(self, experiment_name: str, name: str | None = None):
        sns.set_theme(context="paper", style="white", palette="colorblind")
        self.cwd = init_working_directory(experiment_name, wd_name=name)
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console,
        ).__enter__()
        atexit.register(self.progress.stop)
        logger.remove()
        logger.add(
            self.console.print,
            level="TRACE",
            format=_log_formatter,
            colorize=True,
        )
        logger.add(
            self.cwd / "logs.log",
            level="TRACE",
            format="{time:DD.MM.YYYY HH:mm:ss:ssss} | {level} - {message}",
        )
        logger.info(f"Starting experiment {experiment_name} in {self.cwd}")
        snapshot_python_files(self.cwd / "code")
        self.fieldnames = None
        self.writer = None
        self.csvfile = open(self.cwd / "metrics.csv", "w", newline="")
        self.last_indices: dict[str, jax.Array | None] = defaultdict(lambda: None)
        atexit.register(self.csvfile.close)

    @property
    def stl_dir(self) -> Path:
        """Directory for storing STL files.

        Returns:
            Path: Directory for STL file outputs
        """
        directory = self.cwd / "device" / "stl"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def params_dir(self) -> Path:
        """Directory for storing parameter files.

        Returns:
            Path: Directory for parameter file outputs
        """
        directory = self.cwd / "device" / "params"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def savefig(self, directory: Path, filename: str, fig: Figure, dpi: int = 300):
        """Save a matplotlib figure to file.

        Creates a figures subdirectory if needed and saves the figure with specified settings.

        Args:
            directory (Path): Base directory to save in
            filename (str): Name for the figure file
            fig (Figure): Matplotlib figure to save
            dpi (int, optional): Resolution in dots per inch. Defaults to 300.
        """
        figure_directory = directory / "figures"
        figure_directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "figures" / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def write(self, stats: dict, do_print: bool = True):
        """Write statistics to CSV file and optionally print them.

        Records metrics in a CSV file and optionally displays them in a formatted table.
        Automatically initializes CSV headers on first write.

        Args:
            stats (dict): Dictionary of statistics to record
            do_print (bool, optional): Whether to print stats to console. Defaults to true.
        """
        stats = {
            k: v.item() if isinstance(v, jax.Array) else v
            for k, v in stats.items()
            if isinstance(v, (int, float)) or (isinstance(v, jax.Array) and v.size == 1)
        }
        if self.fieldnames is None:
            self.fieldnames = list(stats.keys())
            self.writer = csv.DictWriter(self.csvfile, fieldnames=self.fieldnames)
            self.writer.writeheader()
        assert self.writer is not None
        self.writer.writerow(stats)
        self.csvfile.flush()
        if do_print:
            table = Table(box=None)
            for k, v in stats.items():
                table.add_column(k)
                table.add_column(str(v))
            self.console.print(table)

    def log_detectors(
        self,
        iter_idx: int,
        objects: ObjectContainer,
        detector_states: dict[str, DetectorState],
        exclude: Sequence[str] = [],
    ):
        """Log detector states and generate visualization plots.

        Creates plots for each detector's state and saves them to the detector's output directory.
        Handles both figure outputs and other detector-specific file formats.

        Args:
            iter_idx (int): Current iteration index
            objects (ObjectContainer): Container with simulation objects
            detector_states (dict[str, DetectorState]): Dictionary mapping detector names to their states
            exclude (Sequence[str], optional): List of detector names to exclude from logging
        """
        for detector in [d for d in objects.detectors if d.name not in exclude]:
            cur_state = jax.device_get(detector_states[detector.name])
            cur_state = cast_floating_to_numpy(cur_state, float)

            if not detector.plot:
                continue
            figure_dict = detector.draw_plot(
                state=cur_state,
                progress=self.progress,
            )

            detector_dir = self.cwd / "detectors" / detector.name
            detector_dir.mkdir(parents=True, exist_ok=True)

            for k, v in figure_dict.items():
                if isinstance(v, Figure):
                    self.savefig(
                        detector_dir,
                        f"{detector.name}_{k}_{iter_idx}.png",
                        v,
                        dpi=detector.plot_dpi,  # type: ignore
                    )
                elif isinstance(v, str):
                    shutil.copy(
                        v,
                        detector_dir / f"{detector.name}_{k}_{iter_idx}{Path(v).suffix}",
                    )
                else:
                    raise Exception(f"invalid detector output for plotting: {k}, {v}")

    def log_params(
        self,
        iter_idx: int,
        params: ParameterContainer,
        objects: ObjectContainer,
        export_figure: bool = False,
        export_stl: bool = False,
        export_background_stl: bool = False,
        **transformation_kwargs,
    ) -> int:
        """Log parameter states and export device visualizations.

        Saves device parameters and optionally exports visualizations as figures or STL files.
        Tracks changes in device voxels between iterations.

        Args:
            iter_idx (int): Current iteration index
            params (ParameterContainer): Container with device parameters
            objects (ObjectContainer): Container with simulation objects
            export_figure (bool, optional): Whether to export index matrix figures
            export_stl (bool, optional): Whether to export device geometry as STL
            export_background_stl (bool, optional): Whether to export air regions as STL
            **transformation_kwargs: keyword arguments passed to the parameter transformation

        Returns:
            int: Number of voxels that changed since last iteration
        """
        changed_voxels = 0
        for device in objects.devices:
            device_params = params[device.name]
            indices = device(device_params, **transformation_kwargs)

            # raw parameters and indices
            if isinstance(device_params, dict):
                for k, v in device_params.items():
                    jnp.save(self.params_dir / f"params_{iter_idx}_{device.name}_{k}.npy", v)
            else:
                jnp.save(self.params_dir / f"params_{iter_idx}_{device.name}.npy", device_params)
            jnp.save(self.params_dir / f"matrix_{iter_idx}_{device.name}.npy", indices)

            has_previous = self.last_indices[device.name] is not None
            cur_changed_voxels = 0
            if has_previous:
                last_device_indices = self.last_indices[device.name]
                cur_changed_voxels = int(jnp.sum(indices != last_device_indices))
            changed_voxels += cur_changed_voxels
            self.last_indices[device.name] = indices
            if cur_changed_voxels == 0 and has_previous:
                continue
            if export_stl:
                background_name = get_background_material_name(device.materials)
                ordered_name_list = compute_ordered_names(device.materials)
                background_idx = ordered_name_list.index(background_name)
                for idx in range(len(device.materials)):
                    if idx == background_idx and not export_background_stl:
                        continue
                    name = ordered_name_list[idx]
                    export_stl_fn(
                        matrix=np.round(indices) == idx,
                        stl_filename=self.stl_dir / f"matrix_{iter_idx}_{device.name}_{name}.stl",
                        voxel_grid_size=device.single_voxel_grid_shape,
                    )
                if len(device.materials) > 2:
                    export_stl_fn(
                        matrix=np.round(indices) != background_idx,
                        stl_filename=self.stl_dir / f"matrix_{iter_idx}_{device.name}_non_air.stl",
                        voxel_grid_size=device.single_voxel_grid_shape,
                    )

            # image of indices
            if export_figure:
                fig = device_matrix_index_figure(
                    device_matrix_indices=indices,
                    material=device.materials,
                    parameter_type=device.output_type,
                )
                self.savefig(
                    self.cwd / "device",
                    f"matrix_indices_{iter_idx}_{device.name}.png",
                    fig,
                )

        return changed_voxels
