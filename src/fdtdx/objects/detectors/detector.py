from abc import ABC, abstractmethod
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.figure import Figure
from rich.progress import Progress

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, frozen_field, frozen_private_field, private_field
from fdtdx.core.plotting.colors import LIGHT_GREEN
from fdtdx.core.switch import OnOffSwitch
from fdtdx.objects.detectors.plotting.line_plot import plot_line_over_time, plot_waterfall_over_time
from fdtdx.objects.detectors.plotting.plot2d import plot_2d_from_slices
from fdtdx.objects.detectors.plotting.video import generate_video_from_slices, plot_from_slices
from fdtdx.objects.object import SimulationObject
from fdtdx.typing import SliceTuple3D

DetectorState = dict[str, jax.Array]


@autoinit
class Detector(SimulationObject, ABC):
    """Base class for electromagnetic field detectors in FDTD simulations.

    This class provides core functionality for recording and analyzing electromagnetic field data
    during FDTD simulations. It supports flexible timing control, data collection intervals,
    and visualization of results.

    Attributes:
        dtype (jnp.dtype): Data type for detector arrays, defaults to float32.
        exact_interpolation (bool, optional): Whether to use exact field interpolation. Defaults to True.
        inverse (bool, optional): Whether to record fields in inverse time order. Defaults to false.
        switch (OnOffSwitch, optional): This switch controls the time steps that the detector is on, i.e. records data.
            Defaults to all time steps.
        plot (bool, optional): Whether to generate plots of recorded data. Defaults to true.
        if_inverse_plot_backwards (bool): Plot inverse data in reverse time order.
        num_video_workers (int | None): Number of workers for video generation. If None (default), then no
            multiprocessing is used. Note that the combination of multiprocessing and matplotlib is known to produce
            problems and can cause the entire system to freeze. It does make the video generation much faster though.
        color (tuple[float, float, float] | None, optional): RGB color for plotting. Defaults to light green.
        plot_interpolation (str, optional): Interpolation method for plots. Defualts to "gaussian".
        plot_dpi (int | None, optional): DPI resolution for plots. Defaults to None.
    """

    dtype: jnp.dtype = frozen_field(default=jnp.float32)
    exact_interpolation: bool = frozen_field(default=True)
    inverse: bool = frozen_field(default=False)
    switch: OnOffSwitch = frozen_field(default=OnOffSwitch())
    plot: bool = frozen_field(default=True)
    if_inverse_plot_backwards: bool = frozen_field(default=True)
    num_video_workers: int | None = frozen_field(default=None)  # only used when generating video
    color: tuple[float, float, float] | None = frozen_field(default=LIGHT_GREEN)
    plot_interpolation: str = frozen_field(default="gaussian")
    plot_dpi: int | None = frozen_field(default=None)

    _num_time_steps_on: int = frozen_private_field()
    _is_on_at_time_step_arr: jax.Array = private_field()
    _time_step_to_arr_idx: jax.Array = private_field()

    @property
    def num_time_steps_recorded(self) -> int:
        """Gets the total number of time steps that will be recorded.

        Returns:
            int: Number of time steps where detector will record data.

        Raises:
            Exception: If detector is not yet initialized.
        """
        if self._num_time_steps_on is None:
            raise Exception("Detector is not yet initialized")
        return self._num_time_steps_on

    def _calculate_on_list(
        self,
    ) -> list[bool]:
        return self.switch.calculate_on_list(
            num_total_time_steps=self._config.time_steps_total,
            time_step_duration=self._config.time_step_duration,
        )

    def _num_latent_time_steps(self) -> int:
        """Calculates total number of time steps that will be recorded.

        Returns:
            int: Number of time steps where detector will be active.
        """
        on_list = self._calculate_on_list()
        return sum(on_list)

    @abstractmethod
    def _shape_dtype_single_time_step(
        self,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        """Gets shape and dtype information for a single time step recording.

        Returns:
            dict[str, jax.ShapeDtypeStruct]: Dictionary mapping field names to their
                shape and dtype specifications.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )
        # determine number of time steps on
        on_list = self._calculate_on_list()
        on_arr = jnp.asarray(on_list, dtype=jnp.bool)
        self = self.aset("_is_on_at_time_step_arr", on_arr, create_new_ok=True)
        self = self.aset("_num_time_steps_on", sum(on_list), create_new_ok=True)
        # calculate mapping time step -> arr index
        counter = 0
        num_t = self._config.time_steps_total
        time_to_arr_idx_list = [-1 for _ in range(num_t)]
        for t in range(num_t):
            if on_list[t]:
                time_to_arr_idx_list[t] = counter
                counter += 1
        time_to_arr_idx_arr = jnp.asarray(time_to_arr_idx_list, dtype=jnp.int32)
        self = self.aset("_time_step_to_arr_idx", time_to_arr_idx_arr, create_new_ok=True)
        return self

    def init_state(
        self: Self,
    ) -> DetectorState:
        # init arrays
        shape_dtype_dict = self._shape_dtype_single_time_step()
        state = {}
        latent_time_size = self._num_latent_time_steps()
        for name, shape_dtype in shape_dtype_dict.items():
            cur_arr = jnp.zeros(
                shape=(latent_time_size, *shape_dtype.shape),
                dtype=shape_dtype.dtype,
            )
            state[name] = cur_arr
        return state

    @abstractmethod
    def update(
        self,
        time_step: jax.Array,
        E: jax.Array,
        H: jax.Array,
        state: DetectorState,
        inv_permittivity: jax.Array,
        inv_permeability: jax.Array | float,
    ) -> DetectorState:
        """Updates detector state with current field values.

        Args:
            time_step (jax.Array): Current simulation time step.
            E (jax.Array): Electric field array.
            H (jax.Array): Magnetic field array.
            state (DetectorState): Current detector state.
            inv_permittivity (jax.Array): Inverse permittivity array.
            inv_permeability (jax.Array | float): Inverse permeability array.

        Returns:
            DetectorState: Updated detector state.
        """
        del (
            time_step,
            E,
            H,
            state,
            inv_permittivity,
            inv_permeability,
        )
        raise NotImplementedError()

    def draw_plot(
        self,
        state: dict[str, np.ndarray],
        progress: Progress | None = None,
    ) -> dict[str, Figure | str]:
        """Generates plots or videos from recorded detector data.

        Creates visualizations based on dimensionality of recorded data and detector
        configuration. Supports 1D line plots, 2D heatmaps, and video generation
        for time-varying data.

        Args:
            state (dict[str, np.ndarray]): Dictionary containing recorded field data arrays.
            progress (Progress | None, optional): Optional progress bar for video generation.

        Returns:
            dict[str, Figure | str]: Dictionary mapping plot names to either
                matplotlib Figure objects or paths to generated video files.
        """
        squeezed_arrs = {}
        squeezed_ndim = None
        for k, v in state.items():
            v_squeezed = v.squeeze()
            if self.inverse and self.if_inverse_plot_backwards and self.num_time_steps_recorded > 1:
                squeezed_arrs[k] = v_squeezed[::-1, ...]
            else:
                squeezed_arrs[k] = v_squeezed
            if squeezed_ndim is None:
                squeezed_ndim = len(v_squeezed.shape)
            else:
                if len(v_squeezed.shape) != squeezed_ndim:
                    raise Exception("Cannot plot multiple arrays with different ndim")
        if squeezed_ndim is None:
            raise Exception(f"empty state: {state}")

        figs = {}
        if squeezed_ndim == 1 and self.num_time_steps_recorded > 1:
            # do line plot
            time_steps = np.where(np.asarray(self._is_on_at_time_step_arr))[0]
            time_steps = time_steps * self._config.time_step_duration
            for k, v in squeezed_arrs.items():
                fig = plot_line_over_time(arr=v, time_steps=time_steps.tolist(), metric_name=f"{self.name}: {k}")
                figs[k] = fig
        elif squeezed_ndim == 1 and self.num_time_steps_recorded == 1:
            SCALE = 10
            xlabel = None
            if self.grid_shape[0] > 1 and self.grid_shape[1] <= 1 and self.grid_shape[2] <= 1:
                xlabel = "X axis (μm)"
            elif self.grid_shape[0] <= 1 and self.grid_shape[1] > 1 and self.grid_shape[2] <= 1:
                xlabel = "Y axis (μm)"
            elif self.grid_shape[0] <= 1 and self.grid_shape[1] <= 1 and self.grid_shape[2] > 1:
                xlabel = "Z axis (μm)"
            assert xlabel is not None, "This should never happen"
            for k, v in squeezed_arrs.items():
                spatial_axis = np.arange(len(v)) / SCALE
                fig = plot_line_over_time(
                    arr=v, time_steps=spatial_axis, metric_name=f"{self.name}: {k}", xlabel=xlabel
                )
                figs[k] = fig
        elif squeezed_ndim == 2 and self.num_time_steps_recorded > 1:
            # multiple time steps, 1d spatial data - visualize as 2D waterfall plot
            time_steps = np.where(np.asarray(self._is_on_at_time_step_arr))[0]
            time_steps = time_steps * self._config.time_step_duration

            # Determine spatial axis based on which dimension has size > 1
            SCALE = 10  # μm per grid point

            for k, v in squeezed_arrs.items():
                # Determine which dimension is spatial (not time)
                spatial_dim = 1 if v.shape[1] > 1 else 0
                if spatial_dim == 0:
                    # Transpose if needed so time is always first dimension
                    v = v.T

                # Create spatial axis in μm
                spatial_points = np.arange(v.shape[1]) / SCALE

                fig = plot_waterfall_over_time(
                    arr=v,
                    time_steps=time_steps,
                    spatial_steps=spatial_points,
                    metric_name=f"{self.name}: {k}",
                    spatial_unit="μm",
                )
                figs[k] = fig
        elif squeezed_ndim == 2 and self.num_time_steps_recorded == 1:
            # single time step, 2d-plot  # TODO:
            if all([x in squeezed_arrs.keys() for x in ["XY Plane", "XZ Plane", "YZ Plane"]]):
                fig = plot_2d_from_slices(
                    xy_slice=squeezed_arrs["XY Plane"],
                    xz_slice=squeezed_arrs["XZ Plane"],
                    yz_slice=squeezed_arrs["YZ Plane"],
                    resolutions=(
                        self._config.resolution,
                        self._config.resolution,
                        self._config.resolution,
                    ),
                    plot_dpi=self.plot_dpi,
                    plot_interpolation=self.plot_interpolation,
                )
                figs["sliced_plot"] = fig
            else:
                raise Exception(f"Cannot plot {squeezed_arrs.keys()}")
        elif squeezed_ndim == 3 and self.num_time_steps_recorded > 1:
            # multiple time steps, 2d-plots
            if all([x in squeezed_arrs.keys() for x in ["XY Plane", "XZ Plane", "YZ Plane"]]):
                path = generate_video_from_slices(
                    plt_fn=plot_from_slices,
                    xy_slice=squeezed_arrs["XY Plane"],
                    xz_slice=squeezed_arrs["XZ Plane"],
                    yz_slice=squeezed_arrs["YZ Plane"],
                    progress=progress,
                    num_worker=self.num_video_workers,
                    resolutions=(
                        self._config.resolution,
                        self._config.resolution,
                        self._config.resolution,
                    ),
                    plot_dpi=self.plot_dpi,
                    plot_interpolation=self.plot_interpolation,
                )
                figs["sliced_video"] = path
            else:
                raise Exception(
                    f"Cannot plot {squeezed_arrs.keys()}. "
                    f"Consider setting plot=False for Object {self.name} ({self.__class__=})"
                )
        elif squeezed_ndim == 3 and self.num_time_steps_recorded == 1:
            # single step, 3d-plot. # TODO: do as mean over planes
            for k, v in squeezed_arrs.items():
                xy_slice = squeezed_arrs[k].mean(axis=0)
                xz_slice = squeezed_arrs[k].mean(axis=1)
                yz_slice = squeezed_arrs[k].mean(axis=2)
                fig = plot_2d_from_slices(
                    xy_slice=xy_slice,
                    xz_slice=xz_slice,
                    yz_slice=yz_slice,
                    resolutions=(
                        self._config.resolution,
                        self._config.resolution,
                        self._config.resolution,
                    ),
                    plot_dpi=self.plot_dpi,
                    plot_interpolation=self.plot_interpolation,
                )
                figs[k] = fig
        elif squeezed_ndim == 4 and self.num_time_steps_recorded > 1:
            # video with 3d-volume in each time step. plot as slices
            for k, v in squeezed_arrs.items():
                xy_slice = squeezed_arrs[k].mean(axis=1)
                xz_slice = squeezed_arrs[k].mean(axis=2)
                yz_slice = squeezed_arrs[k].mean(axis=3)
                path = generate_video_from_slices(
                    plt_fn=plot_from_slices,
                    xy_slice=xy_slice,
                    xz_slice=xz_slice,
                    yz_slice=yz_slice,
                    progress=progress,
                    num_worker=self.num_video_workers,
                    resolutions=(
                        self._config.resolution,
                        self._config.resolution,
                        self._config.resolution,
                    ),
                    plot_dpi=self.plot_dpi,
                    plot_interpolation=self.plot_interpolation,
                )
                figs[k] = path
        else:
            raise Exception("Cannot plot detector with more than three dimensions")
        return figs
