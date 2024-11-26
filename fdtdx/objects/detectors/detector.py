from abc import ABC, abstractmethod
from typing import Self
import jax
import jax.numpy as jnp
from matplotlib.figure import Figure
import numpy as np
import pytreeclass as tc
from rich.progress import Progress

from fdtdx.core.misc import is_on_at_time_step
from fdtdx.core.plotting.colors import LIGHT_GREEN
from fdtdx.objects.detectors.plotting.line_plot import plot_line_over_time
from fdtdx.objects.detectors.plotting.plot2d import plot_2d_from_slices
from fdtdx.objects.detectors.plotting.video import generate_video_from_slices, plot_from_slices
from fdtdx.objects.material import NoMaterial
from fdtdx.core.jax.typing import SliceTuple3D
from fdtdx.core.config import SimulationConfig

DetectorState = dict[str, jax.Array]

@tc.autoinit
class Detector(NoMaterial, ABC):
    name: str = tc.field( # type: ignore
        init=True, 
        kind="KW_ONLY",
        on_getattr=[tc.unfreeze],
        on_setattr=[tc.freeze],
    )
    dtype: jnp.dtype = tc.field( # type: ignore
        init=True, 
        default=jnp.float32,
        kind="KW_ONLY",
        on_getattr=[tc.unfreeze],
        on_setattr=[tc.freeze],
    )
    exact_interpolation: bool = False
    inverse: bool = False
    interval: int = 1
    start_after_periods: float | None = None
    end_after_periods: float | None = None
    period_length: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    on_for_time: float | None = None
    on_for_periods: float | None = None
    time_steps: list[int] = tc.field(default=None, init=True)  # type: ignore
    plot: bool = True
    if_inverse_plot_backwards: bool = True
    num_video_workers: int = 8  # only used when generating video
    _is_on_at_time_step_arr: jax.Array = tc.field(default=None, init=False) # type: ignore
    _time_step_to_arr_idx: jax.Array = tc.field(default=None, init=False) # type: ignore
    _num_time_steps_on: int = tc.field(default=None, init=False) # type: ignore
    color: tuple[float, float, float] = LIGHT_GREEN
    plot_interpolation: str = tc.field(  # type: ignore
        init=True,
        default="gaussian",
        kind="KW_ONLY",
        on_getattr=[tc.unfreeze],
        on_setattr=[tc.freeze],
    )
    plot_dpi: int | None = tc.field(  # type: ignore
        init=True,
        default=None,
        kind="KW_ONLY",
        on_getattr=[tc.unfreeze],
        on_setattr=[tc.freeze],
    )
    
    @property
    def num_time_steps_recorded(self) -> int:
        if self._num_time_steps_on is None:
            raise Exception(f"Detector is not yet initialized")
        return self._num_time_steps_on
    
    def _calculate_on_list(
        self,
    ) -> list[bool]:
        if self.time_steps is not None:
            on_list = [False for _ in range(self._config.time_steps_total)]
            for t_idx in self.time_steps:
                on_list[t_idx] = True
            return on_list
        on_list = []
        for t in range(self._config.time_steps_total):
            cur_on = is_on_at_time_step(
                is_on=True,
                start_time=self.start_time,
                start_after_periods=self.start_after_periods,
                end_time=self.end_time,
                end_after_periods=self.end_after_periods,
                time_step=t,
                time_step_duration=self._config.time_step_duration,
                period=self.period_length,
                on_for_time=self.on_for_time,
                on_for_periods=self.on_for_periods,
            )
            cur_on = cur_on and t % self.interval == 0
            on_list.append(cur_on)
        return on_list
    
    def _num_latent_time_steps(self) -> int:
        on_list = self._calculate_on_list()
        return sum(on_list)
    
    @abstractmethod
    def _shape_dtype_single_time_step(
        self,
    ) -> dict[str, jax.ShapeDtypeStruct]:
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
        self = self.aset('_is_on_at_time_step_arr', on_arr)
        self = self.aset('_num_time_steps_on', sum(on_list))
        # calculate mapping time step -> arr index
        counter = 0
        num_t = self._config.time_steps_total
        time_to_arr_idx_list = [-1 for _ in range(num_t)]
        for t in range(num_t):
            if on_list[t]:
                time_to_arr_idx_list[t] = counter
                counter += 1
        time_to_arr_idx_arr = jnp.asarray(time_to_arr_idx_list, dtype=jnp.int32)
        self = self.aset("_time_step_to_arr_idx", time_to_arr_idx_arr)
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
        inv_permeability: jax.Array,
    ) -> DetectorState:
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
        squeezed_arrs = {}
        sqeezed_ndim = None
        for k, v in state.items():
            v_squeezed = v.squeeze()
            if (
                self.inverse and self.if_inverse_plot_backwards 
                and self.num_time_steps_recorded > 1
            ):
                squeezed_arrs[k] = v_squeezed[::-1, ...]
            else:
                squeezed_arrs[k] = v_squeezed
            if sqeezed_ndim is None:
                sqeezed_ndim = len(v_squeezed.shape)
            else:
                if len(v_squeezed.shape) != sqeezed_ndim:
                    raise Exception(f"Cannot plot multiple arrays with differemt ndim")
        if sqeezed_ndim is None:
            raise Exception(f"empty state: {state}")
        
        figs = {}
        if sqeezed_ndim == 1 and self.num_time_steps_recorded > 1:
            # do line plot
            time_steps = np.where(np.asarray(self._is_on_at_time_step_arr))[0]
            time_steps = time_steps * self._config.time_step_duration
            for k, v in squeezed_arrs.items():
                fig = plot_line_over_time(
                    arr=v,
                    time_steps=time_steps.tolist(),
                    metric_name=f"{self.name}: {k}"
                )
                figs[k] = fig
        elif sqeezed_ndim == 1 and self.num_time_steps_recorded == 1:
            # single time step, 1d-plot  # TODO. same as above, change x-axis name
            raise NotImplementedError()
        elif sqeezed_ndim == 2 and self.num_time_steps_recorded > 1:
            # multiple time steps, 1d-plots  # TODO: do as 2d-plot
            raise NotImplementedError()
        elif sqeezed_ndim == 2 and self.num_time_steps_recorded == 1:
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
        elif sqeezed_ndim == 3 and self.num_time_steps_recorded > 1:
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
                raise Exception(f"Cannot plot {squeezed_arrs.keys()}")
        elif sqeezed_ndim == 3 and self.num_time_steps_recorded == 1:
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
        elif sqeezed_ndim == 4 and self.num_time_steps_recorded > 1:
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
            raise Exception(f"Cannot plot detector with more than three dimensions")
        return figs
            

