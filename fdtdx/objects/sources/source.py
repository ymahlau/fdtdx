from abc import ABC, abstractmethod
from typing import Literal, Self
import jax
import pytreeclass as tc

from fdtdx.core.config import SimulationConfig
from fdtdx.core.misc import is_on_at_time_step
from fdtdx.core.plotting.colors import ORANGE
from fdtdx.objects.wavelength import WaveLengthDependentNoMaterial
from fdtdx.core.jax.typing import SliceTuple3D
import jax.numpy as jnp

@tc.autoinit
class Source(WaveLengthDependentNoMaterial, ABC):
    phase_shift: float = 0.0
    scale_factor: float = 1.0
    is_on: bool = True
    start_after_periods: float | None = None
    end_after_periods: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    on_for_time: float | None = None
    on_for_periods: float | None = None
    time_steps: list[int] = tc.field(default=None, init=True)  # type: ignore
    color: tuple[float, float, float] = ORANGE
    _is_on_at_time_step_arr: jax.Array = tc.field(default=None, init=False) # type: ignore
    _time_step_to_on_idx: jax.Array = tc.field(default=None, init=False) # type: ignore
    _num_time_steps_on: int = tc.field(default=None, init=False) # type: ignore

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
                period=self.period,
                on_for_time=self.on_for_time,
                on_for_periods=self.on_for_periods,
            )
            on_list.append(cur_on)
        return on_list
    
    
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
        # calculate mapping time step -> on index
        counter = 0
        num_t = self._config.time_steps_total
        time_to_arr_idx_list = [-1 for _ in range(num_t)]
        for t in range(num_t):
            if on_list[t]:
                time_to_arr_idx_list[t] = counter
                counter += 1
        time_to_arr_idx_arr = jnp.asarray(time_to_arr_idx_list, dtype=jnp.int32)
        self = self.aset("_time_step_to_on_idx", time_to_arr_idx_arr)
        return self
    

    @abstractmethod
    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        raise NotImplementedError()

    @abstractmethod
    def update_H(
        self,
        H: jax.Array,
        inv_permeabilities: jax.Array,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        raise NotImplementedError()
    
    @abstractmethod
    def apply(
        self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
    ) -> Self:
        raise NotImplementedError()


@tc.autoinit
class DirectionalPlaneSourceBase(Source, ABC):
    direction: Literal["+", "-"] = tc.field(  # type: ignore
        init=True,
        kind="KW_ONLY",
        on_getattr=[tc.unfreeze],
        on_setattr=[tc.freeze],
    )
    
    @property
    def propagation_axis(self) -> int:
        return self.grid_shape.index(1)
    
    @property
    def horizontal_axis(self) -> int:
        return (self.propagation_axis + 1) % 3
    
    @property
    def vertical_axis(self) -> int:
        return (self.propagation_axis + 2) % 3
    
    def _get_wave_vector_raw(
        self,
    ) -> jax.Array:  # shape (3,)
        vec_list = [0, 0, 0]
        sign = 1 if self.direction == "+" else -1
        vec_list[self.propagation_axis] = sign
        return jnp.array(vec_list, dtype=jnp.float32)
    
    def _orthogonal_vector(
        self,
        v_E: jax.Array | None = None,
        v_H: jax.Array | None = None,
    ) -> jax.Array:
        if v_E is None == v_H is None:
            raise Exception(
                f"Invalid input to orthogonal vector computation: {v_E=}, {v_H=}"
            )
        wave_vector = self._get_wave_vector_raw()
        if v_E is not None:
            orthogonal = jnp.cross(wave_vector, v_E)
        elif v_H is not None:
            orthogonal = jnp.cross(v_H, wave_vector)
        else:
            raise Exception(f"This should never happen")
        return orthogonal
