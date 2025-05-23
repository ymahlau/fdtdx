from abc import ABC, abstractmethod
from typing import Literal, Self

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core import WaveCharacter
from fdtdx.core.jax.pytrees import extended_autoinit, field, frozen_field
from fdtdx.core.misc import linear_interpolated_indexing, normalize_polarization_for_source
from fdtdx.core.plotting.colors import ORANGE
from fdtdx.core.switch import OnOffSwitch
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.sources.profile import SingleFrequencyProfile, TemporalProfile
from fdtdx.typing import SliceTuple3D


@extended_autoinit
class Source(SimulationObject, ABC):
    wave_character: WaveCharacter = frozen_field()
    temporal_profile: TemporalProfile = SingleFrequencyProfile()
    amplitude_scale: float = frozen_field(default=1.0)
    switch: OnOffSwitch = frozen_field(default=OnOffSwitch())
    color: tuple[float, float, float] | None = frozen_field(default=ORANGE)
    
    _is_on_at_time_step_arr: jax.Array = private_field()
    _time_step_to_on_idx: jax.Array = private_field()

    def is_on_at_time_step(self, time_step: jax.Array) -> jax.Array:
        return self._is_on_at_time_step_arr[time_step]

    def adjust_time_step_by_on_off(self, time_step: jax.Array) -> jax.Array:
        time_step = linear_interpolated_indexing(
            point=time_step.reshape(1),
            arr=self._time_step_to_on_idx,
        )
        return time_step

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        """Place the source on the simulation grid and initialize timing arrays.

        Args:
            grid_slice_tuple: Tuple of slices defining source's position on grid.
            config: Simulation configuration parameters.
            key: JAX random key for stochastic operations.

        Returns:
            Self with initialized grid position and timing arrays.
        """
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )
        # determine number of time steps on
        on_list = self.switch.calculate_on_list(
            time_step_duration=self._config.time_step_duration,
            num_total_time_steps=self._config.time_steps_total,
        )
        on_arr = jnp.asarray(on_list, dtype=jnp.bool)
        self = self.aset("_is_on_at_time_step_arr", on_arr)
        # calculate mapping time step -> on index
        time_to_arr_idx_list = self.switch.calculate_time_step_to_on_arr_idx(
            time_step_duration=self._config.time_step_duration,
            num_total_time_steps=self._config.time_steps_total,
        )
        time_to_arr_idx_arr = jnp.asarray(time_to_arr_idx_list, dtype=jnp.int32)
        self = self.aset("_time_step_to_on_idx", time_to_arr_idx_arr)
        return self

    @abstractmethod
    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        """Update the electric field component.

        Args:
            E: Current electric field array.
            inv_permittivities: Inverse permittivity values.
            inv_permeabilities: Inverse permeability values.
            time_step: Current simulation time step.
            inverse: Whether to perform inverse update for backpropagation.

        Returns:
            Updated electric field array.
        """
        raise NotImplementedError()

    @abstractmethod
    def update_H(
        self,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        """Update the magnetic field component.

        Args:
            H: Current magnetic field array.
            inv_permittivities: Inverse permittivity values.
            inv_permeabilities: Inverse permeability values.
            time_step: Current simulation time step.
            inverse: Whether to perform inverse update for backpropagation.

        Returns:
            Updated magnetic field array.
        """
        raise NotImplementedError()

    @abstractmethod
    def apply(
        self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
    ) -> Self:
        """Apply source-specific initialization and setup.

        Args:
            key: JAX random key for stochastic operations.
            inv_permittivities: Inverse permittivity values.
            inv_permeabilities: Inverse permeability values.

        Returns:
            Initialized source instance.
        """
        raise NotImplementedError()


@extended_autoinit
class DirectionalPlaneSourceBase(Source, ABC):
    """Base class for directional plane wave sources.

    Implements common functionality for plane wave sources that propagate in a specific
    direction. Provides methods for calculating wave vectors and orthogonal field components.

    Attributes:
        direction: Direction of propagation ('+' or '-' along propagation axis).
    """

    direction: Literal["+", "-"] = frozen_field(kind="KW_ONLY")  # type: ignore

    @property
    def propagation_axis(self) -> int:
        return self.grid_shape.index(1)

    @property
    def horizontal_axis(self) -> int:
        return (self.propagation_axis + 1) % 3

    @property
    def vertical_axis(self) -> int:
        return (self.propagation_axis + 2) % 3


@extended_autoinit
class HardConstantAmplitudePlanceSource(DirectionalPlaneSourceBase):
    amplitude: float = frozen_field(default=1.0)
    fixed_E_polarization_vector: tuple[float, float, float] | None = frozen_field(default=None)
    fixed_H_polarization_vector: tuple[float, float, float] | None = frozen_field(default=None)

    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permittivities, inv_permeabilities
        if inverse:
            return E
        delta_t = self._config.time_step_duration
        time_phase = 2 * jnp.pi * time_step * delta_t / self.wave_character.period + self.wave_character.phase_shift
        magnitude = jnp.real(self.amplitude * jnp.exp(-1j * time_phase))
        e_pol, _ = normalize_polarization_for_source(
            direction=self.direction,
            propagation_axis=self.propagation_axis,
            fixed_E_polarization_vector=self.fixed_E_polarization_vector,
            fixed_H_polarization_vector=self.fixed_E_polarization_vector,
        )
        E_update = e_pol[:, None, None, None] * magnitude

        E = E.at[:, *self.grid_slice].set(E_update.astype(E.dtype))
        return E

    def update_H(
        self,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ):
        del inv_permeabilities, inv_permittivities
        if inverse:
            return H
        delta_t = self._config.time_step_duration
        time_phase = 2 * jnp.pi * time_step * delta_t / self.wave_character.period + self.wave_character.phase_shift
        magnitude = jnp.real(self.amplitude * jnp.exp(-1j * time_phase))
        _, h_pol = normalize_polarization_for_source(
            direction=self.direction,
            propagation_axis=self.propagation_axis,
            fixed_E_polarization_vector=self.fixed_E_polarization_vector,
            fixed_H_polarization_vector=self.fixed_E_polarization_vector,
        )
        H_update = h_pol[:, None, None, None] * magnitude

        H = H.at[:, *self.grid_slice].set(H_update.astype(H.dtype))
        return H

    def apply(
        self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
    ) -> Self:
        del key, inv_permittivities, inv_permeabilities
        return self
