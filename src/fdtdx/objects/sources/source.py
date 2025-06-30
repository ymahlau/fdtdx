from abc import ABC, abstractmethod
from typing import Literal, Self

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.misc import linear_interpolated_indexing, normalize_polarization_for_source
from fdtdx.core.plotting.colors import ORANGE
from fdtdx.core.switch import OnOffSwitch
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.sources.profile import SingleFrequencyProfile, TemporalProfile
from fdtdx.typing import SliceTuple3D


@autoinit
class Source(SimulationObject, ABC):
    wave_character: WaveCharacter = frozen_field()
    temporal_profile: TemporalProfile = SingleFrequencyProfile()
    static_amplitude_factor: float = frozen_field(default=1.0)
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
        self = self.aset("_is_on_at_time_step_arr", on_arr, create_new_ok=True)
        # calculate mapping time step -> on index
        time_to_arr_idx_list = self.switch.calculate_time_step_to_on_arr_idx(
            time_step_duration=self._config.time_step_duration,
            num_total_time_steps=self._config.time_steps_total,
        )
        time_to_arr_idx_arr = jnp.asarray(time_to_arr_idx_list, dtype=jnp.int32)
        self = self.aset("_time_step_to_on_idx", time_to_arr_idx_arr, create_new_ok=True)
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
            E (jax.Array): Current electric field array.
            inv_permittivities (jax.Array): Inverse permittivity values.
            inv_permeabilities (jax.Array | float): Inverse permeability values.
            time_step (jax.Array): Current simulation time step.
            inverse (bool): Whether to perform inverse update for backpropagation.

        Returns:
            jax.Array: Updated electric field array.
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
            H (jax.Array): Current magnetic field array.
            inv_permittivities (jax.Array): Inverse permittivity values.
            inv_permeabilities (jax.Array | float): Inverse permeability values.
            time_step (jax.Array): Current simulation time step.
            inverse (bool): Whether to perform inverse update for backpropagation.

        Returns:
            jax.Array: Updated magnetic field array.
        """
        raise NotImplementedError()


@autoinit
class DirectionalPlaneSourceBase(Source, ABC):
    """Base class for directional plane wave sources.

    Implements common functionality for plane wave sources that propagate in a specific
    direction. Provides methods for calculating wave vectors and orthogonal field components.

    Attributes:
        direction (Literal["+", "-"]): Direction of propagation ('+' or '-' along propagation axis).
    """

    direction: Literal["+", "-"] = frozen_field()

    @property
    def propagation_axis(self) -> int:
        return self.grid_shape.index(1)

    @property
    def horizontal_axis(self) -> int:
        return (self.propagation_axis + 1) % 3

    @property
    def vertical_axis(self) -> int:
        return (self.propagation_axis + 2) % 3


@autoinit
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
        magnitude = magnitude * self.static_amplitude_factor
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
        magnitude = magnitude * self.static_amplitude_factor
        _, h_pol = normalize_polarization_for_source(
            direction=self.direction,
            propagation_axis=self.propagation_axis,
            fixed_E_polarization_vector=self.fixed_E_polarization_vector,
            fixed_H_polarization_vector=self.fixed_E_polarization_vector,
        )
        H_update = h_pol[:, None, None, None] * magnitude

        H = H.at[:, *self.grid_slice].set(H_update.astype(H.dtype))
        return H
