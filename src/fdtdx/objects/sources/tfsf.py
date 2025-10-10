from abc import ABC, abstractmethod
from typing import Self

import jax
import numpy as np

from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.objects.sources.source import DirectionalPlaneSourceBase
from fdtdx.units.unitful import Unitful


@autoinit
class TFSFPlaneSource(DirectionalPlaneSourceBase, ABC):
    """
    Total-Field/Scattered-Field (TFSF) implementation of a source.
    The boundary between the scattered field and total field is at a
    positive offset of 0.25 in the yee grid in the axis of propagation.
    """

    azimuth_angle: float = frozen_field(default=0.0)
    elevation_angle: float = frozen_field(default=0.0)

    _E: Unitful = private_field()
    _H: Unitful = private_field()
    _time_offset_E: Unitful = private_field()
    _time_offset_H: Unitful = private_field()

    @property
    def azimuth_radians(self) -> float:
        """Convert azimuth angle from degrees to radians.

        Returns:
            float: Azimuth angle in radians.
        """
        return np.deg2rad(self.azimuth_angle).item()

    @property
    def elevation_radians(self) -> float:
        """Convert elevation angle from degrees to radians.

        Returns:
            float: Elevation angle in radians.
        """
        return np.deg2rad(self.elevation_angle).item()

    def _get_center(self) -> tuple[float, float]:
        # Calculate center position with random offset
        horizontal_size = self.grid_shape[self.horizontal_axis]
        vertical_size = self.grid_shape[self.vertical_axis]

        # account for zero-indexing in center calculation
        center_horizontal = (horizontal_size - 1) / 2
        center_vertical = (vertical_size - 1) / 2

        return (center_horizontal, center_vertical)

    @abstractmethod
    def get_EH_variation(
        self,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
    ) -> tuple[
        Unitful,  # E: (3, *grid_shape)
        Unitful,  # H: (3, *grid_shape)
        Unitful,  # time_offset_E: (3, *grid_shape)
        Unitful,  # time_offset_H: (3, *grid_shape)
    ]:
        # normal coordinates
        raise NotImplementedError()

    def apply(
        self: Self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
    ) -> Self:
        self = super().apply(
            key=key,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
        )
        E, H, time_offset_E, time_offset_H = self.get_EH_variation(
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
        )
        self = self.aset("_E", E, create_new_ok=True)
        self = self.aset("_H", H, create_new_ok=True)
        self = self.aset("_time_offset_E", time_offset_E, create_new_ok=True)
        self = self.aset("_time_offset_H", time_offset_H, create_new_ok=True)
        return self

    def update_E(
        self,
        E: Unitful,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> Unitful:
        del inv_permeabilities
        if self._E is None or self._H is None or self._time_offset_E is None or self._time_offset_H is None:
            raise Exception("Need to apply random key before calling update")

        delta_t = self._config.time_step_duration
        inv_permittivity_slice = inv_permittivities[*self.grid_slice]

        # Calculate time points for E and H fields
        time_H_h = (delta_t * time_step) + self._time_offset_H[self.horizontal_axis]
        time_H_v = (delta_t * time_step) + self._time_offset_H[self.vertical_axis]

        # Get temporal amplitudes from profile
        amplitude_H_h = self.temporal_profile.get_amplitude(
            time=time_H_h,
            period=self.wave_character.get_period(),
            phase_shift=self.wave_character.phase_shift,
        )
        amplitude_H_h = amplitude_H_h * self.static_amplitude_factor
        amplitude_H_v = self.temporal_profile.get_amplitude(
            time=time_H_v,
            period=self.wave_character.get_period(),
            phase_shift=self.wave_character.phase_shift,
        )
        amplitude_H_v = amplitude_H_v * self.static_amplitude_factor

        # vertical incident wave part
        H_v_inc = self._H[self.vertical_axis] * amplitude_H_v
        H_v_inc = H_v_inc * self._config.courant_number * inv_permittivity_slice
        H_v_inc = jax.lax.stop_gradient(H_v_inc)

        # horizontal incident wave part
        H_h_inc = self._H[self.horizontal_axis] * amplitude_H_h
        H_h_inc = H_h_inc * self._config.courant_number * inv_permittivity_slice
        H_h_inc = jax.lax.stop_gradient(H_h_inc)

        # if direction is negative, updates are reversed
        sign = 1 if self.direction == "+" else -1

        # inverse update is inverted
        if inverse:
            sign = -sign

        # update uses -H_v, we have to subtract update, resulting in +H_v
        E = E.at[self.horizontal_axis, *self.grid_slice].add(sign * H_v_inc)
        # update uses +H_h, we have to subtract update, resulting in -H_h
        E = E.at[self.vertical_axis, *self.grid_slice].add(-sign * H_h_inc)

        return E

    def update_H(
        self,
        H: Unitful,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> Unitful:
        del inv_permittivities
        if self._E is None or self._H is None or self._time_offset_E is None or self._time_offset_H is None:
            raise Exception("Need to apply random key before calling update")

        delta_t = self._config.time_step_duration
        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            inv_permeability_slice = inv_permeabilities[*self.grid_slice]
        else:
            inv_permeability_slice = inv_permeabilities

        # Calculate time points for E and H fields
        time_E_h = (delta_t * time_step) + self._time_offset_E[self.horizontal_axis]
        time_E_v = (delta_t * time_step) + self._time_offset_E[self.vertical_axis]

        # Get temporal amplitudes from profile
        amplitude_E_h = self.temporal_profile.get_amplitude(
            time=time_E_h,
            period=self.wave_character.get_period(),
            phase_shift=self.wave_character.phase_shift,
        )
        amplitude_E_h = amplitude_E_h * self.static_amplitude_factor
        amplitude_E_v = self.temporal_profile.get_amplitude(
            time=time_E_v,
            period=self.wave_character.get_period(),
            phase_shift=self.wave_character.phase_shift,
        )
        amplitude_E_v = amplitude_E_v * self.static_amplitude_factor

        # horizontal incident wave part
        E_h_inc = self._E[self.horizontal_axis] * amplitude_E_h
        E_h_inc = E_h_inc * self._config.courant_number * inv_permeability_slice
        E_h_inc = jax.lax.stop_gradient(E_h_inc)

        # vertical incident wave part
        E_v_inc = self._E[self.vertical_axis] * amplitude_E_v
        E_v_inc = E_v_inc * self._config.courant_number * inv_permeability_slice
        E_v_inc = jax.lax.stop_gradient(E_v_inc)

        # if direction is negative, updates are reversed
        sign = 1 if self.direction == "+" else -1

        # inverse update is inverted
        if inverse:
            sign = -sign

        # update used +E_h, we have to add update, resulting in +E_h
        H = H.at[self.vertical_axis, *self.grid_slice].add(sign * E_h_inc)
        # update used -E_v, we have to add update, resulting in -E_v
        H = H.at[self.horizontal_axis, *self.grid_slice].add(-sign * E_v_inc)

        return H
