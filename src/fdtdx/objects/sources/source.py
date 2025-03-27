from abc import ABC, abstractmethod
from typing import Literal, Self

import jax
import jax.numpy as jnp

from fdtdx.core import WaveCharacter
from fdtdx.core.jax.pytrees import extended_autoinit, frozen_field
from fdtdx.core.linalg import get_orthogonal_vector, get_wave_vector_raw
from fdtdx.core.misc import normalize_polarization_for_source
from fdtdx.core.plotting.colors import ORANGE
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.sources.profile import SingleFrequencyProfile, TemporalProfile


@extended_autoinit
class Source(SimulationObject, ABC):
    wave_character: WaveCharacter = frozen_field(kind="KW_ONLY")  # type: ignore
    temporal_profile: TemporalProfile = SingleFrequencyProfile()
    amplitude_scale: float = 1.0
    is_on: bool = True
    color: tuple[float, float, float] = ORANGE

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
    amplitude: float = 1.0
    fixed_E_polarization_vector: tuple[float, float, float] | None = None
    fixed_H_polarization_vector: tuple[float, float, float] | None = None

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