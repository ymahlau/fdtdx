from abc import ABC, abstractmethod
from typing import Literal, Self

import jax
import jax.numpy as jnp
import pytreeclass as tc

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import extended_autoinit, frozen_field
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.wavelength import WaveCharacter
from fdtdx.typing import SliceTuple3D
from fdtdx.core.plotting.colors import ORANGE
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
        inv_permeabilities: jax.Array,
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
        inv_permeabilities: jax.Array,
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
        inv_permeabilities: jax.Array,
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

    def _get_wave_vector_raw(
        self,
    ) -> jax.Array:  # shape (3,)
        """Calculate the raw wave vector for the plane wave.

        Returns:
            3D array representing wave vector direction (normalized unit vector).
        """
        vec_list = [0, 0, 0]
        sign = 1 if self.direction == "+" else -1
        vec_list[self.propagation_axis] = sign
        return jnp.array(vec_list, dtype=jnp.float32)

    def _orthogonal_vector(
        self,
        v_E: jax.Array | None = None,
        v_H: jax.Array | None = None,
    ) -> jax.Array:
        """Calculate vector orthogonal to wave vector and given E or H field vector.

        Args:
            v_E: Electric field vector (optional).
            v_H: Magnetic field vector (optional).

        Returns:
            Orthogonal vector computed via cross product.

        Raises:
            Exception: If neither or both v_E and v_H are provided.
        """
        if v_E is None == v_H is None:
            raise Exception(f"Invalid input to orthogonal vector computation: {v_E=}, {v_H=}")
        wave_vector = self._get_wave_vector_raw()
        if v_E is not None:
            orthogonal = jnp.cross(wave_vector, v_E)
        elif v_H is not None:
            orthogonal = jnp.cross(v_H, wave_vector)
        else:
            raise Exception("This should never happen")
        return orthogonal
