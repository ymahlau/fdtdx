from typing import Literal, Sequence

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import extended_autoinit, frozen_field
from fdtdx.core.physics import constants
from fdtdx.objects.detectors.detector import Detector, DetectorState


@extended_autoinit
class PhasorDetector(Detector):
    """Detector for measuring phasor components of electromagnetic fields.

    This detector computes complex phasor representations of the field components,
    enabling frequency-domain analysis of the electromagnetic fields.

    Attributes:
        as_slices: If True, returns results as slices rather than full volume.
        reduce_volume: If True, reduces the volume of recorded data.
        wavelength: Wavelength of the phasor analysis in meters. Either this or
            period_length must be specified.
        components: Sequence of field components to measure. Can include any of:
            "Ex", "Ey", "Ez", "Hx", "Hy", "Hz".
    """

    as_slices: bool = False
    reduce_volume: bool = False
    wavelength: float | None = None
    components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] = frozen_field(
        default=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
    )
    dtype: jnp.dtype = frozen_field(
        default=jnp.complex64,
        kind="KW_ONLY",
    )

    def __post_init__(
        self,
    ):
        if self.dtype not in [jnp.complex64, jnp.complex128]:
            raise Exception(f"Invalid dtype in PhasorDetector: {self.dtype}")

    @property
    def frequency(self) -> float:
        """Calculate the frequency for phasor analysis.

        Returns:
            float: The frequency in Hz calculated from either wavelength or period_length.

        Raises:
            Exception: If neither wavelength nor period_length is specified.
        """
        if self.period_length is None and self.wavelength is None:
            raise Exception("Specify either wavelength or period_length for PhasorDetector")
        p = self.period_length
        if p is None:
            if self.wavelength is None:
                raise Exception("this should never happen")
            p = self.wavelength / constants.c
        return 1 / p

    def _num_latent_time_steps(self) -> int:
        """Get number of time steps needed for latent computation.

        Returns:
            int: Always returns 1 for phasor detector since only current state is needed.
        """
        return 1

    def _shape_dtype_single_time_step(
        self,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        """Define shape and dtype for a single time step of phasor data.

        Returns:
            dict: Dictionary with 'phasor' key mapping to a ShapeDtypeStruct containing:
                - shape: (num_components, *grid_shape)
                - dtype: Complex64 or Complex128 depending on detector's dtype
        """
        field_dtype = jnp.complex128 if self.dtype == jnp.float64 else jnp.complex64
        num_components = len(self.components)
        phasor_shape = (num_components, *self.grid_shape)
        return {"phasor": jax.ShapeDtypeStruct(shape=phasor_shape, dtype=field_dtype)}

    def update(
        self,
        time_step: jax.Array,
        E: jax.Array,
        H: jax.Array,
        state: DetectorState,
        inv_permittivity: jax.Array,
        inv_permeability: jax.Array,
    ) -> DetectorState:
        """Update the phasor state with current field values.

        Computes the phasor representation by multiplying field components with a complex
        exponential at the detector's frequency.

        Args:
            time_step: Current simulation time step
            E: Electric field array
            H: Magnetic field array
            state: Current detector state
            inv_permittivity: Inverse permittivity array (unused)
            inv_permeability: Inverse permeability array (unused)

        Returns:
            DetectorState: Updated state containing new phasor values
        """
        del inv_permeability, inv_permittivity
        delta_t = self._config.time_step_duration
        scale = delta_t / jnp.sqrt(2 * jnp.pi)
        angular_frequency = 2 * jnp.pi * self.frequency
        time_passed = time_step * delta_t
        phase_angle = angular_frequency * time_passed
        phasor = jnp.exp(1j * phase_angle)

        E, H = E[:, *self.grid_slice], H[:, *self.grid_slice]
        fields = []
        if "Ex" in self.components:
            fields.append(E[0])
        if "Ey" in self.components:
            fields.append(E[1])
        if "Ez" in self.components:
            fields.append(E[2])
        if "Hx" in self.components:
            fields.append(H[0])
        if "Hy" in self.components:
            fields.append(H[1])
        if "Hz" in self.components:
            fields.append(H[2])

        EH = jnp.stack(fields, axis=0)

        new_phasor = EH * phasor * scale

        if self.inverse:
            result = state["phasor"] - new_phasor[None, ...]
        else:
            result = state["phasor"] + new_phasor[None, ...]
        return {"phasor": result.astype(self.dtype)}
