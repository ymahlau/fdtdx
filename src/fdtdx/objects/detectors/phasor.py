from typing import Literal, Sequence

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import extended_autoinit, frozen_field
from fdtdx.objects.detectors.detector import Detector, DetectorState


@extended_autoinit
class PhasorDetector(Detector):
    """Detector for measuring phasor components of electromagnetic fields.

    This detector computes complex phasor representations of the field components at specified
    frequencies, enabling frequency-domain analysis of the electromagnetic fields.

    Attributes:
        frequencies: Sequence of frequencies to analyze (in Hz)
        as_slices: If True, returns results as slices rather than full volume.
        reduce_volume: If True, reduces the volume of recorded data.
        components: Sequence of field components to measure. Can include any of:
            "Ex", "Ey", "Ez", "Hx", "Hy", "Hz".
    """

    frequencies: Sequence[float] = (None,)
    as_slices: bool = False
    reduce_volume: bool = False
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

        # Precompute angular frequencies for vectorization
        self._angular_frequencies = 2 * jnp.pi * jnp.array(self.frequencies)
        self._scale = self._config.time_step_duration / jnp.sqrt(2 * jnp.pi)

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
                - shape: (num_frequencies, num_components, *grid_shape)
                - dtype: Complex64 or Complex128 depending on detector's dtype
        """
        field_dtype = jnp.complex128 if self.dtype == jnp.float64 else jnp.complex64
        num_components = len(self.components)
        num_frequencies = len(self.frequencies)
        phasor_shape = (num_frequencies, num_components, *self.grid_shape)
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

        Computes the phasor representation by multiplying field components with complex
        exponentials at each of the detector's frequencies.

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
        time_passed = time_step * self._config.time_step_duration

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

        # Vectorized phasor calculation for all frequencies
        phase_angles = self._angular_frequencies[:, None] * time_passed  # Shape: (num_freqs, 1)
        phasors = jnp.exp(1j * phase_angles)  # Shape: (num_freqs, 1)
        new_phasors = EH[None, ...] * phasors[..., None] * self._scale  # Broadcasting handles the multiplication

        if self.reduce_volume:
            # Average over all spatial dimensions
            spatial_axes = tuple(range(2, new_phasors.ndim))  # Skip freq and component axes
            new_phasors = new_phasors.mean(axis=spatial_axes) if spatial_axes else new_phasors

        if self.inverse:
            result = state["phasor"] - new_phasors[None, ...]
        else:
            result = state["phasor"] + new_phasors[None, ...]
        return {"phasor": result.astype(self.dtype)}
