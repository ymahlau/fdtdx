from typing import Literal, Sequence

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, field, frozen_field
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.detectors.detector import Detector, DetectorState


@autoinit
class PhasorDetector(Detector):
    """Detector for measuring frequency components of electromagnetic fields using an efficient Phasor Implementation.

    This detector computes complex phasor representations of the field components at specified
    frequencies, enabling frequency-domain analysis of the electromagnetic fields.
    The amplitude and phase of the original phase can be reconstructed using jnp.abs(phasor) and jnp.angle(phasor).
    The reconstruction itself can then be achieved using amplitude * jnp.cos(2 * jnp.pi * freq * t + phase).

    Attributes:
        wave_characters (Sequence[WaveCharacter]): WaveCharacters to analyze.
        reduce_volume (bool, optional): If True, reduces the volume of recorded data. Defaults to False.
        components (Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]]): Sequence of field components to measure.
            Can include any of: "Ex", "Ey", "Ez", "Hx", "Hy", "Hz".
        dtype (jnp.dtype, optional): data type of the saved fields. Defaults to jnp.complex64
        plot (bool, optional): Wether to plot the measured data. Defaults to False.
    """

    wave_characters: Sequence[WaveCharacter] = field()
    reduce_volume: bool = frozen_field(default=False)
    components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] = frozen_field(
        default=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
    )
    dtype: jnp.dtype = frozen_field(
        default=jnp.complex64,
    )
    plot: bool = frozen_field(default=False)

    def __post_init__(
        self,
    ):
        if self.dtype not in [jnp.complex64, jnp.complex128]:
            raise Exception(f"Invalid dtype in PhasorDetector: {self.dtype}")

    @property
    def _angular_frequencies(self) -> jax.Array:
        freqs = [wc.frequency for wc in self.wave_characters]
        return 2 * jnp.pi * jnp.array(freqs)

    def _num_latent_time_steps(self) -> int:
        return 1

    def _shape_dtype_single_time_step(
        self,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        field_dtype = jnp.complex128 if self.dtype == jnp.float64 else jnp.complex64
        num_components = len(self.components)
        num_frequencies = len(self._angular_frequencies)
        grid_shape = self.grid_shape if not self.reduce_volume else tuple([])
        phasor_shape = (num_frequencies, num_components, *grid_shape)
        return {"phasor": jax.ShapeDtypeStruct(shape=phasor_shape, dtype=field_dtype)}

    def update(
        self,
        time_step: jax.Array,
        E: jax.Array,
        H: jax.Array,
        state: DetectorState,
        inv_permittivity: jax.Array,
        inv_permeability: jax.Array | float,
    ) -> DetectorState:
        del inv_permeability, inv_permittivity
        time_passed = time_step * self._config.time_step_duration
        static_scale = 2 / self.num_time_steps_recorded

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
        new_phasors = EH[None, ...] * phasors[..., None] * static_scale  # Broadcasting handles the multiplication

        if self.reduce_volume:
            # Average over all spatial dimensions
            spatial_axes = tuple(range(2, new_phasors.ndim))  # Skip freq and component axes
            new_phasors = new_phasors.mean(axis=spatial_axes) if spatial_axes else new_phasors

        if self.inverse:
            result = state["phasor"] - new_phasors[None, ...]
        else:
            result = state["phasor"] + new_phasors[None, ...]
        return {"phasor": result.astype(self.dtype)}
