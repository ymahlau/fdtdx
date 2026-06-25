from typing import Literal, Self, Sequence

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, field, frozen_field, frozen_private_field, private_field
from fdtdx.core.temporal.profile import TemporalProfile
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.detectors.detector import Detector, DetectorState
from fdtdx.typing import SliceTuple3D


@autoinit
class PhasorDetector(Detector):
    """Detector for measuring frequency components of electromagnetic fields using an efficient Phasor Implementation.

    This detector computes complex phasor representations of the field components at specified
    frequencies, enabling frequency-domain analysis of the electromagnetic fields.
    The amplitude and phase of the original phase can be reconstructed using jnp.abs(phasor) and jnp.angle(phasor).
    The reconstruction itself can then be achieved using amplitude * jnp.cos(2 * jnp.pi * freq * t + phase).
    """

    #: WaveCharacters to analyze.
    wave_characters: Sequence[WaveCharacter] = field()

    #: If True, reduces the volume of recorded data. Defaults to False.
    reduce_volume: bool = frozen_field(default=False)

    #: Sequence of field components to measure.
    #: Can include any of: "Ex", "Ey", "Ez", "Hx", "Hy", "Hz".
    components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] = frozen_field(
        default=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
    )
    dtype: jnp.dtype = frozen_field(
        default=jnp.complex64,
    )

    #: Whether to plot the measured data. Defaults to False.
    plot: bool = frozen_field(default=False)

    #: Scaling of the resulting phasor. In continuous mode, the result is scaled by a factor of 2 / N, where N is
    #: the number of time steps recorded. This allows accurate reconstruction of a continuous signal.
    #: In pulse mode, the result is not scaled.
    scaling_mode: Literal["continuous", "pulse"] = frozen_field(default="continuous")

    #: Optional smooth temporal **apodization** window applied before the DFT, given as a
    #: :class:`~fdtdx.core.temporal.profile.TemporalProfile` (e.g. ``TukeyWindowProfile`` /
    #: ``GaussianWindowProfile``, evaluated as a carrier-free envelope). ``None`` (default) is
    #: the historical hard rectangular gate. The continuous-mode amplitude scale is corrected
    #: by the window's coherent gain (``2 / sum(w)``) so reconstructed amplitudes stay correct.
    apodization: TemporalProfile | None = frozen_field(default=None)

    #: Per-time-step window weights (on-mask times apodization), length ``time_steps_total``.
    _window_at_time_step_arr: jax.Array = private_field()
    #: Sum of the window weights over the recorded steps (coherent gain times N).
    _window_sum: float = frozen_private_field()

    def __post_init__(
        self,
    ):
        if self.dtype not in [jnp.complex64, jnp.complex128]:
            raise Exception(f"Invalid dtype in PhasorDetector: {self.dtype}")

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(grid_slice_tuple=grid_slice_tuple, config=config, key=key)
        on_arr = self._is_on_at_time_step_arr  # bool, (time_steps_total,)
        if self.apodization is not None:
            num_steps = self._config.time_steps_total
            wc = self.wave_characters[0]
            time = jnp.arange(num_steps) * self._config.time_step_duration
            window = jnp.real(self.apodization.get_amplitude(time, wc.get_period(), wc.phase_shift))
            window = window * on_arr.astype(window.dtype)
        else:
            window = on_arr.astype(jnp.float32)
        self = self.aset("_window_at_time_step_arr", window, create_new_ok=True)
        self = self.aset("_window_sum", float(jnp.sum(window)), create_new_ok=True)
        return self

    @property
    def _angular_frequencies(self) -> jax.Array:
        freqs = [wc.get_frequency() for wc in self.wave_characters]
        return 2 * jnp.pi * jnp.array(freqs)

    def _num_latent_time_steps(self) -> int:
        return 1

    def _shape_dtype_single_time_step(
        self,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        field_dtype = jnp.complex128 if self.dtype == jnp.complex128 else jnp.complex64
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
        if self.scaling_mode == "continuous":
            # Coherent-gain correction: sum(w) == num_time_steps_recorded when no apodization.
            static_scale = 2 / self._window_sum
        elif self.scaling_mode == "pulse":
            static_scale = 1
        else:
            raise Exception(f"Invalid scaling mode: {self.scaling_mode=}")

        window_weight = self._window_at_time_step_arr[time_step]

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
        phase_angles = self._angular_frequencies * time_passed  # Shape: (num_freqs,)
        phasors = jnp.exp(1j * phase_angles)  # Shape: (num_freqs,)
        # Reshape phasors to (num_freqs, 1, 1, 1, 1) for proper broadcasting with EH (num_components, x, y, z)
        phasors = phasors.reshape((len(self._angular_frequencies),) + (1,) * EH.ndim)
        new_phasors = EH * phasors * static_scale * window_weight  # (num_freqs, num_components, *grid_shape)

        if self.reduce_volume:
            # Average over spatial dimensions using physical cell volumes.
            new_phasors = self._volume_weighted_spatial_mean(new_phasors, leading_dims=2)

        if self.inverse:
            result = state["phasor"] - new_phasors[None, ...]
        else:
            result = state["phasor"] + new_phasors[None, ...]
        return {"phasor": result.astype(self.dtype)}
