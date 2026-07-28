import math
from typing import Literal, Self, Sequence

import jax
import jax.numpy as jnp
from loguru import logger

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, field, frozen_field, frozen_private_field
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.detectors.detector import Detector, DetectorState
from fdtdx.typing import SliceTuple3D

# Oversampling margin for the "auto" DFT subsampling stride: keep about this many samples per
# period of the highest recorded frequency.
_DFT_OVERSAMPLE = 12


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

    #: Subsampling stride for the phasor DFT. Only every stride-th active time step is recorded,
    #: with the kept samples rescaled to match every-step recording. If set to "auto", the stride
    #: is derived from the highest recorded frequency and the time step duration. Defaults to 1,
    #: which records every active time step.
    dft_subsample: int | Literal["auto"] = frozen_field(default=1)

    #: Concrete recording stride, resolved from dft_subsample at placement (1 = every active step).
    _dft_stride: int = frozen_private_field(default=1)

    def __post_init__(
        self,
    ):
        if self.dtype not in [jnp.complex64, jnp.complex128]:
            raise Exception(f"Invalid dtype in PhasorDetector: {self.dtype}")

    @property
    def _angular_frequencies(self) -> jax.Array:
        freqs = [wc.get_frequency() for wc in self.wave_characters]
        return 2 * jnp.pi * jnp.array(freqs)

    def _resolve_dft_stride(self) -> int:
        """Resolves dft_subsample to a concrete stride (>= 1). Requires the detector to be placed."""
        sub = self.dft_subsample
        if isinstance(sub, str):
            if sub != "auto":
                raise Exception(f"Invalid dft_subsample: {sub!r}")
            dt = float(self._config.time_step_duration)
            f_max = max((abs(float(wc.get_frequency())) for wc in self.wave_characters), default=0.0)
            if f_max <= 0.0 or dt <= 0.0:
                return 1
            return max(1, math.floor(1.0 / (_DFT_OVERSAMPLE * f_max * dt)))
        return max(1, int(sub))

    def _calculate_on_list(self) -> list[bool]:
        # Thin the base on-list to every stride-th active step; num_time_steps_recorded then
        # reflects the kept count.
        on_list = super()._calculate_on_list()
        stride = self._resolve_dft_stride()
        if stride <= 1:
            return on_list
        active = [t for t, on in enumerate(on_list) if on]
        kept = [False] * len(on_list)
        for t in active[::stride]:
            kept[t] = True
        return kept

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(grid_slice_tuple=grid_slice_tuple, config=config, key=key)
        stride = self._resolve_dft_stride()
        if stride > 1:
            # Warn about explicit strides close to the Nyquist limit ("auto" never trips this).
            dt = float(self._config.time_step_duration)
            f_max = max((abs(float(wc.get_frequency())) for wc in self.wave_characters), default=0.0)
            if stride * dt * f_max > 0.25:
                logger.warning(
                    f"Detector '{self.name}': dft_subsample={stride} leaves fewer than 4 samples per "
                    f"period of the highest recorded frequency ({f_max:.3e} Hz); the phasor may alias. "
                    'Reduce the stride or use dft_subsample="auto".'
                )
        # Store the concrete stride so update() reads a plain int (no host concretization under jit).
        self = self.aset("_dft_stride", stride, create_new_ok=True)
        return self

    def _static_scale(self) -> float | int:
        """Computes the static scale factor for the configured scaling mode.

        In continuous mode, the result is scaled by 2 / N with N the number of recorded time
        steps. In pulse mode, each kept sample is weighted by the dft_subsample stride so that
        subsampled recording matches the every-step DFT sum.

        Returns:
            float | int: Scale factor applied to each recorded sample.
        """
        if self.scaling_mode == "continuous":
            return 2 / self.num_time_steps_recorded
        if self.scaling_mode == "pulse":
            return self._dft_stride
        raise Exception(f"Invalid scaling mode: {self.scaling_mode=}")

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
        static_scale = self._static_scale()

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
        new_phasors = EH * phasors * static_scale  # Shape: (num_freqs, num_components, *grid_shape)

        if self.reduce_volume:
            # Average over spatial dimensions using physical cell volumes.
            new_phasors = self._volume_weighted_spatial_mean(new_phasors, leading_dims=2)

        if self.inverse:
            result = state["phasor"] - new_phasors[None, ...]
        else:
            result = state["phasor"] + new_phasors[None, ...]
        return {"phasor": result.astype(self.dtype)}
