from typing import Literal, Sequence
import jax
import jax.numpy as jnp
from fdtdx.core.physics import constants
from fdtdx.core.jax.pytrees import extended_autoinit, frozen_field
from fdtdx.objects.detectors.detector import Detector, DetectorState

@extended_autoinit
class PhasorDetector(Detector):
    as_slices: bool = False
    reduce_volume: bool = False
    wavelength: float | None = None
    components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] = frozen_field(
        default=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
    )
    
    @property
    def frequency(self):
        if self.period_length is None and self.wavelength is None:
            raise Exception(f"Specify either wavelength or period_length for PhasorDetector")
        p = self.period_length
        if p is None:
            if self.wavelength is None:
                raise Exception(f"this should never happen")
            p = self.wavelength / constants.c
        return 1 / p
    
    def _num_latent_time_steps(self) -> int:
        return 1
    
    def _shape_dtype_single_time_step(
        self,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        field_dtype = jnp.complex128 if self.dtype == jnp.float64 else jnp.complex64
        num_components = len(self.components)
        phasor_shape = (num_components, *self.grid_shape)
        return {
            "phasor": jax.ShapeDtypeStruct(shape=phasor_shape, dtype=field_dtype)
        }
    
    def update(
        self,
        time_step: jax.Array,
        E: jax.Array,
        H: jax.Array,
        state: DetectorState,
        inv_permittivity: jax.Array,
        inv_permeability: jax.Array,
    ) -> DetectorState:
        
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
        return {"phasor": result}