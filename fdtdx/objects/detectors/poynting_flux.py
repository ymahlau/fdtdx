from typing import Literal
import jax

from fdtdx.core.physics.metrics import poynting_flux
import pytreeclass as tc

from fdtdx.objects.detectors.detector import Detector, DetectorState


@tc.autoinit
class PoyntingFluxDetector(Detector):
    direction: Literal["+", "-"] = tc.field( # type: ignore
        init=True, 
        kind="KW_ONLY",
        on_getattr=[tc.unfreeze],
        on_setattr=[tc.freeze],
    )
    reduce_volume: bool = True
    
    @property
    def propagation_axis(self) -> int:
        if sum([a == 1 for a in self.grid_shape]) != 1:
            raise Exception(f"Invalid poynting flux detector shape: {self.grid_shape}")
        return self.grid_shape.index(1)
    
    def _shape_dtype_single_time_step(
        self,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        if self.reduce_volume:
            return {"poynting_flux": jax.ShapeDtypeStruct(
                (1,), self.dtype
            )}
        return {
            "poynting_flux": jax.ShapeDtypeStruct(
                self.grid_shape, self.dtype
            )
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
        cur_E = E[:, *self.grid_slice]
        cur_H = H[:, *self.grid_slice]
        
        pf = poynting_flux(
            cur_E, cur_H
        )[self.propagation_axis]
        if self.direction == "-":
            pf = -pf
        if self.reduce_volume:
            pf = pf.sum()
        arr_idx = self._time_step_to_arr_idx[time_step]
        new_full_arr = state['poynting_flux'].at[arr_idx].set(pf)
        new_state = {
            'poynting_flux': new_full_arr
        }
        return new_state




