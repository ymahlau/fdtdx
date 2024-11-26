import jax
import pytreeclass as tc

from fdtdx.core.physics.metrics import compute_energy
from fdtdx.objects.detectors.detector import Detector, DetectorState


@tc.autoinit
class EnergyDetector(Detector):
    as_slices: bool = False
    reduce_volume: bool = False
    
    def _shape_dtype_single_time_step(
        self,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        if self.as_slices and self.reduce_volume:
            raise Exception(f"Cannot both reduce volume and save mean slices!")
        if self.as_slices:
            gs = self.grid_shape
            return {
                "XY Plane": jax.ShapeDtypeStruct(
                    (gs[0], gs[1]),
                    self.dtype
                ),
                "XZ Plane": jax.ShapeDtypeStruct(
                    (gs[0], gs[2]),
                    self.dtype
                ),
                "YZ Plane": jax.ShapeDtypeStruct(
                    (gs[1], gs[2]),
                    self.dtype
                ),
            }
        if self.reduce_volume:
            return {
                "energy": jax.ShapeDtypeStruct(
                    (1,),
                    self.dtype
                )
            }
        return {
            "energy": jax.ShapeDtypeStruct(
                self.grid_shape,
                self.dtype
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
        cur_E = E[:, *self.grid_slice]
        cur_H = H[:, *self.grid_slice]
        cur_inv_permittivity = inv_permittivity[self.grid_slice]
        cur_inv_permeability = inv_permeability[self.grid_slice]
        energy = compute_energy(
            E=cur_E,
            H=cur_H,
            inv_permittivity=cur_inv_permittivity,
            inv_permeability=cur_inv_permeability,
        )
        arr_idx = self._time_step_to_arr_idx[time_step]
        if self.as_slices:
            energy_xy = energy.mean(axis=2)
            new_xy = state["XY Plane"].at[arr_idx].set(energy_xy)
            energy_xz = energy.mean(axis=1)
            new_xz = state["XZ Plane"].at[arr_idx].set(energy_xz)
            energy_yz = energy.mean(axis=0)
            new_yz = state["YZ Plane"].at[arr_idx].set(energy_yz)
            return {
                "XY Plane": new_xy,
                "XZ Plane": new_xz,
                "YZ Plane": new_yz,
            }
        if self.reduce_volume:
            energy = energy.sum()        
        new_full_arr = state['energy'].at[arr_idx].set(energy)
        new_state = {
            'energy': new_full_arr
        }
        return new_state
    
