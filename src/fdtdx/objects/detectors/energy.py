import jax
import pytreeclass as tc

from fdtdx.core.physics.metrics import compute_energy
from fdtdx.objects.detectors.detector import Detector, DetectorState


@tc.autoinit
class EnergyDetector(Detector):
    """Detector for measuring electromagnetic energy distribution.

    This detector computes and records the electromagnetic energy density at specified
    points in the simulation volume. It can operate in different modes to either record
    full 3D data, 2D slices, or reduced volume measurements.

    Attributes:
        as_slices: If True, returns energy measurements as 2D slices through the volume
            center. If False, returns full 3D volume or reduced measurements.
        reduce_volume: If True, reduces the volume data to a single energy value by
            summing. If False, maintains spatial distribution of energy.
    """

    as_slices: bool = False
    reduce_volume: bool = False

    def _shape_dtype_single_time_step(
        self,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        """Define shape and dtype for a single time step of energy data.

        Returns:
            dict: Dictionary mapping data keys to ShapeDtypeStruct containing shape and
                dtype information. Keys depend on detector mode:
                - If as_slices: Returns XY/XZ/YZ plane data as 2D arrays
                - If reduce_volume: Returns single energy value
                - Otherwise: Returns full 3D energy distribution

        Raises:
            Exception: If both as_slices and reduce_volume are True
        """
        if self.as_slices and self.reduce_volume:
            raise Exception("Cannot both reduce volume and save mean slices!")
        if self.as_slices:
            gs = self.grid_shape
            return {
                "XY Plane": jax.ShapeDtypeStruct((gs[0], gs[1]), self.dtype),
                "XZ Plane": jax.ShapeDtypeStruct((gs[0], gs[2]), self.dtype),
                "YZ Plane": jax.ShapeDtypeStruct((gs[1], gs[2]), self.dtype),
            }
        if self.reduce_volume:
            return {"energy": jax.ShapeDtypeStruct((1,), self.dtype)}
        return {"energy": jax.ShapeDtypeStruct(self.grid_shape, self.dtype)}

    def update(
        self,
        time_step: jax.Array,
        E: jax.Array,
        H: jax.Array,
        state: DetectorState,
        inv_permittivity: jax.Array,
        inv_permeability: jax.Array,
    ) -> DetectorState:
        """Update the energy detector state with current field values.

        Computes electromagnetic energy density from E and H fields and material
        properties. Can record full 3D distribution, 2D slices, or reduced values.

        Args:
            time_step: Current simulation time step
            E: Electric field array
            H: Magnetic field array
            state: Current detector state
            inv_permittivity: Inverse permittivity array
            inv_permeability: Inverse permeability array

        Returns:
            DetectorState: Updated state containing new energy values
        """
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
        new_full_arr = state["energy"].at[arr_idx].set(energy)
        new_state = {"energy": new_full_arr}
        return new_state
