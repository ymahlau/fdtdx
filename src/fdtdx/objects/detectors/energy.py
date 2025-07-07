import jax

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.physics.metrics import compute_energy
from fdtdx.objects.detectors.detector import Detector, DetectorState


@autoinit
class EnergyDetector(Detector):
    """Detector for measuring electromagnetic energy distribution.

    This detector computes and records the electromagnetic energy density at specified
    points in the simulation volume. It can operate in different modes to either record
    full 3D data, 2D slices, or reduced volume measurements.

    Attributes:
        as_slices (bool, optional): If True, returns energy measurements as 2D slices through the volume.
            Defaults to False.
        reduce_volume (bool, optional): If True, reduces the volume data to a single energy value.
            Defaults to False.
        x_slice (float | None, optional): real-world positions for slice extraction. Defaults to None.
        y_slice (float | None, optional): real-world positions for slice extraction. Defaults to None.
        z_slice (float | None, optional): real-world positions for slice extraction. Defaults to None.
        aggregate (str | None, optional): If "mean", aggregates slices by averaging instead of using position.
            If None, mean is used. Defaults to None.
    """

    as_slices: bool = frozen_field(default=False)
    reduce_volume: bool = frozen_field(default=False)
    x_slice: float | None = frozen_field(default=None)
    y_slice: float | None = frozen_field(default=None)
    z_slice: float | None = frozen_field(default=None)
    aggregate: str | None = frozen_field(default=None)  # e.g., "mean"

    def _shape_dtype_single_time_step(
        self,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        if self.as_slices and self.reduce_volume:
            raise Exception("Cannot both reduce volume and save slices!")
        gs = self.grid_shape
        if self.as_slices:
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
        inv_permeability: jax.Array | float,
    ) -> DetectorState:
        cur_E = E[:, *self.grid_slice]
        cur_H = H[:, *self.grid_slice]
        cur_inv_permittivity = inv_permittivity[self.grid_slice]
        if isinstance(inv_permeability, jax.Array) and inv_permeability.ndim > 0:
            cur_inv_permeability = inv_permeability[self.grid_slice]
        else:
            cur_inv_permeability = inv_permeability

        energy = compute_energy(
            E=cur_E,
            H=cur_H,
            inv_permittivity=cur_inv_permittivity,
            inv_permeability=cur_inv_permeability,
        )

        arr_idx = self._time_step_to_arr_idx[time_step]

        if self.as_slices:
            use_mean = self.aggregate == "mean" or any(
                slice_ is None for slice_ in (self.x_slice, self.y_slice, self.z_slice)
            )

            if use_mean:
                energy_xy = energy.mean(axis=2)
                energy_xz = energy.mean(axis=1)
                energy_yz = energy.mean(axis=0)
            else:
                # Convert real-world positions to indices
                origin_x = self.grid_slice[0].start * self._config.resolution
                origin_y = self.grid_slice[1].start * self._config.resolution
                origin_z = self.grid_slice[2].start * self._config.resolution

                def to_index(real_pos, origin, axis_len):
                    if real_pos is not None:
                        idx = int((real_pos - origin) / self._config.resolution)
                        return max(0, min(idx, axis_len - 1))
                    return axis_len // 2

                x_idx = to_index(self.x_slice, origin_x, energy.shape[0])
                y_idx = to_index(self.y_slice, origin_y, energy.shape[1])
                z_idx = to_index(self.z_slice, origin_z, energy.shape[2])

                energy_xy = energy[:, :, z_idx]
                energy_xz = energy[:, y_idx, :]
                energy_yz = energy[x_idx, :, :]

            new_xy = state["XY Plane"].at[arr_idx].set(energy_xy)
            new_xz = state["XZ Plane"].at[arr_idx].set(energy_xz)
            new_yz = state["YZ Plane"].at[arr_idx].set(energy_yz)

            return {
                "XY Plane": new_xy,
                "XZ Plane": new_xz,
                "YZ Plane": new_yz,
            }

        if self.reduce_volume:
            total_energy = energy.sum()
            new_arr = state["energy"].at[arr_idx].set(total_energy)
            return {"energy": new_arr}

        new_arr = state["energy"].at[arr_idx].set(energy)
        return {"energy": new_arr}
