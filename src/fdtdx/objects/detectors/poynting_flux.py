from typing import Literal

import jax

from fdtdx.core.jax.pytrees import extended_autoinit, frozen_field
from fdtdx.core.physics.metrics import poynting_flux
from fdtdx.objects.detectors.detector import Detector, DetectorState


@extended_autoinit
class PoyntingFluxDetector(Detector):
    """Detector for measuring Poynting flux in electromagnetic simulations.

    This detector computes the Poynting flux (power flow) through a specified surface
    in the simulation volume. It can measure flux in either positive or negative
    direction along the propagation axis, and optionally reduce measurements to a
    single value by summing over the detection surface.

    Attributes:
        direction: Direction of flux measurement, either "+" for positive or "-" for
            negative along the propagation axis.
        reduce_volume: If True, reduces measurements to a single value by summing
            over the detection surface. If False, maintains spatial distribution.
    """

    direction: Literal["+", "-"] = frozen_field(kind="KW_ONLY")  # type: ignore
    reduce_volume: bool = True

    @property
    def propagation_axis(self) -> int:
        """Determines the axis along which Poynting flux is measured.

        The propagation axis is identified as the dimension with size 1 in the
        detector's grid shape, representing a plane perpendicular to the flux
        measurement direction.

        Returns:
            int: Index of the propagation axis (0 for x, 1 for y, 2 for z)

        Raises:
            Exception: If detector shape does not have exactly one dimension of size 1
        """
        if sum([a == 1 for a in self.grid_shape]) != 1:
            raise Exception(f"Invalid poynting flux detector shape: {self.grid_shape}")
        return self.grid_shape.index(1)

    def _shape_dtype_single_time_step(
        self,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        if self.reduce_volume:
            return {"poynting_flux": jax.ShapeDtypeStruct((1,), self.dtype)}
        return {"poynting_flux": jax.ShapeDtypeStruct(self.grid_shape, self.dtype)}

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
        cur_E = E[:, *self.grid_slice]
        cur_H = H[:, *self.grid_slice]

        pf = poynting_flux(cur_E, cur_H)[self.propagation_axis]
        if self.direction == "-":
            pf = -pf
        if self.reduce_volume:
            pf = pf.sum()
        arr_idx = self._time_step_to_arr_idx[time_step]
        new_full_arr = state["poynting_flux"].at[arr_idx].set(pf)
        new_state = {"poynting_flux": new_full_arr}
        return new_state
