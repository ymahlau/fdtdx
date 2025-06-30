from typing import Literal

import jax

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.physics.metrics import compute_poynting_flux
from fdtdx.objects.detectors.detector import Detector, DetectorState


@autoinit
class PoyntingFluxDetector(Detector):
    """Detector for measuring Poynting flux in electromagnetic simulations.

    This detector computes the Poynting flux (power flow) through a specified surface
    in the simulation volume. It can measure flux in either positive or negative
    direction along the propagation axis, and optionally reduce measurements to a
    single value by summing over the detection surface.

    Attributes:
        direction (Literal["+", "-"]): Direction of flux measurement, either "+" for positive or "-" for
            negative along the propagation axis.
        reduce_volume (bool, optional): If True, reduces measurements to a single value by summing
            over the detection surface. If False, maintains spatial distribution. Defaults to True.
        fixed_propagation_axis (int | None, optional): By default, the propagation axis for calculating the poynting
            flux is the axis, where the detector has a grid shape of 1. If the detector has a shape of 1 in more than
            one axes or a different axis should be used, then this attribute can/has to be set. Defaults to None.
        keep_all_components (bool, optional): By default, only the poynting flux component for the propagation axis
            is returned (scalar). If true, all three vector components are returned. Defaults to False.
    """

    direction: Literal["+", "-"] = frozen_field()
    reduce_volume: bool = frozen_field(default=True)
    fixed_propagation_axis: int | None = frozen_field(default=None)
    keep_all_components: bool = frozen_field(default=False)

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
        if self.fixed_propagation_axis is not None:
            if self.fixed_propagation_axis not in [0, 1, 2]:
                raise Exception(f"Invalid: {self.fixed_propagation_axis=}")
            return self.fixed_propagation_axis
        if sum([a == 1 for a in self.grid_shape]) != 1:
            raise Exception(f"Invalid poynting flux detector shape: {self.grid_shape}")
        return self.grid_shape.index(1)

    def _shape_dtype_single_time_step(
        self,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        if self.keep_all_components:
            shape = (3,) if self.reduce_volume else (3, *self.grid_shape)
        else:
            shape = (1,) if self.reduce_volume else self.grid_shape
        return {"poynting_flux": jax.ShapeDtypeStruct(shape, self.dtype)}

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

        pf = compute_poynting_flux(cur_E, cur_H)
        if not self.keep_all_components:
            pf = pf[self.propagation_axis]
        if self.direction == "-":
            pf = -pf
        if self.reduce_volume:
            if self.keep_all_components:
                pf = pf.sum(axis=(1, 2, 3))
            else:
                pf = pf.sum()
        arr_idx = self._time_step_to_arr_idx[time_step]
        new_full_arr = state["poynting_flux"].at[arr_idx].set(pf)
        new_state = {"poynting_flux": new_full_arr}
        return new_state
