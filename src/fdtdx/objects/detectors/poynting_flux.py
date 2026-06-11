from typing import Literal, Self

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.physics.metrics import compute_poynting_flux
from fdtdx.objects.detectors.detector import Detector, DetectorState
from fdtdx.typing import SliceTuple3D


@autoinit
class PoyntingFluxDetector(Detector):
    """Detector for measuring Poynting flux in electromagnetic simulations.

    This detector computes the Poynting flux (power flow) through a specified surface
    in the simulation volume. It can measure flux in either positive or negative
    direction along the propagation axis, and optionally reduce measurements to a
    single value by summing over the detection surface.
    """

    #: Direction of flux measurement, either "+" for positive or "-" for negative along the propagation axis.
    direction: Literal["+", "-"] = frozen_field()

    #: If True, reduces measurements to a single value by summing over the detection surface.
    #: If False, maintains spatial distribution. Defaults to True.
    reduce_volume: bool = frozen_field(default=True)

    #: By default, the propagation axis for calculating the poynting
    #: flux is the axis, where the detector has a grid shape of 1. If the detector has a shape of 1 in more than
    #: one axes or a different axis should be used, then this attribute can/has to be set. Defaults to None.
    fixed_propagation_axis: int | None = frozen_field(default=None)

    #: By default, only the poynting flux component for the propagation axis
    #: is returned (scalar). If true, all three vector components are returned. Defaults to False.
    keep_all_components: bool = frozen_field(default=False)

    _cached_face_area_weights: jax.Array = private_field()

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

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(grid_slice_tuple=grid_slice_tuple, config=config, key=key)
        can_determine_axis = self.keep_all_components or (
            self.fixed_propagation_axis is not None or sum(a == 1 for a in self.grid_shape) == 1
        )
        if can_determine_axis:
            grid = self._config.resolved_grid
            if grid is not None:
                if self.keep_all_components:
                    weights = jnp.stack(
                        [grid.face_area(axis=axis, slice_tuple=self.grid_slice_tuple) for axis in range(3)]
                    )
                else:
                    weights = grid.face_area(axis=self.propagation_axis, slice_tuple=self.grid_slice_tuple)
            else:
                spacing = self._config.uniform_spacing()
                area = jnp.ones(self.grid_shape, dtype=self.dtype) * spacing * spacing
                weights = jnp.stack([area, area, area]) if self.keep_all_components else area
            self = self.aset("_cached_face_area_weights", weights, create_new_ok=True)
        return self

    def _shape_dtype_single_time_step(
        self,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        if self.keep_all_components:
            shape = (3,) if self.reduce_volume else (3, *self.grid_shape)
        else:
            shape = (1,) if self.reduce_volume else self.grid_shape
        return {"poynting_flux": jax.ShapeDtypeStruct(shape, self.dtype)}

    def _face_area_weights(self) -> jax.Array:
        """Return face-area weights matching this detector's grid slice."""
        return self._cached_face_area_weights

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

        pf = compute_poynting_flux(cur_E, cur_H).real
        if not self.keep_all_components:
            pf = pf[self.propagation_axis]
        if self.direction == "-":
            pf = -pf
        if self.reduce_volume:
            pf = pf * self._face_area_weights()
            if self.keep_all_components:
                pf = pf.sum(axis=(1, 2, 3))
            else:
                pf = pf.sum()
        arr_idx = self._time_step_to_arr_idx[time_step]
        new_full_arr = state["poynting_flux"].at[arr_idx].set(pf)
        new_state = {"poynting_flux": new_full_arr}
        return new_state
