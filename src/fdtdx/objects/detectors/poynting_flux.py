from typing import ClassVar, Literal, Self

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.physics.metrics import compute_poynting_flux, net_poynting_flux_through_box
from fdtdx.objects.detectors.detector import Detector, DetectorState
from fdtdx.typing import SliceTuple3D


def _resolve_face_area_weights(
    config: SimulationConfig,
    slice_tuple: SliceTuple3D,
    axis: int,
    dtype: jnp.dtype,
) -> jax.Array:
    """Per-cell face-area weights for a face normal to ``axis`` over ``slice_tuple``.

    Uses the resolved grid's ``face_area`` (which returns per-cell transverse
    areas, so non-uniform steps within a single face are captured exactly) and
    falls back to ``spacing**2`` for a pure uniform grid with no resolved
    rectilinear grid. Shared by :class:`PoyntingFluxDetector` (single plane) and
    :class:`ClosedSurfacePoyntingFluxDetector` (box) so the grid/uniform handling
    lives in one place.

    Args:
        config (SimulationConfig): Simulation config providing the resolved grid.
        slice_tuple (SliceTuple3D): Grid slice of the detector region.
        axis (int): Normal axis of the face.
        dtype (jnp.dtype): Output dtype for the uniform fallback.

    Returns:
        jax.Array: Area weights broadcastable to the Poynting component on ``axis``.
    """
    grid = config.resolved_grid
    if grid is not None:
        return grid.face_area(axis=axis, slice_tuple=slice_tuple)
    spacing = config.uniform_spacing()
    shape = tuple(upper - lower for lower, upper in slice_tuple)
    return jnp.ones(shape, dtype=dtype) * spacing * spacing


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

    # Poynting flux is positive.
    _signed_data: ClassVar[bool] = False

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
            if self.keep_all_components:
                weights = jnp.stack(
                    [
                        _resolve_face_area_weights(self._config, self.grid_slice_tuple, axis, self.dtype)
                        for axis in range(3)
                    ]
                )
            elif self._config.resolved_grid is not None:
                # Only touch propagation_axis when a rectilinear grid needs it; the uniform
                # fallback is axis-independent, matching the legacy behavior where an invalid
                # fixed_propagation_axis does not raise until propagation_axis is accessed.
                weights = _resolve_face_area_weights(
                    self._config, self.grid_slice_tuple, self.propagation_axis, self.dtype
                )
            else:
                # Uniform fallback: area is spacing**2 for every cell regardless of axis.
                weights = _resolve_face_area_weights(self._config, self.grid_slice_tuple, 0, self.dtype)
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
        pf = compute_poynting_flux(E, H).real
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


@autoinit
class ClosedSurfacePoyntingFluxDetector(Detector):
    """Net Poynting flux through the closed surface of a rectangular box.

    Integrates the outward-normal Poynting component over all faces of the box
    spanned by this detector's grid slice, giving a single scalar net power. It
    is the natural probe for scattering/absorption power: a box in the pure
    scattered-field region (outside a TFSF box) yields the scattered power, while
    a box around an absorber yields the absorbed power (use ``orientation="inward"``).

    Unlike :class:`PoyntingFluxDetector`, which measures flux through a single
    plane, this closes the surface and sums all faces. The per-cell face-area
    weighting makes the surface integral exact on non-uniform grids, where cell
    areas differ across a single face.

    A face pair on an axis of size one cancels to zero, so the default
    ``axes`` (all axes with more than one cell) naturally reduces to a 4-face
    surface for a quasi-2D / periodic setup and a 6-face surface in full 3D.
    """

    #: ``"outward"`` (default) counts net power leaving the box as positive.
    #: ``"inward"`` flips the sign (net power entering, e.g. absorbed power).
    orientation: Literal["outward", "inward"] = frozen_field(default="outward")

    #: Axes whose two faces contribute to the surface integral. ``None`` (default)
    #: uses every axis with a grid extent greater than one.
    axes: tuple[int, ...] | None = frozen_field(default=None)

    _face_area_weights_per_axis: tuple | None = private_field(default=None)

    # Net flux is signed (can be positive or negative).
    _signed_data: ClassVar[bool] = True

    def _resolve_active_axes(self) -> tuple[int, ...]:
        """Return the axes whose faces contribute (validated, size-one skipped by default)."""
        if self.axes is not None:
            return tuple(self.axes)
        return tuple(a for a in range(3) if self.grid_shape[a] > 1)

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        if self.orientation not in ("outward", "inward"):
            raise ValueError(f"orientation must be 'outward' or 'inward', got {self.orientation!r}")
        if self.axes is not None and any(a not in (0, 1, 2) for a in self.axes):
            raise ValueError(f"axes entries must be in (0, 1, 2), got {self.axes}")
        self = super().place_on_grid(grid_slice_tuple=grid_slice_tuple, config=config, key=key)
        # Per-cell face-area weights for every axis (only the active ones are used
        # at record time). Computing all three keeps this a fixed-shape pytree.
        weights = tuple(
            _resolve_face_area_weights(self._config, self.grid_slice_tuple, axis, self.dtype) for axis in range(3)
        )
        self = self.aset("_face_area_weights_per_axis", weights, create_new_ok=True)
        return self

    def _shape_dtype_single_time_step(
        self,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        return {"poynting_flux": jax.ShapeDtypeStruct((1,), self.dtype)}

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
        if self._face_area_weights_per_axis is None:
            raise Exception("Detector is not yet placed on the grid")
        pf = compute_poynting_flux(E, H).real
        net = net_poynting_flux_through_box(
            poynting_vector=pf,
            active_axes=self._resolve_active_axes(),
            area_weights=self._face_area_weights_per_axis,
        )
        if self.orientation == "inward":
            net = -net
        net = net.reshape((1,)).astype(self.dtype)
        arr_idx = self._time_step_to_arr_idx[time_step]
        new_full_arr = state["poynting_flux"].at[arr_idx].set(net)
        return {"poynting_flux": new_full_arr}
