from typing import ClassVar, Literal, Self, Sequence

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.physics.metrics import compute_poynting_flux, net_poynting_flux_through_box
from fdtdx.objects.detectors.detector import Detector, DetectorState
from fdtdx.objects.detectors.phasor import PhasorDetector
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


def _slice_face(arr: jax.Array, axis: int, side: Literal["min", "max"]) -> jax.Array:
    """Slice ``arr`` to a size-one face along ``axis``, keeping the dimension.

    ``side="min"`` selects index 0 (the minimum face), ``side="max"`` the last
    index (the maximum face). Used both to reduce a full-box area-weight array to
    a single face and to pick the two boundary planes out of the phasor volume.
    """
    idxer: list[slice] = [slice(None)] * arr.ndim
    idxer[axis] = slice(0, 1) if side == "min" else slice(-1, None)
    return arr[tuple(idxer)]


def _phasor_poynting_vector(phasors: jax.Array) -> jax.Array:
    """Real Poynting vector ``Re(E x conj(H))`` from a phasor stack.

    Args:
        phasors (jax.Array): Complex phasors of shape ``(num_freqs, 6, *spatial)``
            with the six components ordered ``(Ex, Ey, Ez, Hx, Hy, Hz)`` on axis 1.

    Returns:
        jax.Array: Real Poynting vector of shape ``(num_freqs, 3, *spatial)``. The
        ``1/2`` time-average factor is *not* applied here; callers add it for the
        continuous scaling mode (see the detector ``compute_*`` methods).
    """
    E_ph, H_ph = phasors[:, :3], phasors[:, 3:]
    return compute_poynting_flux(E_ph, H_ph, axis=1).real


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


@autoinit
class PhasorPoyntingFluxDetector(PhasorDetector):
    """Time-averaged Poynting flux through a single plane in the frequency domain.

    Frequency-domain analog of :class:`PoyntingFluxDetector`. Instead of recording
    the instantaneous flux at every time step, it accumulates the complex field
    phasors (inheriting all of :class:`PhasorDetector`'s DFT / subsampling /
    scaling machinery) and forms the time-averaged Poynting flux
    ``<S> = 1/2 Re(E(w) x H*(w))`` in the post-processing method
    :meth:`compute_poynting_flux`. This mirrors how :class:`ModeOverlapDetector`
    computes its (also bilinear) overlap after the run.

    Because the flux is a *product* of two independently accumulated DFTs, the
    surface integral cannot be folded into the per-step update the way the
    time-domain detector does -- the phasors must be complete first. All six field
    components are therefore always recorded.
    """

    #: Direction of flux measurement, either "+" for positive or "-" for negative along the propagation axis.
    direction: Literal["+", "-"] = frozen_field()

    #: By default, the propagation axis is the axis where the detector has a grid shape of 1. If the detector has
    #: a shape of 1 in more than one axis or a different axis should be used, set this attribute. Defaults to None.
    fixed_propagation_axis: int | None = frozen_field(default=None)

    #: By default, only the Poynting flux component for the propagation axis is returned (scalar per wavelength).
    #: If True, all three vector components are returned. Defaults to False.
    keep_all_components: bool = frozen_field(default=False)

    #: Always all six field components -- both E and H are needed for the Poynting flux. Not user-configurable.
    components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] = frozen_field(
        default=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        init=False,
    )

    #: Spatial phasors must be kept to perform the area integral, so volume reduction is disabled.
    reduce_volume: bool = frozen_field(default=False, init=False)

    #: Raw phasor auto-plotting is not meaningful; consume the ``compute_poynting_flux`` result instead.
    plot: bool = frozen_field(default=False, init=False)

    _cached_face_area_weights: jax.Array = private_field()

    # Poynting flux magnitude is non-negative, but the signed convention matches PoyntingFluxDetector.
    _signed_data: ClassVar[bool] = False

    @property
    def propagation_axis(self) -> int:
        """Axis along which the Poynting flux is measured (the size-one grid dimension)."""
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
        real_dtype = jnp.float64 if self.dtype == jnp.complex128 else jnp.float32
        if self.keep_all_components:
            weights = jnp.stack(
                [_resolve_face_area_weights(self._config, self.grid_slice_tuple, axis, real_dtype) for axis in range(3)]
            )
        else:
            weights = _resolve_face_area_weights(self._config, self.grid_slice_tuple, self.propagation_axis, real_dtype)
        self = self.aset("_cached_face_area_weights", weights, create_new_ok=True)
        return self

    def compute_poynting_flux(self, state: DetectorState) -> jax.Array:
        """Time-averaged Poynting flux through the plane at every recorded wavelength.

        Args:
            state (DetectorState): Detector state holding the accumulated phasors of
                shape ``(1, num_freqs, 6, *grid_shape)``.

        Returns:
            jax.Array: Real flux of shape ``(num_freqs,)`` (net power through the
            plane along the propagation axis), or ``(num_freqs, 3)`` when
            ``keep_all_components`` is set (all three Poynting components).
        """
        phasors = state["phasor"][0]  # (num_freqs, 6, *grid_shape)
        pv = _phasor_poynting_vector(phasors)  # (num_freqs, 3, *grid_shape)
        if self.direction == "-":
            pv = -pv
        weights = self._cached_face_area_weights
        if self.keep_all_components:
            # weights: (3, *grid_shape); pv: (num_freqs, 3, *grid_shape) -> sum over spatial.
            flux = jnp.sum(pv * weights[None, ...], axis=(2, 3, 4))
        else:
            # Only the propagation-axis component; weights broadcast over the frequency axis.
            comp = pv[:, self.propagation_axis]  # (num_freqs, *grid_shape)
            flux = jnp.sum(comp * weights, axis=(1, 2, 3))
        if self.scaling_mode == "continuous":
            flux = 0.5 * flux
        return flux


@autoinit
class ClosedSurfacePhasorPoyntingFluxDetector(PhasorDetector):
    """Net time-averaged Poynting flux through a closed box surface (frequency domain).

    Frequency-domain analog of :class:`ClosedSurfacePoyntingFluxDetector`. It
    accumulates field phasors and forms the net time-averaged outward flux
    ``sum_faces 1/2 Re(E(w) x H*(w)) . dA`` in :meth:`compute_net_flux`.

    Only the **hollow shell** is recorded: for each active axis just the two
    boundary planes are stored, never the box interior. The persistent detector
    state is therefore ``O(surface)`` rather than ``O(volume)`` -- the interior
    phasors would be pure waste since the surface integral reads only the faces.

    A face pair on an axis of size one cancels, so the default ``axes`` (every
    axis with more than one cell) reduces to a 4-face surface for a quasi-2D setup
    and a 6-face surface in full 3D, exactly like the time-domain version.
    """

    #: ``"outward"`` (default) counts net power leaving the box as positive.
    #: ``"inward"`` flips the sign (net power entering, e.g. absorbed power).
    orientation: Literal["outward", "inward"] = frozen_field(default="outward")

    #: Axes whose two faces contribute to the surface integral. ``None`` (default)
    #: uses every axis with a grid extent greater than one.
    axes: tuple[int, ...] | None = frozen_field(default=None)

    #: Always all six field components -- both E and H are needed for the Poynting flux. Not user-configurable.
    components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] = frozen_field(
        default=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        init=False,
    )

    #: Storage is per-face and handled explicitly, so the base volume reduction is disabled.
    reduce_volume: bool = frozen_field(default=False, init=False)

    #: Raw phasor auto-plotting is not meaningful; consume the ``compute_net_flux`` result instead.
    plot: bool = frozen_field(default=False, init=False)

    #: Per-axis face-area weights, each already reduced to a single boundary plane (size one on the normal axis).
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
        real_dtype = jnp.float64 if self.dtype == jnp.complex128 else jnp.float32
        # Reduce each axis' area weights to a single boundary plane (transverse area is
        # independent of position along the normal, so either face works). This matches the
        # hollow per-face storage below, where the field faces are already extracted.
        weights = tuple(
            _slice_face(_resolve_face_area_weights(self._config, self.grid_slice_tuple, axis, real_dtype), axis, "min")
            for axis in range(3)
        )
        self = self.aset("_face_area_weights_per_axis", weights, create_new_ok=True)
        return self

    def _shape_dtype_single_time_step(
        self,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        field_dtype = jnp.complex128 if self.dtype == jnp.complex128 else jnp.complex64
        num_components = len(self.components)
        num_frequencies = len(self._angular_frequencies)
        result: dict[str, jax.ShapeDtypeStruct] = {}
        for a in self._resolve_active_axes():
            plane_shape = tuple(1 if i == a else self.grid_shape[i] for i in range(3))
            shape = (num_frequencies, num_components, *plane_shape)
            result[f"phasor_axis{a}_min"] = jax.ShapeDtypeStruct(shape, field_dtype)
            result[f"phasor_axis{a}_max"] = jax.ShapeDtypeStruct(shape, field_dtype)
        return result

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

        EH = jnp.stack([E[0], E[1], E[2], H[0], H[1], H[2]], axis=0)  # (6, nx, ny, nz)
        phase_angles = self._angular_frequencies * time_passed  # (num_freqs,)
        phasors = jnp.exp(1j * phase_angles).reshape((len(self._angular_frequencies),) + (1,) * EH.ndim)
        new_phasors = EH * phasors * static_scale  # (num_freqs, 6, nx, ny, nz)

        new_state = dict(state)
        for a in self._resolve_active_axes():
            # Spatial axis a maps to array axis a + 2 (leading freq and component axes).
            for side in ("min", "max"):
                key = f"phasor_axis{a}_{side}"
                face = _slice_face(new_phasors, a + 2, side)[None, ...]  # (1, num_freqs, 6, *plane)
                if self.inverse:
                    new_state[key] = (state[key] - face).astype(self.dtype)
                else:
                    new_state[key] = (state[key] + face).astype(self.dtype)
        return new_state

    def compute_net_flux(self, state: DetectorState) -> jax.Array:
        """Net time-averaged Poynting flux through the closed surface at every wavelength.

        Args:
            state (DetectorState): Detector state holding the per-face phasors.

        Returns:
            jax.Array: Real net flux of shape ``(num_freqs,)``. Positive means net
            power leaving the box for ``orientation="outward"``.
        """
        if self._face_area_weights_per_axis is None:
            raise Exception("Detector is not yet placed on the grid")
        active_axes = self._resolve_active_axes()
        num_freqs = len(self._angular_frequencies)
        real_dtype = jnp.float64 if self.dtype == jnp.complex128 else jnp.float32
        net = jnp.zeros((num_freqs,), dtype=real_dtype)
        for a in active_axes:
            area = self._face_area_weights_per_axis[a]  # (*plane) with normal axis size one
            for side, sign in (("max", 1.0), ("min", -1.0)):
                phasors = state[f"phasor_axis{a}_{side}"][0]  # (num_freqs, 6, *plane)
                s_a = _phasor_poynting_vector(phasors)[:, a]  # (num_freqs, *plane)
                net = net + sign * jnp.sum(s_a * area, axis=(1, 2, 3))
        if self.orientation == "inward":
            net = -net
        if self.scaling_mode == "continuous":
            net = 0.5 * net
        return net
