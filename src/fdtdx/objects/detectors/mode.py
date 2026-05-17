from typing import Literal, Self, Sequence

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.null import Null
from fdtdx.core.physics.metrics import bidirectional_mode_overlap
from fdtdx.core.physics.modes import compute_modes_multi_freq
from fdtdx.objects.detectors.detector import DetectorState
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.typing import SliceTuple3D


@autoinit
class ModeOverlapDetector(PhasorDetector):
    """
    Detector for measuring the overlap of a waveguide mode with the simulation fields.
    This detector computes the overlap integral at every frequency in ``wave_characters``,
    enabling broadband frequency-domain analysis of the electromagnetic fields.

    The mode overlap is calculated by integrating the cross product of the mode fields
    with the simulation fields over a cross-sectional plane. This is useful for
    analyzing waveguide coupling efficiency, transmission coefficients, and modal
    decomposition of electromagnetic fields.

    ``compute_overlap()`` returns a complex array of shape ``(num_freqs,)``, where
    ``num_freqs = len(wave_characters)``.  All frequencies are solved in a single
    ``jax.pure_callback`` call during ``apply()``, with neff-proximity tracking to
    prevent mode hopping across frequencies.
    """

    #: Direction of mode propagation, either "+" (forward) or "-" (backward).
    #: Determines which direction along the waveguide axis the mode is assumed to propagate.
    direction: Literal["+", "-"] = frozen_field()

    #: Index of the waveguide mode to use for overlap calculation.
    #: Defaults to 0 (fundamental mode). Higher indices correspond to higher-order modes.
    mode_index: int = frozen_field(default=0)

    #: Optional polarization filter for the mode calculation.
    #: Can be "te" (transverse electric), "tm" (transverse magnetic), or None (no filtering).
    #: When specified, only modes of the given polarization type are considered. Defaults to None.
    filter_pol: Literal["te", "tm"] | None = frozen_field(default=None)

    #: Bend radius of the waveguide in meters. When set, the mode solver accounts for the conformal
    #: transformation introduced by the bend. Must be set together with bend_axis. Defaults to None
    #: (straight waveguide).
    bend_radius: float | None = frozen_field(default=None)

    #: Physical axis index (0=x, 1=y, 2=z) pointing from the waveguide center toward the center of
    #: curvature. Must differ from the propagation axis. Required when bend_radius is set.
    bend_axis: int | None = frozen_field(default=None)

    #: Symmetry-plane condition at the min edge of each transverse axis (the two non-propagation
    #: physical axes, in increasing-index order): ``0`` = PEC mirror (electric wall, the default),
    #: ``1`` = PMC mirror (magnetic wall). Set this when the detector plane's waveguide lies on a
    #: symmetry plane of a reduced (half/quarter) domain so the reference mode is solved with the
    #: same wall the FDTD uses (e.g. ``(0, 1)`` for PEC at y=0 and PMC at the z Si-mid plane).
    #: Must match the corresponding ModePlaneSource for a consistent overlap.
    symmetry: tuple[int, int] = frozen_field(default=(0, 0))

    #: Cannot be specified here since the detector needs all components.
    components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] = frozen_field(
        default=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        init=False,  # in this detector, we always want all components. Do not give user a choice
    )

    #: Cannot be specified here since plotting a single scalar is useless.
    plot: bool = frozen_field(default=False, init=False)  # single scalar is useless for plotting

    _mode_E: jax.Array = private_field()
    _mode_H: jax.Array = private_field()
    _mode_neff: jax.Array = private_field()  # shape (num_freqs,) after apply(); for inspection only
    _cached_area_EuHv: jax.Array = private_field()  # EuxHv Yee face areas; set in place_on_grid
    _cached_area_EvHu: jax.Array = private_field()  # EvxHu Yee face areas; set in place_on_grid

    @property
    def propagation_axis(self) -> int:
        if sum([a == 1 for a in self.grid_shape]) != 1:
            raise Exception(f"Invalid ModeOverlapDetector shape: {self.grid_shape}")
        return self.grid_shape.index(1)

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )
        if (self.bend_radius is None) != (self.bend_axis is None):
            raise ValueError("bend_radius and bend_axis must both be set or both be None")
        if self.bend_axis is not None and self.bend_axis == self.propagation_axis:
            raise ValueError(
                f"bend_axis ({self.bend_axis}) must differ from the propagation axis ({self.propagation_axis})"
            )
        grid = self._config.resolved_grid
        if grid is not None:
            area_EuHv, area_EvHu = grid.yee_face_areas(
                propagation_axis=self.propagation_axis,
                slice_tuple=self.grid_slice_tuple,
            )
        else:
            spacing = self._config.uniform_spacing()
            area_EuHv = jnp.ones(self.grid_shape, dtype=jnp.float32) * spacing * spacing
            area_EvHu = area_EuHv
        self = self.aset("_cached_area_EuHv", area_EuHv, create_new_ok=True)
        self = self.aset("_cached_area_EvHu", area_EvHu, create_new_ok=True)
        return self

    def _transverse_edge_coordinates(self) -> tuple[jax.Array, jax.Array] | None:
        """Return physical transverse edge coordinates for the mode solver.

        Tidy3D can solve modes on rectilinear non-uniform grids when supplied
        with edge-coordinate arrays.  Returning ``None`` keeps the uniform scalar
        spacing path for legacy configurations and older tests.
        """
        grid = self._config.resolved_grid
        if grid is None:
            return None

        transverse_edges = []
        for axis in range(3):
            if axis == self.propagation_axis:
                continue
            lower, upper = self.grid_slice_tuple[axis]
            transverse_edges.append(grid.edges(axis)[lower : upper + 1])
        e0, e1 = transverse_edges
        return e0, e1

    def _mode_solver_resolution(self) -> float:
        """Return scalar resolution only for legacy uniform mode-solver setup.

        ``compute_mode`` ignores this value when explicit transverse coordinates
        are supplied.  For non-uniform grids we pass a harmless finite value so
        the compatibility argument does not force a uniform-grid check.
        """
        if self._config.has_nonuniform_grid:
            assert self._config.resolved_grid is not None
            return self._config.resolved_grid.min_spacing
        return self._config.uniform_spacing()

    def apply(
        self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        dispersive_c1: jax.Array | None = None,
        dispersive_c2: jax.Array | None = None,
        dispersive_c3: jax.Array | None = None,
    ) -> Self:
        del key
        inv_permittivity_slice = inv_permittivities[:, *self.grid_slice]
        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            inv_permeability_slice = inv_permeabilities[:, *self.grid_slice]
        else:
            inv_permeability_slice = inv_permeabilities

        # All frequencies solved in one call with neff-proximity continuity tracking.
        mode_Es, mode_Hs, mode_neffs = compute_modes_multi_freq(
            frequencies=[wc.get_frequency() for wc in self.wave_characters],
            inv_permittivities=inv_permittivity_slice,
            inv_permeabilities=inv_permeability_slice,
            resolution=self._mode_solver_resolution(),
            direction=self.direction,
            mode_index=self.mode_index,
            filter_pol=self.filter_pol,
            dtype=self._config.dtype,
            bend_radius=self.bend_radius,
            bend_axis=self.bend_axis,
            symmetry=self.symmetry,
            transverse_coords=self._transverse_edge_coordinates(),
        )

        self = self.aset("_mode_E", mode_Es, create_new_ok=True)
        self = self.aset("_mode_H", mode_Hs, create_new_ok=True)
        self = self.aset("_mode_neff", mode_neffs, create_new_ok=True)
        return self

    def compute_overlap_to_mode(
        self,
        state: DetectorState,
        mode_E: jax.Array,
        mode_H: jax.Array,
        freq_idx: int = 0,
    ) -> jax.Array:
        """Compute the mode overlap integral for a single mode and frequency slot.

        Args:
            state: Detector state containing accumulated phasors, shape
                ``(1, num_freqs, 6, *spatial)``.
            mode_E: Electric field of the target mode, shape ``(3, *spatial)``.
            mode_H: Magnetic field of the target mode, shape ``(3, *spatial)``.
            freq_idx: Index into the frequency axis of the phasor state.  Defaults
                to 0 (first / only frequency).

        Returns:
            Complex scalar overlap coefficient.
        """
        phasors = state["phasor"]
        phasors_E, phasors_H = phasors[0, freq_idx, :3], phasors[0, freq_idx, 3:]

        alpha_coeff = bidirectional_mode_overlap(
            mode_E=mode_E,
            mode_H=mode_H,
            sim_E=phasors_E,
            sim_H=phasors_H,
            propagation_axis=self.propagation_axis,
            area_EuHv=self._cached_area_EuHv,
            area_EvHu=self._cached_area_EvHu,
        )

        # in pulsed mode return unscaled coefficient
        if self.scaling_mode != "pulse":
            alpha_coeff = alpha_coeff / 4.0

        return alpha_coeff

    def compute_overlap(
        self,
        state: DetectorState,
    ) -> jax.Array:
        """Compute mode overlaps for all frequencies.

        Returns:
            Complex array of shape ``(num_freqs,)`` where ``num_freqs =
            len(wave_characters)``.  Each element is the overlap coefficient
            at the corresponding frequency.  Callers that previously consumed a
            scalar result (single-frequency case) must index the returned array,
            e.g. ``result[0]``.
        """
        if isinstance(self._mode_E, Null) or isinstance(self._mode_H, Null):
            raise Exception("Need to call apply on ModeOverlapDetector before calling compute_mode_overlap!")
        results = [
            self.compute_overlap_to_mode(
                state=state,
                mode_E=self._mode_E[i],
                mode_H=self._mode_H[i],
                freq_idx=i,
            )
            for i in range(self._mode_E.shape[0])
        ]
        return jnp.stack(results)
