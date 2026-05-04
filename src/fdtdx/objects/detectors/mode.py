from functools import cached_property
from typing import Literal, Self, Sequence

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.null import Null
from fdtdx.core.physics.modes import compute_mode
from fdtdx.objects.detectors.detector import DetectorState
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.typing import SliceTuple3D


@autoinit
class ModeOverlapDetector(PhasorDetector):
    """
    Detector for measuring the overlap of a waveguide mode with the simulation fields.
    This detector computes the overlap of a mode with the phasor fields at a specified
    frequency, enabling frequency-domain analysis of the electromagnetic fields.

    The mode overlap is calculated by integrating the cross product of the mode fields
    with the simulation fields over a cross-sectional plane. This is useful for
    analyzing waveguide coupling efficiency, transmission coefficients, and modal
    decomposition of electromagnetic fields.

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

    #: Cannot be specified here since the detector needs all components.
    components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] = frozen_field(
        default=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        init=False,  # in this detector, we always want all components. Do not give user a choice
    )

    #: Cannot be specified here since plotting a single scalar is useless.
    plot: bool = frozen_field(default=False, init=False)  # single scalar is useless for plotting

    _mode_E: jax.Array = private_field()
    _mode_H: jax.Array = private_field()
    _mode_neff: jax.Array = private_field()  # not required for detection, used for inspection

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
        if len(self.wave_characters) > 1:
            raise NotImplementedError()
        if (self.bend_radius is None) != (self.bend_axis is None):
            raise ValueError("bend_radius and bend_axis must both be set or both be None")
        if self.bend_axis is not None and self.bend_axis == self.propagation_axis:
            raise ValueError(
                f"bend_axis ({self.bend_axis}) must differ from the propagation axis ({self.propagation_axis})"
            )
        return self

    @cached_property
    def _cached_face_area_weights(self) -> jax.Array:
        """Return detector-plane face areas for mode-overlap integration.

        The propagation axis is the plane normal.  For legacy construction paths
        without an explicit ``RectilinearGrid``, the scalar grid spacing supplies the
        uniform face area.
        """
        grid = self._config.realized_grid
        if grid is not None:
            return grid.face_area(axis=self.propagation_axis, slice_tuple=self.grid_slice_tuple)

        spacing = self._config.require_uniform_grid()
        return jnp.ones(self.grid_shape, dtype=self.dtype) * spacing * spacing

    def _face_area_weights(self) -> jax.Array:
        """Return cached detector-plane face areas for mode-overlap integration."""
        return self._cached_face_area_weights

    def _transverse_edge_coordinates(self) -> tuple[jax.Array, jax.Array] | None:
        """Return physical transverse edge coordinates for the mode solver.

        Tidy3D can solve modes on rectilinear non-uniform grids when supplied
        with edge-coordinate arrays.  Returning ``None`` keeps the uniform scalar
        spacing path for legacy configurations and older tests.
        """
        grid = self._config.realized_grid
        if grid is None:
            return None

        transverse_edges = []
        for axis in range(3):
            if axis == self.propagation_axis:
                continue
            lower, upper = self.grid_slice_tuple[axis]
            transverse_edges.append(grid.edges(axis)[lower : upper + 1])
        return tuple(transverse_edges)

    def _mode_solver_resolution(self) -> float:
        """Return scalar resolution only for legacy uniform mode-solver setup.

        ``compute_mode`` ignores this value when explicit transverse coordinates
        are supplied.  For non-uniform grids we pass a harmless finite value so
        the compatibility argument does not force a uniform-grid check.
        """
        if self._config.has_nonuniform_grid:
            assert self._config.realized_grid is not None
            return self._config.realized_grid.min_spacing
        return self._config.require_uniform_grid()

    def apply(
        self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
    ) -> Self:
        del key

        inv_permittivity_slice = inv_permittivities[:, *self.grid_slice]
        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            inv_permeability_slice = inv_permeabilities[:, *self.grid_slice]
        else:
            inv_permeability_slice = inv_permeabilities

        mode_E, mode_H, mode_neff = compute_mode(
            frequency=self.wave_characters[0].get_frequency(),
            inv_permittivities=inv_permittivity_slice,
            inv_permeabilities=inv_permeability_slice,
            resolution=self._mode_solver_resolution(),
            direction=self.direction,
            mode_index=self.mode_index,
            filter_pol=self.filter_pol,
            dtype=self._config.dtype,
            bend_radius=self.bend_radius,
            bend_axis=self.bend_axis,
            transverse_coords=self._transverse_edge_coordinates(),
        )

        self = self.aset("_mode_E", mode_E, create_new_ok=True)
        self = self.aset("_mode_H", mode_H, create_new_ok=True)
        self = self.aset("_mode_neff", mode_neff, create_new_ok=True)
        return self

    def compute_overlap_to_mode(
        self,
        state: DetectorState,
        mode_E: jax.Array,
        mode_H: jax.Array,
    ) -> jax.Array:
        # shape (time step, num_freqs, num_components, *spatial)
        # time steps is always 1 and num_components always 6
        phasors = state["phasor"]
        phasors_E, phasors_H = phasors[0, 0, :3], phasors[0, 0, 3:]

        E_cross_H_star_sim = jnp.cross(
            mode_E,
            jnp.conj(phasors_H),
            axis=0,
        )[self.propagation_axis]

        E_star_cross_H_sim = jnp.cross(
            jnp.conj(phasors_E),
            mode_H,
            axis=0,
        )[self.propagation_axis]

        integrand = E_cross_H_star_sim + E_star_cross_H_sim
        integrand = integrand * self._face_area_weights()
        alpha_coeff = jnp.sum(integrand)

        # in pulsed mode return unscaled coefficient
        if self.scaling_mode != "pulse":
            alpha_coeff = alpha_coeff / 4.0

        return alpha_coeff

    def compute_overlap(
        self,
        state: DetectorState,
    ) -> jax.Array:
        if isinstance(self._mode_E, Null) or isinstance(self._mode_H, Null):
            raise Exception("Need to call apply on ModeOverlapDetector before calling compute_mode_overlap!")
        return self.compute_overlap_to_mode(
            state=state,
            mode_E=self._mode_E,
            mode_H=self._mode_H,
        )
