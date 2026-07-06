import jax
import jax.numpy as jnp
from typing_extensions import override

from fdtdx import Color
from fdtdx.colors import XKCD_DARK_GREY
from fdtdx.constants import c, eps0, eta0
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.objects.boundaries.boundary import BaseBoundary


@autoinit
class PerfectlyMatchedLayer(BaseBoundary):
    """Implements a Convolutional Perfectly Matched Layer (CPML) boundary condition.

    The CPML absorbs outgoing electromagnetic waves with minimal reflection by using
    a complex coordinate stretching approach. This implementation supports arbitrary
    axis orientation and both positive/negative directions.
    """

    #: Initial loss parameter for complex frequency shifting. Defaults to 0.01 * 2 * jnp.pi * c / wavelength * eps0 if not provided.
    alpha_start: float | None = frozen_field(default=None)

    #: Final loss parameter for complex frequency shifting. Defaults to 0.0 if not provided.
    alpha_end: float | None = frozen_field(default=None)

    #: Polynomial order for alpha grading. Defaults to 1.0 if not provided.
    alpha_order: float | None = frozen_field(default=None)

    #: Initial kappa stretching coefficient. Defaults to 0.0 if not provided.
    kappa_start: float | None = frozen_field(default=None)

    #: Final kappa stretching coefficient. Defaults to 0.0 if not provided.
    kappa_end: float | None = frozen_field(default=None)

    #: Polynomial order for kappa grading. Defaults to 1.0 if not provided.
    kappa_order: float | None = frozen_field(default=None)

    #: Initial sigma value. Defaults to 0.0 if not provided.
    sigma_start: float | None = frozen_field(default=None)

    #: Final sigma value. Defaults to 1.0 if not provided.
    sigma_end: float | None = frozen_field(default=None)

    #: Polynomial order for sigma grading. Defaults to 3.0 if not provided.
    sigma_order: float | None = frozen_field(default=None)

    #: RGB color tuple for visualization. defaults to dark grey.
    color: Color | None = frozen_field(default=XKCD_DARK_GREY)

    #: CPML 'a' coefficient array for Electric field updates.
    pml_a_E: jax.Array | None = frozen_field(default=None)

    #: CPML 'b' coefficient array for Electric field updates.
    pml_b_E: jax.Array | None = frozen_field(default=None)

    #: Inverse of the kappa stretching parameter array for the Electric field.
    inv_kappa_E: jax.Array | None = frozen_field(default=None)

    #: CPML 'a' coefficient array for Magnetic field updates.
    pml_a_H: jax.Array | None = frozen_field(default=None)

    #: CPML 'b' coefficient array for Magnetic field updates.
    pml_b_H: jax.Array | None = frozen_field(default=None)

    #: Inverse of the kappa stretching parameter array for the Magnetic field.
    inv_kappa_H: jax.Array | None = frozen_field(default=None)

    def __post_init__(self):
        """Sets default PML parameters if not provided."""
        # Set default values if None is provided
        # Simple defaults that don't depend on grid properties
        if self.alpha_start is None:
            object.__setattr__(self, "alpha_start", 0.01 * 2 * jnp.pi * c / 1.55e-6 * eps0)

        if self.alpha_end is None:
            object.__setattr__(self, "alpha_end", 0.0)

        if self.alpha_order is None:
            object.__setattr__(self, "alpha_order", 1.0)

        if self.kappa_start is None:
            object.__setattr__(self, "kappa_start", 1.0)

        if self.kappa_end is None:
            object.__setattr__(self, "kappa_end", 1.0)

        if self.kappa_order is None:
            object.__setattr__(self, "kappa_order", 3.0)

        if self.sigma_start is None:
            object.__setattr__(self, "sigma_start", 0.0)

        if self.sigma_order is None:
            object.__setattr__(self, "sigma_order", 3.0)

    def place_on_grid(self, grid_slice_tuple, config, key):
        """Place the PML on the grid and calculate any remaining defaults.

        This is called after initialization, so grid_shape and config are available.
        """
        # First call the parent implementation to set grid_slice_tuple and config
        self = super().place_on_grid(grid_slice_tuple, config, key)

        # Now calculate sigma_end if it wasn't provided by the user
        if self.sigma_end is None:
            assert self.sigma_order is not None, "sigma_order should be set by __post_init__"
            pml_thickness = self._physical_thickness()
            sigma_end_calculated = -(self.sigma_order + 1) * jnp.log(1e-6) / (2 * (eta0 / 1.0) * pml_thickness)
            self = self.aset("sigma_end", sigma_end_calculated.astype(float))

        dtype = config.dtype
        dt = config.time_step_duration

        assert self.sigma_start is not None and self.sigma_end is not None and self.sigma_order is not None
        assert self.kappa_start is not None and self.kappa_end is not None and self.kappa_order is not None
        assert self.alpha_start is not None and self.alpha_end is not None and self.alpha_order is not None

        sigma_E, sigma_H = self._compute_pml_profile(self.sigma_start, self.sigma_end, self.sigma_order, dtype)
        kappa_E, kappa_H = self._compute_pml_profile(self.kappa_start, self.kappa_end, self.kappa_order, dtype)
        alpha_E, alpha_H = self._compute_pml_profile(self.alpha_start, self.alpha_end, self.alpha_order, dtype)

        b_E = jnp.expm1(-dt / eps0 * (sigma_E / kappa_E + alpha_E)) + 1
        a_E = jnp.nan_to_num((b_E - 1.0) * sigma_E / (sigma_E + alpha_E * kappa_E) / kappa_E, nan=0.0)

        b_H = jnp.expm1(-dt / eps0 * (sigma_H / kappa_H + alpha_H)) + 1
        a_H = jnp.nan_to_num((b_H - 1.0) * sigma_H / (sigma_H + alpha_H * kappa_H) / kappa_H, nan=0.0)

        self = self.aset("pml_a_E", a_E)
        self = self.aset("pml_b_E", b_E)
        self = self.aset("inv_kappa_E", 1.0 / kappa_E)
        self = self.aset("pml_a_H", a_H)
        self = self.aset("pml_b_H", b_H)
        self = self.aset("inv_kappa_H", 1.0 / kappa_H)

        return self

    def step_cpml(
        self,
        d_field_1: jax.Array,
        d_field_2: jax.Array,
        psi_1: jax.Array,
        psi_2: jax.Array,
        is_curl_E: bool,
        simulate_boundaries: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Performs localized CPML correction for the two derivatives along this boundary's axis.

        Uses Auxiliary Differential Equations (ADEs) to update the psi arrays and applies
        the complex coordinate stretching corrections to the spatial derivatives.

        Args:
            d_field_1: The first spatial derivative array needing PML correction.
            d_field_2: The second spatial derivative array needing PML correction.
            psi_1: The accumulator array (psi) corresponding to the first derivative.
            psi_2: The accumulator array (psi) corresponding to the second derivative.
            is_curl_E: Flag determining whether to use H-field coefficients (True, when computing
                curl(E) to update H) or E-field coefficients (False, when computing curl(H) to update E).
            simulate_boundaries: Flag to toggle whether the boundary memory variables (psi)
                should actually be updated in this step.

        Returns:
            tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                A tuple containing:
                - corr_1: The PML-corrected first spatial derivative.
                - corr_2: The PML-corrected second spatial derivative.
                - psi_1_new: The updated accumulator array for the first derivative.
                - psi_2_new: The updated accumulator array for the second derivative.
        """
        assert self.pml_a_E is not None and self.pml_b_E is not None and self.inv_kappa_E is not None
        assert self.pml_a_H is not None and self.pml_b_H is not None and self.inv_kappa_H is not None

        if is_curl_E:
            a, b, inv_kappa = self.pml_a_H, self.pml_b_H, self.inv_kappa_H
        else:
            a, b, inv_kappa = self.pml_a_E, self.pml_b_E, self.inv_kappa_E

        if simulate_boundaries:
            psi_1_new = b * psi_1 + a * d_field_1
            psi_2_new = b * psi_2 + a * d_field_2
        else:
            psi_1_new, psi_2_new = psi_1, psi_2

        corr_1 = (inv_kappa - 1.0) * d_field_1 + psi_1_new
        corr_2 = (inv_kappa - 1.0) * d_field_2 + psi_2_new

        return corr_1, corr_2, psi_1_new, psi_2_new

    def _physical_thickness(self) -> float:
        """Return PML thickness in metres.

        Uniform-grid simulations keep the historical ``cell_count * spacing``
        behavior.  Non-uniform grids derive thickness from physical grid edges so
        the same PML cell count can represent stretched physical layers.
        """
        grid = self._config.resolved_grid
        if grid is not None:
            return grid.axis_extent(self.axis, self.grid_slice_tuple[self.axis])
        return self.thickness * self._config.uniform_spacing()

    @property
    @override
    def descriptive_name(self) -> str:
        """Gets a human-readable name describing this PML boundary's location.

        Returns:
            str: Description like "min_x" or "max_z" indicating position
        """
        axis_str = "x" if self.axis == 0 else "y" if self.axis == 1 else "z"
        direction_str = "min" if self.direction == "-" else "max"
        return f"{direction_str}_{axis_str}"

    @property
    @override
    def thickness(self) -> int:
        """Gets the thickness of the PML layer in grid points.

        Returns:
            int: Number of grid points in the PML along its axis
        """
        return self.grid_shape[self.axis]

    @override
    def apply_field_reset(self, fields: dict[str, jax.Array]) -> dict[str, jax.Array]:
        """Zero all field components within the PML region."""
        return {name: field.at[:, *self.grid_slice].set(0) for name, field in fields.items()}

    def _compute_pml_profile(
        self,
        value_start: float,
        value_end: float,
        order: float,
        dtype,
    ) -> tuple[jax.Array, jax.Array]:
        """Computes a graded PML profile using polynomial scaling.

        Args:
            value_start: Value at the interface (inner boundary)
            value_end: Value at the outer boundary
            order: Polynomial order for grading
            dtype: Data type for the array

        Returns:
            jax.Array: Graded profile array with shape self.grid_shape
        """
        L = self.thickness  # Total thickness of PML

        # Create distance array along the PML axis
        # d varies from 0 (at interface) to L (at outer edge)
        if self._config.has_nonuniform_grid:
            dE, dH, norm = self._compute_nonuniform_pml_depths(dtype)
        elif self.direction == "-":
            # For min boundary, distance increases as we go towards lower indices
            dE = jnp.arange(L - 1, -1, -1, dtype=dtype)
            dH = jnp.append(jnp.arange(L - 1.5, -0.5, -1, dtype=dtype), 0)
            norm = L
        else:
            # For max boundary, distance increases as we go towards higher indices
            dE = jnp.insert(jnp.arange(0.5, L - 0.5, 1, dtype=dtype), 0, 0)
            dH = jnp.arange(0, L, 1, dtype=dtype)
            norm = L

        # Compute polynomial grading: value_start + (value_end - value_start) * (d/L)^order
        profileE_1d = value_start + (value_end - value_start) * jnp.power(dE / norm, order)
        profileH_1d = value_start + (value_end - value_start) * jnp.power(dH / norm, order)

        # Create shape matching PML region with grading only along self.axis
        shape = [1, 1, 1]
        shape[self.axis] = L
        profileE_reshaped = profileE_1d.reshape(shape)
        profileH_reshaped = profileH_1d.reshape(shape)
        # Broadcast to full grid_shape
        profileE = jnp.broadcast_to(profileE_reshaped, self.grid_shape)
        profileH = jnp.broadcast_to(profileH_reshaped, self.grid_shape)

        return profileE, profileH

    def _compute_nonuniform_pml_depths(self, dtype) -> tuple[jax.Array, jax.Array, float]:
        """Return E/H physical depths into a non-uniform PML.

        Depth is measured from the interior PML interface toward the outer
        boundary.  The E profile uses cell-edge depth so the interface cell has
        zero depth, matching the existing uniform-grid endpoint convention.  The
        H profile uses cell-center depth except at the interface cell, where it is
        pinned to zero for continuity with the historical CPML staggering.
        """
        grid = self._config.resolved_grid
        assert grid is not None
        lower, upper = self.grid_slice_tuple[self.axis]
        edges = grid.edges(self.axis)[lower : upper + 1].astype(dtype)
        norm = float(edges[-1] - edges[0])
        centers = 0.5 * (edges[:-1] + edges[1:])

        zero = jnp.asarray(0.0, dtype=dtype)
        if self.direction == "-":
            interface = edges[-1]
            dE = interface - edges[1:]
            dH = jnp.concatenate([interface - centers[1:], zero.reshape(1)])
        else:
            interface = edges[0]
            dE = jnp.concatenate([zero.reshape(1), centers[:-1] - interface])
            dH = edges[:-1] - interface

        return dE, dH, norm
