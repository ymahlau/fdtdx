import jax
import jax.numpy as jnp
from typing_extensions import override

from fdtdx.constants import c, eps0, eta0
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.plotting.colors import XKCD_DARK_GREY
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
    color: tuple[float, float, float] | None = frozen_field(default=XKCD_DARK_GREY)

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
            pml_thickness = self.thickness * self._config.resolution
            sigma_end_calculated = -(self.sigma_order + 1) * jnp.log(1e-6) / (2 * (eta0 / 1.0) * pml_thickness)
            self = self.aset("sigma_end", float(sigma_end_calculated))

        return self

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
        if self.direction == "-":
            # For min boundary, distance increases as we go towards lower indices
            dE = jnp.arange(L - 1, -1, -1, dtype=dtype)
            dH = jnp.append(jnp.arange(L - 1.5, -0.5, -1, dtype=dtype), 0)
        else:
            # For max boundary, distance increases as we go towards higher indices
            dE = jnp.insert(jnp.arange(0.5, L - 0.5, 1, dtype=dtype), 0, 0)
            dH = jnp.arange(0, L, 1, dtype=dtype)

        # Compute polynomial grading: value_start + (value_end - value_start) * (d/L)^order
        profileE_1d = value_start + (value_end - value_start) * jnp.power(dE / L, order)
        profileH_1d = value_start + (value_end - value_start) * jnp.power(dH / L, order)

        # Create shape matching PML region with grading only along self.axis
        shape = [1, 1, 1]
        shape[self.axis] = L
        profileE_reshaped = profileE_1d.reshape(shape)
        profileH_reshaped = profileH_1d.reshape(shape)
        # Broadcast to full grid_shape
        profileE = jnp.broadcast_to(profileE_reshaped, self.grid_shape)
        profileH = jnp.broadcast_to(profileH_reshaped, self.grid_shape)

        return profileE, profileH

    def modify_arrays(
        self,
        alpha: jax.Array,
        kappa: jax.Array,
        sigma: jax.Array,
        electric_conductivity,
        magnetic_conductivity,
    ) -> dict[str, jax.Array]:
        """Modifies simulation arrays to include PML parameters.

        Args:
            alpha: Alpha array for PML calculations (shape: (3, *volume_shape))
            kappa: Kappa array for PML calculations (shape: (3, *volume_shape))
            sigma: Sigma array for PML calculations (shape: (3, *volume_shape))
            electric_conductivity: Electric conductivity array (shape: volume_shape)
            magnetic_conductivity: Magnetic conductivity array (shape: volume_shape)

        Returns:
            dict: Dictionary with modified 'alpha', 'kappa', and 'sigma' arrays
        """

        assert self.alpha_start is not None, "alpha_start should be set by __post_init__"
        assert self.alpha_end is not None, "alpha_end   should be set by __post_init__"
        assert self.alpha_order is not None, "alpha_order should be set by __post_init__"
        assert self.kappa_start is not None, "kappa_start should be set by __post_init__"
        assert self.kappa_end is not None, "kappa_end   should be set by __post_init__"
        assert self.kappa_order is not None, "kappa_order should be set by __post_init__"
        assert self.sigma_start is not None, "sigma_start should be set by __post_init__"
        assert self.sigma_end is not None, "sigma_end   should be set by __post_init__"
        assert self.sigma_order is not None, "sigma_order should be set by __post_init__"

        dtype = self._config.dtype

        # Compute PML parameters using polynomial grading
        sigma_E, sigma_H = self._compute_pml_profile(
            value_start=self.sigma_start,
            value_end=self.sigma_end,
            order=self.sigma_order,
            dtype=dtype,
        )

        kappa_E, kappa_H = self._compute_pml_profile(
            value_start=self.kappa_start,
            value_end=self.kappa_end,
            order=self.kappa_order,
            dtype=dtype,
        )

        alpha_E, alpha_H = self._compute_pml_profile(
            value_start=self.alpha_start,
            value_end=self.alpha_end,
            order=self.alpha_order,
            dtype=dtype,
        )

        # Update arrays in the PML region
        # The PML parameters vary along self.axis, so we need to broadcast them correctly
        alpha = alpha.at[self.axis, *self.grid_slice].set(alpha_E)
        kappa = kappa.at[self.axis, *self.grid_slice].set(kappa_E)
        sigma = sigma.at[self.axis, *self.grid_slice].set(sigma_E)
        alpha = alpha.at[self.axis + 3, *self.grid_slice].set(alpha_H)
        kappa = kappa.at[self.axis + 3, *self.grid_slice].set(kappa_H)
        sigma = sigma.at[self.axis + 3, *self.grid_slice].set(sigma_H)

        return {
            "alpha": alpha,
            "kappa": kappa,
            "sigma": sigma,
            "electric_conductivity": electric_conductivity,
            "magnetic_conductivity": magnetic_conductivity,
        }
