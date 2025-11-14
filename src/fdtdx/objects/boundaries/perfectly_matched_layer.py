import jax
import jax.numpy as jnp
from typing_extensions import override

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.plotting.colors import DARK_GREY
from fdtdx.objects.boundaries.boundary import BaseBoundary
from fdtdx.objects.boundaries.utils import (
    alpha_from_direction_axis,
    kappa_from_direction_axis,
    standard_sigma_from_direction_axis,
)


@autoinit
class PerfectlyMatchedLayer(BaseBoundary):
    """Implements a Convolutional Perfectly Matched Layer (CPML) boundary condition.

    The CPML absorbs outgoing electromagnetic waves with minimal reflection by using
    a complex coordinate stretching approach. This implementation supports arbitrary
    axis orientation and both positive/negative directions.
    """

    #: Initial loss parameter for complex frequency shifting. Defaults to 1e-8.
    alpha_start: float = frozen_field(default=1.0e-8)

    #: Final loss parameter for complex frequency shifting. Defaults to 1e-8.
    alpha_end: float = frozen_field(default=1.0e-8)

    #: Polynomial order for alpha grading. Defaults to 1.0.
    alpha_order: float = frozen_field(default=1.0)

    #: Initial kappa stretching coefficient. Defaults to 1.0.
    kappa_start: float = frozen_field(default=1.0)

    #: Final kappa stretching coefficient. Defaults to 1.5.
    kappa_end: float = frozen_field(default=1.5)

    #: Polynomial order for kappa grading. Defaults to 1.0.
    kappa_order: float = frozen_field(default=1.0)

    #: Initial sigma value. Defaults to 0.0.
    sigma_start: float = frozen_field(default=0.0)

    #: Final sigma value. Defaults to 1.0.
    sigma_end: float = frozen_field(default=1.0)

    #: Polynomial order for sigma grading. Defaults to 3.0.
    sigma_order: float = frozen_field(default=3.0)

    #: RGB color tuple for visualization. defaults to dark grey.
    color: tuple[float, float, float] | None = frozen_field(default=DARK_GREY)

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

    def _get_dtype_update_coefficients(self):
        dtype = self._config.dtype
        sigma_E, sigma_H = standard_sigma_from_direction_axis(
            thickness=self.thickness,
            direction=self.direction,
            axis=self.axis,
            dtype=dtype,
        )

        kappa = kappa_from_direction_axis(
            kappa_start=self.kappa_start,
            kappa_end=self.kappa_end,
            thickness=self.thickness,
            direction=self.direction,
            axis=self.axis,
            dtype=dtype,
        )

        alpha = alpha_from_direction_axis(
            alpha_start=self.alpha_start,
            alpha_end=self.alpha_end,
            thickness=self.thickness,
            direction=self.direction,
            axis=self.axis,
            dtype=dtype,
        )

        bE = jnp.exp(-self._config.courant_number * (sigma_E / kappa + alpha))
        bH = jnp.exp(-self._config.courant_number * (sigma_H / kappa + alpha))

        cE = (bE - 1) * sigma_E / (sigma_E * kappa + kappa**2 * alpha)
        cH = (bH - 1) * sigma_H / (sigma_H * kappa + kappa**2 * alpha)

        return dtype, bE, bH, cE, cH, kappa

    def _compute_pml_profile(
        self,
        value_start: float,
        value_end: float,
        order: float,
        dtype,
    ) -> jax.Array:
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
            d = jnp.arange(L - 1, -1, -1, dtype=dtype)
        else:
            # For max boundary, distance increases as we go towards higher indices
            d = jnp.arange(0, L, dtype=dtype)

        # Compute polynomial grading: value_start + (value_end - value_start) * (d/L)^order
        profile_1d = value_start + (value_end - value_start) * jnp.power(d / L, order)

        # Create shape matching PML region with grading only along self.axis
        shape = [1, 1, 1]
        shape[self.axis] = L
        profile_reshaped = profile_1d.reshape(shape)
        # Broadcast to full grid_shape
        profile = jnp.broadcast_to(profile_reshaped, self.grid_shape)

        return profile

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
        dtype = self._config.dtype

        # Compute PML parameters using polynomial grading
        sigma_E = self._compute_pml_profile(
            value_start=self.sigma_start,
            value_end=self.sigma_end,
            order=self.sigma_order,
            dtype=dtype,
        )

        kappa_pml = self._compute_pml_profile(
            value_start=self.kappa_start,
            value_end=self.kappa_end,
            order=self.kappa_order,
            dtype=dtype,
        )

        alpha_pml = self._compute_pml_profile(
            value_start=self.alpha_start,
            value_end=self.alpha_end,
            order=self.alpha_order,
            dtype=dtype,
        )

        # Update arrays in the PML region
        # The PML parameters vary along self.axis, so we need to broadcast them correctly
        alpha = alpha.at[self.axis, *self.grid_slice].set(alpha_pml)
        kappa = kappa.at[self.axis, *self.grid_slice].set(kappa_pml)
        sigma = sigma.at[self.axis, *self.grid_slice].set(sigma_E)

        return {
            "alpha": alpha,
            "kappa": kappa,
            "sigma": sigma,
            "electric_conductivity": electric_conductivity,
            "magnetic_conductivity": magnetic_conductivity,
        }
