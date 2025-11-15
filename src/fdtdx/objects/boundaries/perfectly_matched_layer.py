import jax
import jax.numpy as jnp
from typing_extensions import override

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.plotting.colors import DARK_GREY
from fdtdx.objects.boundaries.boundary import BaseBoundary


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
            d1 = jnp.arange(L - 1, -1, -1, dtype=dtype)
            d2 = jnp.append(jnp.arange(L - 1.5, -0.5, -1, dtype=dtype), 0)
        else:
            # For max boundary, distance increases as we go towards higher indices
            d1 = jnp.insert(jnp.arange(0.5, L - 0.5, 1, dtype=dtype), 0, 0)
            d2 = jnp.arange(0, L, 1, dtype=dtype)

        # Compute polynomial grading: value_start + (value_end - value_start) * (d/L)^order
        profile1_1d = value_start + (value_end - value_start) * jnp.power(d1 / L, order)
        profile2_1d = value_start + (value_end - value_start) * jnp.power(d2 / L, order)

        # Create shape matching PML region with grading only along self.axis
        shape = [1, 1, 1]
        shape[self.axis] = L
        profile1_reshaped = profile1_1d.reshape(shape)
        profile2_reshaped = profile2_1d.reshape(shape)
        # Broadcast to full grid_shape
        profile1 = jnp.broadcast_to(profile1_reshaped, self.grid_shape)
        profile2 = jnp.broadcast_to(profile2_reshaped, self.grid_shape)

        return profile1, profile2

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
        sigma_E1, sigma_E2 = self._compute_pml_profile(
            value_start=self.sigma_start,
            value_end=self.sigma_end,
            order=self.sigma_order,
            dtype=dtype,
        )

        kappa_pml1, kappa_pml2 = self._compute_pml_profile(
            value_start=self.kappa_start,
            value_end=self.kappa_end,
            order=self.kappa_order,
            dtype=dtype,
        )

        alpha_pml1, alpha_pml2 = self._compute_pml_profile(
            value_start=self.alpha_start,
            value_end=self.alpha_end,
            order=self.alpha_order,
            dtype=dtype,
        )

        # Update arrays in the PML region
        # The PML parameters vary along self.axis, so we need to broadcast them correctly
        alpha = alpha.at[self.axis, *self.grid_slice].set(alpha_pml1)
        kappa = kappa.at[self.axis, *self.grid_slice].set(kappa_pml1)
        sigma = sigma.at[self.axis, *self.grid_slice].set(sigma_E1)
        alpha = alpha.at[self.axis + 3, *self.grid_slice].set(alpha_pml2)
        kappa = kappa.at[self.axis + 3, *self.grid_slice].set(kappa_pml2)
        sigma = sigma.at[self.axis + 3, *self.grid_slice].set(sigma_E2)

        return {
            "alpha": alpha,
            "kappa": kappa,
            "sigma": sigma,
            "electric_conductivity": electric_conductivity,
            "magnetic_conductivity": magnetic_conductivity,
        }
