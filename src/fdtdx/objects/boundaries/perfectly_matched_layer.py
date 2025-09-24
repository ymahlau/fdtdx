import jax
import jax.numpy as jnp
from typing_extensions import override

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.plotting.colors import DARK_GREY
from fdtdx.objects.boundaries.boundary import BaseBoundary, BaseBoundaryState
from fdtdx.objects.boundaries.utils import (
    kappa_from_direction_axis,
    standard_sigma_from_direction_axis,
)


@autoinit
class PMLBoundaryState(BaseBoundaryState):
    """State container for PML boundary conditions.

    Stores the auxiliary field variables and coefficients needed to implement
    the convolutional perfectly matched layer (CPML) boundary conditions.
    """

    #: Auxiliary field for Ex component
    psi_Ex: jax.Array

    #: Auxiliary field for Ey component
    psi_Ey: jax.Array

    #: Auxiliary field for Ez component
    psi_Ez: jax.Array

    #:  Auxiliary field for Hx component
    psi_Hx: jax.Array

    #: Auxiliary field for Hy component
    psi_Hy: jax.Array

    #: Auxiliary field for Hz component
    psi_Hz: jax.Array

    #: Electric field scaling coefficient
    bE: jax.Array

    #: Magnetic field scaling coefficient
    bH: jax.Array

    #: Electric field update coefficient
    cE: jax.Array

    #: Magnetic field update coefficient
    cH: jax.Array

    #: PML stretching coefficient
    kappa: jax.Array


@autoinit
class PerfectlyMatchedLayer(BaseBoundary[PMLBoundaryState]):
    """Implements a Convolutional Perfectly Matched Layer (CPML) boundary condition.

    The CPML absorbs outgoing electromagnetic waves with minimal reflection by using
    a complex coordinate stretching approach. This implementation supports arbitrary
    axis orientation and both positive/negative directions.
    """

    #: Loss parameter for complex frequency shifting. Defaults to 1e-8.
    alpha: float = frozen_field(default=1.0e-8)

    #: Initial kappa stretching coefficient. Defaults to 1.0.
    kappa_start: float = frozen_field(default=1.0)

    #: Final kappa stretching coefficient. Defaults to 1.5.
    kappa_end: float = frozen_field(default=1.5)

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

        bE = jnp.exp(-self._config.courant_number * (sigma_E / kappa + self.alpha))
        bH = jnp.exp(-self._config.courant_number * (sigma_H / kappa + self.alpha))

        cE = (bE - 1) * sigma_E / (sigma_E * kappa + kappa**2 * self.alpha)
        cH = (bH - 1) * sigma_H / (sigma_H * kappa + kappa**2 * self.alpha)

        return dtype, bE, bH, cE, cH, kappa

    @override
    def init_state(
        self,
    ) -> PMLBoundaryState:
        dtype, bE, bH, cE, cH, kappa = self._get_dtype_update_coefficients()
        ext_shape = (3,) + self.grid_shape

        boundary_state = PMLBoundaryState(
            psi_Ex=jnp.zeros(shape=ext_shape, dtype=dtype),
            psi_Ey=jnp.zeros(shape=ext_shape, dtype=dtype),
            psi_Ez=jnp.zeros(shape=ext_shape, dtype=dtype),
            psi_Hx=jnp.zeros(shape=ext_shape, dtype=dtype),
            psi_Hy=jnp.zeros(shape=ext_shape, dtype=dtype),
            psi_Hz=jnp.zeros(shape=ext_shape, dtype=dtype),
            bE=bE.astype(dtype),
            bH=bH.astype(dtype),
            cE=cE.astype(dtype),
            cH=cH.astype(dtype),
            kappa=kappa.astype(dtype),
        )
        return boundary_state

    @override
    def reset_state(self, state: PMLBoundaryState) -> PMLBoundaryState:
        dtype, bE, bH, cE, cH, kappa = self._get_dtype_update_coefficients()

        new_state = PMLBoundaryState(
            psi_Ex=state.psi_Ex * 0,
            psi_Ey=state.psi_Ey * 0,
            psi_Ez=state.psi_Ez * 0,
            psi_Hx=state.psi_Hx * 0,
            psi_Hy=state.psi_Hy * 0,
            psi_Hz=state.psi_Hz * 0,
            bE=bE.astype(dtype),
            bH=bH.astype(dtype),
            cE=cE.astype(dtype),
            cH=cH.astype(dtype),
            kappa=kappa.astype(dtype),
        )
        return new_state

    @override
    def update_E_boundary_state(
        self,
        boundary_state: PMLBoundaryState,
        H: jax.Array,
    ) -> PMLBoundaryState:
        Hx = H[0, *self.grid_slice]
        Hy = H[1, *self.grid_slice]
        Hz = H[2, *self.grid_slice]

        psi_Ex = boundary_state.psi_Ex * boundary_state.bE
        psi_Ey = boundary_state.psi_Ey * boundary_state.bE
        psi_Ez = boundary_state.psi_Ez * boundary_state.bE

        psi_Ex = psi_Ex.at[1, :, 1:, :].add(
            (Hz[:, 1:, :] - Hz[:, :-1, :])
            * (boundary_state.cE[1, :, 1:, :] if self.axis == 1 else boundary_state.cE[1])
        )
        psi_Ex = psi_Ex.at[2, :, :, 1:].add(
            (Hy[:, :, 1:] - Hy[:, :, :-1])
            * (boundary_state.cE[2, :, :, 1:] if self.axis == 2 else boundary_state.cE[2])
        )

        psi_Ey = psi_Ey.at[2, :, :, 1:].add(
            (Hx[:, :, 1:] - Hx[:, :, :-1])
            * (boundary_state.cE[2, :, :, 1:] if self.axis == 2 else boundary_state.cE[2])
        )
        psi_Ey = psi_Ey.at[0, 1:, :, :].add(
            (Hz[1:, :, :] - Hz[:-1, :, :])
            * (boundary_state.cE[0, 1:, :, :] if self.axis == 0 else boundary_state.cE[0])
        )

        psi_Ez = psi_Ez.at[0, 1:, :, :].add(
            (Hy[1:, :, :] - Hy[:-1, :, :])
            * (boundary_state.cE[0, 1:, :, :] if self.axis == 0 else boundary_state.cE[0])
        )
        psi_Ez = psi_Ez.at[1, :, 1:, :].add(
            (Hx[:, 1:, :] - Hx[:, :-1, :])
            * (boundary_state.cE[1, :, 1:, :] if self.axis == 1 else boundary_state.cE[1])
        )

        boundary_state = boundary_state.at["psi_Ex"].set(psi_Ex)
        boundary_state = boundary_state.at["psi_Ey"].set(psi_Ey)
        boundary_state = boundary_state.at["psi_Ez"].set(psi_Ez)

        return boundary_state

    @override
    def update_H_boundary_state(
        self,
        boundary_state: PMLBoundaryState,
        E: jax.Array,
    ) -> PMLBoundaryState:
        Ex = E[0, *self.grid_slice]
        Ey = E[1, *self.grid_slice]
        Ez = E[2, *self.grid_slice]

        psi_Hx = boundary_state.psi_Hx * boundary_state.bH
        psi_Hy = boundary_state.psi_Hy * boundary_state.bH
        psi_Hz = boundary_state.psi_Hz * boundary_state.bH

        psi_Hx = psi_Hx.at[1, :, :-1, :].add(
            (Ez[:, 1:, :] - Ez[:, :-1, :])
            * (boundary_state.cH[1, :, :-1, :] if self.axis == 1 else boundary_state.cH[1])
        )
        psi_Hx = psi_Hx.at[2, :, :, :-1].add(
            (Ey[:, :, 1:] - Ey[:, :, :-1])
            * (boundary_state.cH[2, :, :, :-1] if self.axis == 2 else boundary_state.cH[2])
        )

        psi_Hy = psi_Hy.at[2, :, :, :-1].add(
            (Ex[:, :, 1:] - Ex[:, :, :-1])
            * (boundary_state.cH[2, :, :, :-1] if self.axis == 2 else boundary_state.cH[2])
        )
        psi_Hy = psi_Hy.at[0, :-1, :, :].add(
            (Ez[1:, :, :] - Ez[:-1, :, :])
            * (boundary_state.cH[0, :-1, :, :] if self.axis == 0 else boundary_state.cH[0])
        )

        psi_Hz = psi_Hz.at[0, :-1, :, :].add(
            (Ey[1:, :, :] - Ey[:-1, :, :])
            * (boundary_state.cH[0, :-1, :, :] if self.axis == 0 else boundary_state.cH[0])
        )
        psi_Hz = psi_Hz.at[1, :, :-1, :].add(
            (Ex[:, 1:, :] - Ex[:, :-1, :])
            * (boundary_state.cH[1, :, :-1, :] if self.axis == 1 else boundary_state.cH[1])
        )

        boundary_state = boundary_state.at["psi_Hx"].set(psi_Hx)
        boundary_state = boundary_state.at["psi_Hy"].set(psi_Hy)
        boundary_state = boundary_state.at["psi_Hz"].set(psi_Hz)

        return boundary_state

    @override
    def update_E(
        self,
        E: jax.Array,
        boundary_state: PMLBoundaryState,
        inverse_permittivity: jax.Array,
    ) -> jax.Array:
        phi_Ex = boundary_state.psi_Ex[1] - boundary_state.psi_Ex[2]
        phi_Ey = boundary_state.psi_Ey[2] - boundary_state.psi_Ey[0]
        phi_Ez = boundary_state.psi_Ez[0] - boundary_state.psi_Ez[1]
        phi_E = jnp.stack((phi_Ex, phi_Ey, phi_Ez), axis=0)

        E = E.at[:, *self.grid_slice].divide(boundary_state.kappa)
        inv_perm_slice = inverse_permittivity[self.grid_slice]
        update = self._config.courant_number * inv_perm_slice * phi_E
        E = E.at[:, *self.grid_slice].add(update)
        return E

    @override
    def update_H(
        self,
        H: jax.Array,
        boundary_state: PMLBoundaryState,
        inverse_permeability: jax.Array | float,
    ) -> jax.Array:
        phi_Hx = boundary_state.psi_Hx[1] - boundary_state.psi_Hx[2]
        phi_Hy = boundary_state.psi_Hy[2] - boundary_state.psi_Hy[0]
        phi_Hz = boundary_state.psi_Hz[0] - boundary_state.psi_Hz[1]
        phi_H = jnp.stack((phi_Hx, phi_Hy, phi_Hz), axis=0)

        H = H.at[:, *self.grid_slice].divide(boundary_state.kappa)
        if isinstance(inverse_permeability, jax.Array) and inverse_permeability.ndim > 0:
            inverse_permeability = inverse_permeability[self.grid_slice]
        update = -self._config.courant_number * inverse_permeability * phi_H
        H = H.at[:, *self.grid_slice].add(update)
        return H
