from typing import Literal, Sequence, Tuple

import jax
import jax.numpy as jnp

from fdtdx import constants
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.objects.detectors.detector import Detector, DetectorState


@autoinit
class DiffractiveDetector(Detector):
    """Detector for computing Fourier transforms of fields at specific frequencies and diffraction orders.

    This detector computes field amplitudes for specific diffraction orders and frequencies through
    a specified plane in the simulation volume. It can measure diffraction in either positive or negative
    direction along the propagation axis.

    Attributes:
        frequencies (Sequence[float]): List of frequencies to analyze (in Hz)
        direction (Literal["+", "-"]): Either "+" or "-" for positive or negative direction.
        orders (Sequence[Tuple[int, int]], optional): Tuple of (nx, ny) pairs specifying diffraction orders to compute
        dtype (jnp.dtype, optional): Data type of the saved data.
    """

    frequencies: Sequence[float] = frozen_field()
    direction: Literal["+", "-"] = frozen_field()
    orders: Sequence[Tuple[int, int]] = frozen_field(default=((0, 0),))
    dtype: jnp.dtype = frozen_field(default=jnp.complex64)

    def __post_init__(self):
        if self.dtype not in [jnp.complex64, jnp.complex128]:
            raise Exception(f"Invalid dtype in DiffractiveDetector: {self.dtype}")

    @property
    def propagation_axis(self) -> int:
        """Determines the axis along which diffraction is measured.

        The propagation axis is identified as the dimension with size 1 in the
        detector's grid shape, representing a plane perpendicular to the diffraction
        measurement direction.

        Returns:
            int: Index of the propagation axis (0 for x, 1 for y, 2 for z)

        Raises:
            Exception: If detector shape does not have exactly one dimension of size 1
        """
        if sum([a == 1 for a in self.grid_shape]) != 1:
            raise Exception(f"Invalid diffractive detector shape: {self.grid_shape}")
        return self.grid_shape.index(1)

    # def _validate_orders(self, wavelength: float) -> None:
    #     """Validate that requested diffraction orders are physically realizable.

    #     Args:
    #         wavelength: Wavelength of the light in meters

    #     Raises:
    #         Exception: If any requested order is not physically realizable
    #     """
    #     if self._Nx is None:
    #         raise Exception("Order info not yet computed. Run update first.")

    #     # Maximum possible orders based on grid
    #     max_nx = self._Nx // 2
    #     max_ny = self._Ny // 2

    #     # Check Nyquist limits for all orders at once
    #     nx_valid = jnp.all(jnp.abs(jnp.array([o[0] for o in self.orders])) <= max_nx)
    #     ny_valid = jnp.all(jnp.abs(jnp.array([o[1] for o in self.orders])) <= max_ny)

    #     if not (nx_valid and ny_valid):
    #         raise Exception(f"Some orders exceed Nyquist limit for grid size ({self._Nx}, {self._Ny})")

    #     # Check physical realizability for all orders at once
    #     k0 = 2 * jnp.pi / wavelength
    #     kt_squared = self._kx_normalized**2 + self._ky_normalized**2

    #     if jnp.any(kt_squared > k0**2):
    #         raise Exception(f"Some orders are evanescent at wavelength {wavelength*1e9:.1f}nm")

    def _shape_dtype_single_time_step(self) -> dict[str, jax.ShapeDtypeStruct]:
        num_freqs = len(self.frequencies)
        num_orders = len(self.orders)

        shape = (num_freqs, num_orders)

        # Ensure we're using a complex dtype
        field_dtype = jnp.complex128 if self.dtype == jnp.float64 else jnp.complex64
        return {"diffractive": jax.ShapeDtypeStruct(shape=shape, dtype=field_dtype)}

    def _num_latent_time_steps(self) -> int:
        return 1

    def update(
        self,
        time_step: jax.Array,
        E: jax.Array,
        H: jax.Array,
        state: DetectorState,
        inv_permittivity: jax.Array,
        inv_permeability: jax.Array | float,
    ) -> DetectorState:
        del inv_permittivity, inv_permeability

        # Get grid dimensions for the plane perpendicular to propagation axis
        prop_axis = self.propagation_axis
        plane_dims = [i for i in range(3) if i != prop_axis]
        Nx, Ny = [self.grid_shape[i] for i in plane_dims]

        # Get current field values at the specified plane
        cur_E = E[:, *self.grid_slice]  # Shape: (3, nx, ny, 1)
        cur_H = H[:, *self.grid_slice]  # Shape: (3, nx, ny, 1)

        # Remove the normal axis dimension since it should be 1
        cur_E = jnp.squeeze(cur_E, axis=prop_axis + 1)  # Shape: (3, nx, ny)
        cur_H = jnp.squeeze(cur_H, axis=prop_axis + 1)  # Shape: (3, nx, ny)

        # Compute FFT of each field component
        E_k = jnp.fft.fft2(cur_E, axes=tuple(d + 1 for d in plane_dims))  # FFT in spatial dimensions
        H_k = jnp.fft.fft2(cur_H, axes=tuple(d + 1 for d in plane_dims))

        # Convert orders to array for vectorization
        orders = jnp.array(self.orders)  # Shape: (num_orders, 2)

        # Compute FFT indices for all orders
        kx_indices = jnp.where(orders[:, 0] >= 0, orders[:, 0], Nx + orders[:, 0])
        ky_indices = jnp.where(orders[:, 1] >= 0, orders[:, 1], Ny + orders[:, 1])

        # Compute wavevectors
        dx = dy = self._config.resolution
        kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, dx)
        ky = 2 * jnp.pi * jnp.fft.fftfreq(Ny, dy)
        k0 = 2 * jnp.pi * self.frequencies[0] / constants.c  # Use first frequency for now

        # For each requested order, compute the diffracted power
        order_amplitudes = []
        for kx_idx, ky_idx in zip(kx_indices, ky_indices):
            # Get the field components for this k-point
            E_order = E_k[:, kx_idx, ky_idx]
            H_order = H_k[:, kx_idx, ky_idx]

            # Compute kz for propagating waves
            kz = jnp.sqrt(k0**2 - kx[kx_idx] ** 2 - ky[ky_idx] ** 2 + 0j)
            k_vec = jnp.array([kx[kx_idx], ky[ky_idx], kz])

            # Project fields to be transverse to k
            E_t = E_order - jnp.dot(E_order, k_vec) * k_vec / jnp.dot(k_vec, k_vec)
            H_t = H_order - jnp.dot(H_order, k_vec) * k_vec / jnp.dot(k_vec, k_vec)

            # Compute power in this order
            P_order = jnp.abs(jnp.cross(E_t, jnp.conj(H_t)).sum())
            if self.direction == "-":
                P_order = -P_order
            order_amplitudes.append(P_order)

        order_amplitudes = jnp.array(order_amplitudes)

        # Time domain analysis - vectorized for all frequencies
        t = time_step * self._config.time_step_duration
        angular_frequencies = 2 * jnp.pi * jnp.array(self.frequencies)
        phase_angles = angular_frequencies[:, None] * t  # Shape: (num_freqs, 1)
        phasors = jnp.exp(-1j * phase_angles)  # Shape: (num_freqs, 1)

        # Compute all frequency components for all orders at once
        order_amplitudes = order_amplitudes[None, :]  # Shape: (1, num_orders)
        new_values = order_amplitudes * phasors  # Shape: (num_freqs, num_orders)

        # Update state
        arr_idx = self._time_step_to_arr_idx[time_step]
        new_state = state.copy()
        new_state["diffractive"] = new_state["diffractive"].at[arr_idx].set(new_values)

        return new_state
