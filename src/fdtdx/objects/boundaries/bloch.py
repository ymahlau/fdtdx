import functools
from typing import cast

import jax
import jax.numpy as jnp
from typing_extensions import override

from fdtdx.colors import XKCD_WARM_PURPLE, Color
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.objects.boundaries.boundary import BaseBoundary


@autoinit
class BlochBoundary(BaseBoundary):
    """Implements Bloch periodic boundary conditions.

    Generalizes periodic boundary conditions with a phase shift:
        F(x + L) = F(x) * exp(i * k_bloch * L)

    When the Bloch vector is zero, this is equivalent to a standard periodic
    boundary. Complex-valued field arrays are only required when the Bloch
    vector has non-zero components.
    """

    #: Bloch wave vector components (k_x, k_y, k_z) in units of rad/m.
    bloch_vector: tuple[float, float, float] = frozen_field(default=(0.0, 0.0, 0.0))

    #: RGB color tuple for visualization. Defaults to warm purple.
    color: Color | None = frozen_field(default=XKCD_WARM_PURPLE)

    @property
    def needs_complex_fields(self) -> bool:
        """Whether this boundary requires complex-valued fields.

        Only True when the Bloch vector component along this boundary's axis
        is non-zero.
        """
        return self.bloch_vector[self.axis] != 0.0

    @property
    @override
    def descriptive_name(self) -> str:
        """Gets a human-readable name describing this Bloch boundary's location."""
        axis_str = "x" if self.axis == 0 else "y" if self.axis == 1 else "z"
        direction_str = "min" if self.direction == "-" else "max"
        return f"{direction_str}_{axis_str}"

    @property
    @override
    def uses_wrap_padding(self) -> bool:
        """Bloch boundaries use wrap padding (with Bloch phase correction applied separately)."""
        return True

    @property
    @override
    def thickness(self) -> int:
        """Gets the thickness of the Bloch boundary layer in grid points (always 1)."""
        return 1

    @override
    def apply_pad_correction(
        self, padded_fields: jax.Array, volume_shape: tuple[int, int, int], resolution: float
    ) -> jax.Array:
        """Apply Bloch phase shift to ghost cells of padded fields.

        For the '-' direction boundary: left ghost cell (index 0 on padded axis)
        is multiplied by conj(phase).
        For the '+' direction boundary: right ghost cell (index -1 on padded axis)
        is multiplied by phase.

        Args:
            padded_fields: Padded field array of shape (3, Nx+2, Ny+2, Nz+2)
            volume_shape: Full simulation volume shape (Nx, Ny, Nz)
            resolution: Grid resolution in meters

        Returns:
            Padded fields with Bloch phase corrections applied
        """
        if not self.needs_complex_fields:
            return padded_fields
        phase = self.get_bloch_phase(volume_shape, resolution)
        # padded axis index is self.axis + 1 (field arrays have leading component dim)
        ax = self.axis + 1
        if self.direction == "-":
            # Left ghost wraps from the right end: multiply by conj(phase)
            idx = cast(list[slice | int], [slice(None)] * padded_fields.ndim)
            idx[ax] = 0
            idx_tuple = tuple(idx)
            padded_fields = padded_fields.at[idx_tuple].set(padded_fields[idx_tuple] * jnp.conj(phase))
        else:
            # Right ghost wraps from the left end: multiply by phase
            idx = cast(list[slice | int], [slice(None)] * padded_fields.ndim)
            idx[ax] = -1
            idx_tuple = tuple(idx)
            padded_fields = padded_fields.at[idx_tuple].set(padded_fields[idx_tuple] * phase)
        return padded_fields

    @override
    def apply_field_reset(self, fields: dict[str, jax.Array]) -> dict[str, jax.Array]:
        """Copy field values from this boundary face to maintain periodicity."""
        result = {}
        for name, field in fields.items():
            field_values = field[..., *self.boundary_slice]
            result[name] = field.at[..., *self.grid_slice].set(field_values)
        return result

    @functools.cached_property
    def boundary_slice(self) -> tuple[slice, ...]:
        """Get the slice for the current boundary."""
        boundary_slice = list(self.grid_slice)
        if self.direction == "+":
            boundary_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][0], self._grid_slice_tuple[self.axis][0] + 1
            )
        else:
            boundary_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][1] - 1, self._grid_slice_tuple[self.axis][1]
            )
        return tuple(boundary_slice)

    @functools.cached_property
    def opposite_slice(self) -> tuple[slice, ...]:
        """Get the slice for the opposite boundary."""
        opposite_slice = list(self.grid_slice)
        if self.direction == "+":
            opposite_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][1] - 1, self._grid_slice_tuple[self.axis][1]
            )
        else:
            opposite_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][0], self._grid_slice_tuple[self.axis][0] + 1
            )
        return tuple(opposite_slice)

    def get_bloch_phase(self, volume_shape: tuple[int, int, int], resolution: float) -> jax.Array:
        """Compute the complex phase factor exp(i * k_bloch * L) for this axis.

        The right ghost cell (wrapping from the left side) is multiplied by this phase.
        The left ghost cell (wrapping from the right side) is multiplied by the conjugate.

        Args:
            volume_shape: Full simulation volume shape (Nx, Ny, Nz)
            resolution: Grid resolution in meters

        Returns:
            Complex scalar exp(i * k_axis * L) where L = volume_shape[axis] * resolution
        """
        k = self.bloch_vector[self.axis]
        L = volume_shape[self.axis] * resolution
        return jnp.exp(1j * k * L)
