from pathlib import Path
from typing import Literal, Self

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from fdtdx.core.grid import calculate_time_offset_yee
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.linalg import get_wave_vector_raw
from fdtdx.core.physics.metrics import compute_energy
from fdtdx.core.physics.modes import compute_mode
from fdtdx.objects.sources.tfsf import TFSFPlaneSource


@autoinit
class ModePlaneSource(TFSFPlaneSource):
    #: index of the mode
    mode_index: int = frozen_field(default=0)

    #: a literal value 'te', 'tm' to filter
    filter_pol: Literal["te", "tm"] | None = frozen_field(default=None)

    _inv_permittivity: jax.Array = private_field()
    _inv_permeability: jax.Array | float = private_field()

    _neff: jax.Array = private_field()  # not required for sim, used for inspection

    def _local_edge_coordinates(self) -> tuple[jax.Array, jax.Array, jax.Array] | None:
        """Return local physical edge coordinates for this source slice.

        Non-uniform mode sources need edge coordinates for both Tidy3D mode
        solving and Yee time offsets.  Coordinates are shifted so the source
        slice lower corner is at zero on each axis.
        """
        grid = self._config.realized_grid
        if grid is None:
            return None

        local_edges = []
        for axis in range(3):
            lower, upper = self.grid_slice_tuple[axis]
            edges = grid.edges(axis)[lower : upper + 1]
            local_edges.append(edges - edges[0])
        return tuple(local_edges)

    def _transverse_edge_coordinates(self) -> tuple[jax.Array, jax.Array] | None:
        """Return local transverse edge coordinates for Tidy3D mode solving."""
        local_edges = self._local_edge_coordinates()
        if local_edges is None:
            return None
        return tuple(local_edges[axis] for axis in range(3) if axis != self.propagation_axis)

    def _mode_solver_resolution(self) -> float:
        """Return scalar resolution only for legacy uniform mode-solver setup.

        ``compute_mode`` ignores this value when explicit transverse coordinates
        are provided, but the argument remains part of the compatibility API.
        """
        if self._config.has_nonuniform_grid:
            assert self._config.realized_grid is not None
            return self._config.realized_grid.min_spacing
        return self._config.require_uniform_grid()

    def _source_center_physical(self) -> jax.Array | None:
        """Return the physical source center for grid-aware Yee time offsets."""
        local_edges = self._local_edge_coordinates()
        if local_edges is None:
            return None
        center = []
        for axis, edges in enumerate(local_edges):
            if axis == self.propagation_axis:
                center.append(jnp.asarray(0.0, dtype=self._config.dtype))
            else:
                center.append(0.5 * edges[-1])
        return jnp.asarray(center, dtype=self._config.dtype)

    def apply(
        self: Self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
    ) -> Self:
        del key
        if (
            self.azimuth_angle != 0
            or self.elevation_angle != 0
            or self.max_angle_random_offset != 0
            or self.max_vertical_offset != 0
            or self.max_horizontal_offset != 0
        ):
            raise NotImplementedError()

        # inv_permittivities shape: (3, Nx, Ny, Nz) - slice with component dimension
        inv_permittivity_slice = inv_permittivities[:, *self.grid_slice]
        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            # inv_permeabilities shape: (3, Nx, Ny, Nz) - slice with component dimension
            inv_permeability_slice = inv_permeabilities[:, *self.grid_slice]
        else:
            inv_permeability_slice = inv_permeabilities

        self = self.aset("_inv_permittivity", inv_permittivity_slice, create_new_ok=True)
        self = self.aset("_inv_permeability", inv_permeability_slice, create_new_ok=True)

        # compute mode
        mode_E, mode_H, eff_index = compute_mode(
            frequency=self.wave_character.get_frequency(),
            inv_permittivities=inv_permittivity_slice,
            inv_permeabilities=inv_permeability_slice,
            resolution=self._mode_solver_resolution(),
            direction=self.direction,
            mode_index=self.mode_index,
            filter_pol=self.filter_pol,
            dtype=self._config.dtype,
            transverse_coords=self._transverse_edge_coordinates(),
        )
        mode_E, mode_H = jnp.real(mode_E), jnp.real(mode_H)

        self = self.aset("_E", mode_E, create_new_ok=True)
        self = self.aset("_H", mode_H, create_new_ok=True)
        self = self.aset("_neff", eff_index, create_new_ok=True)

        center = jnp.asarray(
            [round(self.grid_shape[self.horizontal_axis]), round(self.grid_shape[self.vertical_axis])], dtype=jnp.int32
        )
        raw_wave_vector = get_wave_vector_raw(
            direction=self.direction,
            propagation_axis=self.propagation_axis,
            dtype=self._config.dtype,
        )
        time_offset_E, time_offset_H = calculate_time_offset_yee(
            center=center,
            wave_vector=raw_wave_vector,
            inv_permittivities=inv_permittivity_slice,
            inv_permeabilities=jnp.ones_like(inv_permeability_slice),
            resolution=self._mode_solver_resolution(),
            time_step_duration=self._config.time_step_duration,
            effective_index=jnp.real(eff_index),
            coordinate_edges=self._local_edge_coordinates(),
            center_physical=self._source_center_physical(),
        )

        self = self.aset("_time_offset_E", time_offset_E, create_new_ok=True)
        self = self.aset("_time_offset_H", time_offset_H, create_new_ok=True)

        return self

    def plot(self, save_path: str | Path):
        if self._H is None or self._E is None:
            raise Exception("Cannot plot mode without init to grid and apply params first")
        energy = compute_energy(
            E=self._E,
            H=self._H,
            inv_permittivity=self._inv_permittivity,
            inv_permeability=self._inv_permeability,
        )

        energy_2d = energy.squeeze().T

        plt.clf()
        fig = plt.figure(figsize=(10, 10))
        mode_cmap = "inferno"

        im = plt.imshow(
            energy_2d,
            cmap=mode_cmap,
            origin="lower",
            aspect="equal",
        )
        plt.colorbar(im)

        # Ensure the plot takes up the entire figure
        plt.tight_layout(pad=0)

        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
