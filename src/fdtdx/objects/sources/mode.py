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
    mode_index: int = frozen_field(default=0)
    filter_pol: Literal["te", "tm"] | None = frozen_field(default=None)

    _inv_permittivity: jax.Array = private_field()
    _inv_permeability: jax.Array | float = private_field()

    def apply(
        self: Self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
    ) -> Self:
        self = super().apply(
            key=key,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
        )
        if (
            self.azimuth_angle != 0
            or self.elevation_angle != 0
            or self.max_angle_random_offset != 0
            or self.max_vertical_offset != 0
            or self.max_horizontal_offset != 0
        ):
            raise NotImplementedError()

        inv_permittivity_slice = inv_permittivities[*self.grid_slice]
        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            inv_permeability_slice = inv_permeabilities[*self.grid_slice]
        else:
            inv_permeability_slice = inv_permeabilities

        self = self.aset("_inv_permittivity", inv_permittivity_slice, create_new_ok=True)
        self = self.aset("_inv_permeability", inv_permeability_slice, create_new_ok=True)

        return self

    def get_EH_variation(
        self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
    ) -> tuple[
        jax.Array,  # E: (3, *grid_shape)
        jax.Array,  # H: (3, *grid_shape)
        jax.Array,  # time_offset_E: (3, *grid_shape)
        jax.Array,  # time_offset_H: (3, *grid_shape)
    ]:
        del key

        center = jnp.asarray(
            [round(self.grid_shape[self.horizontal_axis]), round(self.grid_shape[self.vertical_axis])], dtype=jnp.int32
        )

        inv_permittivity_slice = inv_permittivities[*self.grid_slice]
        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            inv_permeability_slice = inv_permeabilities[*self.grid_slice]
        else:
            inv_permeability_slice = inv_permeabilities

        raw_wave_vector = get_wave_vector_raw(
            direction=self.direction,
            propagation_axis=self.propagation_axis,
        )

        # compute mode
        mode_E, mode_H, eff_index = compute_mode(
            frequency=self.wave_character.frequency,
            inv_permittivities=inv_permittivity_slice,
            inv_permeabilities=inv_permeability_slice,  # type: ignore
            resolution=self._config.resolution,
            direction=self.direction,
            mode_index=self.mode_index,
            filter_pol=self.filter_pol,
        )

        time_offset_E, time_offset_H = calculate_time_offset_yee(
            center=center,
            wave_vector=raw_wave_vector,
            inv_permittivities=inv_permittivity_slice,
            inv_permeabilities=jnp.ones_like(inv_permeability_slice),
            resolution=self._config.resolution,
            time_step_duration=self._config.time_step_duration,
            effective_index=jnp.real(eff_index),
        )

        return mode_E, mode_H, time_offset_E, time_offset_H

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
