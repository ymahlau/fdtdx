from pathlib import Path
from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from fdtdx.core.grid import calculate_time_offset_yee
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.linalg import get_wave_vector_raw
from fdtdx.core.physics.metrics import compute_energy
from fdtdx.core.physics.modes import compute_mode
from fdtdx.dispersion import effective_inv_permittivity
from fdtdx.objects.sources.tfsf import TFSFPlaneSource, _build_dispersive_H_filter


@autoinit
class ModePlaneSource(TFSFPlaneSource):
    #: index of the mode
    mode_index: int = frozen_field(default=0)

    #: a literal value 'te', 'tm' to filter
    filter_pol: Literal["te", "tm"] | None = frozen_field(default=None)

    _inv_permittivity: jax.Array = private_field()
    _inv_permeability: jax.Array | float = private_field()

    _neff: jax.Array = private_field()  # not required for sim, used for inspection

    def apply(
        self: Self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        dispersive_c1: jax.Array | None = None,
        dispersive_c2: jax.Array | None = None,
        dispersive_c3: jax.Array | None = None,
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

        # Preserve the raw ε∞ slice before the carrier-frequency correction —
        # the broadband impedance filter needs ε∞ to reconstruct ε(ω).
        inv_eps_inf_slice = inv_permittivity_slice

        # Frequency-correct the permittivity seen by the mode solver so that
        # mode profiles computed inside a dispersive medium reflect the true
        # epsilon at the carrier frequency, not epsilon_infinity.
        c1_slice = c2_slice = c3_slice = None
        if dispersive_c1 is not None and dispersive_c2 is not None and dispersive_c3 is not None:
            c1_slice = dispersive_c1[:, :, *self.grid_slice]
            c2_slice = dispersive_c2[:, :, *self.grid_slice]
            c3_slice = dispersive_c3[:, :, *self.grid_slice]
            inv_permittivity_slice = effective_inv_permittivity(
                inv_eps=inv_permittivity_slice,
                c1=c1_slice,
                c2=c2_slice,
                c3=c3_slice,
                omega=2.0 * np.pi * self.wave_character.get_frequency(),
                dt=self._config.time_step_duration,
            )

        self = self.aset("_inv_permittivity", inv_permittivity_slice, create_new_ok=True)
        self = self.aset("_inv_permeability", inv_permeability_slice, create_new_ok=True)

        # compute mode
        mode_E, mode_H, eff_index = compute_mode(
            frequency=self.wave_character.get_frequency(),
            inv_permittivities=inv_permittivity_slice,
            inv_permeabilities=inv_permeability_slice,
            resolution=self._config.resolution,
            direction=self.direction,
            mode_index=self.mode_index,
            filter_pol=self.filter_pol,
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

        self = self.aset("_time_offset_E", time_offset_E, create_new_ok=True)
        self = self.aset("_time_offset_H", time_offset_H, create_new_ok=True)

        # Broadband impedance correction for dispersive media. The mode solver
        # above used ε(ω_c), so the resulting H profile already carries the
        # correct scalar impedance at the carrier frequency. For a broadband
        # pulse the medium's ε(ω) varies across the source spectrum, which
        # mismatches η away from ω_c and radiates spurious reflections through
        # the TFSF surface. Precompute a filtered H-side temporal profile
        # whose spectrum is S(ω)·√(ε(ω)/ε(ω_c)) to bake in the frequency-
        # dependent correction.
        #
        # Note: bulk ε(ω) is averaged uniformly over the source cells; this
        # does not capture geometric modal dispersion (the fact that a
        # waveguide mode's effective index also depends on frequency).
        if c1_slice is not None and c2_slice is not None and c3_slice is not None:
            filtered = _build_dispersive_H_filter(
                temporal_profile=self.temporal_profile,
                wave_character=self.wave_character,
                dt=self._config.time_step_duration,
                num_time_steps=self._config.time_steps_total,
                c1_slice=c1_slice,
                c2_slice=c2_slice,
                c3_slice=c3_slice,
                inv_eps_inf_slice=inv_eps_inf_slice,
            )
            self = self.aset("_temporal_H_filter", filtered, create_new_ok=True)

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
