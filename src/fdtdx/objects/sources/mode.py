from pathlib import Path
from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from fdtdx import constants
from fdtdx.core.axis import get_transverse_axes
from fdtdx.core.grid import calculate_time_offset_yee
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.linalg import get_wave_vector_raw
from fdtdx.core.physics.metrics import compute_energy
from fdtdx.core.physics.modes import compute_mode
from fdtdx.dispersion import effective_complex_inv_permittivity, effective_inv_permittivity
from fdtdx.objects.sources.tfsf import TFSFPlaneSource, _build_dispersive_H_filter


@autoinit
class ModePlaneSource(TFSFPlaneSource):
    #: index of the mode
    mode_index: int = frozen_field(default=0)

    #: a literal value 'te', 'tm' to filter
    filter_pol: Literal["te", "tm"] | None = frozen_field(default=None)

    #: Symmetry-plane condition at the min edge of each transverse axis (the two
    #: non-propagation physical axes, in increasing-index order): ``0`` = PEC
    #: mirror (electric wall, the default), ``1`` = PMC mirror (magnetic wall).
    #: Set this when the source plane's waveguide lies on a symmetry plane of a
    #: reduced (half/quarter) domain, so the mode solver imposes the same wall
    #: the FDTD uses there (e.g. ``(0, 1)`` for PEC at y=0 and PMC at the z
    #: Si-mid plane of a +x-propagating quarter domain).
    symmetry: tuple[int, int] = frozen_field(default=(0, 0))

    _inv_permittivity: jax.Array = private_field()
    _inv_permeability: jax.Array | float = private_field()

    _neff: jax.Array = private_field()  # not required for sim, used for inspection

    def _local_edge_coordinates(self) -> tuple[jax.Array, jax.Array, jax.Array] | None:
        """Return local physical edge coordinates for this source slice.

        Non-uniform mode sources need edge coordinates for both Tidy3D mode
        solving and Yee time offsets.  Coordinates are shifted so the source
        slice lower corner is at zero on each axis.
        """
        grid = self._config.resolved_grid
        if grid is None:
            return None

        local_edges = []
        for axis in range(3):
            lower, upper = self.grid_slice_tuple[axis]
            edges = grid.edges(axis)[lower : upper + 1]
            local_edges.append(edges - edges[0])
        e0, e1, e2 = local_edges
        return e0, e1, e2

    def _transverse_edge_coordinates(self) -> tuple[jax.Array, jax.Array] | None:
        """Return local transverse edge coordinates for Tidy3D mode solving."""
        local_edges = self._local_edge_coordinates()
        if local_edges is None:
            return None
        axes = get_transverse_axes(self.propagation_axis)
        return local_edges[axes[0]], local_edges[axes[1]]

    def _mode_solver_resolution(self) -> float:
        """Return scalar resolution only for legacy uniform mode-solver setup.

        ``compute_mode`` ignores this value when explicit transverse coordinates
        are provided, but the argument remains part of the compatibility API.
        """
        if self._config.has_nonuniform_grid:
            assert self._config.resolved_grid is not None
            return self._config.resolved_grid.min_spacing
        return self._config.uniform_spacing()

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
        dispersive_c1: jax.Array | None = None,
        dispersive_c2: jax.Array | None = None,
        dispersive_c3: jax.Array | None = None,
        electric_conductivity: jax.Array | None = None,
        dispersive_c4: jax.Array | None = None,
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
        c1_slice = c2_slice = c3_slice = c4_slice = None
        if dispersive_c1 is not None and dispersive_c2 is not None and dispersive_c3 is not None:
            c1_slice = dispersive_c1[:, :, *self.grid_slice]
            c2_slice = dispersive_c2[:, :, *self.grid_slice]
            c3_slice = dispersive_c3[:, :, *self.grid_slice]
            c4_slice = None if dispersive_c4 is None else dispersive_c4[:, :, *self.grid_slice]
            inv_permittivity_slice = effective_inv_permittivity(
                inv_eps=inv_permittivity_slice,
                c1=c1_slice,
                c2=c2_slice,
                c3=c3_slice,
                omega=2.0 * np.pi * self.wave_character.get_frequency(),
                dt=self._config.time_step_duration,
                c4=c4_slice,
            )

        self = self.aset("_inv_permittivity", inv_permittivity_slice, create_new_ok=True)
        self = self.aset("_inv_permeability", inv_permeability_slice, create_new_ok=True)

        # Permittivity handed to the mode solver: the FULL complex epsilon at the
        # carrier frequency (eps_inf + chi(omega) + i*sigma/(eps0*omega)), so the
        # solved mode profile and effective index reflect material loss. This is
        # kept separate from inv_permittivity_slice above, which stays real for the
        # impedance/energy normalization — using the imaginary part there would
        # double-count the absorption already integrated by the FDTD update.
        sigma_slice = None if electric_conductivity is None else electric_conductivity[:, *self.grid_slice]
        mode_inv_permittivity = inv_eps_inf_slice
        if sigma_slice is not None or c1_slice is not None:
            mode_inv_permittivity = effective_complex_inv_permittivity(
                inv_eps=inv_eps_inf_slice,
                omega=2.0 * np.pi * self.wave_character.get_frequency(),
                dt=self._config.time_step_duration,
                c1=c1_slice,
                c2=c2_slice,
                c3=c3_slice,
                c4=c4_slice,
                electric_conductivity=sigma_slice,
                conductivity_spacing=(
                    None
                    if sigma_slice is None
                    else constants.c * self._config.time_step_duration / self._config.courant_number
                ),
            )

        # compute mode
        mode_E, mode_H, eff_index = compute_mode(
            frequency=self.wave_character.get_frequency(),
            inv_permittivities=mode_inv_permittivity,
            inv_permeabilities=inv_permeability_slice,
            resolution=self._mode_solver_resolution(),
            direction=self.direction,
            mode_index=self.mode_index,
            filter_pol=self.filter_pol,
            dtype=self._config.dtype,
            symmetry=self.symmetry,
            transverse_coords=self._transverse_edge_coordinates(),
        )
        # Keep the complex modal fields when the mode was solved against a lossy
        # (conductivity) permittivity, so the launched source carries the
        # eigenmode's transverse phase — TFSFPlaneSource.update_E/update_H inject
        # the complex profile via a quadrature (cos/sin) decomposition. Lossless
        # modes are projected to real (bit-identical to before). The dispersive
        # path also stays real here because its broadband H-filter assumes a real
        # temporal profile.
        keep_complex_mode = sigma_slice is not None and c1_slice is None
        if not keep_complex_mode:
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
                dtype=self._config.dtype,
                c4_slice=c4_slice,
            )
            self = self.aset("_temporal_H_filter", filtered, create_new_ok=True)
        else:
            # Reused source applied in a non-dispersive context: clear any stale
            # filter from a previous dispersive apply.
            self = self.aset("_temporal_H_filter", None, create_new_ok=True)

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
