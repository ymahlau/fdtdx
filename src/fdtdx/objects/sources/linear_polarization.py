from abc import ABC, abstractmethod
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from fdtdx.core.grid import calculate_time_offset_yee
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.misc import (
    linear_interpolated_indexing,
    normalize_polarization_for_source,
    tilted_polarization_vectors,
)
from fdtdx.core.physics.metrics import compute_energy
from fdtdx.core.physics.symmetry import field_component_parity
from fdtdx.dispersion import effective_inv_permittivity
from fdtdx.objects.sources.tfsf import TFSFPlaneSource, _build_dispersive_H_filter, _source_impedance


def _linear_interpolate_rectilinear_2d(
    point: jax.Array,
    x_coords: jax.Array,
    y_coords: jax.Array,
    values: jax.Array,
) -> jax.Array:
    """Bilinearly interpolate ``values`` sampled on rectilinear cell centers.

    The tilted-source projection for non-uniform grids works in physical
    transverse coordinates rather than legacy index coordinates.  This helper is
    intentionally small and JAX-friendly: it finds the local bracketing centers
    on each axis, clamps outside samples to the nearest center, and forms the
    separable bilinear blend.  Clamping matches the practical behavior of the
    legacy index-space interpolation near finite source boundaries.
    """

    def axis_weights(coords: jax.Array, val: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        if coords.shape[0] == 1:
            zero = jnp.asarray(0, dtype=jnp.int32)
            return zero, zero, jnp.asarray(1.0, dtype=val.dtype), jnp.asarray(0.0, dtype=val.dtype)
        upper = jnp.searchsorted(coords, val, side="right")
        upper = jnp.clip(upper, 1, coords.shape[0] - 1)
        lower = upper - 1
        lower_coord = coords[lower]
        upper_coord = coords[upper]
        fraction = jnp.where(upper_coord == lower_coord, 0.0, (val - lower_coord) / (upper_coord - lower_coord))
        fraction = jnp.clip(fraction, 0.0, 1.0)
        return lower, upper, 1.0 - fraction, fraction

    x0, x1, wx0, wx1 = axis_weights(x_coords, point[0])
    y0, y1, wy0, wy1 = axis_weights(y_coords, point[1])
    return (
        wx0 * wy0 * values[x0, y0]
        + wx0 * wy1 * values[x0, y1]
        + wx1 * wy0 * values[x1, y0]
        + wx1 * wy1 * values[x1, y1]
    )


@autoinit
class LinearlyPolarizedPlaneSource(TFSFPlaneSource, ABC):
    #: the electric polarization vector
    fixed_E_polarization_vector: tuple[float, float, float] | None = frozen_field(default=None)

    #: the magnetic polarization vector
    fixed_H_polarization_vector: tuple[float, float, float] | None = frozen_field(default=None)

    #: whether to normalize the polarization vector
    normalize_by_energy: bool = frozen_field(default=True)

    def _local_edge_coordinates(self) -> tuple[jax.Array, jax.Array, jax.Array] | None:
        """Return source-local physical edge coordinates for Yee metrics.

        Coordinates are shifted so the lower corner of this source slice is the
        local origin.  Uniform grids can use the legacy scalar path, but
        non-uniform grids need these explicit edge arrays for time-of-flight
        corrections and physical profile sampling.
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

    def _source_center_physical(self, source_center: jax.Array) -> jax.Array | None:
        """Return the physical center used for grid-aware Yee time offsets."""
        local_edges = self._local_edge_coordinates()
        if local_edges is None:
            return None
        physical_center = []
        for axis, edges in enumerate(local_edges):
            if axis == self.propagation_axis:
                physical_center.append(jnp.asarray(0.0, dtype=self._config.dtype))
            elif self._config.has_nonuniform_grid:
                center_axis = 0 if axis == self.horizontal_axis else 1
                physical_center.append(jnp.asarray(source_center[center_axis], dtype=self._config.dtype))
            else:
                transverse_center = source_center[0] if axis == self.horizontal_axis else source_center[1]
                physical_center.append(transverse_center * self._source_resolution())
        return jnp.asarray(physical_center, dtype=self._config.dtype)

    def _source_resolution(self) -> float:
        """Return scalar spacing only for legacy source APIs.

        ``calculate_time_offset_yee`` ignores this value when explicit
        ``coordinate_edges`` are provided.  The min-spacing fallback keeps the
        call signature usable for rectilinear grids without pretending the mesh
        is uniform.
        """
        if self._config.has_nonuniform_grid:
            assert self._config.resolved_grid is not None
            return self._config.resolved_grid.min_spacing
        return self._config.uniform_spacing()

    def _uses_physical_source_coordinates(self) -> bool:
        """Whether transverse source coordinates are represented in metres."""
        return self._config.has_nonuniform_grid

    def validate_placement(self, objects) -> list[str]:
        """Warn when the polarization is incompatible with the symmetry walls it crosses.

        The injected transverse profile is mirror-*even* about every symmetry plane the source
        straddles. A wall makes some field components even and others odd (see
        :func:`~fdtdx.core.physics.symmetry.field_component_parity`), so if a nonzero component is odd there,
        the wall drives it toward zero and the reduced simulation models a different field than the
        user drew. Picking the wall type per polarization is the single most common mistake with
        ``config.symmetry``, and it is silent otherwise: this reports it with the fix.
        """
        errors = list(super().validate_placement(objects))
        straddled = [a for a in self.transverse_axes if self.straddles_symmetry_plane(a)]
        if not straddled:
            return errors

        e_pol, h_pol = normalize_polarization_for_source(
            direction=self.direction,
            propagation_axis=self.propagation_axis,
            fixed_E_polarization_vector=self.fixed_E_polarization_vector,
            fixed_H_polarization_vector=self.fixed_H_polarization_vector,
            dtype=self._config.dtype,
        )
        for axis in straddled:
            wall = self._config.symmetry[axis]
            odd: list[str] = []
            for field_type, polarization in (("E", e_pol), ("H", h_pol)):
                for component in range(3):
                    if abs(float(polarization[component])) < 1e-6:
                        continue
                    if field_component_parity(field_type, component, axis, wall) == -1:
                        odd.append(f"{field_type}{'xyz'[component]}")
            if odd:
                wall_name = "PMC" if wall == 1 else "PEC"
                other = "PEC (-1)" if wall == 1 else "PMC (+1)"
                logger.warning(
                    f"Source '{self.name}' crosses the {'xyz'[axis]}-symmetry plane, where "
                    f"config.symmetry[{axis}]={wall:+d} ({wall_name}) makes {', '.join(odd)} odd — but the "
                    f"source injects a mirror-even profile in those components. The wall will suppress "
                    f"the injected field. For this polarization use {other} on that axis, or place the "
                    f"source so it does not cross the plane."
                )
        return errors

    def apply(
        self: Self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        dispersive_c1: jax.Array | None = None,
        dispersive_c2: jax.Array | None = None,
        dispersive_c3: jax.Array | None = None,
        electric_conductivity: jax.Array | None = None,
    ):
        del electric_conductivity
        # inv_permittivities shape: (3, Nx, Ny, Nz) - slice with component dimension
        inv_permittivities = inv_permittivities[:, *self.grid_slice]
        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            # inv_permeabilities shape: (3, Nx, Ny, Nz) - slice with component dimension
            inv_permeabilities = inv_permeabilities[:, *self.grid_slice]

        # Keep a handle to the raw (ε∞) inverse permittivity before any
        # carrier-frequency correction — the broadband impedance filter
        # computed below needs ε∞ to reconstruct the full ε(ω) spectrum.
        inv_eps_inf_slice = inv_permittivities

        # If the simulation is dispersive, evaluate the real effective inverse
        # permittivity at the source carrier frequency so that the impedance and
        # energy normalization reflect the true medium the source sits in,
        # not just the high-frequency permittivity epsilon_infinity.
        c1_slice = c2_slice = c3_slice = None
        if dispersive_c1 is not None and dispersive_c2 is not None and dispersive_c3 is not None:
            # dispersive_c* shape: (num_poles, 1, Nx, Ny, Nz) → slice spatial axes
            c1_slice = dispersive_c1[:, :, *self.grid_slice]
            c2_slice = dispersive_c2[:, :, *self.grid_slice]
            c3_slice = dispersive_c3[:, :, *self.grid_slice]
            inv_permittivities = effective_inv_permittivity(
                inv_eps=inv_permittivities,
                c1=c1_slice,
                c2=c2_slice,
                c3=c3_slice,
                omega=2.0 * np.pi * self.wave_character.get_frequency(),
                dt=self._config.time_step_duration,
            )

        center, azimuth, elevation = self._get_random_parts(key)

        # determine E/H polarization and the (tilted) wave vector — shared with the
        # analytic Gaussian mode-overlap detector via tilted_polarization_vectors.
        e_pol, h_pol, wave_vector = tilted_polarization_vectors(
            direction=self.direction,
            propagation_axis=self.propagation_axis,
            fixed_E_polarization_vector=self.fixed_E_polarization_vector,
            fixed_H_polarization_vector=self.fixed_H_polarization_vector,
            azimuth_radians=azimuth,
            elevation_radians=elevation,
            dtype=self._config.dtype,
        )

        # update is amplitude multiplied by polarization
        amplitude_raw = self._get_amplitude_raw(center)

        # map amplitude to propagation plane.  Uniform grids keep the legacy
        # index-space projection; non-uniform grids project physical transverse
        # coordinates and interpolate against physical cell centers.
        if self._uses_physical_source_coordinates():
            local_edges = self._local_edge_coordinates()
            assert local_edges is not None
            horizontal_edges = local_edges[self.horizontal_axis]
            vertical_edges = local_edges[self.vertical_axis]
            horizontal_centers = 0.5 * (horizontal_edges[:-1] + horizontal_edges[1:])
            vertical_centers = 0.5 * (vertical_edges[:-1] + vertical_edges[1:])
            w, h = jnp.meshgrid(horizontal_centers, vertical_centers, indexing="ij")
        else:
            w, h = jnp.meshgrid(
                jnp.arange(self.grid_shape[self.horizontal_axis]),
                jnp.arange(self.grid_shape[self.vertical_axis]),
                indexing="ij",
            )
        wh_coords = jnp.stack((w, h), axis=-1)
        wh_coords -= center
        # Orthonormal in-plane basis. u follows the horizontal axis (projected perpendicular to the
        # wave vector for tilted incidence); v completes a right-handed triple with the *forward*
        # propagation direction, so the untilted projection below is the identity for both
        # directions. Deriving v from the signed wave vector instead would mirror the transverse
        # profile about the center whenever direction == "-" (invisible for a centered, radially
        # symmetric profile, wrong for every other one).
        h_list = [0, 0, 0]
        h_list[self.horizontal_axis] = 1
        h_axis = jnp.asarray(h_list, dtype=self._config.dtype)
        u_basis = h_axis - jnp.dot(h_axis, wave_vector) * wave_vector
        u_basis = u_basis / jnp.linalg.norm(u_basis)
        direction_sign = 1.0 if self.direction == "+" else -1.0
        v_basis = direction_sign * jnp.cross(wave_vector, u_basis)

        # projection
        def project(point):
            point_list = [jnp.zeros((), dtype=self._config.dtype)] * 3
            point_list[self.horizontal_axis] = point[0]
            point_list[self.vertical_axis] = point[1]
            point = jnp.asarray(point_list, dtype=self._config.dtype)
            projection = point - jnp.dot(point, wave_vector) * wave_vector
            # Convert to plane coordinates
            u = jnp.dot(projection, u_basis)
            v = jnp.dot(projection, v_basis)
            return jnp.asarray((u, v), dtype=self._config.dtype)

        float_projected = jax.vmap(project)(wh_coords.reshape(-1, 2))
        float_projected += center
        profile_hv = self._grid_to_hv(amplitude_raw)
        if self._uses_physical_source_coordinates():
            index_fn = jax.vmap(
                _linear_interpolate_rectilinear_2d,
                in_axes=(0, None, None, None),
            )
            interp = index_fn(float_projected, horizontal_centers, vertical_centers, profile_hv)
        else:
            # interpolate floating indices in original array
            index_fn = jax.vmap(linear_interpolated_indexing, in_axes=(0, None))
            interp = index_fn(float_projected, profile_hv)
        amplitude = self._hv_to_grid(interp.reshape(profile_hv.shape))[None, ...]

        E = amplitude * e_pol[:, None, None, None]
        H = amplitude * h_pol[:, None, None, None]

        if self.normalize_by_energy:
            energy = compute_energy(
                E=E,
                H=H,
                inv_permittivity=inv_permittivities,
                inv_permeability=inv_permeabilities,
            )
            # Normalize by the energy of the *full-domain* source. Under config.symmetry the slice
            # covers only one half/quarter of it, and the profile is mirror-symmetric about each
            # symmetry plane it straddles, so the full-domain sum is the reduced sum times the
            # plane multiplicity. Without this the reduced run would inject 2**(k/2) times the
            # amplitude of the equivalent full-domain run.
            total_energy_root = jnp.sqrt(energy.sum() * self.symmetry_profile_multiplicity)
            E = E / total_energy_root
            H = H / total_energy_root

        # adjust H for impedance of the medium (isotropic/diagonal or full-tensor)
        impedance = _source_impedance(inv_permittivities, inv_permeabilities, e_pol, h_pol)

        H = H / impedance

        time_offset_E, time_offset_H = calculate_time_offset_yee(
            center=center,
            wave_vector=wave_vector,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            resolution=self._source_resolution(),
            time_step_duration=self._config.time_step_duration,
            e_polarization=e_pol,
            h_polarization=h_pol,
            coordinate_edges=self._local_edge_coordinates(),
            center_physical=self._source_center_physical(center),
        )

        self = self.aset("_E", E, create_new_ok=True)
        self = self.aset("_H", H, create_new_ok=True)
        self = self.aset("_time_offset_E", time_offset_E, create_new_ok=True)
        self = self.aset("_time_offset_H", time_offset_H, create_new_ok=True)

        # Broadband impedance correction. The carrier-frequency rescale above
        # only matches η at ω_c; a wide-bandwidth pulse (e.g. GaussianPulseProfile)
        # sees a frequency-dependent impedance in a dispersive medium and the
        # TFSF boundary leaks unphysical reflections for frequencies away from
        # ω_c. Precompute a filtered H-side temporal profile s_H(t) whose
        # spectrum is S(ω)·√(ε(ω)/ε(ω_c)) so that the injected H field has
        # the frequency-dependent impedance correction baked in.
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
            )
            self = self.aset("_temporal_H_filter", filtered, create_new_ok=True)
        else:
            # Reused source applied in a non-dispersive context: clear any stale
            # H-side filter left over from a previous dispersive apply, otherwise
            # the TFSF inner loop would keep injecting filtered amplitudes.
            self = self.aset("_temporal_H_filter", None, create_new_ok=True)

        return self

    @abstractmethod
    def _get_amplitude_raw(
        self,
        center: jax.Array,
    ) -> jax.Array:  # shape (*grid_shape)
        # in normal coordinates, not yee grid
        del center
        raise NotImplementedError()


@autoinit
class GaussianPlaneSource(LinearlyPolarizedPlaneSource):
    #: the radius of the gaussian source
    radius: float = frozen_field()

    #:  the standard deviation of the gaussian source
    std: float = frozen_field(default=1 / 3)  # relative to radius

    @staticmethod
    def _gauss_profile_2d(
        width: int,
        height: int,
        center: tuple[float, float] | jax.Array,
        radii: tuple[float, float],
        std: float,
        normalization_multiplicity: int = 1,
    ) -> jax.Array:  # shape (width, height)
        """Truncated Gaussian spot on a ``(width, height)`` transverse grid, normalized to unit sum.

        ``width``/``height``, ``center`` and ``radii`` are all in the same (horizontal, vertical)
        order; an xy-indexed meshgrid would swap the coordinates on non-square planes and misplace
        the spot. ``normalization_multiplicity`` divides out the number of copies of this plane in
        the full domain (see ``TFSFPlaneSource.symmetry_profile_multiplicity``), so a plane clipped
        by a symmetry plane still carries the amplitude of the full-domain profile it is part of.
        """
        grid = (
            jnp.stack(jnp.meshgrid(jnp.arange(width), jnp.arange(height), indexing="ij"), axis=-1) - jnp.asarray(center)
        ) / jnp.asarray(radii)
        euc_dist = (grid**2).sum(axis=-1)

        mask = euc_dist < 1
        exp_part = jnp.exp(-0.5 * euc_dist / std**2)

        profile = jnp.where(mask, exp_part, 0)
        return profile / (profile.sum() * normalization_multiplicity)

    @staticmethod
    def _gauss_profile(
        width: int,
        height: int,
        axis: int,
        center: tuple[float, float] | jax.Array,
        radii: tuple[float, float],
        std: float,
        normalization_multiplicity: int = 1,
    ) -> jax.Array:  # shape (*grid_shape)
        """:meth:`_gauss_profile_2d` with a singleton inserted at ``axis``.

        ``width``/``height`` are the sizes along the two axes *other than* ``axis``, in ascending
        array-axis order — which is (horizontal, vertical) for propagation along x or z but
        (vertical, horizontal) for propagation along y.
        """
        profile = GaussianPlaneSource._gauss_profile_2d(
            width=width,
            height=height,
            center=center,
            radii=radii,
            std=std,
            normalization_multiplicity=normalization_multiplicity,
        )
        return jnp.expand_dims(profile, axis=axis)

    def _get_amplitude_raw(
        self,
        center: jax.Array,
    ) -> jax.Array:
        multiplicity = self.symmetry_profile_multiplicity
        if self._config.has_nonuniform_grid:
            local_edges = self._local_edge_coordinates()
            assert local_edges is not None
            horizontal_edges = local_edges[self.horizontal_axis]
            vertical_edges = local_edges[self.vertical_axis]
            horizontal_centers = 0.5 * (horizontal_edges[:-1] + horizontal_edges[1:])
            vertical_centers = 0.5 * (vertical_edges[:-1] + vertical_edges[1:])
            h_grid, v_grid = jnp.meshgrid(horizontal_centers, vertical_centers, indexing="ij")
            h_center = center[0]
            v_center = center[1]
            normalized_radius_squared = ((h_grid - h_center) / self.radius) ** 2 + (
                (v_grid - v_center) / self.radius
            ) ** 2
            mask = normalized_radius_squared < 1
            exp_part = jnp.exp(-0.5 * normalized_radius_squared / self.std**2)
            profile_2d = jnp.where(mask, exp_part, 0)
            h_widths = horizontal_edges[1:] - horizontal_edges[:-1]
            v_widths = vertical_edges[1:] - vertical_edges[:-1]
            cell_areas = h_widths[:, None] * v_widths[None, :]
            profile_2d = profile_2d / ((profile_2d * cell_areas).sum() * multiplicity)
            return self._hv_to_grid(profile_2d)

        grid_radius = self.radius / self._config.uniform_spacing()
        profile_hv = self._gauss_profile_2d(
            width=self.grid_shape[self.horizontal_axis],
            height=self.grid_shape[self.vertical_axis],
            center=center,
            radii=(grid_radius, grid_radius),
            std=self.std,
            normalization_multiplicity=multiplicity,
        )
        return self._hv_to_grid(profile_hv)


@autoinit
class UniformPlaneSource(LinearlyPolarizedPlaneSource):
    #: the amplitude of the uniform source
    amplitude: float = frozen_field(default=1.0)

    def _get_amplitude_raw(
        self,
        center: jax.Array,
    ) -> jax.Array:
        del center
        profile = jnp.ones(shape=self.grid_shape, dtype=self._config.dtype)
        return self.amplitude * profile
