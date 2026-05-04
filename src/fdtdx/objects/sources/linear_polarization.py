from abc import ABC, abstractmethod
from typing import Self

import jax
import jax.numpy as jnp

from fdtdx.core.grid import calculate_time_offset_yee
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.linalg import get_wave_vector_raw, rotate_vector
from fdtdx.core.misc import expand_to_3x3, linear_interpolated_indexing, normalize_polarization_for_source
from fdtdx.core.physics.metrics import compute_energy
from fdtdx.objects.sources.tfsf import TFSFPlaneSource


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
        if self._config.grid is None:
            return None
        local_edges = []
        for axis in range(3):
            lower, upper = self.grid_slice_tuple[axis]
            edges = self._config.grid.edges(axis)[lower : upper + 1]
            local_edges.append(edges - edges[0])
        return tuple(local_edges)

    def _source_center_physical(self, source_center: jax.Array) -> jax.Array | None:
        """Return the physical center used for grid-aware Yee time offsets."""
        local_edges = self._local_edge_coordinates()
        if local_edges is None:
            return None
        physical_center = []
        for axis, edges in enumerate(local_edges):
            if axis == self.propagation_axis:
                physical_center.append(jnp.asarray(0.0, dtype=self._config.dtype))
            elif self._config.grid is not None and not self._config.grid.is_uniform:
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
        if self._config.grid is not None and not self._config.grid.is_uniform:
            return self._config.grid.min_spacing
        return self._config.require_uniform_grid()

    def _uses_physical_source_coordinates(self) -> bool:
        """Whether transverse source coordinates are represented in metres."""
        return self._config.grid is not None and not self._config.grid.is_uniform

    def apply(
        self: Self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
    ):
        # inv_permittivities shape: (3, Nx, Ny, Nz) - slice with component dimension
        inv_permittivities = inv_permittivities[:, *self.grid_slice]
        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            # inv_permeabilities shape: (3, Nx, Ny, Nz) - slice with component dimension
            inv_permeabilities = inv_permeabilities[:, *self.grid_slice]

        # determine E/H polarization
        e_pol_raw, h_pol_raw = normalize_polarization_for_source(
            direction=self.direction,
            propagation_axis=self.propagation_axis,
            fixed_E_polarization_vector=self.fixed_E_polarization_vector,
            fixed_H_polarization_vector=self.fixed_H_polarization_vector,
            dtype=self._config.dtype,
        )
        wave_vector_raw = get_wave_vector_raw(
            direction=self.direction,
            propagation_axis=self.propagation_axis,
            dtype=self._config.dtype,
        )

        center, azimuth, elevation = self._get_random_parts(key)

        # tilt polarizations
        axes_tpl = (self.horizontal_axis, self.vertical_axis, self.propagation_axis)
        wave_vector = rotate_vector(wave_vector_raw, azimuth, elevation, axes_tpl)
        e_pol = rotate_vector(e_pol_raw, azimuth, elevation, axes_tpl)
        h_pol = rotate_vector(h_pol_raw, azimuth, elevation, axes_tpl)

        # update is amplitude multiplied by polarization
        amplitude_raw = self._get_amplitude_raw(center)[None, ...]

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
        # basis in plane
        h_list = [0, 0, 0]
        h_list[self.horizontal_axis] = 1
        h_axis = jnp.asarray(h_list, dtype=self._config.dtype)
        u_basis = h_axis - jnp.dot(h_axis, wave_vector) * wave_vector
        u_basis = u_basis / jnp.linalg.norm(u_basis)
        v_basis = jnp.cross(wave_vector, u_basis)

        # projection
        def project(point):
            point_list = [point[0], point[1]]
            point_list.insert(self.propagation_axis, 0)
            point = jnp.asarray(point_list, dtype=self._config.dtype)
            projection = point - jnp.dot(point, wave_vector) * wave_vector
            # Convert to plane coordinates
            u = jnp.dot(projection, u_basis)
            v = jnp.dot(projection, v_basis)
            return jnp.asarray((u, v), dtype=self._config.dtype)

        float_projected = jax.vmap(project)(wh_coords.reshape(-1, 2))
        float_projected += center
        if self._uses_physical_source_coordinates():
            index_fn = jax.vmap(
                _linear_interpolate_rectilinear_2d,
                in_axes=(0, None, None, None),
            )
            profile_2d = jnp.take(amplitude_raw[0], 0, axis=self.propagation_axis)
            interp = index_fn(float_projected, horizontal_centers, vertical_centers, profile_2d)
        else:
            # interpolate floating indices in original array
            index_fn = jax.vmap(linear_interpolated_indexing, in_axes=(0, None))
            profile_2d = jnp.take(amplitude_raw[0], 0, axis=self.propagation_axis)
            interp = index_fn(float_projected, profile_2d)
        amplitude = interp.reshape(*amplitude_raw.shape)

        E = amplitude * e_pol[:, None, None, None]
        H = amplitude * h_pol[:, None, None, None]

        if self.normalize_by_energy:
            energy = compute_energy(
                E=E,
                H=H,
                inv_permittivity=inv_permittivities,
                inv_permeability=inv_permeabilities,
            )
            total_energy_root = jnp.sqrt(energy.sum())
            E = E / total_energy_root
            H = H / total_energy_root

        # adjust H for impedance of the medium
        # check if fully anisotropic
        if (
            isinstance(inv_permittivities, jax.Array)
            and inv_permittivities.ndim >= 1
            and inv_permittivities.shape[0] == 9
        ) or (
            isinstance(inv_permeabilities, jax.Array)
            and inv_permeabilities.ndim >= 1
            and inv_permeabilities.shape[0] == 9
        ):
            # convert to 3x3 tensors
            inv_eps_tensor = expand_to_3x3(inv_permittivities)  # shape: (3, 3, Nx, Ny, Nz)
            inv_mu_tensor = expand_to_3x3(inv_permeabilities)  # shape: (3, 3, Nx, Ny, Nz)

            # invert to get eps and mu tensors
            perm = (2, 3, 4, 0, 1)  # (3, 3, nx, ny, nz) -> (nx, ny, nz, 3, 3)
            inv_perm = (3, 4, 0, 1, 2)  # (nx, ny, nz, 3, 3) -> (3, 3, nx, ny, nz)
            eps = jnp.linalg.inv(inv_eps_tensor.transpose(perm)).transpose(inv_perm)
            mu = jnp.linalg.inv(inv_mu_tensor.transpose(perm)).transpose(inv_perm)

            # compute effective permittivity and permeability along polarization directions
            eps_eff = jnp.einsum("i,ijxyz,j->xyz", e_pol, eps, e_pol)
            mu_eff = jnp.einsum("i,ijxyz,j->xyz", h_pol, mu, h_pol)
            impedance = jnp.sqrt(mu_eff / eps_eff)
        else:
            impedance = jnp.sqrt(inv_permittivities / inv_permeabilities)

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
    def _gauss_profile(
        width: int,
        height: int,
        axis: int,
        center: tuple[float, float] | jax.Array,
        radii: tuple[float, float],
        std: float,
    ) -> jax.Array:  # shape (*grid_shape)
        grid = (
            jnp.stack(jnp.meshgrid(*map(jnp.arange, (height, width)), indexing="xy"), axis=-1) - jnp.asarray(center)
        ) / jnp.asarray(radii)
        euc_dist = (grid**2).sum(axis=-1)

        mask = euc_dist < 1
        mask = jnp.expand_dims(mask, axis=axis)

        exp_part = jnp.exp(-0.5 * euc_dist / std**2)
        exp_part = jnp.expand_dims(exp_part, axis=axis)

        profile = jnp.where(mask, exp_part, 0)
        profile = profile / profile.sum()

        return profile

    def _get_amplitude_raw(
        self,
        center: jax.Array,
    ) -> jax.Array:
        if self._config.grid is not None and not self._config.grid.is_uniform:
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
            profile_2d = profile_2d / profile_2d.sum()
            return jnp.expand_dims(profile_2d, axis=self.propagation_axis)

        grid_radius = self.radius / self._config.require_uniform_grid()
        profile = self._gauss_profile(
            width=self.grid_shape[self.horizontal_axis],
            height=self.grid_shape[self.vertical_axis],
            axis=self.propagation_axis,
            center=center,
            radii=(grid_radius, grid_radius),
            std=self.std,
        )
        return profile


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
