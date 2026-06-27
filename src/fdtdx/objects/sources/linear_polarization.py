import warnings
from abc import ABC, abstractmethod
from typing import Callable, Self

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import calculate_time_offset_yee
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.misc import (
    expand_to_3x3,
    gaussian_amplitude,
    linear_interpolated_indexing,
    tilted_polarization_vectors,
)
from fdtdx.core.physics.metrics import compute_energy
from fdtdx.dispersion import effective_inv_permittivity
from fdtdx.objects.sources.tfsf import TFSFPlaneSource, _build_dispersive_H_filter
from fdtdx.typing import SliceTuple3D


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

    def apply(
        self: Self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        dispersive_c1: jax.Array | None = None,
        dispersive_c2: jax.Array | None = None,
        dispersive_c3: jax.Array | None = None,
    ):
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
    #: the ``1/e`` amplitude radius of the gaussian beam (``exp(-r^2 / radius^2)``), in metres.
    #: Shares the convention of :func:`~fdtdx.gaussian_mode_fields` via
    #: :func:`~fdtdx.core.misc.gaussian_amplitude` (smooth tail, no hard aperture).
    radius: float = frozen_field()

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(grid_slice_tuple=grid_slice_tuple, config=config, key=key)
        self._warn_if_truncated()
        return self

    def _warn_if_truncated(self) -> None:
        """Warn if the finite source plane truncates the Gaussian above ~1% amplitude.

        The beam is injected only over the source object's rectangular transverse footprint and
        hard-cut to zero outside it (there is no smooth taper at the plane edge itself). If the
        Gaussian amplitude ``exp(-r^2 / radius^2)`` is still above 1% at the nearest plane edge,
        that hard aperture causes non-negligible edge diffraction. A clean source needs the
        transverse half-extent to be ``>~ 3 * radius`` (edge amplitude ``~0.01%``). The check
        assumes the beam is centered in its footprint, so it is the best-case (smallest)
        truncation amplitude.
        """
        grid = self._config.resolved_grid
        half_extents = []
        for axis in (self.horizontal_axis, self.vertical_axis):
            lower, upper = self.grid_slice_tuple[axis]
            if grid is not None:
                extent = grid.axis_extent(axis, (lower, upper))
            else:
                extent = (upper - lower) * self._config.uniform_spacing()
            half_extents.append(0.5 * extent)
        edge_distance = min(half_extents)  # nearest plane edge from the (centered) beam
        truncation_amplitude = float(np.exp(-((edge_distance / self.radius) ** 2)))
        if truncation_amplitude > 0.01:
            warnings.warn(
                f"GaussianPlaneSource '{self.name}': the source plane truncates the beam at "
                f"{truncation_amplitude * 100:.1f}% amplitude (nearest edge at "
                f"{edge_distance / self.radius:.2f} x radius). The hard aperture edge will cause "
                f"diffraction — enlarge the source transverse size to >~ 3 x radius (or reduce "
                f"radius) so the edge amplitude drops below 1%.",
                UserWarning,
                stacklevel=2,
            )

    def _get_amplitude_raw(
        self,
        center: jax.Array,
    ) -> jax.Array:
        if self._config.has_nonuniform_grid:
            local_edges = self._local_edge_coordinates()
            assert local_edges is not None
            horizontal_edges = local_edges[self.horizontal_axis]
            vertical_edges = local_edges[self.vertical_axis]
            horizontal_centers = 0.5 * (horizontal_edges[:-1] + horizontal_edges[1:])
            vertical_centers = 0.5 * (vertical_edges[:-1] + vertical_edges[1:])
            h_grid, v_grid = jnp.meshgrid(horizontal_centers, vertical_centers, indexing="ij")
            profile_2d = gaussian_amplitude(h_grid, v_grid, self.radius, center=(center[0], center[1]))
            return jnp.expand_dims(profile_2d, axis=self.propagation_axis)

        grid_radius = self.radius / self._config.uniform_spacing()
        height = self.grid_shape[self.vertical_axis]
        width = self.grid_shape[self.horizontal_axis]
        coords = jnp.stack(jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="xy"), axis=-1)
        profile = gaussian_amplitude(coords[..., 0], coords[..., 1], grid_radius, center=(center[0], center[1]))
        return jnp.expand_dims(profile, axis=self.propagation_axis)


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


@autoinit
class CustomProfilePlaneSource(LinearlyPolarizedPlaneSource):
    """Plane source with a user-defined transverse amplitude profile.

    The injected transverse amplitude is ``profile_function(t0, t1)``, evaluated on the
    source-plane cell centers in physical metres (centered at the plane center).  ``t0``/``t1`` are
    the two transverse coordinate meshgrids (along ``horizontal_axis``/``vertical_axis``), and the
    callable returns a real amplitude array of the same shape.  Polarization, off-normal tilt,
    impedance, and energy normalization are handled by the base class exactly as for
    :class:`UniformPlaneSource` / :class:`GaussianPlaneSource` — this only customizes the spatial
    weighting.  It is the source-side counterpart to :class:`~fdtdx.CustomModeOverlapDetector`.

    Any JAX callable works — this is the hook for spatial apodization (e.g. a super-Gaussian
    flat-top ``lambda t0, t1: jnp.exp(-((t0**2 + t1**2) / r**2) ** n)``, a Tukey-windowed flat-top,
    a vortex amplitude, or a multi-spot pattern).  It is differentiable, so the illumination profile
    can be inverse-designed.  No ready-made profile builders are provided; supply your own callable.
    """

    #: ``profile_function(transverse_0, transverse_1) -> amplitude``: real transverse amplitude on
    #: the source plane (physical metres, centered at the plane center), same shape as the inputs.
    profile_function: Callable[[jax.Array, jax.Array], jax.Array] = frozen_field()

    def _axis_plane_centers(self, axis: int) -> jax.Array:
        lower, upper = self.grid_slice_tuple[axis]
        grid = self._config.resolved_grid
        if grid is not None:
            centers = jnp.asarray(grid.centers(axis))[lower:upper]
        else:
            spacing = self._config.uniform_spacing()
            centers = (jnp.arange(lower, upper) + 0.5) * spacing
        return centers - 0.5 * (centers[0] + centers[-1])  # center at the plane middle

    def _get_amplitude_raw(
        self,
        center: jax.Array,
    ) -> jax.Array:
        del center  # the profile is centered on the plane; off-normal tilt is handled by the base
        h_centers = self._axis_plane_centers(self.horizontal_axis)
        v_centers = self._axis_plane_centers(self.vertical_axis)
        g_h, g_v = jnp.meshgrid(h_centers, v_centers, indexing="ij")
        amplitude = jnp.asarray(self.profile_function(g_h, g_v))
        return jnp.expand_dims(amplitude, axis=self.propagation_axis)
