from abc import ABC, abstractmethod
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from fdtdx.constants import c as c0
from fdtdx.core.axis import get_oriented_transverse_axes
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.misc import expand_to_3x3
from fdtdx.dispersion import (
    compute_eps_spectrum_from_coefficients,
    compute_impedance_corrected_temporal_profile,
)
from fdtdx.objects.sources.profile import TemporalProfile
from fdtdx.objects.sources.source import DirectionalPlaneSourceBase


def _build_dispersive_H_filter(
    temporal_profile: TemporalProfile,
    wave_character,
    dt: float,
    num_time_steps: int,
    c1_slice: jax.Array,
    c2_slice: jax.Array,
    c3_slice: jax.Array,
    inv_eps_inf_slice: jax.Array,
    dtype: jnp.dtype = jnp.float32,
    c4_slice: jax.Array | None = None,
) -> jax.Array:
    """Precompute the broadband-corrected H-side temporal profile for a TFSF source.

    Builds the FIR filter ``G(ω) = √(ε(ω)/ε(ω_c))`` from the ADE coefficients
    and the raw ε∞ of the cells covered by the source slice, then applies it
    to the raw temporal profile sampled at every integer simulation time step.
    The returned array replaces the per-step ``temporal_profile.get_amplitude``
    call on the H side of the TFSF injection, giving broadband impedance
    matching in dispersive media.

    Averaging over the source slice is uniform — sufficient for plane sources
    (homogeneous background) and a reasonable first-order approximation for
    mode sources (captures bulk dispersion of the guiding medium but not
    geometry-driven modal dispersion).

    Args:
        temporal_profile: The source's raw temporal profile.
        wave_character: The source's carrier WaveCharacter — provides ω_c and
            the phase shift passed to ``get_amplitude``.
        dt: Simulation time step (seconds).
        num_time_steps: ``config.time_steps_total`` — the length of the
            returned filter array.
        c1_slice: ADE coefficient array sliced to the source cells.
        c2_slice: ADE coefficient array sliced to the source cells.
        c3_slice: ADE coefficient array sliced to the source cells.
        inv_eps_inf_slice: Raw ``1/ε∞`` at the source cells
            (before any carrier-frequency correction).
        dtype: Output dtype for the filtered profile. Should match the
            simulation's field dtype so float64 simulations are not
            silently downcast to float32. Defaults to float32.

    Returns:
        JAX array of shape ``(num_time_steps,)`` in ``dtype`` — the filtered
        temporal profile ``s_H[n]``.
    """
    carrier_period = wave_character.get_period()
    phase_shift = wave_character.phase_shift

    # Sample the raw temporal profile at every integer time step. Use numpy
    # here so the computation stays on the host and avoids the JAX dtype
    # downcast warning when x64 is not enabled — the filter is built once
    # at setup time and never traced.
    times = np.arange(num_time_steps, dtype=np.float64) * dt
    raw_samples_jax = temporal_profile.get_amplitude(
        time=jnp.asarray(times),
        period=carrier_period,
        phase_shift=phase_shift,
    )
    raw_samples = np.asarray(raw_samples_jax, dtype=np.float64)

    # Length-M zero-padded FFT for linear convolution.
    m = 1
    while m < 2 * num_time_steps:
        m *= 2
    omegas_rfft = 2.0 * np.pi * np.fft.rfftfreq(m, d=dt)

    c1_np = np.asarray(c1_slice)
    c2_np = np.asarray(c2_slice)
    c3_np = np.asarray(c3_slice)
    c4_np = None if c4_slice is None else np.asarray(c4_slice)
    inv_eps_inf_np = np.asarray(inv_eps_inf_slice)

    # All-zero coupling in the source slice (non-dispersive material at the
    # source plane) means eps(omega) is flat and the filter is the identity —
    # skip it rather than allocate the full broadband spectrum.
    if c3_np.size == 0 or (not np.any(c3_np) and (c4_np is None or not np.any(c4_np))):
        return jnp.asarray(raw_samples, dtype=dtype)

    # With per-axis (anisotropic) dispersion the coefficient arrays carry one
    # value per component; the scalar impedance filter below can only use a
    # component average. Warn once at setup so the approximation is visible.
    if c1_np.shape[0] > 0 and c1_np.shape[1] > 1:
        anisotropic = any(
            bool(np.any(arr != arr[:, :1])) for arr in (c1_np, c2_np, c3_np) + (() if c4_np is None else (c4_np,))
        )
        if anisotropic:
            logger.warning(
                "TFSF source overlaps a material with per-axis (anisotropic) dispersion; the broadband "
                "impedance filter uses a component-averaged eps(omega). The carrier-frequency amplitude "
                "stays exact per component, but broadband impedance matching is approximate here."
            )

    eps_spectrum = compute_eps_spectrum_from_coefficients(
        c1=c1_np,
        c2=c2_np,
        c3=c3_np,
        inv_eps_inf=inv_eps_inf_np,
        omegas=omegas_rfft,
        dt=dt,
        c4=c4_np,
    )
    omega_c = 2.0 * np.pi * wave_character.get_frequency()
    eps_center_arr = compute_eps_spectrum_from_coefficients(
        c1=c1_np,
        c2=c2_np,
        c3=c3_np,
        inv_eps_inf=inv_eps_inf_np,
        omegas=np.array([omega_c], dtype=np.float64),
        dt=dt,
        c4=c4_np,
    )
    eps_center = complex(eps_center_arr[0])

    s_H = compute_impedance_corrected_temporal_profile(
        raw_samples=raw_samples,
        dt=dt,
        eps_spectrum=eps_spectrum,
        eps_center=eps_center,
    )
    return jnp.asarray(s_H, dtype=dtype)


def _source_impedance(
    inv_permittivities: jax.Array,
    inv_permeabilities: jax.Array | float,
    e_pol: jax.Array,
    h_pol: jax.Array,
) -> jax.Array:
    """Wave impedance seen by a plane-wave source along its polarization.

    For isotropic / diagonally anisotropic media this is the scalar
    ``sqrt(inv_eps / inv_mu)`` (i.e. ``sqrt(mu / eps)``). For fully anisotropic
    media (9-component tensors) the permittivity and permeability tensors are
    inverted and projected onto the E- and H-polarization directions before
    taking the ratio. Shared by the single-plane source and the box region so
    the impedance normalization is computed in exactly one place.

    Args:
        inv_permittivities: Inverse permittivity, already sliced to the source
            cells. Scalar/1/3/9-component leading axis.
        inv_permeabilities: Inverse permeability (same layout) or a float.
        e_pol: Electric polarization unit vector, shape ``(3,)``.
        h_pol: Magnetic polarization unit vector, shape ``(3,)``.

    Returns:
        The impedance array (or scalar) to divide the injected H field by.
    """
    if (
        isinstance(inv_permittivities, jax.Array) and inv_permittivities.ndim >= 1 and inv_permittivities.shape[0] == 9
    ) or (
        isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim >= 1 and inv_permeabilities.shape[0] == 9
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
    return impedance


def _tfsf_inject_E_face(
    E: jax.Array,
    *,
    grid_slice,
    normal_axis: int,
    sign: int,
    incident_H: jax.Array,
    time_offset_H: jax.Array,
    temporal_H_filter: jax.Array | None,
    inv_permittivities: jax.Array,
    c: jax.Array | float,
    time_step: jax.Array,
    delta_t: float,
    temporal_profile: TemporalProfile,
    wave_character,
    static_amplitude_factor: float,
) -> jax.Array:
    """Inject the incident-H TFSF correction into E on a single face.

    A TFSF face couples the two field components tangential to the face
    (``a``, ``b`` = the transverse axes of ``normal_axis``): the update of E[a]
    is driven by the incident H[b] and E[b] by the incident H[a], with the
    curl orientation baked into the signs (``g_H[a]=+H_b``, ``g_H[b]=-H_a``).
    The face-normal component only picks up a correction under full anisotropy
    (off-diagonal tensor bleed). The single-plane :class:`TFSFPlaneSource` is
    the special case ``normal_axis == propagation_axis``; the box source calls
    this once per active face with the face's outward-normal ``sign``.

    Args:
        E: Full-domain electric field array ``(3, Nx, Ny, Nz)``.
        grid_slice: The face slice (size 1 along ``normal_axis``).
        normal_axis: Axis normal to the face (0/1/2).
        sign: Geometric injection sign (+1 min face, -1 max face; negated for
            the reverse/adjoint update).
        incident_H: Incident H profile on the face, shape ``(3, *face)``.
        time_offset_H: Per-component Yee time offsets on the face ``(3, *face)``.
        temporal_H_filter: Broadband impedance-corrected H profile (dispersive
            media) or ``None`` to use the raw temporal profile.
        inv_permittivities: Full-domain inverse permittivity array.
        c: Courant number scaled by the local metric factor (backward stencil).
        time_step: Current (possibly on/off-adjusted) time step.
        delta_t: Simulation time-step duration (seconds).
        temporal_profile: Source temporal profile.
        wave_character: Source carrier WaveCharacter.
        static_amplitude_factor: Static amplitude multiplier.

    Returns:
        The E array with the face correction added.
    """
    a_axis, b_axis = get_oriented_transverse_axes(normal_axis)

    is_fully_anisotropic = inv_permittivities.ndim > 0 and inv_permittivities.shape[0] == 9
    inv_permittivity_slice = inv_permittivities[:, *grid_slice]

    amplitude_H = {}
    for axis in (a_axis, b_axis):
        if temporal_H_filter is None:
            time_H = (time_step + time_offset_H[axis]) * delta_t
            amplitude_H[axis] = (
                temporal_profile.get_amplitude(
                    time=time_H,
                    period=wave_character.get_period(),
                    phase_shift=wave_character.phase_shift,
                )
                * static_amplitude_factor
            )
        else:
            idx_arr = time_step + time_offset_H[axis]
            xp = jnp.arange(temporal_H_filter.shape[0], dtype=idx_arr.dtype)
            amplitude_H[axis] = (
                jnp.interp(idx_arr, xp, temporal_H_filter, left=0.0, right=0.0) * static_amplitude_factor
            )

    inject_complex_H = jnp.iscomplexobj(incident_H) and temporal_H_filter is None
    amplitude_H_quad = {}
    if inject_complex_H:
        for axis in (a_axis, b_axis):
            time_H = (time_step + time_offset_H[axis]) * delta_t
            amplitude_H_quad[axis] = (
                temporal_profile.get_amplitude(
                    time=time_H,
                    period=wave_character.get_period(),
                    phase_shift=wave_character.phase_shift - 0.5 * np.pi,
                )
                * static_amplitude_factor
            )

    def incident_H_component(axis):
        if inject_complex_H:
            return jnp.real(incident_H[axis]) * amplitude_H[axis] + jnp.imag(incident_H[axis]) * amplitude_H_quad[axis]
        return incident_H[axis] * amplitude_H[axis]

    if is_fully_anisotropic:
        H_b_inc = jax.lax.stop_gradient(incident_H_component(b_axis))
        H_a_inc = jax.lax.stop_gradient(incident_H_component(a_axis))

        def get_inv_eps(row, col):
            return inv_permittivity_slice[row * 3 + col]

        for row in (normal_axis, a_axis, b_axis):
            correction = c * (get_inv_eps(row, a_axis) * (+H_b_inc) + get_inv_eps(row, b_axis) * (-H_a_inc))
            E = E.at[row, *grid_slice].add(sign * correction)
        return E

    else:
        H_b_inc = incident_H_component(b_axis)
        H_b_inc = H_b_inc * c * inv_permittivity_slice[a_axis]
        H_b_inc = jax.lax.stop_gradient(H_b_inc)

        H_a_inc = incident_H_component(a_axis)
        H_a_inc = H_a_inc * c * inv_permittivity_slice[b_axis]
        H_a_inc = jax.lax.stop_gradient(H_a_inc)

        E = E.at[a_axis, *grid_slice].add(sign * H_b_inc)
        E = E.at[b_axis, *grid_slice].add(-sign * H_a_inc)
        return E


def _tfsf_inject_H_face(
    H: jax.Array,
    *,
    grid_slice,
    normal_axis: int,
    sign: int,
    incident_E: jax.Array,
    time_offset_E: jax.Array,
    inv_permeabilities: jax.Array | float,
    c: jax.Array | float,
    time_step: jax.Array,
    delta_t: float,
    temporal_profile: TemporalProfile,
    wave_character,
    static_amplitude_factor: float,
) -> jax.Array:
    """Inject the incident-E TFSF correction into H on a single face.

    Symmetric counterpart of :func:`_tfsf_inject_E_face`: the tangential H
    components are driven by the incident E of the other tangential axis
    (``g_E[a]=-E_b``, ``g_E[b]=+E_a``). See that function for the parameter
    conventions; ``time_offset_E``/``incident_E`` replace their H analogues and
    ``c`` uses the forward metric stencil.

    Returns:
        The H array with the face correction added.
    """
    a_axis, b_axis = get_oriented_transverse_axes(normal_axis)

    is_fully_anisotropic = (
        isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0 and inv_permeabilities.shape[0] == 9
    )

    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
        inv_permeability_slice = inv_permeabilities[:, *grid_slice]
    else:
        inv_permeability_slice = inv_permeabilities

    amplitude_E = {}
    for axis in (a_axis, b_axis):
        time_E = (time_step + time_offset_E[axis]) * delta_t
        amplitude_E[axis] = (
            temporal_profile.get_amplitude(
                time=time_E,
                period=wave_character.get_period(),
                phase_shift=wave_character.phase_shift,
            )
            * static_amplitude_factor
        )

    inject_complex_E = jnp.iscomplexobj(incident_E)
    amplitude_E_quad = {}
    if inject_complex_E:
        for axis in (a_axis, b_axis):
            time_E = (time_step + time_offset_E[axis]) * delta_t
            amplitude_E_quad[axis] = (
                temporal_profile.get_amplitude(
                    time=time_E,
                    period=wave_character.get_period(),
                    phase_shift=wave_character.phase_shift - 0.5 * np.pi,
                )
                * static_amplitude_factor
            )

    def incident_E_component(axis):
        if inject_complex_E:
            return jnp.real(incident_E[axis]) * amplitude_E[axis] + jnp.imag(incident_E[axis]) * amplitude_E_quad[axis]
        return incident_E[axis] * amplitude_E[axis]

    if is_fully_anisotropic:
        E_a_inc = jax.lax.stop_gradient(incident_E_component(a_axis))
        E_b_inc = jax.lax.stop_gradient(incident_E_component(b_axis))

        def get_inv_mu(row, col):
            return inv_permeability_slice[row * 3 + col]  # type: ignore

        for row in (normal_axis, a_axis, b_axis):
            correction = c * (get_inv_mu(row, a_axis) * (-E_b_inc) + get_inv_mu(row, b_axis) * (+E_a_inc))
            H = H.at[row, *grid_slice].add(sign * correction)
        return H

    else:
        E_a_inc = incident_E_component(a_axis)
        if isinstance(inv_permeability_slice, jax.Array) and inv_permeability_slice.ndim > 1:
            E_a_inc = E_a_inc * c * inv_permeability_slice[b_axis]
        else:
            E_a_inc = E_a_inc * c * inv_permeability_slice
        E_a_inc = jax.lax.stop_gradient(E_a_inc)

        E_b_inc = incident_E_component(b_axis)
        if isinstance(inv_permeability_slice, jax.Array) and inv_permeability_slice.ndim > 1:
            E_b_inc = E_b_inc * c * inv_permeability_slice[a_axis]
        else:
            E_b_inc = E_b_inc * c * inv_permeability_slice
        E_b_inc = jax.lax.stop_gradient(E_b_inc)

        H = H.at[b_axis, *grid_slice].add(sign * E_a_inc)
        H = H.at[a_axis, *grid_slice].add(-sign * E_b_inc)
        return H


@autoinit
class TFSFPlaneSource(DirectionalPlaneSourceBase, ABC):
    """
    Total-Field/Scattered-Field (TFSF) implementation of a source.
    The boundary between the scattered field and total field is at a
    positive offset of 0.25 in the yee grid in the axis of propagation.
    """

    #: the azimuth angle
    azimuth_angle: float = frozen_field(default=0.0)

    #: the elevation angle
    elevation_angle: float = frozen_field(default=0.0)

    #: the max angle random offset
    max_angle_random_offset: float = frozen_field(default=0.0)

    #: the max vertical offset
    max_vertical_offset: float = frozen_field(default=0.0)

    #: the max horizontal offset
    max_horizontal_offset: float = frozen_field(default=0.0)

    _E: jax.Array = private_field()
    _H: jax.Array = private_field()
    _time_offset_E: jax.Array = private_field()
    _time_offset_H: jax.Array = private_field()

    # Precomputed H-side temporal profile for broadband impedance matching in
    # dispersive media. Shape (time_steps_total,). When None the source falls
    # back to the per-step temporal_profile.get_amplitude call, which exactly
    # reproduces the non-dispersive behavior. When set, the inner update_E
    # loop reads the filtered profile with a fractional-index lookup so the
    # existing half-step Yee time offsets are preserved.
    _temporal_H_filter: jax.Array | None = private_field(default=None)

    @property
    def azimuth_radians(self) -> float:
        """Convert azimuth angle from degrees to radians.

        Returns:
            float: Azimuth angle in radians.
        """
        return np.deg2rad(self.azimuth_angle)

    @property
    def elevation_radians(self) -> float:
        """Convert elevation angle from degrees to radians.

        Returns:
            float: Elevation angle in radians.
        """
        return np.deg2rad(self.elevation_angle)

    @property
    def max_angle_random_offset_radians(self) -> float:
        """Convert maximum random angle offset from degrees to radians.

        Returns:
            float: Maximum random angle offset in radians.
        """
        return np.deg2rad(self.max_angle_random_offset)

    @property
    def max_vertical_offset_grid(self) -> float:
        """Return the maximum vertical random offset in source-center units.

        Returns:
            On uniform grids this is the legacy grid-index offset.  On
            non-uniform grids source centers are represented in physical
            transverse coordinates, so the returned value is the requested
            physical offset in metres.
        """
        if self.max_vertical_offset == 0:
            return 0.0
        if self._config.has_nonuniform_grid:
            return self.max_vertical_offset
        return self.max_vertical_offset / self._config.uniform_spacing()

    @property
    def max_horizontal_offset_grid(self) -> float:
        """Return the maximum horizontal random offset in source-center units.

        Returns:
            On uniform grids this is the legacy grid-index offset.  On
            non-uniform grids source centers are represented in physical
            transverse coordinates, so the returned value is the requested
            physical offset in metres.
        """
        if self.max_horizontal_offset == 0:
            return 0.0
        if self._config.has_nonuniform_grid:
            return self.max_horizontal_offset
        return self.max_horizontal_offset / self._config.uniform_spacing()

    def _get_azimuth_elevation(
        self,
        key: jax.Array,
    ) -> tuple[
        jax.Array,  # azimuth (radians)
        jax.Array,  # elevation (radians)
    ]:
        # Generate random azimuth and elevation angles within allowed offset ranges
        key1, key2 = jax.random.split(key)
        elevation_radians = jax.random.uniform(
            key1,
            shape=(),
            minval=self.elevation_radians - self.max_angle_random_offset_radians,
            maxval=self.elevation_radians + self.max_angle_random_offset_radians,
        )
        azimuth_radians = jax.random.uniform(
            key2,
            shape=(),
            minval=self.azimuth_radians - self.max_angle_random_offset_radians,
            maxval=self.azimuth_radians + self.max_angle_random_offset_radians,
        )
        return azimuth_radians, elevation_radians

    def _get_center(self, key: jax.Array) -> jax.Array:  # shape(2,)
        """Calculate the randomized source center.

        Uniform-grid sources use the legacy index-space center.  On a
        non-uniform grid, tilted projections and Gaussian profiles are sampled in
        physical coordinates, so the returned center is measured in metres from
        the source-slice lower edge along the transverse axes.
        """
        if self._config.has_nonuniform_grid:
            grid = self._config.resolved_grid
            assert grid is not None
            local_edges = []
            for axis in (self.horizontal_axis, self.vertical_axis):
                lower, upper = self.grid_slice_tuple[axis]
                edges = grid.edges(axis)[lower : upper + 1]
                local_edges.append(edges - edges[0])
            center_horizontal = 0.5 * local_edges[0][-1]
            center_vertical = 0.5 * local_edges[1][-1]
        else:
            horizontal_size = self.grid_shape[self.horizontal_axis]
            vertical_size = self.grid_shape[self.vertical_axis]
            center_horizontal = (horizontal_size - 1) / 2
            center_vertical = (vertical_size - 1) / 2

        key, subkey = jax.random.split(key)
        horizontal_offset = jax.random.uniform(
            key=subkey,
            shape=(1,),
            minval=-self.max_horizontal_offset_grid,
            maxval=self.max_horizontal_offset_grid,
        )
        vertical_offset = jax.random.uniform(
            key=key,
            shape=(1,),
            minval=-self.max_vertical_offset_grid,
            maxval=self.max_vertical_offset_grid,
        )

        center = jnp.asarray(
            [center_horizontal + horizontal_offset, center_vertical + vertical_offset],
            dtype=self._config.dtype,
        ).squeeze()
        return center

    def _get_random_parts(self, key: jax.Array):
        key, subkey = jax.random.split(key)
        center = self._get_center(subkey)

        key, subkey = jax.random.split(key)
        azimuth, elevation = self._get_azimuth_elevation(subkey)

        return center, azimuth, elevation

    @abstractmethod
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
        # Must populate self._E, self._H, self._time_offset_E, and self._time_offset_H.
        # When dispersive_* are provided, the concrete implementation is expected
        # to use them to compute a frequency-corrected inverse permittivity at
        # the source carrier frequency so that the injected E/H ratio matches
        # the real impedance of the local medium.
        raise NotImplementedError()

    def _metric_scale_at_plane(
        self,
        stencil: str,
        normal_axis: int | None = None,
        start_index: int | None = None,
    ) -> jax.Array | float:
        """Computes the local derivative scale across a face along its normal axis.

        On non-uniform grids, spatial derivatives are scaled by the local cell width.
        Returns 1.0 on uniform grids.

        Args:
            stencil (str): Either "backward" or "forward" difference stencil.
            normal_axis (int | None): Axis normal to the face. Defaults to the
                propagation axis (the single-plane case).
            start_index (int | None): Grid index of the face along ``normal_axis``.
                Defaults to this object's slice start on that axis. The box source
                passes a per-face index so each face uses its own cell width.

        Returns:
            jax.Array | float: Scalar scale factor for the injected field corrections.
        """
        config = self._config
        if not config.has_nonuniform_grid:
            return 1.0
        grid = config.resolved_grid
        assert grid is not None
        if normal_axis is None:
            normal_axis = self.propagation_axis
        if start_index is None:
            start_index = self.grid_slice_tuple[normal_axis][0]
        widths = grid.cell_widths(normal_axis)
        width = widths[start_index]
        if stencil == "backward":
            width = 0.5 * (width + widths[max(start_index - 1, 0)])
        elif stencil != "forward":
            raise ValueError(f"Unknown derivative stencil: {stencil}")
        reference_spacing = c0 * config.time_step_duration / config.courant_number
        return reference_spacing / width

    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permeabilities
        if self._E is None or self._H is None or self._time_offset_E is None or self._time_offset_H is None:
            raise Exception("Need to apply random key before calling update")

        # if direction is negative, updates are reversed; inverse update is inverted
        sign = 1 if self.direction == "+" else -1
        if inverse:
            sign = -sign

        c = self._config.courant_number * self._metric_scale_at_plane("backward")
        return _tfsf_inject_E_face(
            E,
            grid_slice=self.grid_slice,
            normal_axis=self.propagation_axis,
            sign=sign,
            incident_H=self._H,
            time_offset_H=self._time_offset_H,
            temporal_H_filter=self._temporal_H_filter,
            inv_permittivities=inv_permittivities,
            c=c,
            time_step=time_step,
            delta_t=self._config.time_step_duration,
            temporal_profile=self.temporal_profile,
            wave_character=self.wave_character,
            static_amplitude_factor=self.static_amplitude_factor,
        )

    def update_H(
        self,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permittivities
        if self._E is None or self._H is None or self._time_offset_E is None or self._time_offset_H is None:
            raise Exception("Need to apply random key before calling update")

        # if direction is negative, updates are reversed; inverse update is inverted
        sign = 1 if self.direction == "+" else -1
        if inverse:
            sign = -sign

        c = self._config.courant_number * self._metric_scale_at_plane("forward")
        return _tfsf_inject_H_face(
            H,
            grid_slice=self.grid_slice,
            normal_axis=self.propagation_axis,
            sign=sign,
            incident_E=self._E,
            time_offset_E=self._time_offset_E,
            inv_permeabilities=inv_permeabilities,
            c=c,
            time_step=time_step,
            delta_t=self._config.time_step_duration,
            temporal_profile=self.temporal_profile,
            wave_character=self.wave_character,
            static_amplitude_factor=self.static_amplitude_factor,
        )
