from abc import ABC, abstractmethod
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
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

    Returns:
        JAX float32 array of shape ``(num_time_steps,)`` — the filtered
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
    inv_eps_inf_np = np.asarray(inv_eps_inf_slice)

    eps_spectrum = compute_eps_spectrum_from_coefficients(
        c1=c1_np,
        c2=c2_np,
        c3=c3_np,
        inv_eps_inf=inv_eps_inf_np,
        omegas=omegas_rfft,
        dt=dt,
    )
    omega_c = 2.0 * np.pi * wave_character.get_frequency()
    eps_center_arr = compute_eps_spectrum_from_coefficients(
        c1=c1_np,
        c2=c2_np,
        c3=c3_np,
        inv_eps_inf=inv_eps_inf_np,
        omegas=np.array([omega_c], dtype=np.float64),
        dt=dt,
    )
    eps_center = complex(eps_center_arr[0])

    s_H = compute_impedance_corrected_temporal_profile(
        raw_samples=raw_samples,
        dt=dt,
        eps_spectrum=eps_spectrum,
        eps_center=eps_center,
    )
    return jnp.asarray(s_H, dtype=jnp.float32)


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
        """Convert maximum vertical offset from physical units to grid points.

        Returns:
            float: Maximum vertical offset in grid points.
        """
        return self.max_vertical_offset / self._config.resolution

    @property
    def max_horizontal_offset_grid(self) -> float:
        """Convert maximum horizontal offset from physical units to grid points.

        Returns:
            float: Maximum horizontal offset in grid points.
        """
        return self.max_horizontal_offset / self._config.resolution

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
        # Calculate center position with random offset
        horizontal_size = self.grid_shape[self.horizontal_axis]
        vertical_size = self.grid_shape[self.vertical_axis]

        # account for zero-indexing in center calculation
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
            dtype=jnp.float32,
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
    ) -> Self:
        # Must populate self._E, self._H, self._time_offset_E, and self._time_offset_H.
        # When dispersive_* are provided, the concrete implementation is expected
        # to use them to compute a frequency-corrected inverse permittivity at
        # the source carrier frequency so that the injected E/H ratio matches
        # the real impedance of the local medium.
        raise NotImplementedError()

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

        delta_t = self._config.time_step_duration
        c = self._config.courant_number

        # Determine if fully anisotropic
        is_fully_anisotropic = inv_permittivities.ndim > 0 and inv_permittivities.shape[0] == 9

        # Slice the permittivity tensor at the TFSF boundary, shape: (num_components, Nx, Ny, Nz)
        inv_permittivity_slice = inv_permittivities[:, *self.grid_slice]

        # Calculate time points for H fields
        h_axis, v_axis, p_axis = self.horizontal_axis, self.vertical_axis, self.propagation_axis
        time_H = {}
        amplitude_H = {}
        for axis in [h_axis, v_axis, p_axis]:
            time_H[axis] = (time_step + self._time_offset_H[axis]) * delta_t
            if self._temporal_H_filter is None:
                amplitude_H[axis] = (
                    self.temporal_profile.get_amplitude(
                        time=time_H[axis],
                        period=self.wave_character.get_period(),
                        phase_shift=self.wave_character.phase_shift,
                    )
                    * self.static_amplitude_factor
                )
            else:
                # Dispersive background: look up the precomputed broadband
                # impedance-corrected H profile at the Yee-offset time step.
                # The time offset is spatially resolved — one fractional
                # index per source cell — so use jnp.interp which preserves
                # the array shape and zeros out-of-range samples (matching
                # the raw profile's behavior at the source on/off boundaries).
                idx_arr = time_step + self._time_offset_H[axis]
                xp = jnp.arange(self._temporal_H_filter.shape[0], dtype=idx_arr.dtype)
                amplitude_H[axis] = (
                    jnp.interp(idx_arr, xp, self._temporal_H_filter, left=0.0, right=0.0) * self.static_amplitude_factor
                )

        # if direction is negative, updates are reversed
        sign = 1 if self.direction == "+" else -1
        # inverse update is inverted
        if inverse:
            sign = -sign

        if is_fully_anisotropic:
            # vertical incident wave part
            H_v_inc = jax.lax.stop_gradient(self._H[v_axis] * amplitude_H[v_axis])
            # horizontal incident wave part
            H_h_inc = jax.lax.stop_gradient(self._H[h_axis] * amplitude_H[h_axis])

            def get_inv_eps(row, col):
                # get inverse permittivity tensor element at (row, col)
                idx = row * 3 + col
                return inv_permittivity_slice[idx]

            # update uses -H_v, we have to subtract update, resulting in +H_v
            # update uses +H_h, we have to subtract update, resulting in -H_h
            E_p_correction = c * (get_inv_eps(p_axis, h_axis) * (+H_v_inc) + get_inv_eps(p_axis, v_axis) * (-H_h_inc))
            E = E.at[p_axis, *self.grid_slice].add(sign * E_p_correction)
            E_h_correction = c * (get_inv_eps(h_axis, h_axis) * (+H_v_inc) + get_inv_eps(h_axis, v_axis) * (-H_h_inc))
            E = E.at[h_axis, *self.grid_slice].add(sign * E_h_correction)
            E_v_correction = c * (get_inv_eps(v_axis, h_axis) * (+H_v_inc) + get_inv_eps(v_axis, v_axis) * (-H_h_inc))
            E = E.at[v_axis, *self.grid_slice].add(sign * E_v_correction)

            return E

        else:
            # vertical incident wave part
            H_v_inc = self._H[v_axis] * amplitude_H[v_axis]
            # Use horizontal component of inv_permittivity for H_v calculation
            H_v_inc = H_v_inc * c * inv_permittivity_slice[h_axis]
            H_v_inc = jax.lax.stop_gradient(H_v_inc)

            # horizontal incident wave part
            H_h_inc = self._H[h_axis] * amplitude_H[h_axis]
            # Use vertical component of inv_permittivity for H_h calculation
            H_h_inc = H_h_inc * c * inv_permittivity_slice[v_axis]
            H_h_inc = jax.lax.stop_gradient(H_h_inc)

            # update uses -H_v, we have to subtract update, resulting in +H_v
            E = E.at[h_axis, *self.grid_slice].add(sign * H_v_inc)
            # update uses +H_h, we have to subtract update, resulting in -H_h
            E = E.at[v_axis, *self.grid_slice].add(-sign * H_h_inc)

            return E

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

        delta_t = self._config.time_step_duration
        c = self._config.courant_number

        # Determine if fully anisotropic
        is_fully_anisotropic = (
            isinstance(inv_permeabilities, jax.Array)
            and inv_permeabilities.ndim > 0
            and inv_permeabilities.shape[0] == 9
        )

        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            # Slice the permeability tensor at the TFSF boundary, shape: (num_components, Nx, Ny, Nz)
            inv_permeability_slice = inv_permeabilities[:, *self.grid_slice]
        else:
            inv_permeability_slice = inv_permeabilities

        # Calculate time points for E fields
        h_axis, v_axis, p_axis = self.horizontal_axis, self.vertical_axis, self.propagation_axis
        time_E = {}
        amplitude_E = {}
        for axis in [h_axis, v_axis, p_axis]:
            time_E[axis] = (time_step + self._time_offset_E[axis]) * delta_t
            amplitude_E[axis] = (
                self.temporal_profile.get_amplitude(
                    time=time_E[axis],
                    period=self.wave_character.get_period(),
                    phase_shift=self.wave_character.phase_shift,
                )
                * self.static_amplitude_factor
            )

        # if direction is negative, updates are reversed
        sign = 1 if self.direction == "+" else -1
        # inverse update is inverted
        if inverse:
            sign = -sign

        if is_fully_anisotropic:
            # horizontal incident wave part
            E_h_inc = jax.lax.stop_gradient(self._E[h_axis] * amplitude_E[h_axis])
            # vertical incident wave part
            E_v_inc = jax.lax.stop_gradient(self._E[v_axis] * amplitude_E[v_axis])

            def get_inv_mu(row, col):
                # get inverse permeability tensor element at (row, col)
                idx = row * 3 + col
                return inv_permeability_slice[idx]  # type: ignore

            # update uses +E_h, we have to add update, resulting in +E_h
            # update uses -E_v, we have to add update, resulting in -E_v
            H_p_correction = c * (get_inv_mu(p_axis, h_axis) * (-E_v_inc) + get_inv_mu(p_axis, v_axis) * (+E_h_inc))
            H = H.at[p_axis, *self.grid_slice].add(sign * H_p_correction)
            H_h_correction = c * (get_inv_mu(h_axis, h_axis) * (-E_v_inc) + get_inv_mu(h_axis, v_axis) * (+E_h_inc))
            H = H.at[h_axis, *self.grid_slice].add(sign * H_h_correction)
            H_v_correction = c * (get_inv_mu(v_axis, h_axis) * (-E_v_inc) + get_inv_mu(v_axis, v_axis) * (+E_h_inc))
            H = H.at[v_axis, *self.grid_slice].add(sign * H_v_correction)

            return H

        else:
            # horizontal incident wave part
            E_h_inc = self._E[h_axis] * amplitude_E[h_axis]
            # Use vertical component of inv_permeability for E_h calculation
            if isinstance(inv_permeability_slice, jax.Array) and inv_permeability_slice.ndim > 1:
                E_h_inc = E_h_inc * c * inv_permeability_slice[v_axis]
            else:
                E_h_inc = E_h_inc * c * inv_permeability_slice
            E_h_inc = jax.lax.stop_gradient(E_h_inc)

            # vertical incident wave part
            E_v_inc = self._E[v_axis] * amplitude_E[v_axis]
            # Use horizontal component of inv_permeability for E_v calculation
            if isinstance(inv_permeability_slice, jax.Array) and inv_permeability_slice.ndim > 1:
                E_v_inc = E_v_inc * c * inv_permeability_slice[h_axis]
            else:
                E_v_inc = E_v_inc * c * inv_permeability_slice
            E_v_inc = jax.lax.stop_gradient(E_v_inc)

            # update used +E_h, we have to add update, resulting in +E_h
            H = H.at[v_axis, *self.grid_slice].add(sign * E_h_inc)
            # update used -E_v, we have to add update, resulting in -E_v
            H = H.at[h_axis, *self.grid_slice].add(-sign * E_v_inc)

            return H
