from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
import pytreeclass as tc
from matplotlib import pyplot as plt

from fdtdx.core.jax.pytrees import extended_autoinit, frozen_field
from fdtdx.core.misc import linear_interpolated_indexing
from fdtdx.core.physics import constants
from fdtdx.core.physics.metrics import compute_energy, normalize_by_energy
from fdtdx.core.physics.modes import compute_modes
from fdtdx.objects.sources.source import DirectionalPlaneSourceBase


@extended_autoinit
class PlaneSource(DirectionalPlaneSourceBase, ABC):
    """
    Total-Field/Scattered-Field (TFSF) implementation of a source.
    The boundary between the scattered field and total field is at a
    positive offset of 0.25 in the yee grid in the axis of propagation.
    """

    azimuth_angle: float = 0.0
    elevation_angle: float = 0.0
    max_angle_random_offset: float = 0.0
    max_vertical_offset: float = 0.0
    max_horizontal_offset: float = 0.0

    _E: jax.Array = frozen_field(
        default=None,
        init=False,
    )  # type: ignore
    _H: jax.Array = frozen_field(
        default=None,
        init=False,
    )  # type: ignore
    _time_offset_E: jax.Array = frozen_field(
        default=None,
        init=False,
    )  # type: ignore
    _time_offset_H: jax.Array = frozen_field(
        default=None,
        init=False,
    )  # type: ignore

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
        """Generate random azimuth and elevation angles within allowed offset ranges.

        Args:
            key: JAX random key for generating random values.

        Returns:
            tuple containing:
                - azimuth angle in radians
                - elevation angle in radians
        """
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
        """Calculate center position with random offset.

        Args:
            key: JAX random key for generating random offsets.

        Returns:
            jax.Array: Center coordinates with shape (2,) containing [horizontal, vertical]
                      positions with random offsets applied.
        """
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

    def _get_single_directional_rotation_matrix(
        self,
        rotation_axis: int,
        angle_radians: float | jax.Array,
    ) -> jax.Array:  # rotation matrix of shape (3, 3)
        """Generate a 3D rotation matrix for rotation around a specified axis.

        Args:
            rotation_axis: Axis around which to rotate (0=x, 1=y, 2=z)
            angle_radians: Rotation angle in radians

        Returns:
            jax.Array: 3x3 rotation matrix

        Raises:
            Exception: If rotation_axis is not 0, 1, or 2
        """
        if rotation_axis == 0:
            return jnp.asarray(
                [
                    [1, 0, 0],
                    [0, jnp.cos(angle_radians), -jnp.sin(angle_radians)],
                    [0, jnp.sin(angle_radians), jnp.cos(angle_radians)],
                ]
            )
        elif rotation_axis == 1:
            return jnp.asarray(
                [
                    [jnp.cos(angle_radians), 0, -jnp.sin(angle_radians)],
                    [0, 1, 0],
                    [jnp.sin(angle_radians), 0, jnp.cos(angle_radians)],
                ]
            )
        elif rotation_axis == 2:
            return jnp.asarray(
                [
                    [jnp.cos(angle_radians), -jnp.sin(angle_radians), 0],
                    [jnp.sin(angle_radians), jnp.cos(angle_radians), 0],
                    [0, 0, 1],
                ]
            )
        raise Exception(f"Invalid rotation axis: {rotation_axis}")

    def _rotate_vector(
        self,
        vector: jax.Array,
        azimuth_angle: float | jax.Array,
        elevation_angle: float | jax.Array,
    ) -> jax.Array:
        """Rotate a vector by specified azimuth and elevation angles.

        Transforms the vector from the global coordinate system to a rotated coordinate
        system defined by the azimuth and elevation angles.

        Args:
            vector: Input vector to rotate
            azimuth_angle: Rotation angle around vertical axis in radians
            elevation_angle: Rotation angle around horizontal axis in radians

        Returns:
            jax.Array: Rotated vector in global coordinates
        """
        # basis vectors
        e1_list, e2_list, e3_list = [0, 0, 0], [0, 0, 0], [0, 0, 0]
        e1_list[self.horizontal_axis] = 1
        e2_list[self.vertical_axis] = 1
        e3_list[self.propagation_axis] = 1
        e1 = jnp.asarray(e1_list, dtype=jnp.float32)
        e2 = jnp.asarray(e2_list, dtype=jnp.float32)
        e3 = jnp.asarray(e3_list, dtype=jnp.float32)
        global_to_raw_basis = jnp.stack((e1, e2, e3), axis=0)
        raw_to_global_basis = jnp.linalg.inv(global_to_raw_basis)

        # azimuth rotates horizontal around vertical axis
        az_matrix = self._get_single_directional_rotation_matrix(
            rotation_axis=1,
            angle_radians=azimuth_angle,
        )
        u = az_matrix @ jnp.asarray([1, 0, 0], dtype=jnp.float32)

        # elevation rotates vertical around horizontal axis
        el_matrix = self._get_single_directional_rotation_matrix(
            rotation_axis=0,
            angle_radians=elevation_angle,
        )
        v = el_matrix @ jnp.asarray([0, 1, 0], dtype=jnp.float32)
        w = jnp.cross(u, v)

        rotation_basis = jnp.stack((u, v, w), axis=0)

        # vector transformation
        raw_vector = global_to_raw_basis @ vector
        rotated = rotation_basis @ raw_vector
        global_rotated = raw_to_global_basis @ rotated

        return global_rotated

    def _calculate_time_offset_yee(
        self,
        center: jax.Array,
        wave_vector: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Calculate time offsets for E and H fields on the Yee grid.

        Computes phase variations and time offsets for electric and magnetic fields
        based on material properties and wave propagation direction.

        Args:
            center: Center position of the source
            wave_vector: Direction of wave propagation
            inv_permittivities: Inverse permittivity values
            inv_permeabilities: Inverse permeability values

        Returns:
            tuple containing:
                - Time offsets for E field components
                - Time offsets for H field components
        """
        # phase variation
        x, y, z = jnp.meshgrid(
            jnp.arange(self.grid_shape[0]),
            jnp.arange(self.grid_shape[1]),
            jnp.arange(self.grid_shape[2]),
            indexing="ij",
        )
        xyz = jnp.stack([x, y, z], axis=-1)
        center_list = [center[0], center[1]]
        center_list.insert(self.propagation_axis, 0)  # type: ignore
        center_3d = jnp.asarray(center_list, dtype=jnp.float32)[None, None, None, :]
        xyz = xyz - center_3d

        # yee grid offsets
        xyz_E = jnp.stack(
            [
                xyz + jnp.asarray([0.5, 0, 0])[None, None, None, :],
                xyz + jnp.asarray([0, 0.5, 0])[None, None, None, :],
                xyz + jnp.asarray([0, 0, 0.5])[None, None, None, :],
            ]
        )
        xyz_H = jnp.stack(
            [
                xyz + jnp.asarray([0, 0.5, 0.5])[None, None, None, :],
                xyz + jnp.asarray([0.5, 0, 0.5])[None, None, None, :],
                xyz + jnp.asarray([0.5, 0.5, 0])[None, None, None, :],
            ]
        )

        travel_offset_E = -jnp.dot(xyz_E, wave_vector)
        travel_offset_H = -jnp.dot(xyz_H, wave_vector)

        # adjust speed for material and calculate time offset
        delta_s = self._config.resolution
        delta_t = self._config.time_step_duration
        refractive_idx = 1 / jnp.sqrt(inv_permittivities * inv_permeabilities)
        velocity = (constants.c / refractive_idx)[None, ...]
        time_offset_E = travel_offset_E * delta_s / (velocity * delta_t)
        time_offset_H = travel_offset_H * delta_s / (velocity * delta_t)
        return time_offset_E, time_offset_H

    def _get_random_parts(self, key: jax.Array):
        """Generate random center position and angles for the source.

        Args:
            key: JAX random key for generating random values

        Returns:
            tuple containing:
                - center position array
                - azimuth angle in radians
                - elevation angle in radians
        """
        key, subkey = jax.random.split(key)
        center = self._get_center(subkey)

        key, subkey = jax.random.split(key)
        azimuth, elevation = self._get_azimuth_elevation(subkey)

        return center, azimuth, elevation

    @abstractmethod
    def get_EH_variation(
        self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
    ) -> tuple[
        jax.Array,  # E: (3, *grid_shape)
        jax.Array,  # H: (3, *grid_shape)
        jax.Array,  # time_offset_E: (3, *grid_shape)
        jax.Array,  # time_offset_H: (3, *grid_shape)
    ]:
        # normal coordinates
        raise NotImplementedError()

    def apply(
        self: Self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
    ) -> Self:
        E, H, time_offset_E, time_offset_H = self.get_EH_variation(
            key=key,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
        )
        self = self.aset("_E", E)
        self = self.aset("_H", H)
        self = self.aset("_time_offset_E", time_offset_E)
        self = self.aset("_time_offset_H", time_offset_H)
        return self

    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permeabilities
        if self._E is None or self._H is None or self._time_offset_E is None or self._time_offset_H is None:
            raise Exception("Need to apply random key before calling update")
        time_step = linear_interpolated_indexing(
            time_step.reshape(
                1,
            ),
            self._time_step_to_on_idx,
        )

        delta_t = self._config.time_step_duration
        inv_permittivity_slice = inv_permittivities[*self.grid_slice]

        # vertical incident wave part
        time_phase_H_v = (
            2 * jnp.pi * (time_step - 0.5 + self._time_offset_H[self.vertical_axis]) * delta_t / self.period
            + self.phase_shift
        )
        H_v_inc = self._H[self.vertical_axis] * jnp.cos(time_phase_H_v)
        H_v_inc = H_v_inc * self._config.courant_factor * inv_permittivity_slice
        H_v_inc = jax.lax.stop_gradient(H_v_inc)

        # horizontal incident wave part
        time_phase_H_h = (
            2 * jnp.pi * (time_step - 0.5 + self._time_offset_H[self.horizontal_axis]) * delta_t / self.period
            + self.phase_shift
        )
        H_h_inc = self._H[self.horizontal_axis] * jnp.cos(time_phase_H_h)
        H_h_inc = H_h_inc * self._config.courant_factor * inv_permittivity_slice
        H_h_inc = jax.lax.stop_gradient(H_h_inc)

        # if direction is negative, updates are reversed
        sign = 1 if self.direction == "+" else -1

        # inverse update is inverted
        if inverse:
            sign = -sign

        # update uses -H_v, we have to subtract update, resulting in +H_v
        E = E.at[self.horizontal_axis, *self.grid_slice].add(sign * H_v_inc)
        # update uses +H_h, we have to subtract update, resulting in -H_h
        E = E.at[self.vertical_axis, *self.grid_slice].add(-sign * H_h_inc)

        return E

    def update_H(
        self,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permittivities
        if self._E is None or self._H is None or self._time_offset_E is None or self._time_offset_H is None:
            raise Exception("Need to apply random key before calling update")
        time_step = linear_interpolated_indexing(
            time_step.reshape(
                1,
            ),
            self._time_step_to_on_idx,
        )

        delta_t = self._config.time_step_duration
        inv_permeability_slice = inv_permeabilities[*self.grid_slice]

        # horizontal incident wave part
        time_phase_E_h = (
            2 * jnp.pi * (time_step - 0.5 + self._time_offset_E[self.horizontal_axis]) * delta_t / self.period
            + self.phase_shift
        )
        E_h_inc = self._E[self.horizontal_axis] * jnp.cos(time_phase_E_h)
        E_h_inc = E_h_inc * self._config.courant_factor * inv_permeability_slice
        E_h_inc = jax.lax.stop_gradient(E_h_inc)

        # vertical incident wave part
        time_phase_E_v = (
            2 * jnp.pi * (time_step - 0.5 + self._time_offset_E[self.vertical_axis]) * delta_t / self.period
            + self.phase_shift
        )
        E_v_inc = self._E[self.vertical_axis] * jnp.cos(time_phase_E_v)
        E_v_inc = E_v_inc * self._config.courant_factor * inv_permeability_slice
        E_v_inc = jax.lax.stop_gradient(E_v_inc)

        # if direction is negative, updates are reversed
        sign = 1 if self.direction == "+" else -1

        # inverse update is inverted
        if inverse:
            sign = -sign

        # update used +E_h, we have to add update, resulting in +E_h
        H = H.at[self.vertical_axis, *self.grid_slice].add(sign * E_h_inc)
        # update used -E_v, we have to add update, resulting in -E_v
        H = H.at[self.horizontal_axis, *self.grid_slice].add(-sign * E_v_inc)

        return H


@tc.autoinit
class LinearlyPolarizedPlaneSource(PlaneSource, ABC):
    fixed_E_polarization_vector: tuple[float, float, float] | None = None
    fixed_H_polarization_vector: tuple[float, float, float] | None = None

    def _get_raw_EH_polarization(
        self,
    ) -> tuple[jax.Array, jax.Array]:
        # determine E/H polarization
        e_pol = self.fixed_E_polarization_vector
        h_pol = self.fixed_H_polarization_vector
        if h_pol is not None:
            h_pol = jnp.asarray(self.fixed_H_polarization_vector, dtype=jnp.float32)
            h_pol = h_pol / jnp.linalg.norm(h_pol)
        if e_pol is not None:
            e_pol = jnp.asarray(self.fixed_E_polarization_vector, dtype=jnp.float32)
            e_pol = e_pol / jnp.linalg.norm(e_pol)
        if e_pol is None:
            if h_pol is None:
                raise Exception("Need to specify either E or H polarization")
            e_pol = self._orthogonal_vector(v_H=h_pol)
        if h_pol is None:
            if e_pol is None:
                raise Exception("Need to specify either E or H polarization")
            h_pol = self._orthogonal_vector(v_E=e_pol)
        return e_pol, h_pol

    def get_EH_variation(
        self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
    ) -> tuple[
        jax.Array,  # E: (3, *grid_shape)
        jax.Array,  # H: (3, *grid_shape)
        jax.Array,  # time_offset_E: (3, *grid_shape)
        jax.Array,  # time_offset_H: (3, *grid_shape)
    ]:
        inv_permittivities = inv_permittivities[*self.grid_slice]
        inv_permeabilities = inv_permeabilities[*self.grid_slice]

        # determine E/H polarization
        e_pol_raw, h_pol_raw = self._get_raw_EH_polarization()
        wave_vector_raw = self._get_wave_vector_raw()

        center, azimuth, elevation = self._get_random_parts(key)

        # tilt polarizations
        wave_vector = self._rotate_vector(wave_vector_raw, azimuth, elevation)
        e_pol = self._rotate_vector(e_pol_raw, azimuth, elevation)
        h_pol = self._rotate_vector(h_pol_raw, azimuth, elevation)

        # update is amplitude multiplied by polarization
        amplitude_raw = self._get_amplitude_raw(center)[None, ...]

        # map amplitude to propagation plane
        w, h = jnp.meshgrid(
            jnp.arange(self.grid_shape[self.horizontal_axis]),
            jnp.arange(self.grid_shape[self.vertical_axis]),
            indexing="ij",
        )
        wh_indices = jnp.stack((w, h), axis=-1)
        wh_indices -= center
        # basis in plane
        h_list = [0, 0, 0]
        h_list[self.horizontal_axis] = 1
        h_axis = jnp.asarray(h_list, dtype=jnp.float32)
        u_basis = h_axis - jnp.dot(h_axis, wave_vector) * wave_vector
        u_basis = u_basis / jnp.linalg.norm(u_basis)
        v_basis = jnp.cross(wave_vector, u_basis)

        # projection
        def project(point):
            point_list = [point[0], point[1]]
            point_list.insert(self.propagation_axis, 0)
            point = jnp.asarray(point_list, dtype=jnp.float32)
            projection = point - jnp.dot(point, wave_vector) * wave_vector
            # Convert to plane coordinates
            u = jnp.dot(projection, u_basis)
            v = jnp.dot(projection, v_basis)
            return jnp.asarray((u, v), dtype=jnp.float32)

        float_projected = jax.vmap(project)(wh_indices.reshape(-1, 2))
        float_projected += center
        # interpolate floating indices in original array
        index_fn = jax.vmap(linear_interpolated_indexing, in_axes=(0, None))
        interp = index_fn(float_projected, amplitude_raw.squeeze())
        amplitude = interp.reshape(*amplitude_raw.shape)

        E = amplitude * e_pol[:, None, None, None]
        H = amplitude * h_pol[:, None, None, None]

        time_offset_E, time_offset_H = self._calculate_time_offset_yee(
            center=center,
            wave_vector=wave_vector,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
        )

        return E, H, time_offset_E, time_offset_H

    @abstractmethod
    def _get_amplitude_raw(
        self,
        center: jax.Array,
    ) -> jax.Array:  # shape (*grid_shape)
        # in normal coordinates, not yee grid
        del center
        raise NotImplementedError()


@tc.autoinit
class GaussianPlaneSource(LinearlyPolarizedPlaneSource):
    radius: float = tc.field(init=True, kind="KW_ONLY")  # type: ignore
    std: float = 1 / 3  # relative to radius

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
        grid_radius = self.radius / self._config.resolution
        profile = self._gauss_profile(
            width=self.grid_shape[self.horizontal_axis],
            height=self.grid_shape[self.vertical_axis],
            axis=self.propagation_axis,
            center=center,
            radii=(grid_radius, grid_radius),
            std=self.std,
        )
        return profile


@extended_autoinit
class ModePlaneSource(PlaneSource):
    mode_index: int = frozen_field(default=0)
    filter_pol: Literal["te", "tm"] | None = frozen_field(default=None)

    _inv_permittivity: jax.Array = frozen_field(default=None, init=False)  # type: ignore
    _inv_permeability: jax.Array = frozen_field(default=None, init=False)  # type: ignore

    def apply(
        self: Self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
    ) -> Self:
        if (
            self.azimuth_angle != 0
            or self.elevation_angle != 0
            or self.max_angle_random_offset != 0
            or self.max_vertical_offset != 0
            or self.max_horizontal_offset != 0
        ):
            raise NotImplementedError()

        self = super().apply(
            key=key,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
        )
        inv_permittivity_slice = inv_permittivities[*self.grid_slice]
        inv_permeability_slice = inv_permeabilities[*self.grid_slice]

        self = self.aset("_inv_permittivity", inv_permittivity_slice)
        self = self.aset("_inv_permeability", inv_permeability_slice)

        return self

    def _compute_modes(
        self,
        inv_permittivity_slice: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        input_dtype = inv_permittivity_slice.dtype

        permittivity_cross_section = 1 / inv_permittivity_slice
        other_axes = [a for a in range(3) if a != self.propagation_axis]
        coords = [np.arange(permittivity_cross_section.shape[dim] + 1) / 1e-6 for dim in other_axes]
        permittivity_squeezed = jnp.take(
            permittivity_cross_section,
            indices=0,
            axis=self.propagation_axis,
        )

        def mode_helper(permittivity):
            modes = compute_modes(
                frequency=self.frequency,
                permittivity_cross_section=permittivity,  # type: ignore
                coords=coords,
                num_modes=self.mode_index + 1,
                filter_pol=self.filter_pol,
                direction=self.direction,
            )

            mode_E_list, mode_H_list = [], []
            for mode in modes:
                if self.propagation_axis == 0:
                    mode_E, mode_H = (
                        np.stack([mode.Ez, mode.Ex, mode.Ey], axis=0).astype(np.complex64),
                        np.stack([mode.Hz, mode.Hx, mode.Hy], axis=0).astype(np.complex64),
                    )
                elif self.propagation_axis == 1:
                    # mode_E, mode_H = (  # untested
                    #     np.stack([mode.Ex, mode.Ez, mode.Ey], axis=0).astype(np.complex64),
                    #     np.stack([mode.Hx, mode.Hz, mode.Hy], axis=0).astype(np.complex64),
                    # )
                    raise NotImplementedError()
                elif self.propagation_axis == 2:
                    # mode_E, mode_H = (  # untested
                    #     np.stack([mode.Ez, mode.Ex, mode.Ey], axis=0).astype(np.complex64),
                    #     np.stack([mode.Hz, mode.Hx, mode.Hy], axis=0).astype(np.complex64),
                    # )
                    raise NotImplementedError()
                else:
                    raise Exception(f"Invalid popagation axis: {self.propagation_axis}")
                mode_E_list.append(mode_E)
                mode_H_list.append(mode_H)

                return mode_E_list[self.mode_index], mode_H_list[self.mode_index]

        result_shape_dtype = (
            jnp.zeros((3, *permittivity_squeezed.shape), dtype=jnp.complex64),
            jnp.zeros((3, *permittivity_squeezed.shape), dtype=jnp.complex64),
        )
        mode_E_raw, mode_H_raw = jax.pure_callback(
            mode_helper,
            result_shape_dtype,
            jax.lax.stop_gradient(permittivity_squeezed),
        )

        mode_E = jnp.real(jnp.expand_dims(mode_E_raw, axis=self.propagation_axis + 1)).astype(input_dtype)

        mode_H = jnp.real(jnp.expand_dims(mode_H_raw, axis=self.propagation_axis + 1)).astype(input_dtype)
        return mode_E, mode_H

    def get_EH_variation(
        self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
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
        inv_permeability_slice = inv_permeabilities[*self.grid_slice]

        time_offset_E, time_offset_H = self._calculate_time_offset_yee(
            center=center,
            wave_vector=self._get_wave_vector_raw(),
            inv_permittivities=jnp.ones_like(inv_permittivity_slice),
            inv_permeabilities=jnp.ones_like(inv_permeability_slice),
        )

        mode_E, mode_H = self._compute_modes(inv_permittivity_slice=inv_permittivity_slice)

        mode_E2_norm, mode_H2_norm = normalize_by_energy(
            E=mode_E,
            H=mode_H,
            inv_permittivity=inv_permittivity_slice,
            inv_permeability=inv_permeability_slice,
        )

        # energy = compute_energy(
        #     E=mode_E,
        #     H=mode_H,
        #     inv_permittivity=inv_permittivity_slice,
        #     inv_permeability=inv_permeability_slice,
        # )

        # E_l2 = jnp.linalg.norm(mode_E, axis=0)
        # H_l2 = jnp.linalg.norm(mode_H, axis=0)

        # E_normalized = mode_E / (E_l2[None, ...] + 1e-12)
        # H_normalized = mode_H / (H_l2[None, ...] + 1e-12)

        # mode_E2 = E_normalized * jnp.sqrt(energy)
        # mode_H2 = H_normalized * jnp.sqrt(energy)

        # energy2 = compute_energy(
        #     E=mode_E2,
        #     H=mode_H2,
        #     inv_permittivity=inv_permittivity_slice,
        #     inv_permeability=inv_permeability_slice,
        # )
        # energy_root = jnp.sqrt(jnp.sum(energy2))

        # mode_E2_norm = mode_E2 / energy_root
        # mode_H2_norm = mode_H2 / energy_root

        return mode_E2_norm, mode_H2_norm, time_offset_E, time_offset_H

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
        levels = jnp.linspace(energy_2d.min(), energy_2d.max(), 11)[1:]
        mode_cmap = "inferno"

        # Add contour lines on top of the imshow plot
        plt.contour(energy_2d, cmap=mode_cmap, alpha=0.5, levels=levels)
        plt.gca().set_aspect("equal")

        plt.colorbar()

        # Ensure the plot takes up the entire figure
        plt.tight_layout(pad=0)

        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


@tc.autoinit
class ConstantAmplitudePlaneSource(LinearlyPolarizedPlaneSource):
    amplitude: float = 1.0

    def _get_amplitude_raw(
        self,
        center: jax.Array,
    ) -> jax.Array:
        del center
        result = jnp.ones(shape=self.grid_shape, dtype=jnp.float32)
        return result


@tc.autoinit
class HardConstantAmplitudePlanceSource(DirectionalPlaneSourceBase):
    amplitude: float = 1.0
    fixed_E_polarization_vector: tuple[float, float, float] | None = None
    fixed_H_polarization_vector: tuple[float, float, float] | None = None

    def _get_raw_EH_polarization(
        self,
    ) -> tuple[jax.Array, jax.Array]:
        # determine E/H polarization
        e_pol = self.fixed_E_polarization_vector
        h_pol = self.fixed_H_polarization_vector
        if h_pol is not None:
            h_pol = jnp.asarray(self.fixed_H_polarization_vector, dtype=jnp.float32)
            h_pol = h_pol / jnp.linalg.norm(h_pol)
        if e_pol is not None:
            e_pol = jnp.asarray(self.fixed_E_polarization_vector, dtype=jnp.float32)
            e_pol = e_pol / jnp.linalg.norm(e_pol)
        if e_pol is None:
            if h_pol is None:
                raise Exception("Need to specify either E or H polarization")
            e_pol = self._orthogonal_vector(v_H=h_pol)
        if h_pol is None:
            if e_pol is None:
                raise Exception("Need to specify either E or H polarization")
            h_pol = self._orthogonal_vector(v_E=e_pol)
        return e_pol, h_pol

    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permittivities, inv_permeabilities
        if inverse:
            return E
        delta_t = self._config.time_step_duration
        time_phase = 2 * jnp.pi * time_step * delta_t / self.period + self.phase_shift
        magnitude = jnp.real(self.amplitude * jnp.exp(-1j * time_phase))
        e_pol, _ = self._get_raw_EH_polarization()
        E_update = e_pol[:, None, None, None] * magnitude

        E = E.at[:, *self.grid_slice].set(E_update.astype(E.dtype))
        return E

    def update_H(
        self,
        H: jax.Array,
        inv_permeabilities: jax.Array,
        inv_permittivities: jax.Array,
        time_step: jax.Array,
        inverse: bool,
    ):
        del inv_permeabilities, inv_permittivities
        if inverse:
            return H
        delta_t = self._config.time_step_duration
        time_phase = 2 * jnp.pi * time_step * delta_t / self.period + self.phase_shift
        magnitude = jnp.real(self.amplitude * jnp.exp(-1j * time_phase))
        _, h_pol = self._get_raw_EH_polarization()
        H_update = h_pol[:, None, None, None] * magnitude

        H = H.at[:, *self.grid_slice].set(H_update.astype(H.dtype))
        return H

    def apply(
        self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
    ) -> Self:
        del key, inv_permittivities, inv_permeabilities
        return self
