import jax
import jax.numpy as jnp

from fdtdx import constants


def calculate_time_offset_yee(
    center: jax.Array,
    wave_vector: jax.Array,
    inv_permittivities: jax.Array,
    inv_permeabilities: jax.Array | float,
    resolution: float,
    time_step_duration: float,
) -> tuple[jax.Array, jax.Array]:
    if inv_permittivities.squeeze().ndim != 2 or inv_permittivities.ndim != 3:
        raise Exception(f"Invalid permittivity shape: {inv_permittivities.shape=}")
    # phase variation
    x, y, z = jnp.meshgrid(
        jnp.arange(inv_permittivities.shape[0]),
        jnp.arange(inv_permittivities.shape[1]),
        jnp.arange(inv_permittivities.shape[2]),
        indexing="ij",
    )
    xyz = jnp.stack([x, y, z], axis=-1)
    center_list = [center[0], center[1]]
    propagation_axis = inv_permittivities.shape.index(1)
    center_list.insert(propagation_axis, 0)  # type: ignore
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
    refractive_idx = 1 / jnp.sqrt(inv_permittivities * inv_permeabilities)
    velocity = (constants.c / refractive_idx)[None, ...]
    time_offset_E = travel_offset_E * resolution / (velocity * time_step_duration)
    time_offset_H = travel_offset_H * resolution / (velocity * time_step_duration)
    # time_offset_E = travel_offset_E / refractive_idx[None, ...]
    # time_offset_H = travel_offset_H / refractive_idx[None, ...]
    return time_offset_E, time_offset_H
