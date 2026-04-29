from typing import Literal

import jax
import jax.numpy as jnp


def get_wave_vector_raw(
    direction: Literal["+", "-"],
    propagation_axis: int,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:  # shape (3,)
    vec_list = [0, 0, 0]
    sign = 1 if direction == "+" else -1
    vec_list[propagation_axis] = sign
    return jnp.array(vec_list, dtype=dtype)


def get_orthogonal_vector(
    v_E: jax.Array | None = None,
    v_H: jax.Array | None = None,
    wave_vector: jax.Array | None = None,
    direction: Literal["+", "-"] | None = None,
    propagation_axis: int | None = None,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    if (v_E is None) == (v_H is None):
        raise Exception(f"Invalid input to orthogonal vector computation: {v_E=}, {v_H=}")
    if (direction is None and propagation_axis is None) == (wave_vector is None):
        raise Exception(
            f"Invalid input to orthogonal vector computation: {direction=}, {propagation_axis=}, {wave_vector=}"
        )
    if direction is not None and propagation_axis is not None:
        wave_vector = get_wave_vector_raw(
            direction=direction,
            propagation_axis=propagation_axis,
            dtype=dtype,
        )
    elif wave_vector is None:
        raise Exception("Need to specify either wave_vector or direction and propagation axis")
    # assert wave_vector is not None
    if v_E is not None:
        orthogonal = jnp.cross(wave_vector, v_E)
    elif v_H is not None:
        orthogonal = jnp.cross(v_H, wave_vector)
    else:
        raise Exception("This should never happen")
    return orthogonal


def get_single_directional_rotation_matrix(
    rotation_axis: int,
    angle_radians: float | jax.Array,
) -> jax.Array:  # rotation matrix of shape (3, 3)
    """Generate a 3D rotation matrix for rotation around a specified axis.

    Args:
        rotation_axis (int): Axis around which to rotate (0=x, 1=y, 2=z)
        angle_radians (float | jax.Array): Rotation angle in radians

    Returns:
        jax.Array: 3x3 rotation matrix
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


def rotate_vector(
    vector: jax.Array,
    azimuth_angle: float | jax.Array,
    elevation_angle: float | jax.Array,
    axes_tuple: tuple[int, int, int],
) -> jax.Array:
    """Rotate a vector by specified azimuth and elevation angles.

    Transforms the vector from the global coordinate system to a rotated coordinate
    system defined by the azimuth and elevation angles.

    Args:
        vector (jax.Array): Input vector to rotate
        azimuth_angle (float | jax.Array): Rotation angle around vertical axis in radians
        elevation_angle (float | jax.Array): Rotation angle around horizontal axis in radians
        axes_tuple (tuple[int, int, int]): tuple of axes specifying horizontal_axis, vertical_axis,
            and propagation_axis.

    Returns:
        jax.Array: Rotated vector in global coordinates
    """

    horizontal_axis, vertical_axis, propagation_axis = axes_tuple

    # basis vectors — match the input vector's dtype to avoid implicit promotion warnings
    _dtype = vector.dtype
    e1 = jnp.zeros(3, dtype=_dtype).at[horizontal_axis].set(1)
    e2 = jnp.zeros(3, dtype=_dtype).at[vertical_axis].set(1)
    e3 = jnp.zeros(3, dtype=_dtype).at[propagation_axis].set(1)
    global_to_raw_basis = jnp.stack((e1, e2, e3), axis=0)
    raw_to_global_basis = jnp.transpose(global_to_raw_basis)

    # azimuth rotates horizontal around vertical axis
    az_matrix = get_single_directional_rotation_matrix(
        rotation_axis=1,
        angle_radians=azimuth_angle,
    )
    u = az_matrix @ jnp.array([1, 0, 0], dtype=_dtype)

    # elevation rotates vertical around horizontal axis
    el_matrix = get_single_directional_rotation_matrix(
        rotation_axis=0,
        angle_radians=elevation_angle,
    )
    v = el_matrix @ jnp.array([0, 1, 0], dtype=_dtype)
    w = jnp.cross(u, v)
    w = w / jnp.linalg.norm(w)

    rotation_basis = jnp.stack((u, v, w), axis=0)

    # vector transformation
    raw_vector = global_to_raw_basis @ vector
    rotated = rotation_basis @ raw_vector
    global_rotated = raw_to_global_basis @ rotated

    global_rotated = global_rotated / jnp.linalg.norm(global_rotated)
    global_rotated = global_rotated * jnp.linalg.norm(vector)

    return global_rotated
