import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.path import Path

from fdtdx import constants


def calculate_spatial_offsets_yee() -> tuple[jax.Array, jax.Array]:
    offset_E = jnp.stack(
        [
            jnp.asarray([0.5, 0, 0])[None, None, None, :],
            jnp.asarray([0, 0.5, 0])[None, None, None, :],
            jnp.asarray([0, 0, 0.5])[None, None, None, :],
        ]
    )
    offset_H = jnp.stack(
        [
            jnp.asarray([0, 0.5, 0.5])[None, None, None, :],
            jnp.asarray([0.5, 0, 0.5])[None, None, None, :],
            jnp.asarray([0.5, 0.5, 0])[None, None, None, :],
        ]
    )
    return offset_E, offset_H


def calculate_time_offset_yee(
    center: jax.Array,
    wave_vector: jax.Array,
    inv_permittivities: jax.Array,
    inv_permeabilities: jax.Array | float,
    resolution: float,
    time_step_duration: float,
    effective_index: jax.Array | float | None = None,
    e_polarization: jax.Array | None = None,
    h_polarization: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    if inv_permittivities.ndim == 4:
        # Extract spatial shape from (1, Nx, Ny, Nz) or (3, Nx, Ny, Nz) or (9, Nx, Ny, Nz)
        spatial_shape = inv_permittivities.shape[1:]
    elif inv_permittivities.ndim == 3:
        # Legacy shape (Nx, Ny, Nz)
        spatial_shape = inv_permittivities.shape
    else:
        raise Exception(f"Invalid permittivity shape: {inv_permittivities.shape=}")

    if 1 not in spatial_shape:
        raise Exception(f"Expected one spatial axis to be one, but got {spatial_shape}")

    # phase variation
    x, y, z = jnp.meshgrid(
        jnp.arange(spatial_shape[0]),
        jnp.arange(spatial_shape[1]),
        jnp.arange(spatial_shape[2]),
        indexing="ij",
    )
    xyz = jnp.stack([x, y, z], axis=-1)
    center_list = [center[0], center[1]]
    propagation_axis = spatial_shape.index(1)
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

    if effective_index is not None:
        refractive_idx = effective_index * jnp.ones(spatial_shape)
    else:
        # adjust speed for material and calculate time offset
        if inv_permittivities.ndim == 4:
            # Remove when anisotropic case is verified
            if inv_permittivities.shape[0] == 3 or inv_permittivities.shape[0] == 9:
                # Check if diagonal isotropic (shape[0] == 3)
                is_diagonal_isotropic = (
                    (inv_permittivities.shape[0] == 3)
                    & jnp.allclose(inv_permittivities[0], inv_permittivities[1])
                    & jnp.allclose(inv_permittivities[1], inv_permittivities[2])
                )

                # Check if full tensor isotropic (shape[0] == 9)
                is_full_isotropic = (
                    (inv_permittivities.shape[0] == 9)
                    & jnp.allclose(inv_permittivities[0], inv_permittivities[4])
                    & jnp.allclose(inv_permittivities[4], inv_permittivities[8])
                    & jnp.allclose(inv_permittivities[1], 0.0)
                    & jnp.allclose(inv_permittivities[2], 0.0)
                    & jnp.allclose(inv_permittivities[3], 0.0)
                    & jnp.allclose(inv_permittivities[5], 0.0)
                    & jnp.allclose(inv_permittivities[6], 0.0)
                    & jnp.allclose(inv_permittivities[7], 0.0)
                )

                is_isotropic = is_diagonal_isotropic | is_full_isotropic

                def _raise_if_anisotropic(is_iso):
                    if not is_iso:
                        raise NotImplementedError(
                            "Gaussian or planewave sources within anisotropic materials are not supported yet."
                        )

                jax.debug.callback(_raise_if_anisotropic, is_isotropic)

            # For now just return the isotropic result
            inv_perm_eff = inv_permittivities[0]
        else:
            inv_perm_eff = inv_permittivities

        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim == 4:
            # Remove when anisotropic case is verified
            if inv_permeabilities.shape[0] == 3 or inv_permeabilities.shape[0] == 9:
                # Check if diagonal isotropic (shape[0] == 3)
                is_diagonal_isotropic = (
                    (inv_permeabilities.shape[0] == 3)
                    & jnp.allclose(inv_permeabilities[0], inv_permeabilities[1])
                    & jnp.allclose(inv_permittivities[1], inv_permittivities[2])
                )

                # Check if full tensor isotropic (shape[0] == 9)
                is_full_isotropic = (
                    (inv_permeabilities.shape[0] == 9)
                    & jnp.allclose(inv_permeabilities[0], inv_permeabilities[4])
                    & jnp.allclose(inv_permeabilities[4], inv_permeabilities[8])
                    & jnp.allclose(inv_permeabilities[1], 0.0)
                    & jnp.allclose(inv_permeabilities[2], 0.0)
                    & jnp.allclose(inv_permeabilities[3], 0.0)
                    & jnp.allclose(inv_permeabilities[5], 0.0)
                    & jnp.allclose(inv_permeabilities[6], 0.0)
                    & jnp.allclose(inv_permeabilities[7], 0.0)
                )

                is_isotropic = is_diagonal_isotropic | is_full_isotropic

                def _raise_if_anisotropic(is_iso):
                    if not is_iso:
                        raise NotImplementedError(
                            "Gaussian or planewave sources within anisotropic materials are not supported yet."
                        )

                jax.debug.callback(_raise_if_anisotropic, is_isotropic)

            # For now just return the isotropic result
            inv_perm_eff_perm = inv_permeabilities[0]
        else:
            inv_perm_eff_perm = inv_permeabilities

        refractive_idx = 1 / jnp.sqrt(inv_perm_eff * inv_perm_eff_perm)

    velocity = (constants.c / refractive_idx)[None, ...]
    time_offset_E = travel_offset_E * resolution / (velocity * time_step_duration)
    time_offset_H = travel_offset_H * resolution / (velocity * time_step_duration)
    return time_offset_E, time_offset_H


def polygon_to_mask(
    boundary: tuple[float, float, float, float],
    resolution: float,
    polygon_vertices: np.ndarray,
) -> np.ndarray:
    """
    Generate a 2D binary mask from a polygon.

    Args:
        boundary (tuple[float, float, float, float]): tuple of (min_x, min_y, max_x, max_y)
            Rectangular boundary in metrical units (meter).
        resolution (float): float
            Grid resolution (spacing between grid points) in metrical units
        polygon_vertices (np.ndarray): list of (x, y) tuples
            Vertices of the polygon in metrical units. Last point should equal first point.
            Must have shape (N, 2).
    Returns:
        np.ndarray: 2D binary mask where 1 indicates inside polygon, 0 indicates outside
    """
    assert polygon_vertices.ndim == 2
    assert polygon_vertices.shape[1] == 2
    min_x, min_y, max_x, max_y = boundary

    # Create coordinate arrays
    x_coords = np.arange(min_x, max_x + 0.5 * resolution, resolution)
    y_coords = np.arange(min_y, max_y + 0.5 * resolution, resolution)

    # Create meshgrid
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")

    # Flatten coordinates for point-in-polygon test
    points = np.column_stack((X.ravel(), Y.ravel()))

    # Create matplotlib Path object from polygon vertices
    polygon_path = Path(polygon_vertices)

    # Test which points are inside the polygon
    inside_polygon = polygon_path.contains_points(points)

    # Reshape back to 2D grid
    mask = inside_polygon.reshape(Y.shape).astype(bool)

    return mask
