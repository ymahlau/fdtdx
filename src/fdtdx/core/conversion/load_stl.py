"""Utilities for loading and converting STL files to permittivity arrays.

This module provides functionality to load 3D models from STL files and convert them
into voxelized permittivity arrays suitable for FDTD simulations. The conversion
process includes voxelization at a specified resolution and mapping of material
properties.

The main function load_stl() handles both file paths and pre-loaded trimesh objects,
making it flexible for different usage scenarios.
"""

import jax
import jax.numpy as jnp
import trimesh


def load_stl(
    stl: str | trimesh.Trimesh,
    permittivity: float,
    target_shape: tuple[int, int, int],
    voxel_size: float,
    ambient_permittivity: float = 1.0,
) -> jax.Array:
    """Loads an STL file and converts it to a voxelized permittivity array.

    This function takes either an STL file path or a trimesh.Trimesh object,
    voxelizes it at the specified resolution, and converts it to a permittivity
    array where voxels inside the mesh have the specified permittivity value
    and voxels outside have the ambient permittivity value.

    Args:
        stl: Path to an STL file or a trimesh.Trimesh object containing the mesh
        permittivity: Relative permittivity value for the object
        target_shape: Desired output shape as (nx, ny, nz)
        voxel_size: Size of each voxel in the initial voxelization
        ambient_permittivity: Relative permittivity value for the surrounding medium (default: 1.0)

    Returns:
        jax.Array: 4D array of shape (nx, ny, nz, 3) containing the permittivity values,
            where the last dimension represents the permittivity for each spatial direction
    """
    if isinstance(stl, str):
        your_mesh: trimesh.Trimesh = trimesh.load_mesh(stl)  # type: ignore
    else:
        your_mesh = stl

    mesh_voxel = your_mesh.voxelized(voxel_size).fill()

    occupancy_array = mesh_voxel.encoding.dense

    resized_occupancy_array = jax.image.resize(occupancy_array, target_shape, method="nearest")

    object_permittivity = jnp.where(resized_occupancy_array > 0, permittivity, ambient_permittivity)

    return jnp.tile(object_permittivity[..., jnp.newaxis], (1, 1, 1, 3))
