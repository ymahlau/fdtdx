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
    if isinstance(stl, str):
        your_mesh: trimesh.Trimesh = trimesh.load_mesh(stl)  # type: ignore
    else:
        your_mesh = stl

    mesh_voxel = your_mesh.voxelized(voxel_size).fill()

    occupancy_array = mesh_voxel.encoding.dense

    resized_occupancy_array = jax.image.resize(
        occupancy_array, target_shape, method="nearest"
    )

    object_permittivity = jnp.where(
        resized_occupancy_array > 0, permittivity, ambient_permittivity
    )

    return jnp.tile(object_permittivity[..., jnp.newaxis], (1, 1, 1, 3))
