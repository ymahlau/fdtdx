import math
from pathlib import Path
import numpy as np
import trimesh
import gdspy

def idx_to_xyz(
    idx: np.ndarray, 
    shape: tuple[int, int, int]
):
    _, d1, d2 = shape
    x = idx // (d1 * d2)
    y = (idx // d2) % d1
    z = idx % d2
    return x, y, z

def xyz_to_idx(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    shape: tuple[int, int, int],
):
    _, d1, d2 = shape
    return x * (d1 * d2) + y * (d2) + z


def export_stl(
    matrix: np.ndarray, 
    stl_filename: Path | str,
    voxel_grid_size: tuple[int, int, int] = (1, 1, 1),
):
    if matrix.ndim != 3:
        raise Exception(f"Invalid matrix shape: {matrix.shape}")
    scaling_factor = np.asarray(voxel_grid_size, dtype=int)
    
    d0, d1, d2 = (
        matrix.shape[0] + 1,
        matrix.shape[1] + 1,
        matrix.shape[2] + 1,
    )
    vertex_shape = (d0, d1, d2)
    num_vertices = d0 * d1 * d2
    num_voxels = math.prod(matrix.shape)
    
    matrix_shape = (matrix.shape[0], matrix.shape[1], matrix.shape[2])
    x, y, z = idx_to_xyz(np.arange(num_voxels), matrix_shape)
    matrix_flat = matrix.reshape(-1)
    stacked_idx = np.stack([x, y, z], axis=-1)
    
    use_sides_arr = np.zeros((num_voxels, 6), dtype=bool)
    use_sides_arr[matrix_flat & (stacked_idx[..., 0] == 0), 0] = True   # left
    use_sides_arr[matrix_flat & (stacked_idx[..., 1] == 0), 1] = True   # front
    use_sides_arr[matrix_flat & (stacked_idx[..., 2] == 0), 2] = True   # bottom
    use_sides_arr[matrix_flat & (stacked_idx[..., 0] == matrix.shape[0] - 1), 3] = True   # right
    use_sides_arr[matrix_flat & (stacked_idx[..., 1] == matrix.shape[1] - 1), 4] = True   # back
    use_sides_arr[matrix_flat & (stacked_idx[..., 2] == matrix.shape[2] - 1), 5] = True   # top
    
    inv_matrix = ~matrix
    x_p1, y_p1, z_p1 = x+1, y+1, z+1
    x_p1[x_p1 == matrix.shape[0]] = matrix.shape[0] - 1
    y_p1[y_p1 == matrix.shape[1]] = matrix.shape[1] - 1
    z_p1[z_p1 == matrix.shape[2]] = matrix.shape[2] - 1
    
    use_sides_arr[matrix_flat & inv_matrix[x-1, y, z], 0] = True   # left
    use_sides_arr[matrix_flat & inv_matrix[x, y-1, z], 1] = True   # front
    use_sides_arr[matrix_flat & inv_matrix[x, y, z-1], 2] = True   # bottom
    use_sides_arr[matrix_flat & inv_matrix[x_p1, y, z], 3] = True   # right
    use_sides_arr[matrix_flat & inv_matrix[x, y_p1, z], 4] = True   # back
    use_sides_arr[matrix_flat & inv_matrix[x, y, z_p1], 5] = True   # top
    
    v_idx = np.asarray([
        xyz_to_idx(x, y, z, shape=vertex_shape),
        xyz_to_idx(x, y, z+1, shape=vertex_shape),
        xyz_to_idx(x, y+1, z, shape=vertex_shape),
        xyz_to_idx(x, y+1, z+1, shape=vertex_shape),
        xyz_to_idx(x+1, y, z, shape=vertex_shape),
        xyz_to_idx(x+1, y, z+1, shape=vertex_shape),
        xyz_to_idx(x+1, y+1, z, shape=vertex_shape),
        xyz_to_idx(x+1, y+1, z+1, shape=vertex_shape),
    ])
    
    faces_raw = np.asarray([
        np.stack([[v_idx[0], v_idx[1], v_idx[2]], [v_idx[1], v_idx[3], v_idx[2]]], axis=-1), # left
        np.stack([[v_idx[0], v_idx[4], v_idx[5]], [v_idx[5], v_idx[1], v_idx[0]]], axis=-1), # front
        np.stack([[v_idx[0], v_idx[2], v_idx[6]], [v_idx[6], v_idx[4], v_idx[0]]], axis=-1), # bottom
        np.stack([[v_idx[4], v_idx[6], v_idx[7]], [v_idx[7], v_idx[5], v_idx[4]]], axis=-1), # right
        np.stack([[v_idx[2], v_idx[3], v_idx[7]], [v_idx[7], v_idx[6], v_idx[2]]], axis=-1), # back
        np.stack([[v_idx[1], v_idx[5], v_idx[3]], [v_idx[3], v_idx[5], v_idx[7]]], axis=-1), # top
    ])
    faces_raw = faces_raw.transpose((2, 0, 3, 1))
    
    # select used faces only
    faces = faces_raw[use_sides_arr].reshape(-1, 3)
    vertex_arr = np.stack(
        idx_to_xyz(
            np.arange(num_vertices),
            shape=vertex_shape,
        ),
        axis=-1,
    )
    vertex_arr = vertex_arr * scaling_factor
    
    # export to trimesh
    mesh = trimesh.Trimesh(
        vertices=vertex_arr,
        faces=faces,
        validate=False,
    )
    mesh.export(stl_filename)



def export_mask_to_gds(
    arr: np.ndarray, 
    resolution: float,  # in µm
    gds_filename: str | Path,
    cell_name="MASK", 
    layer: int = 0, 
    datatype: int = 0,
):
    # 1. Convert to GDS Polygon Representation
    origins = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j]:
                origins.append((i, j))
    
    # 2. Create GDS Cell and Library
    lib = gdspy.GdsLibrary(
        unit=1e-6,   # µm
    )
    main_cell = lib.new_cell(cell_name)
    
    # 3. Add Polygons to Cell
    for origin in origins:
        rectangle = gdspy.Rectangle(
            (origin[0] * resolution, origin[1] * resolution),
            ((origin[0] + 1) * resolution, (origin[1] + 1) * resolution),
            layer=layer,
            datatype=datatype,
        )
        main_cell.add(rectangle)

    # 4. Save to GDS File
    lib.write_gds(
        gds_filename,
    )