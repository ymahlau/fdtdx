import jax
import jax.numpy as jnp

from fdtdx.core.misc import PaddingConfig, advanced_padding


def remove_floating_polymer(
    matrix: jax.Array,  # 1 is polymer, zero is air, shape (x, y, z)
) -> jax.Array:
    """Removes polymer regions that are not connected to the substrate.

    Uses flood-fill algorithm to identify polymer regions connected to the bottom layer
    and removes any floating polymer regions that are not connected.

    Args:
        matrix (jax.Array): Binary array where 1 represents polymer and 0 represents air.
               Shape is (x, y, z) representing the 3D grid.

    Returns:
        jax.Array: Modified binary array with floating polymer regions removed.
    """
    connected = compute_polymer_connection(matrix)
    non_connected_polymer = jnp.invert(connected) & matrix
    matrix = matrix & jnp.invert(non_connected_polymer)
    return matrix


def remove_polymer_non_connected_to_x_max_middle(
    matrix: jax.Array,  # 1 is polymer, zero is air, shape (x, y, z)
) -> jax.Array:
    """Removes polymer regions not connected to the middle of the x-max boundary.

    Uses flood-fill algorithm starting from the middle point of the x-max boundary
    to identify connected polymer regions and removes unconnected regions.

    Args:
        matrix (jax.Array): Binary array where 1 represents polymer and 0 represents air.
               Shape is (x, y, z) representing the 3D grid.

    Returns:
        jax.Array: Modified binary array with unconnected polymer regions removed.
    """
    y_middle = round(matrix.shape[1] / 2)
    x_middle = round(matrix.shape[0] / 2)
    connected = compute_polymer_connection(
        matrix,
        connected_slice=(x_middle, y_middle, None),
    )
    non_connected_polymer = jnp.invert(connected) & matrix
    matrix = matrix & jnp.invert(non_connected_polymer)
    return matrix


def connect_holes_and_structures(
    matrix: jax.Array,  # 1 is polymer, zero is air, shape (x, y, z)
) -> jax.Array:
    """Connects disconnected polymer structures and air holes in the matrix.

    Performs a two-pass algorithm:
    1. Connects disconnected polymer structures by adding polymer material
    2. Connects disconnected air regions by removing polymer material

    This ensures both structural integrity and proper air ventilation.

    Args:
        matrix (jax.Array): Binary array where 1 represents polymer and 0 represents air.
               Shape is (x, y, z) representing the 3D grid.

    Returns:
        jax.Array: Modified binary array with connected structures and holes.
    """
    matrix = matrix.astype(bool)
    # first connect all polymer structures  # TODO: convert to fori-loops
    for i in range(matrix.shape[-1] - 1):
        connected = compute_polymer_connection(matrix)
        if i == 0:
            lower_slice = jnp.ones_like(matrix[..., 0])
        else:
            lower_slice = matrix[..., i - 1]
        new_middle, new_upper = connect_slice(
            lower_slice=lower_slice,
            middle_slice=matrix[..., i],
            upper_slice=matrix[..., i + 1],
            upper_save_points=connected[..., i + 1],
        )
        matrix = matrix.at[..., i].set(new_middle)
        matrix = matrix.at[..., i + 1].set(new_upper)
    # then connect all air
    for i in range(matrix.shape[-1], 0, -1):
        air_connected = compute_air_connection(matrix)
        if i == matrix.shape[-1]:
            lower_slice = jnp.ones_like(matrix[..., 0])
        else:
            lower_slice = jnp.invert(matrix[..., i + 1])
        new_middle, new_upper = connect_slice(
            lower_slice=lower_slice,
            middle_slice=jnp.invert(matrix[..., i]),
            upper_slice=jnp.invert(matrix[..., i - 1]),
            upper_save_points=air_connected[..., i - 1],
        )
        matrix = matrix.at[..., i].set(jnp.invert(new_middle))
        matrix = matrix.at[..., i - 1].set(jnp.invert(new_upper))
    # throw away all polymer that got free floating due to air connection in last step
    connected = compute_polymer_connection(matrix)
    non_connected_polymer = jnp.invert(connected) & matrix
    matrix = matrix & jnp.invert(non_connected_polymer)
    return matrix


def compute_air_connection(matrix: jax.Array) -> jax.Array:
    """Computes a mask of air regions connected to the boundaries.

    Uses iterative dilation to identify air regions (zeros in the matrix)
    that are connected to the simulation boundaries. This is used to ensure
    proper ventilation in the structure.

    Args:
        matrix (jax.Array): Binary array where 1 represents polymer and 0 represents air.

    Returns:
        jax.Array: Boolean array marking air regions connected to boundaries.
    """
    inv_matrix = jnp.invert(matrix)
    n = max([matrix.shape[0], matrix.shape[1], matrix.shape[2]])
    n4_kernel = jnp.asarray(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    )
    connected = jnp.zeros_like(inv_matrix, dtype=bool)
    connected = connected.at[:, :, -1].set(True)
    connected = connected.at[0, :, :].set(True)
    connected = connected.at[-1, :, :].set(True)
    connected = connected.at[:, 0, :].set(True)
    connected = connected.at[:, -1, :].set(True)
    connected = connected & inv_matrix

    def _body_fn(_, arr):
        arr = seperated_3d_dilation(
            arr_3d=arr,
            kernel_xy=n4_kernel,
            kernel_xz=n4_kernel,
            kernel_yz=n4_kernel,
            reduction_arr=inv_matrix,
        )
        return arr

    connected = jax.lax.fori_loop(0, n, _body_fn, connected)

    return connected


def compute_polymer_connection(
    matrix: jax.Array,
    connected_slice: tuple[int | None, int | None, int | None] | None = None,
) -> jax.Array:
    """Computes a mask of polymer regions connected to specified points.

    Uses iterative dilation to identify polymer regions (ones in the matrix)
    that are connected either to the bottom layer or to specified points
    in connected_slice.

    Args:
        matrix (jax.Array): Binary array where 1 represents polymer and 0 represents air.
        connected_slice (tuple[int | None, int | None, int | None] | None, optional):
            Optional tuple of indices specifying starting points for the connection computation.
            If None, uses bottom layer.

    Returns:
        jax.Array: Boolean array marking connected polymer regions.
    """
    n = max([matrix.shape[0], matrix.shape[1], matrix.shape[2]])
    padded = False
    if matrix.shape[2] == 1:
        padded = True
        matrix = jnp.pad(matrix, pad_width=((0, 0), (0, 0), (1, 1)))
    n4_kernel = jnp.asarray(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    )
    connected = jnp.zeros_like(matrix, dtype=bool)
    if connected_slice is None:
        connected = connected.at[..., 0].set(True)
    else:
        connected = connected.at[connected_slice].set(True)

    def _body_fn(_, arr):
        arr = seperated_3d_dilation(
            arr_3d=arr,
            kernel_xy=n4_kernel,
            kernel_xz=n4_kernel,
            kernel_yz=n4_kernel,
            reduction_arr=matrix,
        )
        return arr

    connected = jax.lax.fori_loop(0, n, _body_fn, connected)

    if padded:
        connected = connected[..., 1:2]
    return connected


def connect_slice(
    lower_slice: jax.Array,
    middle_slice: jax.Array,  # this layer is optimized
    upper_slice: jax.Array,  # this layer is optimized,
    upper_save_points: jax.Array,  # points which are surely connected
) -> tuple[jax.Array, jax.Array]:
    """Connects polymer regions between adjacent slices.

    Attempts to connect disconnected polymer regions between three adjacent
    z-slices by modifying the middle and upper slices while preserving
    specified connection points.

    Args:
        lower_slice (jax.Array): Binary array representing the lower z-slice.
        middle_slice (jax.Array): Binary array representing the middle z-slice to be optimized.
        upper_slice (jax.Array): Binary array representing the upper z-slice to be optimized.
        upper_save_points (jax.Array): Boolean mask of points in upper slice that must remain connected.

    Returns:
        tuple[jax.Array, jax.Array]: Tuple of (modified_middle_slice, modified_upper_slice) with connected regions.
    """
    n = max(lower_slice.shape[0], lower_slice.shape[1])

    # define kernels
    n4_kernel = jnp.asarray(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    )
    n8_kernel = jnp.asarray(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    )
    direction_kernels = jnp.asarray(
        [
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        ],
        dtype=bool,
    )

    # first determine all well-supported points
    connected_points = upper_slice & middle_slice
    connected_points = connected_points | upper_save_points

    for _ in range(n):
        connected_points = dilate_jax(connected_points, n4_kernel)
        connected_points = connected_points & upper_slice
    upper_air = jnp.invert(upper_slice)
    non_connected_points = jnp.invert(upper_air | connected_points)

    # first try to connect by setting additional polymers in lower array
    connectable_region_lower = dilate_jax(middle_slice, n4_kernel)
    connectable_region_lower = connectable_region_lower | lower_slice  # connectable either vertical or horiz.
    connectable_by_lower = non_connected_points & connectable_region_lower
    # update matrix, non-connected and connected points
    middle_slice = middle_slice | connectable_by_lower
    connected_points = connected_points | connectable_by_lower
    for _ in range(n):
        connected_points = dilate_jax(connected_points, n4_kernel)
        connected_points = connected_points & upper_slice
    non_connected_points = jnp.invert(upper_air | connected_points)

    # then try to connect by adding polymer in upper array
    connectable_region_upper = dilate_jax(connected_points, n8_kernel)
    connectable_by_upper = non_connected_points & connectable_region_upper
    connection_points = jax.vmap(dilate_jax, in_axes=(None, 0))(connectable_by_upper, direction_kernels)
    valid_connection_points = connectable_region_upper & connection_points
    valid_connection_points = jnp.sum(valid_connection_points, axis=0).astype(bool)
    # update matrix, non-connected and connected points
    upper_slice = upper_slice | valid_connection_points
    for _ in range(n):
        connected_points = dilate_jax(connected_points, n4_kernel)
        connected_points = connected_points & upper_slice
    non_connected_points = jnp.invert(upper_air | connected_points)

    # delete all non-connected
    upper_slice = upper_slice & jnp.invert(non_connected_points)

    return middle_slice, upper_slice


def dilate_jax(image: jax.Array, kernel: jax.Array) -> jax.Array:
    """Performs morphological dilation on a binary image using JAX.

    Args:
        image (jax.Array): Binary input array to be dilated.
        kernel (jax.Array): Binary kernel defining the dilation shape.

    Returns:
        jax.Array: Dilated binary array.
    """
    conv = jax.scipy.signal.convolve2d(image, kernel, mode="same", boundary="fill")
    binary_arr = jnp.asarray(conv, dtype=bool)
    return binary_arr


def erode_jax(image: jax.Array, kernel: jax.Array) -> jax.Array:
    """Performs morphological erosion on a binary image using JAX.

    Args:
        image (jax.Array): Binary input array to be eroded.
        kernel (jax.Array): Binary kernel defining the erosion shape.

    Returns:
        jax.Array: Eroded binary array.
    """
    conv = jax.scipy.signal.convolve2d(~image, kernel, mode="same", boundary="fill")
    binary_arr = jnp.asarray(conv, dtype=bool)
    return ~binary_arr


def seperated_3d_dilation(
    arr_3d: jax.Array,
    kernel_xy: jax.Array,
    kernel_yz: jax.Array,
    kernel_xz: jax.Array,
    reduction_arr: jax.Array,
) -> jax.Array:
    """Performs separated 3D dilation along each axis plane.

    Applies 2D dilation kernels separately in the XY, YZ, and XZ planes
    to approximate 3D dilation while being more computationally efficient.
    The result is masked by the reduction array after each operation.

    Args:
        arr_3d (jax.Array): 3D binary array to be dilated.
        kernel_xy (jax.Array): 2D kernel for XY plane dilation.
        kernel_yz (jax.Array): 2D kernel for YZ plane dilation.
        kernel_xz (jax.Array): 2D kernel for XZ plane dilation.
        reduction_arr (jax.Array): Binary mask to constrain dilation results.

    Returns:
        jax.Array: Dilated 3D binary array.
    """

    def convolve_partial(image: jax.Array, kernel: jax.Array):
        return jax.scipy.signal.convolve2d(image, kernel, mode="same", boundary="fill")

    arr_3d = jax.vmap(convolve_partial, in_axes=(2, None), out_axes=(2))(arr_3d, kernel_xy)
    arr_3d = jnp.asarray(arr_3d, dtype=bool)
    arr_3d = arr_3d & reduction_arr

    arr_3d = jax.vmap(convolve_partial, in_axes=(1, None), out_axes=(1))(arr_3d, kernel_xz)
    arr_3d = jnp.asarray(arr_3d, dtype=bool)
    arr_3d = arr_3d & reduction_arr

    arr_3d = jax.vmap(convolve_partial, in_axes=(0, None), out_axes=(0))(arr_3d, kernel_yz)
    arr_3d = jnp.asarray(arr_3d, dtype=bool)
    arr_3d = arr_3d & reduction_arr

    return arr_3d


def binary_median_filter(
    arr_3d: jax.Array,
    kernel_sizes: tuple[int, int, int],
    padding_cfg: PaddingConfig,
) -> jax.Array:
    """Applies a binary median filter to a 3D array.

    Implements a median filter for binary data by using convolution
    and thresholding. The filter is applied separately along each axis
    using the specified kernel sizes.

    Args:
        arr_3d (jax.Array): 3D binary input array.
        kernel_sizes (tuple[int, int, int]): Tuple of (kx, ky, kz) specifying filter size in each dimension.
        padding_cfg (PaddingConfig): Configuration for padding the input array.

    Returns:
        jax.Array: Filtered binary array.
    """
    # padding
    padded_arr, orig_slice = advanced_padding(arr_3d, padding_cfg)
    padded_arr = padded_arr.astype(jnp.float32)
    # create filter kernel and convolve (avg filter)
    for axis, k_size in enumerate(kernel_sizes):
        shape_list = [1, 1, 1]
        shape_list[axis] = k_size
        shape_tpl = tuple(shape_list)
        kernel = jnp.ones(shape=shape_tpl, dtype=jnp.float32)
        padded_arr = jax.scipy.signal.convolve(padded_arr, kernel, mode="same", method="direct")
    # discretize again to get median
    kernel_sum = jnp.prod(jnp.asarray(kernel_sizes))
    result = padded_arr / kernel_sum
    result = result[orig_slice]
    result = jnp.round(result).astype(arr_3d.dtype)
    return result
