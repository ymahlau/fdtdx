import itertools
import math
from functools import partial
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from matfree import eig

from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit, frozen_field


def is_on_at_time_step(
    is_on: bool,
    start_time: float | None,
    start_after_periods: float | None,
    end_time: float | None,
    end_after_periods: float | None,
    on_for_time: float | None,
    on_for_periods: float | None,
    time_step: int,
    time_step_duration: float,
    period: float | None,
) -> bool:  # scalar bool
    """Determines if a time-dependent component should be active at a given time step.

    Args:
        is_on: Base on/off state
        start_time: Absolute start time
        start_after_periods: Start time in terms of periods
        end_time: Absolute end time
        end_after_periods: End time in terms of periods
        on_for_time: Duration to stay on in absolute time
        on_for_periods: Duration to stay on in terms of periods
        time_step: Current simulation time step
        time_step_duration: Duration of each time step
        period: Period length for period-based timing

    Returns:
        bool: True if the component should be active at the given time step
    """
    if not is_on:
        return False

    # validate start/end/on time
    if (
        any(
            [
                x is not None
                for x in [
                    start_after_periods,
                    end_after_periods,
                    on_for_periods,
                ]
            ]
        )
        and period is None
    ):
        raise Exception("Need to specify period!")
    num_start_specs = sum(
        [
            start_time is not None,
            start_after_periods is not None,
            on_for_time is not None and end_time is not None,
            on_for_periods is not None and end_time is not None,
            on_for_time is not None and end_after_periods is not None,
            on_for_periods is not None and end_after_periods is not None,
        ]
    )
    if num_start_specs > 1:
        raise Exception("Invalid start time specification!")
    if num_start_specs == 0:
        start_time = 0
    num_end_specs = sum(
        [
            end_time is not None,
            end_after_periods is not None,
            on_for_time is not None and start_time is not None,
            on_for_periods is not None and start_time is not None,
            on_for_time is not None and start_after_periods is not None,
            on_for_periods is not None and start_after_periods is not None,
        ]
    )
    if num_end_specs > 1:
        raise Exception("Invalid end time specification!")
    if num_end_specs == 0:
        end_time = math.inf

    # period to actual time
    if start_after_periods is not None:
        if period is None:
            raise Exception("This should never happen")
        start_time = start_after_periods * period
    if end_after_periods is not None:
        if period is None:
            raise Exception("This should never happen")
        end_time = end_after_periods * period
    if on_for_periods is not None:
        if period is None:
            raise Exception("This should never happen")
        on_for_time = on_for_periods * period

    # determine start/end time
    if start_time is None and on_for_time is not None:
        if end_time is None:
            raise Exception("This should never happen")
        start_time = end_time - on_for_time

    if end_time is None and on_for_time is not None:
        if start_time is None:
            raise Exception("This should never happen")
        end_time = start_time + on_for_time

    # check if on
    if start_time is None or end_time is None:
        raise Exception("This should never happen")
    time_passed = time_step * time_step_duration
    on = True
    on = on and (start_time <= time_passed)
    on = on and (time_passed <= end_time)
    return on


def expand_matrix(matrix: jax.Array, grid_points_per_voxel: tuple[int, ...], add_channels: bool = True) -> jax.Array:
    """Expands a matrix by repeating values along spatial dimensions and optionally adding channels.

    Used to upsample a coarse grid to a finer simulation grid by repeating values. Can also add
    vector field components as channels.

    Args:
        matrix: Input matrix to expand
        grid_points_per_voxel: Number of grid points to expand each voxel into along each dimension
        add_channels: If True, adds and tiles 3 channels for vector field components

    Returns:
        jax.Array: Expanded matrix with repeated values and optional channels
    """
    """Expands a matrix by repeating values along spatial dimensions and optionally adding channels.

    Args:
        matrix: Input matrix to expand
        grid_points_per_voxel: Number of grid points to expand each voxel into along each dimension
        add_channels: If True, adds and tiles 3 channels for vector field components

    Returns:
        jax.Array: Expanded matrix with repeated values and optional channels
    """
    if matrix.ndim == 2:
        matrix = jnp.expand_dims(matrix, axis=-1)
    expanded_matrix = jnp.repeat(matrix, grid_points_per_voxel[0], axis=0)
    expanded_matrix = jnp.repeat(expanded_matrix, grid_points_per_voxel[1], axis=1)
    expanded_matrix = jnp.repeat(expanded_matrix, grid_points_per_voxel[2], axis=2)
    if add_channels:
        if matrix.ndim == 3:
            expanded_matrix = jnp.expand_dims(expanded_matrix, axis=-1)
        expanded_matrix = jnp.tile(expanded_matrix, tuple(1 for _ in grid_points_per_voxel) + (3,))
    return expanded_matrix


def ensure_slice_tuple(t: Sequence[slice | int | Tuple[int, int]]) -> Tuple[slice, ...]:
    """
    Ensures that all elements of the input sequence are converted to slices.

    This function takes a sequence of elements that can be slices, integers,
    or tuples of integers and returns a tuple of slices. Integers are converted
    to slices that select a single item, and tuples are converted to slices
    that select a range of items.

    Args:
        t: A sequence of elements where each element is either a slice, an
            integer, or a tuple of two integers representing the start and end
            of a slice range.

    Returns:
        A tuple of slices corresponding to the input sequence.
    """

    def to_slice(loc):
        if isinstance(loc, int):
            return slice(loc, loc + 1)
        elif isinstance(loc, slice):
            return loc
        elif isinstance(loc, (tuple, list)) and len(loc) == 2 and all(isinstance(i, int) for i in loc):
            return slice(*loc)
        else:
            raise ValueError(f"Invalid location type: {type(loc)}. Expected int, slice, or tuple of ints.")

    return tuple(to_slice(loc) for loc in t)


def is_float_divisible(a, b, tolerance=1e-15):
    """
    Checks if a floating-point number 'a' is divisible by another floating-point number 'b'.

    Args:
        a: The dividend (float).
        b: The divisor (float).
        tolerance: A small tolerance value to account for floating-point imprecision.

    Returns:
        True if 'a' is divisible by 'b' within the specified tolerance, False otherwise.
    """
    # Check if divisor is zero
    if abs(b) < tolerance:
        return False

    # Calculate the remainder
    remainder = a % b

    # Check if the remainder is within the tolerance
    return abs(remainder) <= tolerance or abs(remainder - b) <= tolerance


def is_index_in_slice(index, slice, seq_length):
    """Checks if an index is within a slice, accounting for potential None values"""
    start, stop, _ = slice.indices(seq_length)
    return start <= index < stop


def cast_floating_to_numpy(vals: dict[str, np.ndarray], dtype) -> dict[str, np.ndarray]:
    """Casts floating point arrays in a dictionary to a specified numpy dtype.

    Args:
        vals: Dictionary of numpy arrays to cast
        dtype: Target numpy dtype to cast to

    Returns:
        dict[str, np.ndarray]: Dictionary with arrays cast to specified dtype
    """
    return {
        k: np.real(v).astype(dtype)
        if v.dtype in [jnp.complex64, jnp.complex128] and dtype not in [jnp.complex64, jnp.complex128]
        else v.astype(dtype)
        for k, v in vals.items()
    }


def batched_diag_construct(arr: jax.Array):
    """Constructs diagonal matrices in a batched fashion from the last axis of the input array.

    Creates a batch of diagonal matrices where each matrix's diagonal is populated from
    a slice of the input array's last dimension.

    Args:
        arr: Input array where the last axis will be used to create diagonal matrices

    Returns:
        jax.Array: Batched diagonal matrices with shape [..., N, N] where N is arr.shape[-1]
    """
    """Constructs diagonal matrices in a batched fashion from the last axis of the input array.

    Args:
        arr: Input array where the last axis will be used to create diagonal matrices

    Returns:
        jax.Array: Batched diagonal matrices
    """

    # performs jnp.diag on the last axis in a batched fashion
    def _map_fn(arr_1d: jax.Array):
        return jnp.diag(arr_1d)

    result = jax.vmap(_map_fn)(arr.reshape(-1, arr.shape[-1]))
    result_reshaped = result.reshape(*arr.shape[:-1], arr.shape[-1], arr.shape[-1])
    return result_reshaped


@partial(jax.jit, static_argnames=["use_lanczos", "lanczos_factor", "index"])
def safe_svd(
    arr: jax.Array,
    key: jax.Array,
    index: int,
    use_lanczos: bool = False,
    lanczos_factor: float = 2,
    lanczos_bias: int = 10,
):
    """Performs SVD decomposition with safety checks and optional Lanczos algorithm.

    Args:
        arr: Input array for SVD
        key: JAX random key for Lanczos initialization
        index: Number of singular values/vectors to compute
        use_lanczos: Whether to use Lanczos algorithm instead of full SVD
        lanczos_factor: Factor for determining Lanczos iteration depth
        lanczos_bias: Bias term for Lanczos iteration depth

    Returns:
        tuple: (U, S, Vh) matrices from SVD decomposition
    """
    remove_dim = False
    if len(arr.shape) == 2:
        arr = arr[None, ...]
        remove_dim = True
    if len(arr.shape) != 3:
        raise Exception("Save svd only works with batched 3dims")

    def _zero_result(idx: int):
        del idx
        # min_rank = min(arr.shape[-2], arr.shape[-1])
        u = jnp.zeros(
            shape=tuple([*arr.shape[1:-2], arr.shape[-2], index]),
            dtype=arr.dtype,
        )
        s = jnp.zeros(
            shape=tuple([*arr.shape[1:-2], index]),
            dtype=arr.dtype,
        )
        vT = jnp.zeros(
            shape=tuple([*arr.shape[1:-2], index, arr.shape[-1]]),
            dtype=arr.dtype,
        )
        return u, s, vT

    def svd_normal(idx: int):
        if use_lanczos:
            u, s, vT = svd_lanczos(
                arr[idx],
                index=index,
                key=key,
                lanczos_factor=lanczos_factor,
                lanczos_bias=lanczos_bias,
            )
        else:
            u, s, vT = jnp.linalg.svd(arr[idx], full_matrices=False)
        return u[..., :index], s[..., :index], vT[..., :index, :]

    u_list, s_list, vT_list = [], [], []
    for idx in range(arr.shape[0]):
        u_i, s_i, vT_i = jax.lax.cond(jnp.allclose(arr[idx], 0, rtol=0, atol=1e-10), _zero_result, svd_normal, idx)
        # u_i, s_i, vT_i = svd_normal(idx)

        u_list.append(u_i)
        s_list.append(s_i)
        vT_list.append(vT_i)

    result_u = jnp.stack(u_list, axis=0)
    result_s = jnp.stack(s_list, axis=0)
    result_vT = jnp.stack(vT_list, axis=0)
    if remove_dim:
        result_u = result_u[0]
        result_s = result_s[0]
        result_vT = result_vT[0]
    return result_u, result_s, result_vT


def svd_lanczos(
    arr: jax.Array,
    index: int,
    key: jax.Array,
    lanczos_factor: float = 2,
    lanczos_bias: int = 10,
):
    """Performs SVD using the Lanczos algorithm for improved efficiency on large matrices.

    Args:
        arr: Input array for SVD
        index: Number of singular values/vectors to compute
        key: JAX random key for initialization
        lanczos_factor: Factor for determining iteration depth
        lanczos_bias: Bias term for iteration depth

    Returns:
        tuple: (U, S, Vh) matrices from SVD decomposition
    """
    if len(arr.shape) != 2:
        raise Exception(f"Invalid array shape: {arr.shape}")
    orig_dtype = arr.dtype
    arr = arr.astype(jnp.float32)
    # random initial vector
    m, n = arr.shape[-2], arr.shape[-1]
    v0 = jax.random.uniform(key, shape=(n,), dtype=arr.dtype)
    # Heuristic: Keep a few extra for better accuracy
    depth = min(m - 1, n - 1, round(lanczos_factor * index + lanczos_bias))

    def Av(x):
        return arr @ x

    def vA(x):
        return x @ arr

    U, S, Vh = eig.svd_partial(v0, depth, Av, vA, arr.shape)
    U, S, Vh = U.astype(orig_dtype), S.astype(orig_dtype), Vh.astype(orig_dtype)
    return U, S, Vh


def invert_dict(d: dict):
    """Inverts a dictionary by swapping keys and values.

    Args:
        d: Input dictionary to invert

    Returns:
        dict: Inverted dictionary where original values are keys and original keys are values
    """
    return {v: k for k, v in d.items()}


def prime_factorization(num: int) -> list[int]:
    """Computes the prime factorization of a number.

    Args:
        num: Integer to factorize

    Returns:
        list[int]: List of prime factors in ascending order
    """
    factors = []
    i = 2
    while i * i <= num:
        if num % i:
            i += 1
        else:
            num //= i
            factors.append(i)
    if num > 1:
        factors.append(num)
    return factors


def find_squarest_divisors(num: int) -> tuple[int, int]:
    """Finds two divisors of a number that are as close as possible to being square.

    Uses a greedy approximation to find two factors whose ratio is close to 1.

    Args:
        num: Integer to find divisors for

    Returns:
        tuple[int, int]: Two divisors whose product equals the input number
    """
    factors = prime_factorization(num)
    # greedy approximation of np-hard problem
    a, b = 1, 1
    for f in factors[::-1]:
        if a < b:
            a *= f
        else:
            b *= f
    return a, b


def index_1d_array(arr: jax.Array, val: jax.Array) -> jax.Array:
    """Finds the first index where a 1D array equals a given value.

    Args:
        arr: 1D input array to search
        val: Value to find in the array

    Returns:
        int: Index of first occurrence of val in arr

    Raises:
        Exception: If input array is not 1D
    """
    if len(arr.shape) != 1:
        raise Exception(f"index only works on 1d-array, got shape: {arr.shape}")
    first_idx = jnp.argmax(arr == val)
    return first_idx


def index_by_slice(
    arr: jax.Array,
    start: int | None,
    stop: int | None,
    axis: int,
    step: int = 1,
) -> jax.Array:
    """Indexes an array along a specified axis using slice notation.

    Args:
        arr: Input array to slice
        start: Starting index
        stop: Stopping index
        axis: Axis along which to slice
        step: Step size between elements

    Returns:
        jax.Array: Sliced array
    """
    slice_list = [slice(None) for _ in range(arr.ndim)]
    slice_list[axis] = slice(start, stop, step)
    return arr[tuple(slice_list)]


def index_by_slice_take_1d(
    arr: jax.Array,
    slice: slice,
    axis: int,
) -> jax.Array:
    """Takes elements from an array along one axis using a slice and JAX's take operation.

    Optimized version of array slicing that uses JAX's take operation for better performance
    when taking elements along a single axis.

    Args:
        arr: Input array
        slice: Slice object specifying which elements to take
        axis: Axis along which to take elements

    Returns:
        jax.Array: Array with selected elements

    Raises:
        Exception: If slice would result in empty array
    """
    """Takes elements from an array along one axis using a slice and JAX's take operation.

    Args:
        arr: Input array
        slice: Slice object specifying which elements to take
        axis: Axis along which to take elements

    Returns:
        jax.Array: Array with selected elements

    Raises:
        Exception: If slice would result in empty array
    """
    start, stop, step = slice.indices(arr.shape[axis])
    if start == 0 and stop == arr.shape[axis] and step == 1:
        return arr
    indices = jnp.arange(start, stop, step)
    if len(indices) == 0:
        raise Exception(f"Invalid slice: {slice}")
    arr = jnp.take(arr, indices, axis=axis, unique_indices=True, indices_are_sorted=True)
    return arr


def index_by_slice_take(
    arr: jax.Array,
    slices: Sequence[slice],
) -> jax.Array:
    """Takes elements from an array using multiple slices and JAX's take operation.

    Optimized version of array slicing that uses JAX's take operation for better performance
    when taking elements along multiple axes.

    Args:
        arr: Input array
        slices: Sequence of slice objects, one for each dimension

    Returns:
        jax.Array: Array with selected elements

    Raises:
        Exception: If any slice would result in empty array
    """
    """Takes elements from an array using multiple slices and JAX's take operation.

    Args:
        arr: Input array
        slices: Sequence of slice objects, one for each dimension

    Returns:
        jax.Array: Array with selected elements

    Raises:
        Exception: If any slice would result in empty array
    """
    for axis, s in enumerate(slices):
        start, stop, step = s.indices(arr.shape[axis])
        if start == 0 and stop == arr.shape[axis] and step == 1:
            continue
        indices = jnp.arange(start, stop, step)
        if len(indices) == 0:
            raise Exception(f"Invalid slice: {s}")
        arr = jnp.take(arr, indices, axis=axis, unique_indices=True, indices_are_sorted=True)
    return arr


def mask_1d_from_slice(
    s: slice,
    axis_size: int,
) -> jax.Array:
    """Creates a boolean mask array from a slice specification.

    Args:
        s: Slice object defining which elements should be True
        axis_size: Size of the axis to create mask for

    Returns:
        jax.Array: Boolean mask array with True values where slice selects elements
    """
    """Creates a boolean mask array from a slice specification.

    Args:
        s: Slice object defining which elements should be True
        axis_size: Size of the axis to create mask for

    Returns:
        jax.Array: Boolean mask array with True values where slice selects elements
    """
    start, stop, step = s.indices(axis_size)
    mask = jnp.zeros(shape=(axis_size,), dtype=jnp.bool)
    mask = mask.at[start:stop:step].set(1)
    return mask


def assimilate_shape(
    arr: jax.Array,
    ref_arr: jax.Array,
    ref_axes: tuple[int, ...],
    repeat_single_dims: bool = False,
) -> jax.Array:
    """Reshapes array to match reference array's dimensions for broadcasting.

    Inserts new dimensions of size 1 such that arr has same dimensions as ref_arr
    and can be broadcasted. Optionally repeats single dimensions to match ref_arr's shape.

    Args:
        arr: Array to reshape
        ref_arr: Reference array whose shape to match
        ref_axes: Tuple mapping arr's axes to ref_arr's axes
        repeat_single_dims: If True, repeats size-1 dimensions to match ref_arr

    Returns:
        jax.Array: Reshaped array that can be broadcasted with ref_arr

    Raises:
        Exception: If shapes are incompatible or axes mapping is invalid
    """
    """
    Inserts new dimensions of size 1 such that to_change has same dimensions
    as the reference arr and can be broadcasted.
    """
    if arr.ndim != len(ref_axes):
        raise Exception(f"Invalid axes: {arr.ndim=}, {ref_axes=}")
    if max(ref_axes) >= ref_arr.ndim:
        raise Exception(f"Invalid axes: {ref_arr.ndim=}, {ref_axes=}")
    for a, ra in enumerate(ref_axes):
        if ref_arr.shape[ra] != arr.shape[a] and arr.shape[a] != 1:
            raise Exception(f"Invalid shapes: {arr.shape=}, {ref_arr.shape=}")
    new_shape = [1] * len(ref_arr.shape)
    for a, ra in enumerate(ref_axes):
        new_shape[ra] = arr.shape[a]
    arr = jnp.reshape(arr, new_shape)
    if not repeat_single_dims:
        return arr
    for ra in ref_axes:
        if arr.shape[ra] == 1:
            arr = jnp.repeat(arr, ref_arr.shape[ra], axis=ra)
    return arr


def linear_interpolated_indexing(
    point: jax.Array,
    arr: jax.Array,
) -> jax.Array:
    """Performs linear interpolation at a point in an array.

    Args:
        point: Coordinates at which to interpolate
        arr: Array to interpolate from

    Returns:
        jax.Array: Interpolated value at the specified point

    Raises:
        Exception: If point dimensions don't match array dimensions
    """
    if point.ndim != 1 or point.shape[0] != arr.ndim:
        raise Exception(f"Invalid shape of point ({point.shape}) or arr {arr.shape}")
    indices = [[jnp.floor(point[a]), jnp.ceil(point[a])] for a in range(point.shape[0])]
    to_interpolate = jnp.asarray(list(itertools.product(*indices)), dtype=jnp.int32)
    weights = (1 - jnp.abs(to_interpolate - point[None, :])).prod(axis=-1)
    for axis in range(arr.ndim):
        invalid_mask = (to_interpolate[:, axis] < 0) | (to_interpolate[:, axis] >= arr.shape[axis])
        weights = jnp.where(invalid_mask, 0, weights)
        to_interpolate = jnp.where(invalid_mask[:, None], 0, to_interpolate)
    indexed_vals = arr[tuple(to_interpolate.T)]
    result = (weights * indexed_vals).sum() / (weights.sum() + 1e-8)
    return result


def get_air_name(permittivity_config: dict[str, float]):
    for k, v in permittivity_config.items():
        if v == 1:
            return k
    raise Exception(f"Could not find air in: {permittivity_config}")


@extended_autoinit
class PaddingConfig(ExtendedTreeClass):
    """
    Padding configuration. The order is:
    minx, maxx, miny, maxy, minz, maxz, ...
    or just single value that can be used for all
    """

    widths: Sequence[int] = frozen_field()
    modes: Sequence[str] = frozen_field()
    values: Sequence[float] = frozen_field(
        default=None,  # type: ignore
    )


def advanced_padding(
    arr: jax.Array,
    padding_cfg: PaddingConfig,
) -> tuple[jax.Array, tuple[slice, ...]]:
    """Pads the input array with configurable padding modes and widths.

    Args:
        arr: Input array to pad
        padding_cfg: Configuration object containing:
            - widths: Padding widths for each edge
            - modes: Padding modes (constant, edge, reflect etc)
            - values: Values to use for constant padding

    Returns:
        tuple[jax.Array, tuple[slice, ...]]: Tuple containing:
            - Padded array
            - Slice tuple to extract original array region
    """
    # default values
    if len(padding_cfg.widths) == 1:
        padding_cfg = padding_cfg.aset("widths", [padding_cfg.widths[0] for _ in range(2 * arr.ndim)])
    if len(padding_cfg.modes) == 1:
        padding_cfg = padding_cfg.aset("modes", [padding_cfg.modes[0] for _ in range(2 * arr.ndim)])
    if padding_cfg.values is None:
        padding_cfg = padding_cfg.aset("values", [0 for _ in range(2 * arr.ndim)])
    if len(padding_cfg.values) == 1:
        padding_cfg = padding_cfg.aset("values", [padding_cfg.values[0] for _ in range(2 * arr.ndim)])

    # sanity checks
    if len(padding_cfg.widths) % 2 != 0 or len(padding_cfg.widths) / 2 != arr.ndim:
        raise Exception(f"Invalid padding width: {padding_cfg.widths} for array with {arr.ndim} dimensions")
    if len(padding_cfg.modes) % 2 != 0 or len(padding_cfg.modes) / 2 != arr.ndim:
        raise Exception(f"Invalid padding width: {padding_cfg.modes} for array with {arr.ndim} dimensions")
    if len(padding_cfg.values) % 2 != 0 or len(padding_cfg.values) / 2 != arr.ndim:
        raise Exception(f"Invalid padding width: {padding_cfg.values} for array with {arr.ndim} dimensions")

    slices = [[0, arr.shape[ax]] for ax in range(arr.ndim)]
    for edge in range(2 * arr.ndim):
        is_end = edge % 2 != 0
        axis = math.floor(edge / 2)
        cur_width = padding_cfg.widths[edge]
        cur_mode = padding_cfg.modes[edge]
        cur_value = padding_cfg.values[edge]

        kwargs = {}
        if cur_mode == "constant":
            kwargs["constant_values"] = cur_value
        pad_width_tuple = tuple(
            [(0, 0) if ax != axis else ((0, cur_width) if is_end else (cur_width, 0)) for ax in range(arr.ndim)]
        )
        if not is_end:
            slices[axis][0] = cur_width
            slices[axis][1] += cur_width
        arr = jnp.pad(array=arr, pad_width=pad_width_tuple, mode=cur_mode, **kwargs)
    slices = ensure_slice_tuple(slices)  # type: ignore
    return arr, slices
