"""Low-level mirror-symmetry primitives shared by sources, detectors and the unfold helpers.

This module holds only array/parity mathematics and imports nothing from ``fdtdx.objects`` or
``fdtdx.fdtd``, so object classes can use it without an import cycle. The user-facing symmetry
machinery lives in :mod:`fdtdx.fdtd.symmetry`, which re-exports :func:`field_component_parity`.

Convention (identical to :mod:`fdtdx.fdtd.symmetry`): a symmetry plane normal to ``axis`` sits at
the *min edge* of the reduced domain, the upper half is kept, ``-1`` denotes a PEC (electric) wall
and ``+1`` a PMC (magnetic) wall.
"""

from typing import Literal

import jax
import jax.numpy as jnp


def field_component_parity(
    field_type: Literal["E", "H"],
    component: int,
    axis: int,
    wall: int,
) -> int:
    """Mirror parity of a field component across a symmetry plane.

    For a mirror plane normal to ``axis`` with the kept domain on the ``+axis`` side, this is the
    sign that relates the field on the discarded side to its mirror image on the kept side. A PEC
    wall forces tangential ``E`` (and normal ``H``) to zero on the plane; a PMC wall forces
    tangential ``H`` (and normal ``E``) to zero.

    Args:
        field_type (Literal["E", "H"]): Which field the component belongs to.
        component (int): Component axis index (0=x, 1=y, 2=z).
        axis (int): Axis the mirror plane is normal to (0=x, 1=y, 2=z).
        wall (int): ``-1`` for a PEC (electric) wall, ``+1`` for a PMC (magnetic) wall.

    Returns:
        int: ``+1`` (even / unchanged under reflection) or ``-1`` (odd / sign-flipped).
    """
    normal = component == axis
    if wall == -1:  # PEC: tangential E = 0
        if field_type == "E":
            return 1 if normal else -1
        return -1 if normal else 1
    elif wall == 1:  # PMC: tangential H = 0
        if field_type == "E":
            return -1 if normal else 1
        return 1 if normal else -1
    raise ValueError(f"wall must be -1 (PEC) or +1 (PMC), got {wall}")


def component_sits_on_plane(field_type: Literal["E", "H"], component: int, axis: int) -> bool:
    """Whether a Yee component's samples lie *on* a mirror plane normal to ``axis``.

    ``E_c`` is offset by half a cell along ``c`` only, so along ``axis`` it sits at integer
    positions (i.e. on the plane at index 0) for every component except ``c == axis``; ``H_c`` is
    offset along the two axes other than ``c``, so the opposite holds. The distinction sets the
    mirror index map: on-plane samples pair ``i`` with ``-i`` (index 0 is its own mirror), the
    half-cell-offset ones pair ``i`` with ``-i-1`` (a plain flip).

    Args:
        field_type (Literal["E", "H"]): Which field the component belongs to.
        component (int): Component axis index (0=x, 1=y, 2=z).
        axis (int): Axis the mirror plane is normal to.

    Returns:
        bool: True if the component's samples include the plane itself.
    """
    if field_type == "E":
        return component != axis
    if field_type == "H":
        return component == axis
    raise ValueError(f"field_type must be 'E' or 'H', got {field_type!r}")


def mirror_pairs_on_plane(
    field_type: Literal["E", "H"],
    component: int,
    axis: int,
    wall: int,
) -> bool:
    """Whether a component's mirror map pairs its samples *about* a shared plane row.

    This is the single place that decides the mirror index map, and it depends on the wall type, not
    only on the Yee offsets:

    * An **electric** plane sits on the tangential-``E`` node row at the reduced domain's min edge,
      so a component sampled there (see :func:`component_sits_on_plane`) is its own mirror and the
      pairs are ``m ± j``.
    * A **magnetic** plane sits half a cell *below* the reduced domain. Sources and materials are
      rasterized per cell, so their footprint is symmetric about the cell boundary below the min
      edge — the tangential-``H`` node one cell out — and across that plane *every* component
      mirrors one-to-one (the plain flip), whatever its Yee offset.

    The parity itself (:func:`field_component_parity`) is unaffected: it is a property of the wall
    type, not of where the plane sits.

    Args:
        field_type (Literal["E", "H"]): Which field the component belongs to.
        component (int): Component axis index (0=x, 1=y, 2=z).
        axis (int): Axis the mirror plane is normal to.
        wall (int): ``-1`` for a PEC (electric) wall, ``+1`` for a PMC (magnetic) wall.

    Returns:
        bool: True if the samples pair as ``m ± j`` about a shared row, False for a plain flip.
    """
    return wall == -1 and component_sits_on_plane(field_type, component, axis)


def mirror_material_array(
    array: jax.Array,
    axis: int,
    component_axis: int = 0,
) -> jax.Array:
    """Reflect a material array about a mirror plane at its min edge, along a spatial axis.

    Material tensors are even under reflection except for the off-diagonal entries with exactly one
    index equal to the mirror axis, which change sign (``eps_xy -> -eps_xy`` for a mirror normal to
    x). Isotropic (1-component) and diagonal (3-component) layouts therefore only need the flip.

    Args:
        array (jax.Array): Material array whose three trailing axes are spatial and whose
            ``component_axis`` holds 1, 3 or 9 tensor components (row-major 3x3 for 9).
        axis (int): Physical axis (0=x, 1=y, 2=z) the mirror plane is normal to.
        component_axis (int): Axis holding the tensor components. Defaults to 0.

    Returns:
        jax.Array: The reflected copy, same shape as the input.
    """
    spatial_axis = array.ndim - 3 + axis
    mirrored = jnp.flip(array, axis=spatial_axis)
    num_components = array.shape[component_axis]
    if num_components != 9:
        return mirrored
    signs = [
        -1.0 if ((row == axis) != (col == axis)) else 1.0  # exactly one index along the mirror axis
        for row in range(3)
        for col in range(3)
    ]
    sign_shape = [1] * array.ndim
    sign_shape[component_axis] = 9
    return mirrored * jnp.asarray(signs, dtype=array.dtype).reshape(sign_shape)


def mirror_material_cross_section(
    array: jax.Array,
    axes: tuple[int, ...],
    component_axis: int = 0,
) -> jax.Array:
    """Rebuild the full-domain material array from a symmetry-reduced one.

    For every axis in ``axes`` the reflected copy is prepended, doubling that axis: exactly the
    material distribution the reduced simulation models, since its mirror wall replaces the
    discarded half by the mirror image of the kept half.

    Args:
        array (jax.Array): Reduced material array (three trailing spatial axes).
        axes (tuple[int, ...]): Physical axes to mirror.
        component_axis (int): Axis holding the tensor components. Defaults to 0.

    Returns:
        jax.Array: Full-domain material array, doubled along each mirrored axis.
    """
    for axis in axes:
        spatial_axis = array.ndim - 3 + axis
        array = jnp.concatenate([mirror_material_array(array, axis, component_axis), array], axis=spatial_axis)
    return array


def mirror_edge_coordinates(edges: jax.Array) -> jax.Array:
    """Rebuild full-domain edge coordinates from the kept half's, mirrored about ``edges[0]``.

    Args:
        edges (jax.Array): Monotone edge coordinates of the kept half, ``n + 1`` entries.

    Returns:
        jax.Array: ``2n + 1`` edge coordinates, mirror-symmetric about the original first edge.
    """
    widths = edges[1:] - edges[:-1]
    full_widths = jnp.concatenate([jnp.flip(widths), widths])
    return jnp.concatenate([jnp.zeros((1,), dtype=edges.dtype), jnp.cumsum(full_widths)]) + (edges[0] - edges[-1])


def _slice_axis(array: jax.Array, axis: int, start: int, stop: int | None = None) -> jax.Array:
    """``array[..., start:stop, ...]`` along a single axis."""
    index: list[slice] = [slice(None)] * array.ndim
    index[axis] = slice(start, stop)
    return array[tuple(index)]


def mirror_about_interior_plane(array: jax.Array, axis: int, on_plane: bool) -> jax.Array:
    """Mirror image of an array whose mirror plane lies at the centre of ``axis``.

    The array holds ``2n`` samples split by a plane between index ``n - 1`` and ``n``. Samples that
    sit half a cell off the plane pair up as ``n - 1 - j <-> n + j`` (a plain flip); samples that sit
    *on* the plane pair up as ``n - j <-> n + j``, leaving index ``n`` as its own mirror and index
    ``0`` without a partner (its mirror would be one cell past the array). Index ``0`` is returned
    unchanged, which is harmless because it belongs to the discarded half.

    Args:
        array (jax.Array): Array to mirror along ``axis``.
        axis (int): Array axis to mirror.
        on_plane (bool): Whether the samples include the plane itself
            (see :func:`component_sits_on_plane`).

    Returns:
        jax.Array: The mirrored array, same shape as the input.
    """
    flipped = jnp.flip(array, axis=axis)
    if not on_plane:
        return flipped
    shifted = jnp.roll(flipped, shift=1, axis=axis)
    # roll wraps the far edge into index 0, which has no mirror partner: restore the original value.
    return jnp.concatenate([_slice_axis(array, axis, 0, 1), _slice_axis(shifted, axis, 1)], axis=axis)


def mirror_extend_low_side(array: jax.Array, axis: int, parity: int, on_plane: bool) -> jax.Array:
    """Low-side (discarded-half) block for an array whose mirror plane is at its min edge.

    Given the ``n`` kept samples along ``axis``, returns the ``n`` samples of the mirrored half in
    ascending order, so that ``concatenate([low, array])`` is the full-domain array. Samples half a
    cell off the plane mirror one-to-one (a plain flip). Samples *on* the plane have index 0 as
    their own mirror, so only indices ``1..n-1`` produce mirror images; the outermost missing sample
    (whose partner lies past the kept half) is filled by repeating its neighbour, which is the
    far-boundary cell of the reconstructed domain.

    Args:
        array (jax.Array): Kept half, mirrored along ``axis``.
        axis (int): Array axis to mirror.
        parity (int): ``+1`` (even) or ``-1`` (odd).
        on_plane (bool): Whether the samples include the plane itself.

    Returns:
        jax.Array: The mirrored low-side block, same shape as ``array``.
    """
    if not on_plane:
        return parity * jnp.flip(array, axis=axis)
    mirrored = parity * jnp.flip(_slice_axis(array, axis, 1), axis=axis)
    return jnp.concatenate([_slice_axis(mirrored, axis, 0, 1), mirrored], axis=axis)


def project_onto_parity(
    field: jax.Array,
    field_type: Literal["E", "H"],
    walls: dict[int, int],
) -> tuple[jax.Array, jax.Array]:
    """Project a full-cross-section ``(3, Nx, Ny, Nz)`` field onto the parity subspace of the walls.

    Keeps only the part of the field the walls admit: the even part of every component a wall makes
    even and the odd part of every component it makes odd. That part is exactly representable on the
    reduced domain — in particular the components a wall drives to zero on the plane come out zero
    there — so injecting the result does not fight the wall. The returned residual is the fraction of
    the field that was removed: a few tens of percent for a coarsely resolved mode (the mode solver's
    discrete mode is only mirror-symmetric to first order in the cell size), close to one when the
    wall type does not match the field at all.

    Each component is mirrored with the index map of its wall type (see
    :func:`mirror_pairs_on_plane` and :func:`mirror_about_interior_plane`), matching how the reduced
    FDTD run treats it.

    Args:
        field (jax.Array): Field array with a leading component axis and three spatial axes, spanning
            the full (mirrored) cross-section.
        field_type (Literal["E", "H"]): Whether ``field`` is electric or magnetic.
        walls (dict[int, int]): Mapping of mirror axis to wall type (``-1`` PEC, ``+1`` PMC).

    Returns:
        tuple[jax.Array, jax.Array]: The projected field and the removed fraction (relative L2 norm),
        as a JAX scalar so this stays usable inside ``jax.jit`` — a mode source or mode-overlap
        detector overlapping a :class:`~fdtdx.Device` solves its mode inside
        :func:`fdtdx.apply_params`, which callers do trace. Inspect the value only where it is
        concrete (see :func:`fdtdx.core.jax.utils.is_jax_tracer`).
    """
    projected = field
    for axis, wall in walls.items():
        components = []
        for component in range(3):
            parity = field_component_parity(field_type, component, axis, wall)
            on_plane = mirror_pairs_on_plane(field_type, component, axis, wall)
            single = projected[component : component + 1]
            mirrored = mirror_about_interior_plane(single, axis + 1, on_plane)
            components.append(0.5 * (single + parity * mirrored))
        projected = jnp.concatenate(components, axis=0)
    norm = jnp.linalg.norm(jnp.abs(field))
    residual = jnp.where(norm > 0, jnp.linalg.norm(jnp.abs(field - projected)) / norm, 0.0)
    return projected, residual


def restrict_to_kept_half(array: jax.Array, axes: tuple[int, ...]) -> jax.Array:
    """Slice the upper (kept) half off a full-domain array along each mirrored axis.

    Inverse of :func:`mirror_material_cross_section` for arrays whose three trailing axes are
    spatial.

    Args:
        array (jax.Array): Full-domain array (three trailing spatial axes).
        axes (tuple[int, ...]): Physical axes that were mirrored.

    Returns:
        jax.Array: The kept half, halved along each axis in ``axes``.
    """
    index: list[slice] = [slice(None)] * array.ndim
    for axis in axes:
        spatial_axis = array.ndim - 3 + axis
        index[spatial_axis] = slice(array.shape[spatial_axis] // 2, None)
    return array[tuple(index)]
