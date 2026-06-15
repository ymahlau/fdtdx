from collections import namedtuple
from types import SimpleNamespace
from typing import List, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import tidy3d
from jax.typing import ArrayLike
from tidy3d.components.mode.solver import compute_modes as _compute_modes

from fdtdx.core.axis import get_transverse_axes
from fdtdx.core.grid import yee_face_areas_from_edges
from fdtdx.core.misc import expand_to_3x3
from fdtdx.core.physics.metrics import normalize_by_poynting_flux

ModeTupleType = namedtuple("ModeTupleType", ["neff", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
"""A named tuple containing the mode fields and effective index.

Attributes:
    neff: Complex effective refractive index of the mode
    Ex: x-component of the electric field
    Ey: y-component of the electric field
    Ez: z-component of the electric field
    Hx: x-component of the magnetic field
    Hy: y-component of the magnetic field
    Hz: z-component of the magnetic field
"""


def _perm_mode_E(m: ModeTupleType, propagation_axis: int, dtype) -> np.ndarray:
    """Permute tidy3d mode E-field components into fdtdx physical axis order.

    tidy3d always places propagation along z; this maps fields to the fdtdx
    convention where the propagation axis may be 0, 1, or 2.  Returns a
    (3, n0, n1) array — the propagation-axis singleton is NOT included here;
    callers that need the full 4-D shape should add it with ``np.expand_dims``.
    """
    if propagation_axis == 0:
        return np.stack([m.Ez, m.Ex, m.Ey], axis=0).astype(dtype)
    elif propagation_axis == 1:
        return np.stack([m.Ex, m.Ez, m.Ey], axis=0).astype(dtype)
    else:
        return np.stack([m.Ex, m.Ey, m.Ez], axis=0).astype(dtype)


def compute_mode_polarization_fraction(
    mode: ModeTupleType,
    tangential_axes: tuple[int, int],
    pol: Literal["te", "tm"],
) -> float:
    """Mode polarization fraction.

    Args:
        mode (ModeTupleType): a ModeTupleType instance
        tangential_axes (tuple[int, int]): indices of transverse E-field component axes.
        pol (Literal["te", "tm"]): "te" or "tm" determines which axis is 'E1'

    Returns:
        float: Polarization fraction between 0 and 1.
    """

    E_fields = [mode.Ex, mode.Ey, mode.Ez]
    E1 = E_fields[tangential_axes[0]]
    E2 = E_fields[tangential_axes[1]]

    if pol == "te":
        numerator = np.sum(np.abs(E1) ** 2)
    elif pol == "tm":
        numerator = np.sum(np.abs(E2) ** 2)
    else:
        raise ValueError(f"pol must be 'te' or 'tm', but got {pol}")

    denominator = np.sum(np.abs(E1) ** 2 + np.abs(E2) ** 2) + 1e-18
    return numerator / denominator


def sort_modes(
    modes: list[ModeTupleType],
    filter_pol: Literal["te", "tm"] | None,
    tangential_axes: tuple[int, int],
) -> list[ModeTupleType]:
    """
    Sort modes by polarization.

    Args:
        modes (list[ModeTupleType]): list of modes.
        filter_pol (Literal["te", "tm"] | None): If not none, sort by polarization specificaton.
        tangential_axes (tuple[int, int]): indices of transverse E-field component axes.

    Returns:
        list[ModeTupleType]: sorted list of modes.
    """
    if filter_pol is None:
        return sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)

    def is_matching(mode):
        frac = compute_mode_polarization_fraction(mode, tangential_axes, filter_pol)
        return frac >= 0.5

    matching = [m for m in modes if is_matching(m)]
    non_matching = [m for m in modes if not is_matching(m)]

    matching_sorted = sorted(matching, key=lambda m: float(np.real(m.neff)), reverse=True)
    non_matching_sorted = sorted(non_matching, key=lambda m: float(np.real(m.neff)), reverse=True)

    return matching_sorted + non_matching_sorted


def compute_mode(
    frequency: float,
    inv_permittivities: jax.Array,  # shape (nx, ny, nz)
    inv_permeabilities: jax.Array | float,
    resolution: float | None = None,
    direction: Literal["+", "-"] = "+",
    mode_index: int = 0,
    filter_pol: Literal["te", "tm"] | None = None,
    dtype: jnp.dtype = jnp.float32,
    bend_radius: float | None = None,
    bend_axis: int | None = None,
    symmetry: tuple[int, int] = (0, 0),
    transverse_coords: Sequence[jax.Array] | None = None,
    target_neff: float | None = None,
    reference_E: np.ndarray | None = None,
) -> tuple[
    jax.Array,  # E
    jax.Array,  # H
    jax.Array,  # complex effective index
]:
    """Compute optical modes of a waveguide cross-section.

    This function uses the Tidy3D mode solver to compute the optical modes of a given waveguide cross-section defined
    by its permittivity distribution.

    By default modes are sorted by their effective index. The mode_index argument indexes this sorted list of modes and
    returns the desired mode. With filter_pol, it is also possible to only index a specific polarization.

    Args:
        frequency (float): Operating frequency in Hz
        inv_permittivities (jax.Array): 3D array of inverse relative permittivity values
        inv_permeabilities (jax.Array | float): 3D array of inverse relative permittivity values or single float for
            uniform permeability distribution.
        resolution (float | None): Uniform-grid spacing in metres. Required when ``transverse_coords`` is not
            provided (uniform-grid path). Ignored when ``transverse_coords`` is given. Defaults to None.
        direction (Literal["+", "-"]): Propagation direction, either "+" or "-".
        mode_index (int, optional): Index of the mode to compute. Defaults to 0.
        filter_pol (Literal["te", "tm"] | None, optional). If not None, modes are filtered by polarization.
        dtype (jnp.dtype, optional): Float dtype of the simulation. Controls whether mode fields are returned
            as complex64 (float32) or complex128 (float64). Defaults to jnp.float32.
        bend_radius (float | None, optional): Bend radius of the waveguide in meters. Must be set together with
            bend_axis. When set, the mode solver uses a conformal transformation to account for the bend. Defaults to
            None (straight waveguide).
        bend_axis (int | None, optional): Physical axis index (0/1/2) pointing from the waveguide toward the center
            of curvature. Must differ from the propagation axis. Required when bend_radius is set. Defaults to None.
        symmetry (tuple[int, int], optional): Symmetry-plane condition at the *min* edge of each transverse axis,
            in the order of the two non-propagation physical axes (increasing index). ``0`` imposes a PEC mirror
            (electric wall — the tidy3d default), ``1`` imposes a PMC mirror (magnetic wall). Use this when the
            waveguide sits on a symmetry plane of a reduced (half/quarter) domain so the mode solver reproduces the
            same boundary the FDTD uses there. For a +x-propagating TE mode on a y/z quarter domain with PEC at y=0
            and PMC at the z Si-mid plane, pass ``(0, 1)``. Defaults to ``(0, 0)`` (PEC on both, i.e. no symmetry).
        transverse_coords: Optional pair of physical edge-coordinate arrays, in metres, for the two axes transverse
            to propagation. Each array must have one more entry than the corresponding transverse cell count.
            When provided, the Tidy3D mode solver receives the non-uniform rectilinear grid directly.
            JAX arrays are accepted; the numpy conversion happens inside the tidy3d callback so the function
            remains compatible with ``jax.jit``.
        target_neff: When set, selects the mode whose ``real(neff)`` is closest to this
            value.  Useful for picking a specific mode by effective index at a single
            frequency.  Overridden by ``reference_E`` when both are provided.
        reference_E: When set, selects the mode with the highest field dot-product overlap
            with this reference electric field (shape ``(3, nx, ny, nz)`` in fdtdx axis
            order, with the propagation-axis dimension equal to 1).  Takes full precedence
            over both ``target_neff`` and ``mode_index``.  Pass ``np.array(mode_E)`` from
            the previous frequency's call to track the same physical mode across a sweep.

    Returns:
        Tuple[jax.Array, jax.Array, jax.Array]:
            Tuple of (E field, H field, effective index) as complex-valued jax arrays.
            Fields are Poynting-flux normalised to unit power.
    """
    # Input validation
    if (
        not (inv_permittivities.ndim == 4 and inv_permittivities.shape[0] in [1, 3, 9])
        or sum(dim == 1 for dim in inv_permittivities.shape[1:]) != 1
    ):
        raise Exception(f"Invalid shape of inv_permittivities: {inv_permittivities.shape}")
    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
        if (
            not (inv_permeabilities.ndim == 4 and inv_permeabilities.shape[0] in [1, 3, 9])
            or sum(dim == 1 for dim in inv_permeabilities.shape[1:]) != 1
        ):
            raise Exception(f"Invalid shape of inv_permeabilities: {inv_permeabilities.shape}")
    if (bend_radius is None) != (bend_axis is None):
        raise ValueError("bend_radius and bend_axis must both be set or both be None")
    if reference_E is not None and target_neff is not None:
        raise ValueError("reference_E and target_neff are mutually exclusive")

    np_complex_dtype = np.complex128 if dtype == jnp.float64 else np.complex64

    def mode_helper(permittivity, permeability, c0_um, c1_um):
        # c0_um, c1_um are concrete numpy arrays here (materialised by pure_callback)
        coords = [np.asarray(c0_um), np.asarray(c1_um)]
        if bend_radius is not None:
            assert bend_axis is not None
            transverse_axes = get_transverse_axes(propagation_axis)
            tidy3d_bend_axis = transverse_axes.index(bend_axis)
            bend_radius_um = bend_radius / 1e-6
            plane_center = (float(0.5 * (coords[0][0] + coords[0][-1])), float(0.5 * (coords[1][0] + coords[1][-1])))
        else:
            tidy3d_bend_axis = None
            bend_radius_um = None
            plane_center = None
        modes = tidy3d_mode_computation_wrapper(
            frequency=frequency,
            permittivity_cross_section=permittivity,
            permeability_cross_section=permeability,
            coords=coords,
            direction=direction,
            num_modes=2 * (mode_index + 1) + 10,
            bend_radius=bend_radius_um,
            bend_axis=tidy3d_bend_axis,
            plane_center=plane_center,
            symmetry=symmetry,
            target_neff=target_neff,
        )

        # tidy3d assumes propagation in the z-direction; tangential axes are x and y.
        sorted_modes = sort_modes(modes, filter_pol, (0, 1))
        if reference_E is not None:
            # Field-overlap selection: pick the candidate whose E field (converted to
            # fdtdx axis order + propagation-axis singleton) has the highest dot-product
            # magnitude with the reference field.
            mode = max(
                sorted_modes,
                key=lambda m: float(
                    np.abs(
                        np.sum(
                            np.conj(reference_E)
                            * np.expand_dims(
                                _perm_mode_E(m, propagation_axis, np_complex_dtype), axis=propagation_axis + 1
                            )
                        )
                    )
                ),
            )
        elif target_neff is not None:
            mode = min(sorted_modes, key=lambda m: abs(float(np.real(m.neff)) - target_neff))
        else:
            mode = sorted_modes[mode_index]

        mode_E = _perm_mode_E(mode, propagation_axis, np_complex_dtype)
        if propagation_axis == 1:
            mode_H = -np.stack([mode.Hx, mode.Hz, mode.Hy], axis=0).astype(np_complex_dtype)
        elif propagation_axis == 0:
            mode_H = np.stack([mode.Hz, mode.Hx, mode.Hy], axis=0).astype(np_complex_dtype)
        else:
            mode_H = np.stack([mode.Hx, mode.Hy, mode.Hz], axis=0).astype(np_complex_dtype)

        neff = np.asarray(mode.neff).astype(np_complex_dtype)
        return mode_E, mode_H, neff

    # compute input to tidy3d Mode solver
    if inv_permittivities.shape[0] == 9:
        eps = expand_to_3x3(inv_permittivities)
        # Invert the 3x3 matrix
        perm = (2, 3, 4, 0, 1)  # (3, 3, nx, ny, nz) -> (nx, ny, nz, 3, 3)
        inv_perm = (3, 4, 0, 1, 2)  # (nx, ny, nz, 3, 3) -> (3, 3, nx, ny, nz)
        permittivities = (
            jnp.linalg.inv(eps.transpose(perm)).transpose(inv_perm).reshape(9, *inv_permittivities.shape[1:])
        )
    else:
        permittivities = 1 / inv_permittivities
    other_axes = [a for a in range(1, 4) if permittivities.shape[a] != 1]
    propagation_axis = permittivities.shape[1:].index(1)
    if transverse_coords is None:
        if resolution is None:
            raise ValueError("resolution is required when transverse_coords is not provided")
        # Uniform grid: build concrete coordinate arrays in µm and pass as callback args.
        c0_um = jnp.asarray(np.arange(permittivities.shape[other_axes[0]] + 1) * resolution / 1e-6)
        c1_um = jnp.asarray(np.arange(permittivities.shape[other_axes[1]] + 1) * resolution / 1e-6)
        normalization_area_EuHv = None
        normalization_area_EvHu = None
    else:
        if len(transverse_coords) != 2:
            raise ValueError(
                f"transverse_coords must contain exactly two coordinate arrays, got {len(transverse_coords)}"
            )
        # Shape validation uses .shape which is always concrete, even for JAX tracers.
        expected_lengths = [permittivities.shape[dim] + 1 for dim in other_axes]
        for axis_idx, (coord, expected_length) in enumerate(zip(transverse_coords, expected_lengths, strict=True)):
            if coord.ndim != 1 or coord.shape[0] != expected_length:
                raise ValueError(
                    f"transverse_coords[{axis_idx}] must be 1D with length {expected_length}, got {coord.shape}"
                )
        # Convert to µm for tidy3d; keep as JAX arrays so jax.jit can trace through.
        c0_um = jnp.asarray(transverse_coords[0]) / 1e-6
        c1_um = jnp.asarray(transverse_coords[1]) / 1e-6
        # Yee-staggered areas for normalization — consistent with bidirectional_mode_overlap.
        # other_axes are tensor axes (1-indexed); subtract 1 to get physical axes (0-indexed).
        # transverse_coords[i] corresponds to physical axis other_axes[i] - 1, which may not match
        # the Yee u/v ordering, so we explicitly map to the correct edge arrays.
        u, v = [(1, 2), (2, 0), (0, 1)][propagation_axis]
        phys_axis_0 = other_axes[0] - 1
        edges_u = jnp.asarray(transverse_coords[0] if phys_axis_0 == u else transverse_coords[1])
        edges_v = jnp.asarray(transverse_coords[1] if phys_axis_0 == u else transverse_coords[0])
        normalization_area_EuHv, normalization_area_EvHu = yee_face_areas_from_edges(
            edges_u=edges_u,
            edges_v=edges_v,
            u_axis=u,
            v_axis=v,
            dtype=dtype,
        )
    permittivity_squeezed = jnp.take(
        permittivities,
        indices=0,
        axis=propagation_axis + 1,
    )

    # Rotate permittivity components to match tidy3d coordinate system
    # tidy3d assumes propagation along z, so we need to map physical axes to tidy3d axes:
    # - tidy3d x → first transverse axis
    # - tidy3d y → second transverse axis
    # - tidy3d z → propagation axis
    if propagation_axis == 0:
        # propagation along x: tidy3d (x,y,z) → physical (y,z,x)
        perm_idx = [1, 2, 0]
        perm_idx_full_anisotropy = [4, 5, 3, 7, 8, 6, 1, 2, 0]
    elif propagation_axis == 1:
        # propagation along y: tidy3d (x,y,z) → physical (x,z,y)
        perm_idx = [0, 2, 1]
        perm_idx_full_anisotropy = [0, 2, 1, 6, 8, 7, 3, 5, 4]
    else:  # propagation_axis == 2
        # propagation along z: tidy3d (x,y,z) → physical (x,y,z)
        perm_idx = [0, 1, 2]
        perm_idx_full_anisotropy = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Only apply rotation if anisotropic (3 components)
    if permittivity_squeezed.shape[0] == 3:
        permittivity_squeezed = permittivity_squeezed[jnp.array(perm_idx), :, :]
    if permittivity_squeezed.shape[0] == 9:
        permittivity_squeezed = permittivity_squeezed[jnp.array(perm_idx_full_anisotropy), :, :]

    jnp_complex_dtype = jnp.complex128 if dtype == jnp.float64 else jnp.complex64
    result_shape_dtype = (
        jnp.zeros((3, *permittivity_squeezed.shape[1:]), dtype=jnp_complex_dtype),
        jnp.zeros((3, *permittivity_squeezed.shape[1:]), dtype=jnp_complex_dtype),
        jnp.zeros(shape=(), dtype=jnp_complex_dtype),
    )

    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0 and inv_permeabilities.shape[0] == 9:
        mu = expand_to_3x3(inv_permeabilities)
        # Invert the 3x3 matrix
        perm = (2, 3, 4, 0, 1)  # (3, 3, nx, ny, nz) -> (nx, ny, nz, 3, 3)
        inv_perm = (3, 4, 0, 1, 2)  # (nx, ny, nz, 3, 3) -> (3, 3, nx, ny, nz)
        permeabilities = (
            jnp.linalg.inv(mu.transpose(perm)).transpose(inv_perm).reshape(9, *inv_permeabilities.shape[1:])
        )
    else:
        permeabilities = 1 / inv_permeabilities
    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
        permeability_squeezed = jnp.take(
            permeabilities,
            indices=0,
            axis=propagation_axis + 1,
        )
        # Apply same rotation to permeability if anisotropic
        if permeability_squeezed.shape[0] == 3:
            permeability_squeezed = permeability_squeezed[jnp.array(perm_idx), :, :]
        if permeability_squeezed.shape[0] == 9:
            permeability_squeezed = permeability_squeezed[jnp.array(perm_idx_full_anisotropy), :, :]
    else:  # float
        permeability_squeezed = permeabilities

    # pure callback to tidy3d is necessary to work in jitted environment.
    # c0_um and c1_um are passed as explicit args so JAX materialises them to
    # concrete numpy arrays before calling mode_helper, allowing np.asarray()
    # inside the callback without raising TracerArrayConversionError.
    mode_E_raw, mode_H_raw, eff_idx = jax.pure_callback(
        mode_helper,
        result_shape_dtype,
        jax.lax.stop_gradient(permittivity_squeezed),
        jax.lax.stop_gradient(permeability_squeezed),
        jax.lax.stop_gradient(c0_um),
        jax.lax.stop_gradient(c1_um),
    )
    mode_E = jnp.expand_dims(mode_E_raw, axis=propagation_axis + 1)
    mode_H = jnp.expand_dims(mode_H_raw, axis=propagation_axis + 1)

    # Tidy3D uses different scaling internally, so convert back
    mode_H = mode_H * tidy3d.constants.ETA_0

    mode_E_norm, mode_H_norm = normalize_by_poynting_flux(
        mode_E,
        mode_H,
        axis=propagation_axis,
        area_EuHv=normalization_area_EuHv,
        area_EvHu=normalization_area_EvHu,
    )

    return mode_E_norm, mode_H_norm, eff_idx


def compute_modes_tracked(
    frequencies: list[float],
    inv_permittivities_stack: jax.Array,
    inv_permeabilities: jax.Array | float,
    resolution: float | None = None,
    direction: Literal["+", "-"] = "+",
    mode_index: int = 0,
    filter_pol: Literal["te", "tm"] | None = None,
    dtype: jnp.dtype = jnp.float32,
    bend_radius: float | None = None,
    bend_axis: int | None = None,
    symmetry: tuple[int, int] = (0, 0),
    transverse_coords: Sequence[jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Multi-frequency mode solving with field-overlap continuity tracking.

    Runs all frequency solves inside a single ``jax.pure_callback``, making the
    function fully compatible with ``jax.jit``.  Inside the callback all arrays
    are concrete numpy, so the previous step's mode_E can be read and passed as
    a reference to the next step without breaking JAX tracing.

    Field-overlap tracking selects — among all ARPACK candidates — the mode with
    the highest ``|<conj(prev_mode_E), candidate_E>|``.  The first frequency uses
    ``mode_index`` as usual (no prior reference).

    Args:
        frequencies: List of N operating frequencies in Hz.
        inv_permittivities_stack: Stacked inverse permittivities, shape
            ``(N, {1, 3, 9}, nx, ny, nz)`` where exactly one of nx/ny/nz is 1.
            Slice ``i`` is used for ``frequencies[i]``.  Build this by stacking
            per-frequency dispersive corrections before calling.
        inv_permeabilities: Same as ``compute_mode`` — shared across all frequencies.
        resolution: Uniform grid spacing in metres; required when
            ``transverse_coords`` is ``None``.
        direction: Propagation direction, ``"+"`` or ``"-"``.
        mode_index: Index into neff-sorted candidate list for the first frequency.
        filter_pol: Optional ``"te"`` / ``"tm"`` polarisation filter.
        dtype: Float dtype; controls complex64 vs complex128 output.
        bend_radius: Waveguide bend radius in metres.
        bend_axis: Physical axis toward the centre of curvature.
        symmetry: Per-transverse-axis symmetry condition at the min edge.
        transverse_coords: Optional pair of physical edge-coordinate arrays in metres.

    Returns:
        Tuple of ``(mode_Es, mode_Hs, neffs)`` with shapes
        ``(N, 3, nx, ny, nz)``, ``(N, 3, nx, ny, nz)``, ``(N,)``.
        Fields are Poynting-flux normalised to unit power.
    """
    num_freqs = len(frequencies)
    if inv_permittivities_stack.shape[0] != num_freqs:
        raise ValueError(
            f"inv_permittivities_stack first dim must equal len(frequencies) ({num_freqs}), "
            f"got {inv_permittivities_stack.shape[0]}"
        )
    sample = inv_permittivities_stack[0]

    if not (sample.ndim == 4 and sample.shape[0] in [1, 3, 9]) or sum(dim == 1 for dim in sample.shape[1:]) != 1:
        raise Exception(f"Invalid shape of inv_permittivities: {sample.shape}")
    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
        if (
            not (inv_permeabilities.ndim == 4 and inv_permeabilities.shape[0] in [1, 3, 9])
            or sum(dim == 1 for dim in inv_permeabilities.shape[1:]) != 1
        ):
            raise Exception(f"Invalid shape of inv_permeabilities: {inv_permeabilities.shape}")
    if (bend_radius is None) != (bend_axis is None):
        raise ValueError("bend_radius and bend_axis must both be set or both be None")

    np_complex_dtype = np.complex128 if dtype == jnp.float64 else np.complex64
    jnp_complex_dtype = jnp.complex128 if dtype == jnp.float64 else jnp.complex64

    propagation_axis = sample.shape[1:].index(1)
    other_axes = [a for a in range(1, 4) if sample.shape[a] != 1]

    if propagation_axis == 0:
        perm_idx = [1, 2, 0]
        perm_idx_full_anisotropy = [4, 5, 3, 7, 8, 6, 1, 2, 0]
    elif propagation_axis == 1:
        perm_idx = [0, 2, 1]
        perm_idx_full_anisotropy = [0, 2, 1, 6, 8, 7, 3, 5, 4]
    else:
        perm_idx = [0, 1, 2]
        perm_idx_full_anisotropy = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    if transverse_coords is None:
        if resolution is None:
            raise ValueError("resolution is required when transverse_coords is not provided")
        c0_um = jnp.asarray(np.arange(sample.shape[other_axes[0]] + 1) * resolution / 1e-6)
        c1_um = jnp.asarray(np.arange(sample.shape[other_axes[1]] + 1) * resolution / 1e-6)
        normalization_area_EuHv = None
        normalization_area_EvHu = None
    else:
        if len(transverse_coords) != 2:
            raise ValueError(
                f"transverse_coords must contain exactly two coordinate arrays, got {len(transverse_coords)}"
            )
        expected_lengths = [sample.shape[dim] + 1 for dim in other_axes]
        for axis_idx, (coord, expected_length) in enumerate(zip(transverse_coords, expected_lengths, strict=True)):
            if coord.ndim != 1 or coord.shape[0] != expected_length:
                raise ValueError(
                    f"transverse_coords[{axis_idx}] must be 1D with length {expected_length}, got {coord.shape}"
                )
        c0_um = jnp.asarray(transverse_coords[0]) / 1e-6
        c1_um = jnp.asarray(transverse_coords[1]) / 1e-6
        u, v = [(1, 2), (2, 0), (0, 1)][propagation_axis]
        phys_axis_0 = other_axes[0] - 1
        edges_u = jnp.asarray(transverse_coords[0] if phys_axis_0 == u else transverse_coords[1])
        edges_v = jnp.asarray(transverse_coords[1] if phys_axis_0 == u else transverse_coords[0])
        normalization_area_EuHv, normalization_area_EvHu = yee_face_areas_from_edges(
            edges_u=edges_u,
            edges_v=edges_v,
            u_axis=u,
            v_axis=v,
            dtype=dtype,
        )

    # Process permeability once (frequency-independent) — mirrors compute_mode exactly.
    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0 and inv_permeabilities.shape[0] == 9:
        mu = expand_to_3x3(inv_permeabilities)
        perm = (2, 3, 4, 0, 1)
        inv_perm = (3, 4, 0, 1, 2)
        permeabilities_mu = (
            jnp.linalg.inv(mu.transpose(perm)).transpose(inv_perm).reshape(9, *inv_permeabilities.shape[1:])
        )
    else:
        permeabilities_mu = 1 / inv_permeabilities
    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
        permeability_squeezed = jnp.take(permeabilities_mu, indices=0, axis=propagation_axis + 1)
        if permeability_squeezed.shape[0] == 3:
            permeability_squeezed = permeability_squeezed[jnp.array(perm_idx), :, :]
        if permeability_squeezed.shape[0] == 9:
            permeability_squeezed = permeability_squeezed[jnp.array(perm_idx_full_anisotropy), :, :]
    else:
        permeability_squeezed = permeabilities_mu

    # Output shape: (N, 3, n0, n1) — propagation singleton added after callback.
    spatial_2d = tuple(s for i, s in enumerate(sample.shape[1:]) if i != propagation_axis)
    result_shape_dtype = (
        jnp.zeros((num_freqs, 3, *spatial_2d), dtype=jnp_complex_dtype),
        jnp.zeros((num_freqs, 3, *spatial_2d), dtype=jnp_complex_dtype),
        jnp.zeros((num_freqs,), dtype=jnp_complex_dtype),
    )

    def multi_freq_helper(inv_perm_stack, perm_mu, c0_arr, c1_arr):
        """All arrays are concrete numpy here (pure_callback materialises them)."""
        coords = [np.asarray(c0_arr), np.asarray(c1_arr)]
        if bend_radius is not None:
            assert bend_axis is not None
            transverse_axes = get_transverse_axes(propagation_axis)
            tidy3d_bend_axis = transverse_axes.index(bend_axis)
            bend_radius_um = bend_radius / 1e-6
            plane_center = (
                float(0.5 * (coords[0][0] + coords[0][-1])),
                float(0.5 * (coords[1][0] + coords[1][-1])),
            )
        else:
            tidy3d_bend_axis = None
            bend_radius_um = None
            plane_center = None

        prev_mode_E_4d: np.ndarray | None = None
        all_Es: list[np.ndarray] = []
        all_Hs: list[np.ndarray] = []
        all_neffs: list[np.ndarray] = []

        for i, freq in enumerate(frequencies):
            # Invert, squeeze, rotate permittivity for this frequency.
            inv_perm_i = inv_perm_stack[i]
            if inv_perm_i.shape[0] == 9:
                reshaped = inv_perm_i.reshape(3, 3, *inv_perm_i.shape[1:])
                t_fwd, t_back = (2, 3, 4, 0, 1), (3, 4, 0, 1, 2)
                perm_i = np.linalg.inv(reshaped.transpose(t_fwd)).transpose(t_back)
                perm_i = perm_i.reshape(9, *inv_perm_i.shape[1:])
            else:
                perm_i = 1.0 / inv_perm_i
            perm_i_sq = np.squeeze(perm_i, axis=propagation_axis + 1)
            if perm_i_sq.shape[0] == 3:
                perm_i_sq = perm_i_sq[perm_idx, :, :]
            elif perm_i_sq.shape[0] == 9:
                perm_i_sq = perm_i_sq[perm_idx_full_anisotropy, :, :]

            modes = tidy3d_mode_computation_wrapper(
                frequency=freq,
                permittivity_cross_section=perm_i_sq,
                permeability_cross_section=perm_mu,
                coords=coords,
                direction=direction,
                num_modes=2 * (mode_index + 1) + 10,
                bend_radius=bend_radius_um,
                bend_axis=tidy3d_bend_axis,
                plane_center=plane_center,
                symmetry=symmetry,
            )

            sorted_modes = sort_modes(modes, filter_pol, (0, 1))
            if prev_mode_E_4d is not None:
                mode = max(
                    sorted_modes,
                    key=lambda m: float(
                        np.abs(
                            np.sum(
                                np.conj(prev_mode_E_4d)
                                * np.expand_dims(
                                    _perm_mode_E(m, propagation_axis, np_complex_dtype),
                                    axis=propagation_axis + 1,
                                )
                            )
                        )
                    ),
                )
            else:
                mode = sorted_modes[mode_index]

            mode_E_3d = _perm_mode_E(mode, propagation_axis, np_complex_dtype)
            prev_mode_E_4d = np.expand_dims(mode_E_3d, axis=propagation_axis + 1)

            if propagation_axis == 1:
                mode_H_3d = -np.stack([mode.Hx, mode.Hz, mode.Hy], axis=0).astype(np_complex_dtype)
            elif propagation_axis == 0:
                mode_H_3d = np.stack([mode.Hz, mode.Hx, mode.Hy], axis=0).astype(np_complex_dtype)
            else:
                mode_H_3d = np.stack([mode.Hx, mode.Hy, mode.Hz], axis=0).astype(np_complex_dtype)

            all_Es.append(mode_E_3d)
            all_Hs.append(mode_H_3d)
            all_neffs.append(np.asarray(mode.neff).astype(np_complex_dtype))

        return (
            np.stack(all_Es, axis=0),
            np.stack(all_Hs, axis=0),
            np.array(all_neffs),
        )

    mode_Es_raw, mode_Hs_raw, neffs = jax.pure_callback(
        multi_freq_helper,
        result_shape_dtype,
        jax.lax.stop_gradient(inv_permittivities_stack),
        jax.lax.stop_gradient(permeability_squeezed),
        jax.lax.stop_gradient(c0_um),
        jax.lax.stop_gradient(c1_um),
    )

    # Add propagation-axis singleton (axis 0 is the frequency batch).
    mode_Es = jnp.expand_dims(mode_Es_raw, axis=propagation_axis + 2)
    mode_Hs = jnp.expand_dims(mode_Hs_raw, axis=propagation_axis + 2) * tidy3d.constants.ETA_0

    def _normalize_one(E: jax.Array, H: jax.Array) -> tuple[jax.Array, jax.Array]:
        return normalize_by_poynting_flux(
            E,
            H,
            axis=propagation_axis,
            area_EuHv=normalization_area_EuHv,
            area_EvHu=normalization_area_EvHu,
        )

    mode_Es_norm, mode_Hs_norm = jax.vmap(_normalize_one)(mode_Es, mode_Hs)
    return mode_Es_norm, mode_Hs_norm, neffs


def tidy3d_mode_computation_wrapper(
    frequency: float,
    permittivity_cross_section: ArrayLike,
    coords: List[np.ndarray],
    direction: Literal["+", "-"],
    permeability_cross_section: ArrayLike | float | None = None,
    target_neff: float | None = None,
    angle_theta: float = 0.0,
    angle_phi: float = 0.0,
    num_modes: int = 10,
    precision: Literal["single", "double"] = "double",
    bend_radius: float | None = None,
    bend_axis: int | None = None,
    plane_center: tuple[float, float] | None = None,
    symmetry: tuple[int, int] = (0, 0),
) -> List[ModeTupleType]:
    """Compute optical modes of a waveguide cross-section.

    This function uses the Tidy3D mode solver to compute the optical modes of a given
    waveguide cross-section defined by its permittivity distribution.

    Args:
        frequency (float): Operating frequency in Hz
        permittivity_cross_section (jax.Array): 2D array of relative permittivity values
        coords (List[np.ndarray]): List of coordinate arrays [x, y] defining the grid
        direction (Literal["+", "-"], optional): Propagation direction, either "+" or "-"
        permeability_cross_section (jax.Array | float | None, optional): 2D array of relative permeability values.
            Defauts to None.
        target_neff (float | None, optional): Target effective index to search around. Defaults to None.
        angle_theta (float, optional): Polar angle in radians. Defaults to 0.0.
        angle_phi (float, optional): Azimuthal angle in radians. Defaults to 0.0.
        num_modes (int, optional): Number of modes to compute. Defaults to 10.
        precision (Literal["single", "double"], optional): Numerical precision. Defaults to "double".
        bend_radius (float | None, optional): Bend radius in microns (tidy3d units). Defaults to None.
        bend_axis (int | None, optional): Axis index (0 or 1) of the center of curvature in tidy3d's transverse
            coordinate frame. Defaults to None.
        plane_center (tuple[float, float] | None, optional): Center of the mode plane in the same units as coords.
            Required by tidy3d when bend_radius is set. Defaults to None.
        symmetry (tuple[int, int], optional): Per-transverse-axis symmetry condition at the min edge, forwarded to
            the tidy3d mode solver. ``1`` imposes a PMC (magnetic) wall there; ``0`` (default) leaves the solver's
            PEC (electric) wall. Order matches ``coords``. Defaults to ``(0, 0)``.

    Notes:
        tidy3d assumes propagation in z-direction. The output fields should be handled accordingly.

    Returns:
        List[ModeTupleType]: List of computed modes sorted by decreasing real part of
            effective index. Each mode contains the field components and effective index.
    """
    # see https://docs.flexcompute.com/projects/tidy3d/en/latest/_autosummary/tidy3d.ModeSpec.html#tidy3d.ModeSpec
    mode_spec = SimpleNamespace(
        # Note that the filter_pol argument is not used here since it does not work from tidy3d
        num_modes=num_modes,
        target_neff=target_neff,
        num_pml=(0, 0),
        angle_theta=angle_theta,
        angle_phi=angle_phi,
        bend_radius=bend_radius,
        bend_axis=bend_axis,
        precision=precision,
        track_freq="central",
        group_index_step=False,
    )
    permittivity_cross_section = jnp.asarray(permittivity_cross_section)
    permittivity_cross_section = expand_to_3x3(permittivity_cross_section)
    permittivity_cross_section = permittivity_cross_section.reshape(9, *permittivity_cross_section.shape[2:])
    eps_cross = [
        permittivity_cross_section[0],
        permittivity_cross_section[1],
        permittivity_cross_section[2],
        permittivity_cross_section[3],
        permittivity_cross_section[4],
        permittivity_cross_section[5],
        permittivity_cross_section[6],
        permittivity_cross_section[7],
        permittivity_cross_section[8],
    ]
    mu_cross = None
    if permeability_cross_section is not None:
        permeability_cross_section = jnp.asarray(permeability_cross_section)
        permeability_cross_section = expand_to_3x3(permeability_cross_section)
        permeability_cross_section = permeability_cross_section.reshape(9, *permeability_cross_section.shape[2:])

        mu_cross = [
            permeability_cross_section[0],
            permeability_cross_section[1],
            permeability_cross_section[2],
            permeability_cross_section[3],
            permeability_cross_section[4],
            permeability_cross_section[5],
            permeability_cross_section[6],
            permeability_cross_section[7],
            permeability_cross_section[8],
        ]

    EH, neffs, _ = _compute_modes(
        eps_cross=eps_cross,
        coords=coords,
        freq=frequency,
        precision=precision,
        mode_spec=mode_spec,
        direction=direction,
        mu_cross=mu_cross,
        plane_center=plane_center,
        symmetry=symmetry,
    )
    ((Ex, Ey, Ez), (Hx, Hy, Hz)) = EH.squeeze()

    if num_modes == 1:
        modes = [
            ModeTupleType(
                Ex=Ex,
                Ey=Ey,
                Ez=Ez,
                Hx=Hx,
                Hy=Hy,
                Hz=Hz,
                neff=float(neffs.real) + 1j * float(neffs.imag),
            )
            for _ in range(num_modes)
        ]
    else:
        modes = [
            ModeTupleType(
                Ex=Ex[..., i],
                Ey=Ey[..., i],
                Ez=Ez[..., i],
                Hx=Hx[..., i],
                Hy=Hy[..., i],
                Hz=Hz[..., i],
                neff=neffs[i],
            )
            for i in range(num_modes)
        ]
    return modes
