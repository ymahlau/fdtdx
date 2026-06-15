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
    """Permute tidy3d E-field into fdtdx axis order. Returns (3, n0, n1) without propagation singleton."""
    if propagation_axis == 0:
        return np.stack([m.Ez, m.Ex, m.Ey], axis=0).astype(dtype)
    elif propagation_axis == 1:
        return np.stack([m.Ex, m.Ez, m.Ey], axis=0).astype(dtype)
    else:
        return np.stack([m.Ex, m.Ey, m.Ez], axis=0).astype(dtype)


def _perm_mode_H(m: ModeTupleType, propagation_axis: int, dtype) -> np.ndarray:
    """Permute tidy3d H-field into fdtdx axis order. Returns (3, n0, n1) without propagation singleton."""
    if propagation_axis == 0:
        return np.stack([m.Hz, m.Hx, m.Hy], axis=0).astype(dtype)
    elif propagation_axis == 1:
        return -np.stack([m.Hx, m.Hz, m.Hy], axis=0).astype(dtype)
    else:
        return np.stack([m.Hx, m.Hy, m.Hz], axis=0).astype(dtype)


def _axis_perm_indices(propagation_axis: int) -> tuple[list[int], list[int]]:
    """Component permutation indices for rotating into tidy3d's z-propagation frame."""
    return [
        ([1, 2, 0], [4, 5, 3, 7, 8, 6, 1, 2, 0]),  # x-propagation
        ([0, 2, 1], [0, 2, 1, 6, 8, 7, 3, 5, 4]),  # y-propagation
        ([0, 1, 2], [0, 1, 2, 3, 4, 5, 6, 7, 8]),  # z-propagation
    ][propagation_axis]


def _prepare_permittivity(
    inv_perm: np.ndarray,
    propagation_axis: int,
    perm_idx: list[int],
    perm_idx_9: list[int],
) -> np.ndarray:
    """Invert, squeeze the propagation axis, and rotate into tidy3d's z-propagation frame.

    Input: (K, nx, ny, nz) with K ∈ {1, 3, 9} and exactly one spatial dim == 1.
    Output: (K, n0, n1) — the two non-propagation transverse dimensions.
    """
    if inv_perm.shape[0] == 9:
        m = inv_perm.reshape(3, 3, *inv_perm.shape[1:])
        fwd, back = (2, 3, 4, 0, 1), (3, 4, 0, 1, 2)
        perm = np.linalg.inv(m.transpose(fwd)).transpose(back).reshape(9, *inv_perm.shape[1:])
    else:
        perm = 1.0 / inv_perm
    sq = np.asarray(np.squeeze(perm, axis=propagation_axis + 1))
    if sq.shape[0] == 3:
        return sq[perm_idx, :, :]
    if sq.shape[0] == 9:
        return sq[perm_idx_9, :, :]
    return sq


def _process_permeability(
    inv_permeabilities: jax.Array | float,
    propagation_axis: int,
    perm_idx: list[int],
    perm_idx_9: list[int],
) -> jax.Array | float:
    """Invert, squeeze, and rotate permeability in JAX (runs outside callback, frequency-independent)."""
    if not (isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0):
        return 1.0 / inv_permeabilities
    if inv_permeabilities.shape[0] == 9:
        mu = expand_to_3x3(inv_permeabilities)
        fwd, back = (2, 3, 4, 0, 1), (3, 4, 0, 1, 2)
        perms = jnp.linalg.inv(mu.transpose(fwd)).transpose(back).reshape(9, *inv_permeabilities.shape[1:])
    else:
        perms = 1.0 / inv_permeabilities
    sq = jnp.take(perms, indices=0, axis=propagation_axis + 1)
    if sq.shape[0] == 3:
        return sq[jnp.array(perm_idx)]
    if sq.shape[0] == 9:
        return sq[jnp.array(perm_idx_9)]
    return sq


def _build_transverse_coords(
    sample_shape: tuple[int, ...],
    propagation_axis: int,
    other_axes: list[int],
    resolution: float | None,
    transverse_coords: Sequence[jax.Array] | None,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array | None]:
    """Build tidy3d coordinate arrays (µm) and optional Yee area weights.

    Returns (c0_um, c1_um, area_EuHv, area_EvHu); area_* are None on uniform grids.
    """
    if transverse_coords is None:
        if resolution is None:
            raise ValueError("resolution is required when transverse_coords is not provided")
        c0_um = jnp.asarray(np.arange(sample_shape[other_axes[0]] + 1) * resolution / 1e-6)
        c1_um = jnp.asarray(np.arange(sample_shape[other_axes[1]] + 1) * resolution / 1e-6)
        return c0_um, c1_um, None, None

    if len(transverse_coords) != 2:
        raise ValueError(f"transverse_coords must have exactly 2 arrays, got {len(transverse_coords)}")
    for i, (coord, expected) in enumerate(zip(transverse_coords, [sample_shape[a] + 1 for a in other_axes])):
        if coord.ndim != 1 or coord.shape[0] != expected:
            raise ValueError(f"transverse_coords[{i}] must be 1D with length {expected}, got {coord.shape}")

    c0_um = jnp.asarray(transverse_coords[0]) / 1e-6
    c1_um = jnp.asarray(transverse_coords[1]) / 1e-6
    u, v = [(1, 2), (2, 0), (0, 1)][propagation_axis]
    phys_axis_0 = other_axes[0] - 1
    edges_u = jnp.asarray(transverse_coords[0] if phys_axis_0 == u else transverse_coords[1])
    edges_v = jnp.asarray(transverse_coords[1] if phys_axis_0 == u else transverse_coords[0])
    area_EuHv, area_EvHu = yee_face_areas_from_edges(edges_u=edges_u, edges_v=edges_v, u_axis=u, v_axis=v, dtype=dtype)
    return c0_um, c1_um, area_EuHv, area_EvHu


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


def _validate_mode_inputs(
    sample: jax.Array,
    inv_permeabilities: jax.Array | float,
    bend_radius: float | None,
    bend_axis: int | None,
) -> None:
    if not (sample.ndim == 4 and sample.shape[0] in [1, 3, 9]) or sum(s == 1 for s in sample.shape[1:]) != 1:
        raise ValueError(f"Invalid inv_permittivities shape: {sample.shape}")
    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
        if (
            not (inv_permeabilities.ndim == 4 and inv_permeabilities.shape[0] in [1, 3, 9])
            or sum(s == 1 for s in inv_permeabilities.shape[1:]) != 1
        ):
            raise ValueError(f"Invalid inv_permeabilities shape: {inv_permeabilities.shape}")
    if (bend_radius is None) != (bend_axis is None):
        raise ValueError("bend_radius and bend_axis must both be set or both be None")


def _build_bend_params(
    bend_radius: float | None,
    bend_axis: int | None,
    propagation_axis: int,
    coords: list[np.ndarray],
) -> tuple[float | None, int | None, tuple[float, float] | None]:
    if bend_radius is None:
        return None, None, None
    transverse_axes = get_transverse_axes(propagation_axis)
    bend_radius_um = bend_radius / 1e-6
    tidy3d_bend_axis = transverse_axes.index(bend_axis)
    plane_center = (float(0.5 * (coords[0][0] + coords[0][-1])), float(0.5 * (coords[1][0] + coords[1][-1])))
    return bend_radius_um, tidy3d_bend_axis, plane_center


def _select_by_field_overlap(
    sorted_modes: list[ModeTupleType],
    ref_E_4d: np.ndarray,
    propagation_axis: int,
    np_complex_dtype,
) -> ModeTupleType:
    return max(
        sorted_modes,
        key=lambda m: float(
            np.abs(
                np.sum(
                    np.conj(ref_E_4d)
                    * np.expand_dims(
                        _perm_mode_E(m, propagation_axis, np_complex_dtype),
                        axis=propagation_axis + 1,
                    )
                )
            )
        ),
    )


_ModeSetup = namedtuple(
    "_ModeSetup",
    [
        "propagation_axis",
        "perm_idx",
        "perm_idx_9",
        "c0_um",
        "c1_um",
        "area_EuHv",
        "area_EvHu",
        "permeability_squeezed",
        "np_complex_dtype",
        "jnp_complex_dtype",
        "spatial_2d",
    ],
)


def _build_mode_setup(
    sample: jax.Array,
    inv_permeabilities: jax.Array | float,
    resolution: float | None,
    transverse_coords: Sequence[jax.Array] | None,
    dtype: jnp.dtype,
) -> "_ModeSetup":
    """Derive geometry, coordinates and permeability from a single-frequency sample slice."""
    propagation_axis = sample.shape[1:].index(1)
    other_axes = [a for a in range(1, 4) if sample.shape[a] != 1]
    perm_idx, perm_idx_9 = _axis_perm_indices(propagation_axis)
    c0_um, c1_um, area_EuHv, area_EvHu = _build_transverse_coords(
        sample.shape, propagation_axis, other_axes, resolution, transverse_coords, dtype
    )
    permeability_squeezed = _process_permeability(inv_permeabilities, propagation_axis, perm_idx, perm_idx_9)
    np_complex_dtype = np.complex128 if dtype == jnp.float64 else np.complex64
    jnp_complex_dtype = jnp.complex128 if dtype == jnp.float64 else jnp.complex64
    spatial_2d = tuple(s for i, s in enumerate(sample.shape[1:]) if i != propagation_axis)
    return _ModeSetup(
        propagation_axis=propagation_axis,
        perm_idx=perm_idx,
        perm_idx_9=perm_idx_9,
        c0_um=c0_um,
        c1_um=c1_um,
        area_EuHv=area_EuHv,
        area_EvHu=area_EvHu,
        permeability_squeezed=permeability_squeezed,
        np_complex_dtype=np_complex_dtype,
        jnp_complex_dtype=jnp_complex_dtype,
        spatial_2d=spatial_2d,
    )


def _postprocess_modes(
    mode_Es_raw: jax.Array,
    mode_Hs_raw: jax.Array,
    neffs: jax.Array,
    setup: "_ModeSetup",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Add propagation singleton, apply ETA_0, and Poynting-flux normalise."""
    ax = setup.propagation_axis
    mode_Es = jnp.expand_dims(mode_Es_raw, axis=ax + 2)
    mode_Hs = jnp.expand_dims(mode_Hs_raw, axis=ax + 2) * tidy3d.constants.ETA_0

    def _normalize_one(E: jax.Array, H: jax.Array) -> tuple[jax.Array, jax.Array]:
        return normalize_by_poynting_flux(E, H, axis=ax, area_EuHv=setup.area_EuHv, area_EvHu=setup.area_EvHu)

    mode_Es_norm, mode_Hs_norm = jax.vmap(_normalize_one)(mode_Es, mode_Hs)
    return mode_Es_norm, mode_Hs_norm, neffs


def compute_mode(
    frequencies: list[float],
    inv_permittivities: jax.Array,
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
    """Compute optical modes at N frequencies with automatic field-overlap tracking.

    Runs all N solves inside a single ``jax.pure_callback`` — safe under ``jax.jit``.
    Field-overlap tracking selects — at each frequency after the first — the ARPACK
    candidate with the highest ``|<conj(prev_E), candidate_E>|``, preventing mode
    hopping at neff branch crossings.  Pass a single-element list for one frequency.

    Args:
        frequencies: List of N operating frequencies in Hz.
        inv_permittivities: Stacked inverse permittivity, shape ``(N, K, nx, ny, nz)``
            where ``K ∈ {1, 3, 9}`` and exactly one spatial dimension is 1.
            Slice ``i`` is used at ``frequencies[i]``.
        inv_permeabilities: Inverse relative permeability — scalar float or same-shape
            array shared across all frequencies.
        resolution: Uniform grid spacing in metres; required when ``transverse_coords`` is not provided.
        direction: Propagation direction, ``"+"`` or ``"-"``.
        mode_index: Index into neff-sorted list for the first frequency. Defaults to 0.
        filter_pol: Optional ``"te"`` / ``"tm"`` polarisation filter.
        dtype: Float dtype; controls complex64 vs complex128 output.
        bend_radius: Waveguide bend radius in metres; requires ``bend_axis``.
        bend_axis: Physical axis index toward the centre of curvature.
        symmetry: Per-transverse-axis symmetry condition at the min edge.
        transverse_coords: Optional pair of physical edge-coordinate arrays in metres.

    Returns:
        ``(mode_Es, mode_Hs, neffs)`` with shapes ``(N, 3, nx, ny, nz)`` and ``(N,)``.
        Fields are Poynting-flux normalised to unit power.
    """
    N = len(frequencies)
    sample = inv_permittivities[0]
    _validate_mode_inputs(sample, inv_permeabilities, bend_radius, bend_axis)

    setup = _build_mode_setup(sample, inv_permeabilities, resolution, transverse_coords, dtype)

    def _solve(inv_perm_stack_np, perm_mu, c0_arr, c1_arr):
        coords = [np.asarray(c0_arr), np.asarray(c1_arr)]
        bend_radius_um, tidy3d_bend_axis, plane_center = _build_bend_params(
            bend_radius, bend_axis, setup.propagation_axis, coords
        )
        prev_E_4d: np.ndarray | None = None
        all_Es: list[np.ndarray] = []
        all_Hs: list[np.ndarray] = []
        all_neffs: list[np.ndarray] = []

        for i, freq in enumerate(frequencies):
            perm_sq = _prepare_permittivity(
                inv_perm_stack_np[i], setup.propagation_axis, setup.perm_idx, setup.perm_idx_9
            )
            modes = tidy3d_mode_computation_wrapper(
                frequency=freq,
                permittivity_cross_section=perm_sq,
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

            if prev_E_4d is not None:
                mode = _select_by_field_overlap(sorted_modes, prev_E_4d, setup.propagation_axis, setup.np_complex_dtype)
            else:
                mode = sorted_modes[mode_index]

            mode_E = _perm_mode_E(mode, setup.propagation_axis, setup.np_complex_dtype)
            mode_H = _perm_mode_H(mode, setup.propagation_axis, setup.np_complex_dtype)
            prev_E_4d = np.expand_dims(mode_E, axis=setup.propagation_axis + 1)

            all_Es.append(mode_E)
            all_Hs.append(mode_H)
            all_neffs.append(np.asarray(mode.neff).astype(setup.np_complex_dtype))

        return np.stack(all_Es, 0), np.stack(all_Hs, 0), np.array(all_neffs)

    result_shape_dtype = (
        jnp.zeros((N, 3, *setup.spatial_2d), dtype=setup.jnp_complex_dtype),
        jnp.zeros((N, 3, *setup.spatial_2d), dtype=setup.jnp_complex_dtype),
        jnp.zeros((N,), dtype=setup.jnp_complex_dtype),
    )
    mode_Es_raw, mode_Hs_raw, neffs = jax.pure_callback(
        _solve,
        result_shape_dtype,
        jax.lax.stop_gradient(inv_permittivities),
        jax.lax.stop_gradient(setup.permeability_squeezed),
        jax.lax.stop_gradient(setup.c0_um),
        jax.lax.stop_gradient(setup.c1_um),
    )
    return _postprocess_modes(mode_Es_raw, mode_Hs_raw, neffs, setup)


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
