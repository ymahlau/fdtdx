"""Automatic mirror-symmetry support for FDTD simulations.

When :attr:`fdtdx.SimulationConfig.symmetry` has a nonzero entry, :func:`fdtdx.place_objects`
reduces the domain to the symmetric half/quarter/octant before running the FDTD. This module
provides the two halves of that machinery:

* **Reduction** (used internally by ``place_objects``): clip every resolved grid slice onto the
  kept upper half along each symmetric axis, drop objects that fall entirely in the discarded
  half, and insert a PEC wall on each electric symmetry plane.
* **Unfolding** (public helpers): reconstruct full-domain field and detector arrays from the
  reduced simulation, applying the correct per-component mirror parity.

Symmetry encoding (per axis, order ``(x, y, z)``): ``0`` = none, ``-1`` = PEC (electric wall),
``+1`` = PMC (magnetic wall). The symmetry plane is the geometric center of the axis; the upper
half is kept so the plane lands at the reduced domain's min edge.

**Where the two wall types sit.** An electric plane sits *on* the reduced domain's min edge: the
tangential ``E`` components are sampled there and the odd symmetry makes them vanish, which is a
PEC face. A magnetic plane sits *half a cell below* it: sources and materials are rasterized per
cell, so the discrete problem is mirror symmetric about the tangential-``H`` node one cell out,
where tangential ``H`` vanishes — already supplied by the zero halo of the field padding, so no wall
object is created for it (see :func:`make_symmetry_walls`). The same distinction sets the mirror
index map used when unfolding (:func:`~fdtdx.core.physics.symmetry.mirror_pairs_on_plane`): across
an electric plane, components sampled on it pair as ``m ± j``; across a magnetic plane every
component mirrors one-to-one.

Mode sources and mode-overlap detectors clipped by a symmetry plane do not use the mode solver's own
symmetric solve; they mirror the reduced cross-section back to the full one, solve there and
restrict (see :func:`~fdtdx.core.physics.modes.compute_mode_symmetry_reduced`).

The parity table itself (:func:`field_component_parity`) lives in
:mod:`fdtdx.core.physics.symmetry` so that source and detector classes can use it without an
import cycle; it is re-exported here, where it is documented.
"""

from typing import Literal

import jax
import jax.numpy as jnp
from loguru import logger

from fdtdx.config import SimulationConfig
from fdtdx.core.misc import validate_symmetric_axis_cells
from fdtdx.core.null import Null
from fdtdx.core.physics.symmetry import (
    field_component_parity,
    mirror_extend_low_side,
    mirror_pairs_on_plane,
)
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.boundaries.boundary import BaseBoundary
from fdtdx.objects.boundaries.pec import PerfectElectricConductor
from fdtdx.objects.detectors.diffractive import DiffractiveDetector
from fdtdx.objects.detectors.energy import EnergyDetector
from fdtdx.objects.detectors.field import FieldDetector
from fdtdx.objects.detectors.mode import ModeOverlapDetector
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.detectors.poynting_flux import PoyntingFluxDetector
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.sources.mode import ModePlaneSource
from fdtdx.typing import SliceTuple3D

#: Canonical order in which the field detectors stack their components (see e.g.
#: ``PhasorDetector.update``): ``Ex, Ey, Ez, Hx, Hy, Hz``. Each entry is ``(field_type, axis)``.
_COMPONENT_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_COMPONENT_SPEC: tuple[tuple[Literal["E", "H"], int], ...] = (
    ("E", 0),
    ("E", 1),
    ("E", 2),
    ("H", 0),
    ("H", 1),
    ("H", 2),
)

_AXIS_NAMES = ("x", "y", "z")


# ──────────────────────────────────────────────────────────────────────────────
# Reduction (used by place_objects)
# ──────────────────────────────────────────────────────────────────────────────


def reduce_resolved_slices(
    resolved_slices: dict[str, SliceTuple3D],
    object_map: dict[str, SimulationObject],
    config: SimulationConfig,
    volume_name: str,
) -> tuple[dict[str, SliceTuple3D], dict[str, SliceTuple3D], set[str], tuple[int, int, int]]:
    """Clip full-domain grid slices onto the symmetric (kept upper) half along each axis.

    Args:
        resolved_slices (dict[str, SliceTuple3D]): Full-domain grid slices, keyed by object name.
        object_map (dict[str, SimulationObject]): Object name → object, used for type-aware warnings.
        config (SimulationConfig): Simulation config; ``config.symmetry`` selects the axes.
        volume_name (str): Name of the simulation volume object.

    Returns:
        tuple: ``(new_slices, unreduced_slices, dropped_names, reduced_volume_shape)`` where
        ``new_slices`` excludes the dropped objects, ``unreduced_slices`` holds each surviving
        object's *unclipped* slice shifted into the reduced coordinate frame (negative start on
        axes where the object crossed the symmetry plane), ``dropped_names`` lists objects entirely
        in the discarded half, and ``reduced_volume_shape`` is the reduced volume grid shape
        ``(Nx', Ny', Nz')``.
    """
    symmetry = config.symmetry
    vol_slice = resolved_slices[volume_name]

    # Per-axis absolute index of the symmetry plane (= shift applied to map full → reduced),
    # and the reduced volume slice.
    mid_abs = [0, 0, 0]
    new_vol: list[tuple[int, int]] = []
    for a in range(3):
        vs0, vs1 = vol_slice[a]
        n = vs1 - vs0
        if symmetry[a] != 0:
            validate_symmetric_axis_cells(n, _AXIS_NAMES[a], subject="simulation volume")
            mid_abs[a] = vs0 + n // 2
            new_vol.append((0, vs1 - mid_abs[a]))
        else:
            mid_abs[a] = vs0
            new_vol.append((vs0, vs1))

    reduced_volume_shape = (
        new_vol[0][1] - new_vol[0][0],
        new_vol[1][1] - new_vol[1][0],
        new_vol[2][1] - new_vol[2][0],
    )

    new_slices: dict[str, SliceTuple3D] = {}
    unreduced_slices: dict[str, SliceTuple3D] = {}
    dropped: set[str] = set()
    for name, sl in resolved_slices.items():
        if name == volume_name:
            new_slices[name] = (new_vol[0], new_vol[1], new_vol[2])
            shifted = [(sl[a][0] - mid_abs[a], sl[a][1] - mid_abs[a]) if symmetry[a] != 0 else sl[a] for a in range(3)]
            unreduced_slices[name] = (shifted[0], shifted[1], shifted[2])
            continue

        obj = object_map[name]
        clipped: list[tuple[int, int]] = []
        unclipped: list[tuple[int, int]] = []
        drop = False
        for a in range(3):
            s0, s1 = sl[a]
            if symmetry[a] != 0:
                m = mid_abs[a]
                ns0 = max(s0, m) - m
                ns1 = min(s1, vol_slice[a][1]) - m
                if ns1 <= ns0:
                    drop = True
                elif s0 < m < s1 and (s0 + s1) != 2 * m:
                    logger.warning(
                        f"Object '{name}' straddles the {_AXIS_NAMES[a]}-symmetry plane "
                        f"asymmetrically (grid [{s0}, {s1}] about center {m}); the reduced "
                        f"simulation assumes mirror symmetry, so the discarded half is replaced by "
                        f"the mirror of the kept half."
                    )
                clipped.append((ns0, ns1))
                unclipped.append((s0 - m, s1 - m))
            else:
                clipped.append((s0, s1))
                unclipped.append((s0, s1))

        if drop:
            dropped.add(name)
            if isinstance(obj, BlochBoundary):
                logger.warning(
                    f"Dropped periodic/Bloch boundary '{name}' on a symmetry axis. Periodic "
                    f"boundaries and mirror symmetry are mutually exclusive on the same axis; use "
                    f"PML (or nothing) on the far side of a symmetric axis."
                )
            elif not isinstance(obj, BaseBoundary):
                logger.warning(
                    f"Object '{name}' lies entirely in the discarded (lower) half of a symmetric "
                    f"axis and was removed. Place it in the upper half or straddling the symmetry "
                    f"plane (the geometric center) so it survives the reduction."
                )
            continue

        new_slices[name] = (clipped[0], clipped[1], clipped[2])
        unreduced_slices[name] = (unclipped[0], unclipped[1], unclipped[2])

        if isinstance(obj, BlochBoundary) and symmetry[obj.axis] != 0:
            # The min-side boundary of a symmetric axis is dropped above; a far-side one survives.
            # Its wrap padding must not reach the min-side halo, which belongs to the mirror (see
            # pad_fields_for_boundaries) - say so rather than let the axis look fully periodic.
            logger.warning(
                f"Periodic/Bloch boundary '{name}' sits on the {_AXIS_NAMES[obj.axis]}-symmetry "
                f"axis. Periodic boundaries and mirror symmetry are mutually exclusive on the same "
                f"axis: the halo behind the symmetry plane is set by the mirror, not by wrapping to "
                f"this boundary. Use PML (or nothing) on the far side of a symmetric axis."
            )

    return new_slices, unreduced_slices, dropped, reduced_volume_shape


def make_symmetry_walls(
    config: SimulationConfig,
    reduced_volume_shape: tuple[int, int, int],
    key: jax.Array,
    existing_names: set[str],
) -> list[SimulationObject]:
    """Create the PEC wall objects sitting on each *electric* symmetry plane (reduced min edge).

    Only an electric (``-1``) plane gets a wall object. It sits on the tangential-``E`` node row at
    the reduced domain's min edge, where the odd symmetry does make those components vanish, so the
    condition is exactly a PEC face.

    A magnetic (``+1``) plane gets **no wall object**. Its plane is half a cell below the reduced
    domain: sources and materials are rasterized per cell, so the discrete problem is mirror
    symmetric about the tangential-``H`` node one cell below the min edge, where tangential ``H``
    vanishes — which is precisely what the zero halo of the field padding already supplies. Placing a
    :class:`~fdtdx.PerfectMagneticConductor` there instead (zeroing tangential ``H`` at the first
    cell, half a cell inside the domain) or filling that halo with ``-H_t[0]`` forces the mirror onto
    the cell edge and displaces the plane by half a cell: a consistent discretization of the *wrong*
    problem, which converges cleanly at first order and so looks plausible while being wrong.

    Args:
        config (SimulationConfig): Simulation config; ``config.symmetry`` selects the axes/types.
        reduced_volume_shape (tuple[int, int, int]): Reduced volume grid shape.
        key (jax.Array): JAX key for placement.
        existing_names (set[str]): Names already in use, so wall names can be made unique.

    Returns:
        list[SimulationObject]: PEC boundary objects for the electric symmetry planes, already
        placed on the reduced grid. Empty if every symmetric axis is magnetic.
    """
    walls: list[SimulationObject] = []
    used = set(existing_names)
    for a in range(3):
        if config.symmetry[a] != -1:
            continue
        name = f"_sym_wall_{_AXIS_NAMES[a]}"
        suffix = 0
        while name in used:
            suffix += 1
            name = f"_sym_wall_{_AXIS_NAMES[a]}_{suffix}"
        used.add(name)

        partial_grid_shape: tuple[int | None, int | None, int | None] = (
            1 if a == 0 else None,
            1 if a == 1 else None,
            1 if a == 2 else None,
        )
        wall = PerfectElectricConductor(
            name=name,
            axis=a,
            partial_grid_shape=partial_grid_shape,
            direction="-",
        )

        grid_slice: list[tuple[int, int]] = [(0, reduced_volume_shape[b]) for b in range(3)]
        grid_slice[a] = (0, 1)

        key, subkey = jax.random.split(key)
        wall = wall.place_on_grid(
            grid_slice_tuple=(grid_slice[0], grid_slice[1], grid_slice[2]),
            config=config,
            key=subkey,
        )
        wall = wall.aset("_is_symmetry_wall", True)
        walls.append(wall)
    return walls


def apply_mode_symmetry(
    placed_objects: list[SimulationObject],
    config: SimulationConfig,
) -> list[SimulationObject]:
    """Warn about explicit mode-solver symmetry tuples that ``config.symmetry`` supersedes.

    Mode objects clipped by a symmetry plane no longer ask the mode solver for a symmetric solve:
    :func:`~fdtdx.core.physics.modes.compute_mode_symmetry_reduced` mirrors the reduced
    cross-section back to the full one, solves there and restricts the result, which reproduces the
    mode the unreduced simulation would use (the solver's own symmetric solve does not, because
    FDTDX hands it one cell-centred permittivity array per component while the solver samples
    materials on its staggered grid). A user-set ``symmetry`` tuple is therefore inert here; say so
    rather than silently ignoring it.

    Args:
        placed_objects (list[SimulationObject]): Objects already placed on the reduced grid.
        config (SimulationConfig): Simulation config.

    Returns:
        list[SimulationObject]: The objects, unchanged.
    """
    del config
    for obj in placed_objects:
        if not isinstance(obj, (ModePlaneSource, ModeOverlapDetector)) or obj.symmetry == (0, 0):
            continue
        shape = obj.grid_shape
        if 1 not in shape:
            continue
        propagation_axis = shape.index(1)
        if obj.symmetry_mirror_axes(exclude_axis=propagation_axis):
            logger.warning(
                f"Mode object '{obj.name}' sets symmetry={obj.symmetry} for the mode solver, but "
                f"config.symmetry already reduces its cross-section. The explicit value is ignored: the "
                f"mode is solved on the mirrored full cross-section and restricted to the kept half. "
                f"Remove it, or drop config.symmetry on those axes and keep your own PEC/PMC boundary."
            )
    return placed_objects


# ──────────────────────────────────────────────────────────────────────────────
# Unfolding (public helpers)
# ──────────────────────────────────────────────────────────────────────────────


def _check_has_symmetry(symmetry: tuple[int, int, int]) -> None:
    if not any(s != 0 for s in symmetry):
        raise ValueError(
            "Nothing to unfold: this simulation has no symmetry (config.symmetry is (0, 0, 0)). "
            "The unfold helpers are only meaningful for symmetry-reduced simulations."
        )


def unfold_fields(
    field: jax.Array,
    symmetry: tuple[int, int, int],
    field_type: Literal["E", "H"],
) -> jax.Array:
    """Reconstruct a full-domain ``(3, Nx, Ny, Nz)`` field array from the reduced field.

    Mirrors the field across each symmetric axis with the correct per-component parity and
    concatenates the mirror image in front of the kept half, so the symmetry plane ends up at the
    center of the reconstructed array.

    The index map is chosen per axis from its wall type (see
    :func:`~fdtdx.core.physics.symmetry.mirror_pairs_on_plane`). Across an **electric** plane, which
    sits on the reduced domain's min edge, a component sampled *on* it (tangential ``E``, normal
    ``H``) has its first sample as its own mirror, so only the remaining ones produce images, while
    the half-cell-offset components mirror one-to-one. Across a **magnetic** plane, which sits half a
    cell below the min edge, every component mirrors one-to-one. In the on-plane case one
    reconstructed cell per axis — the outermost of the mirrored half, whose partner lies outside the
    kept domain — is filled by repeating its neighbour; it is the far-boundary cell of the
    reconstructed domain (inside the absorbing layer in a typical setup).

    Args:
        field (jax.Array): Reduced field, shape ``(3, Nx, Ny, Nz)`` (component axis first).
        symmetry (tuple[int, int, int]): Per-axis symmetry ``(x, y, z)``; see module docstring.
        field_type (Literal["E", "H"]): Whether ``field`` is the electric or magnetic field.

    Returns:
        jax.Array: Full-domain field; each symmetric axis is doubled in size.
    """
    if field_type not in ("E", "H"):
        raise ValueError(f"field_type must be 'E' or 'H', got {field_type!r}")
    _check_has_symmetry(symmetry)
    arr = field
    for a in range(3):
        if symmetry[a] == 0:
            continue
        components = []
        for c in range(3):
            single = arr[c : c + 1]
            low = mirror_extend_low_side(
                single,
                axis=a + 1,
                parity=field_component_parity(field_type, c, a, symmetry[a]),
                on_plane=mirror_pairs_on_plane(field_type, c, a, symmetry[a]),
            )
            components.append(jnp.concatenate([low, single], axis=a + 1))
        arr = jnp.concatenate(components, axis=0)
    return arr


def unfold_source_mode(
    source,
    config: SimulationConfig,
) -> tuple[jax.Array, jax.Array]:
    """Reconstruct the full-domain ``(E, H)`` mode profile a mode source injects.

    A :class:`ModePlaneSource` solves and stores its mode on the *reduced* cross-section
    (``source._E`` / ``source._H``, real, shape ``(3, Nx, Ny, Nz)`` with a singleton on the
    propagation axis). This mirrors that profile back to the full transverse cross-section with the
    correct per-component parity. Only the two transverse axes are unfolded — the propagation axis
    is never a symmetry plane for a mode source (its mode plane is one cell thick there).

    Convenience wrapper so callers don't reach into the private ``_E`` / ``_H`` state; for the
    fields *recorded during the run*, prefer a detector on the source plane plus
    :func:`unfold_detector_states`.

    Args:
        source: A placed mode source whose ``apply`` has already run (so ``_E`` / ``_H`` are
            populated), e.g. a :class:`ModePlaneSource`. Run :func:`fdtdx.apply_params` first.
        config (SimulationConfig): Simulation config; must have nonzero ``symmetry`` on at least one
            axis transverse to the source's propagation axis.

    Returns:
        tuple[jax.Array, jax.Array]: Full-domain ``(E, H)`` mode profiles, shape ``(3, Nx, Ny, Nz)``.

    Raises:
        ValueError: If the mode has not been computed yet, the source is not a plane, or there is no
            transverse symmetry to unfold.
    """
    mode_E = getattr(source, "_E", None)
    mode_H = getattr(source, "_H", None)
    if mode_E is None or mode_H is None or isinstance(mode_E, Null) or isinstance(mode_H, Null):
        raise ValueError(
            f"Source '{getattr(source, 'name', source)}' has no computed mode profile yet. Run "
            "fdtdx.apply_params (which invokes the source's mode solver) before unfolding."
        )

    shape = source.grid_shape
    if 1 not in shape:
        raise ValueError(
            f"unfold_source_mode expects a plane source (one grid dimension of size 1); got grid shape {shape}."
        )
    propagation_axis = shape.index(1)

    # Only the transverse axes are symmetry planes for a mode source; mask the propagation axis.
    transverse_symmetry: tuple[int, int, int] = (
        0 if propagation_axis == 0 else config.symmetry[0],
        0 if propagation_axis == 1 else config.symmetry[1],
        0 if propagation_axis == 2 else config.symmetry[2],
    )
    if config.symmetry[propagation_axis] != 0:
        logger.warning(
            f"config.symmetry is nonzero on the source's propagation axis "
            f"{_AXIS_NAMES[propagation_axis]}; ignoring it when unfolding the mode profile (a mode "
            f"plane is never split along its propagation axis)."
        )
    if not any(s != 0 for s in transverse_symmetry):
        raise ValueError(
            "No transverse symmetry to unfold for this source plane: config.symmetry is zero on both "
            "axes transverse to the propagation axis."
        )

    return unfold_fields(mode_E, transverse_symmetry, "E"), unfold_fields(mode_H, transverse_symmetry, "H")


def unfold_array(
    arr: jax.Array,
    symmetry: tuple[int, int, int],
    spatial_axes: tuple[int, int, int],
    signs: dict[int, jax.Array] | None = None,
    on_plane_axes: tuple[int, ...] = (),
) -> jax.Array:
    """Mirror-and-concatenate a spatial array along each symmetric axis.

    Generic building block used by :func:`unfold_detector_states`. For every axis ``a`` with
    ``symmetry[a] != 0`` the array is mirrored along its corresponding array axis
    ``spatial_axes[a]`` (optionally multiplied by a broadcastable per-component sign), and the
    mirror image is concatenated in front of the original.

    Args:
        arr (jax.Array): Array to unfold.
        symmetry (tuple[int, int, int]): Per-axis symmetry ``(x, y, z)``; ``0`` axes are skipped.
        spatial_axes (tuple[int, int, int]): For each physical axis, the corresponding array axis.
        signs (dict[int, jax.Array] | None): Optional mapping physical-axis → broadcastable sign
            array applied to the mirror image (defaults to ``+1``).
        on_plane_axes (tuple[int, ...]): Physical axes whose samples lie *on* the symmetry plane
            rather than half a cell off it. There the first sample is its own mirror, so it is not
            duplicated and the outermost reconstructed sample repeats its neighbour. Only an electric
            plane can be on-plane at all — a magnetic one sits half a cell below the reduced domain,
            so everything mirrors one-to-one across it; see :func:`_colocated_on_plane_axes`.

    Returns:
        jax.Array: The unfolded array.
    """
    _check_has_symmetry(symmetry)
    for a in range(3):
        if symmetry[a] == 0:
            continue
        ax = spatial_axes[a]
        sign = 1.0 if signs is None else signs.get(a, 1.0)
        if a in on_plane_axes:
            mirror = mirror_extend_low_side(arr, axis=ax, parity=1, on_plane=True) * sign
        else:
            mirror = jnp.flip(arr, axis=ax) * sign
        arr = jnp.concatenate([mirror, arr], axis=ax)
    return arr


def _colocated_on_plane_axes(detector, touched: tuple[int, int, int]) -> tuple[int, ...]:
    """Which physical axes a detector's stored samples sit on, for the mirror index map.

    With ``exact_interpolation`` the six components are co-located at the ``E_z`` Yee point
    ``(i, j, k+1/2)``: integer along x and y and half a cell off along z. An integer position lands on
    an **electric** symmetry plane, which sits on the reduced domain's min edge, so those samples pair
    as ``m ± j``. A **magnetic** plane sits half a cell below the min edge, so nothing is on-plane
    there and every sample mirrors one-to-one (see
    :func:`~fdtdx.core.physics.symmetry.mirror_pairs_on_plane`). Without ``exact_interpolation`` the
    raw staggered components are stored, each with its own offset, and no single index map fits; those
    fall back to the plain flip.

    Args:
        detector: The detector whose stored output is being unfolded.
        touched (tuple[int, int, int]): Per-axis wall type of the symmetry planes this detector was
            clipped by (``0`` where it was not).
    """
    if not getattr(detector, "exact_interpolation", False):
        return ()
    return tuple(a for a in (0, 1) if touched[a] == -1)


def _stored_component_spec(components) -> list[tuple[Literal["E", "H"], int]]:
    """Stored ``(field_type, component_axis)`` order for a Field/Phasor detector's components."""
    return [_COMPONENT_SPEC[i] for i, name in enumerate(_COMPONENT_NAMES) if name in components]


def _component_signs(
    spec: list[tuple[Literal["E", "H"], int]],
    touched: tuple[int, int, int],
    ndim: int,
    component_axis: int,
) -> dict[int, jax.Array]:
    """Per-axis, per-component mirror signs broadcastable against a detector array."""
    out: dict[int, jax.Array] = {}
    for a in range(3):
        if touched[a] == 0:
            continue
        vals = [field_component_parity(ft, ca, a, touched[a]) for (ft, ca) in spec]
        shape = [1] * ndim
        shape[component_axis] = len(vals)
        out[a] = jnp.asarray(vals).reshape(shape)
    return out


def _reduce_factor(
    parities: list[list[int]],
    ndim: int,
    component_axis: int,
    mean: bool,
) -> jax.Array:
    """Per-component scale factor that turns a half-domain reduction into the full-domain one.

    For a linear reduction over the detector region, the full-domain value relates to the stored
    half-domain value by ``∏_axes (1 + parity)`` for a *sum* and ``∏_axes (1 + parity) / 2`` for a
    *mean*: an even component doubles (sum) or is unchanged (mean), an odd component vanishes.

    Args:
        parities (list[list[int]]): ``parities[c]`` is the per-touched-axis parity list of component ``c``.
        ndim (int): Rank of the stored array (so the factor broadcasts against it).
        component_axis (int): Array axis holding the components.
        mean (bool): True if the detector reduces by averaging, False if by summing.

    Returns:
        jax.Array: Factor of shape ``[1, ..., num_components, ..., 1]`` (component axis populated).
    """
    factors = []
    for per_axis in parities:
        f = 1.0
        for p in per_axis:
            f *= (1 + p) / 2 if mean else (1 + p)
        factors.append(f)
    shape = [1] * ndim
    shape[component_axis] = len(factors)
    return jnp.asarray(factors).reshape(shape)


def _poynting_parity(component: int, axis: int, wall: int) -> int:
    """Mirror parity of the Poynting component ``S_component = (E x H)_component``.

    ``S_i`` involves the two field components transverse to ``i``, so its parity is the product of
    the corresponding ``E`` and ``H`` parities.
    """
    j, k = (x for x in range(3) if x != component)
    return field_component_parity("E", j, axis, wall) * field_component_parity("H", k, axis, wall)


def _unfold_poynting(detector, state: dict[str, jax.Array], touched: tuple[int, int, int]):
    """Unfold a PoyntingFluxDetector (scalar/vector, summed/spatial) via per-component flux parity."""
    arr = state["poynting_flux"]
    on_plane = _colocated_on_plane_axes(detector, touched)
    components = (0, 1, 2) if detector.keep_all_components else (detector.propagation_axis,)
    touched_axes = [a for a in range(3) if touched[a] != 0]

    if detector.reduce_volume:  # summed flux -> per-component factor ∏ (1 + parity)
        parities = [[_poynting_parity(i, a, touched[a]) for a in touched_axes] for i in components]
        if detector.keep_all_components:
            factor = _reduce_factor(parities, ndim=arr.ndim, component_axis=1, mean=False)  # (T, 3)
            return {"poynting_flux": arr * factor}
        scalar = 1.0
        for p in parities[0]:
            scalar *= 1 + p
        return {"poynting_flux": arr * scalar}

    # Spatial flux density -> mirror with the flux-component parity.
    if detector.keep_all_components:  # (T, 3, nx, ny, nz)
        signs = {
            a: jnp.asarray([_poynting_parity(i, a, touched[a]) for i in components]).reshape((1, 3, 1, 1, 1))
            for a in touched_axes
        }
        return {"poynting_flux": unfold_array(arr, touched, (2, 3, 4), signs, on_plane)}
    # (T, nx, ny, nz): single component along the propagation axis.
    p = detector.propagation_axis
    signs = {a: jnp.asarray(float(_poynting_parity(p, a, touched[a]))) for a in touched_axes}
    return {"poynting_flux": unfold_array(arr, touched, (1, 2, 3), signs, on_plane)}


def _unfold_energy_slices(
    state: dict[str, jax.Array],
    touched: tuple[int, int, int],
    on_plane_axes: tuple[int, ...] = (),
):
    """Unfold an EnergyDetector(as_slices=True): mirror each 2D plane along its in-plane symmetric
    axes. Energy density is even, so the parity is ``+1``; the collapsed-axis mean is already the
    full-domain value (energy is symmetric about the plane)."""
    # Each plane stores (T, axis_a, axis_b); map those two physical axes to array axes 1 and 2.
    planes = {
        "XY Plane": (0, 1),
        "XZ Plane": (0, 2),
        "YZ Plane": (1, 2),
    }
    out: dict[str, jax.Array] = {}
    for key, phys_axes in planes.items():
        arr = state[key]
        sub = [0, 0, 0]
        spatial_axes = [0, 0, 0]
        for arr_axis, phys in enumerate(phys_axes, start=1):
            sub[phys] = touched[phys]
            spatial_axes[phys] = arr_axis
        if any(s != 0 for s in sub):
            out[key] = unfold_array(
                arr,
                (sub[0], sub[1], sub[2]),
                (spatial_axes[0], spatial_axes[1], spatial_axes[2]),
                on_plane_axes=tuple(a for a in on_plane_axes if a in phys_axes),
            )
        else:
            out[key] = arr
    return out


def _unfold_one_detector(
    detector,
    state: dict[str, jax.Array],
    touched: tuple[int, int, int],
    count: int,
) -> dict[str, jax.Array]:
    """Unfold a single detector's recorded state to the full domain. See module docstring.

    Everything except :class:`DiffractiveDetector` is recovered purely from the stored reduced
    output plus the parity table: spatial outputs are mirrored per component, and ``reduce_volume``
    sums/means are rescaled per component (even doubles/keeps, odd vanishes).
    """
    # DiffractiveDetector first: it subclasses Detector (not PhasorDetector) and stores per-order
    # efficiencies from an FFT over the plane. Halving the domain changes the diffraction-order
    # basis, so the full-domain orders are not a function of the reduced-domain ones — unrecoverable
    # post-hoc from the stored output.
    if isinstance(detector, DiffractiveDetector):
        raise NotImplementedError(
            "unfold_detector_states cannot unfold a DiffractiveDetector: its diffraction-order basis "
            "depends on the (reduced) domain size, so the full-domain orders are not recoverable from "
            "the stored efficiencies. Unfold the fields with unfold_fields and recompute the orders."
        )

    # PhasorDetector (and ModeOverlapDetector, which subclasses it). Note: for ModeOverlapDetector
    # the mode-overlap S-parameter is already correct on the reduced domain (source and detector
    # share the reduced plane); unfolding only affects the raw stored phasor field.
    if isinstance(detector, PhasorDetector):
        spec = _stored_component_spec(detector.components)
        if detector.reduce_volume:  # (1, num_freqs, num_components); mean reduction
            arr = state["phasor"]
            parities = [
                [field_component_parity(ft, ca, a, touched[a]) for a in range(3) if touched[a]] for ft, ca in spec
            ]
            factor = _reduce_factor(parities, ndim=arr.ndim, component_axis=2, mean=True)
            return {"phasor": arr * factor.astype(arr.dtype)}
        arr = state["phasor"]  # (1, num_freqs, num_components, nx, ny, nz)
        signs = _component_signs(spec, touched, arr.ndim, component_axis=2)
        return {"phasor": unfold_array(arr, touched, (3, 4, 5), signs, _colocated_on_plane_axes(detector, touched))}

    if isinstance(detector, FieldDetector):
        spec = _stored_component_spec(detector.components)
        if detector.reduce_volume:  # (T, num_components); mean reduction
            arr = state["fields"]
            parities = [
                [field_component_parity(ft, ca, a, touched[a]) for a in range(3) if touched[a]] for ft, ca in spec
            ]
            factor = _reduce_factor(parities, ndim=arr.ndim, component_axis=1, mean=True)
            return {"fields": arr * factor.astype(arr.dtype)}
        arr = state["fields"]  # (T, num_components, nx, ny, nz)
        signs = _component_signs(spec, touched, arr.ndim, component_axis=1)
        return {"fields": unfold_array(arr, touched, (2, 3, 4), signs, _colocated_on_plane_axes(detector, touched))}

    if isinstance(detector, EnergyDetector):
        if detector.as_slices:
            return _unfold_energy_slices(state, touched, _colocated_on_plane_axes(detector, touched))
        if detector.reduce_volume:  # summed energy; energy density is even -> x2 per plane
            return {"energy": state["energy"] * (2**count)}
        arr = state["energy"]  # (T, nx, ny, nz)
        return {
            "energy": unfold_array(arr, touched, (1, 2, 3), on_plane_axes=_colocated_on_plane_axes(detector, touched))
        }

    if isinstance(detector, PoyntingFluxDetector):
        return _unfold_poynting(detector, state, touched)

    raise NotImplementedError(
        f"unfold_detector_states does not know how to unfold detector type "
        f"{type(detector).__name__!r}. Unfold the fields with unfold_fields instead."
    )


def unfold_detector_states(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
) -> ArrayContainer:
    """Reconstruct full-domain detector states from a symmetry-reduced simulation.

    This is a pure post-processing step: it transforms each detector's stored reduced-domain output
    into the full-domain result using the parity table, with no work added to the FDTD time loop.
    Each detector that sits on one or more symmetry planes is unfolded per its type — spatial
    outputs are mirrored with the correct per-component parity; ``reduce_volume`` sums/means are
    rescaled per component (even components double/keep, odd components vanish); ``as_slices`` energy
    planes are mirrored along their in-plane symmetric axes. Detectors that do not touch any
    symmetry plane are returned unchanged.

    All detector types are supported except :class:`DiffractiveDetector`, whose diffraction-order
    basis depends on the domain size and so cannot be recovered from the stored efficiencies; for
    that, unfold the fields with :func:`unfold_fields` and recompute.

    Args:
        arrays (ArrayContainer): Arrays returned by the reduced simulation.
        objects (ObjectContainer): The placed (reduced) objects.
        config (SimulationConfig): Simulation config (must have nonzero ``symmetry``).

    Returns:
        ArrayContainer: A copy of ``arrays`` with ``detector_states`` reconstructed to full domain.

    Raises:
        ValueError: If ``config.symmetry`` is ``(0, 0, 0)`` (nothing to unfold).
        NotImplementedError: For a :class:`DiffractiveDetector` (see above).
    """
    _check_has_symmetry(config.symmetry)
    detector_by_name = {d.name: d for d in objects.detectors}

    new_states: dict[str, dict[str, jax.Array]] = {}
    for name, state in arrays.detector_states.items():
        detector = detector_by_name.get(name)
        if detector is None:
            new_states[name] = state
            continue
        # Symmetric axes on which the detector was clipped because it crossed the symmetry plane.
        # A detector that merely begins at the plane but lies entirely inside the kept half was not
        # clipped, so its stored output is already the full-domain result on that axis.
        touched: tuple[int, int, int] = (
            config.symmetry[0] if detector.straddles_symmetry_plane(0) else 0,
            config.symmetry[1] if detector.straddles_symmetry_plane(1) else 0,
            config.symmetry[2] if detector.straddles_symmetry_plane(2) else 0,
        )
        count = sum(1 for a in range(3) if touched[a] != 0)
        if count == 0:
            new_states[name] = state
            continue
        new_states[name] = _unfold_one_detector(detector, state, touched, count)

    return arrays.aset("detector_states", new_states)
