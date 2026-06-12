"""Automatic mirror-symmetry support for FDTD simulations.

When :attr:`fdtdx.SimulationConfig.symmetry` has a nonzero entry, :func:`fdtdx.place_objects`
reduces the domain to the symmetric half/quarter/octant before running the FDTD. This module
provides the two halves of that machinery:

* **Reduction** (used internally by ``place_objects``): clip every resolved grid slice onto the
  kept upper half along each symmetric axis, drop objects that fall entirely in the discarded
  half, insert a PEC/PMC wall on the symmetry plane, and forward the matching per-axis condition
  to mode sources/detectors.
* **Unfolding** (public helpers): reconstruct full-domain field and detector arrays from the
  reduced simulation, applying the correct per-component mirror parity.

Symmetry encoding (per axis, order ``(x, y, z)``): ``0`` = none, ``-1`` = PEC (electric wall),
``+1`` = PMC (magnetic wall). The symmetry plane is the geometric center of the axis; the upper
half is kept so the plane lands at the reduced domain's min edge — matching the mode solver's
"wall at the min edge" convention.
"""

from typing import Literal

import jax
import jax.numpy as jnp
from loguru import logger

from fdtdx.config import SimulationConfig
from fdtdx.core.misc import validate_symmetric_axis_cells
from fdtdx.core.null import Null
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.boundaries.boundary import BaseBoundary
from fdtdx.objects.boundaries.pec import PerfectElectricConductor
from fdtdx.objects.boundaries.pmc import PerfectMagneticConductor
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
# Parity table
# ──────────────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────────────
# Reduction (used by place_objects)
# ──────────────────────────────────────────────────────────────────────────────


def reduce_resolved_slices(
    resolved_slices: dict[str, SliceTuple3D],
    object_map: dict[str, SimulationObject],
    config: SimulationConfig,
    volume_name: str,
) -> tuple[dict[str, SliceTuple3D], set[str], tuple[int, int, int]]:
    """Clip full-domain grid slices onto the symmetric (kept upper) half along each axis.

    Args:
        resolved_slices (dict[str, SliceTuple3D]): Full-domain grid slices, keyed by object name.
        object_map (dict[str, SimulationObject]): Object name → object, used for type-aware warnings.
        config (SimulationConfig): Simulation config; ``config.symmetry`` selects the axes.
        volume_name (str): Name of the simulation volume object.

    Returns:
        tuple: ``(new_slices, dropped_names, reduced_volume_shape)`` where ``new_slices`` excludes
        the dropped objects, ``dropped_names`` lists objects entirely in the discarded half, and
        ``reduced_volume_shape`` is the reduced volume grid shape ``(Nx', Ny', Nz')``.
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
    dropped: set[str] = set()
    for name, sl in resolved_slices.items():
        if name == volume_name:
            new_slices[name] = (new_vol[0], new_vol[1], new_vol[2])
            continue

        obj = object_map[name]
        clipped: list[tuple[int, int]] = []
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
            else:
                clipped.append((s0, s1))

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

    return new_slices, dropped, reduced_volume_shape


def make_symmetry_walls(
    config: SimulationConfig,
    reduced_volume_shape: tuple[int, int, int],
    key: jax.Array,
    existing_names: set[str],
) -> list[SimulationObject]:
    """Create the PEC/PMC wall objects sitting on each symmetry plane (reduced min edge).

    Args:
        config (SimulationConfig): Simulation config; ``config.symmetry`` selects the axes/types.
        reduced_volume_shape (tuple[int, int, int]): Reduced volume grid shape.
        key (jax.Array): JAX key for placement.
        existing_names (set[str]): Names already in use, so wall names can be made unique.

    Returns:
        list[SimulationObject]: PEC/PMC boundary objects, already placed on the reduced grid.
    """
    walls: list[SimulationObject] = []
    used = set(existing_names)
    for a in range(3):
        if config.symmetry[a] == 0:
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
        wall_cls = PerfectElectricConductor if config.symmetry[a] == -1 else PerfectMagneticConductor
        wall = wall_cls(
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
        walls.append(wall)
    return walls


def _derive_mode_symmetry(
    propagation_axis: int,
    reduced_slice: SliceTuple3D,
    symmetry: tuple[int, int, int],
) -> tuple[int, int]:
    """Derive the mode-solver symmetry 2-tuple for a plane perpendicular to ``propagation_axis``.

    The mode solver expects the two non-propagation axes in increasing index order, with ``0`` =
    PEC and ``1`` = PMC at the min edge. A wall is imposed only when the plane sits on the
    symmetry plane (its reduced slice starts at index 0 on that transverse axis).
    """
    transverse = [a for a in range(3) if a != propagation_axis]
    result = []
    for a in transverse:
        if symmetry[a] == 1 and reduced_slice[a][0] == 0:
            result.append(1)  # PMC
        else:
            result.append(0)  # PEC / electric wall (also the no-symmetry default)
    return (result[0], result[1])


def apply_mode_symmetry(
    placed_objects: list[SimulationObject],
    config: SimulationConfig,
) -> list[SimulationObject]:
    """Forward the simulation symmetry to mode sources/detectors left at their default symmetry.

    For each :class:`ModePlaneSource` / :class:`ModeOverlapDetector` whose ``symmetry`` is still
    the default ``(0, 0)``, set it to the value derived from the simulation symmetry and the
    object's reduced placement. Explicit user-set values are respected (a mismatch warns).

    Args:
        placed_objects (list[SimulationObject]): Objects already placed on the reduced grid.
        config (SimulationConfig): Simulation config.

    Returns:
        list[SimulationObject]: The objects, with mode symmetry tuples filled in where applicable.
    """
    result: list[SimulationObject] = []
    for obj in placed_objects:
        if isinstance(obj, (ModePlaneSource, ModeOverlapDetector)):
            shape = obj.grid_shape
            if 1 not in shape:
                result.append(obj)
                continue
            propagation_axis = shape.index(1)
            derived = _derive_mode_symmetry(propagation_axis, obj._grid_slice_tuple, config.symmetry)
            if obj.symmetry == (0, 0):
                obj = obj.aset("symmetry", derived)
            elif obj.symmetry != derived:
                logger.warning(
                    f"Mode object '{obj.name}' has an explicit symmetry={obj.symmetry} that differs "
                    f"from the value {derived} implied by config.symmetry={config.symmetry}. Keeping "
                    f"the explicit value; ensure this is intended."
                )
        result.append(obj)
    return result


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
        signs = jnp.asarray(
            [field_component_parity(field_type, c, a, symmetry[a]) for c in range(3)],
            dtype=arr.dtype,
        ).reshape((3, 1, 1, 1))
        mirror = jnp.flip(arr, axis=a + 1) * signs
        arr = jnp.concatenate([mirror, arr], axis=a + 1)
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
) -> jax.Array:
    """Mirror-and-concatenate a spatial array along each symmetric axis.

    Generic building block used by :func:`unfold_detector_states`. For every axis ``a`` with
    ``symmetry[a] != 0`` the array is flipped along its corresponding array axis
    ``spatial_axes[a]`` (optionally multiplied by a broadcastable per-component sign), and the
    mirror image is concatenated in front of the original.

    Args:
        arr (jax.Array): Array to unfold.
        symmetry (tuple[int, int, int]): Per-axis symmetry ``(x, y, z)``; ``0`` axes are skipped.
        spatial_axes (tuple[int, int, int]): For each physical axis, the corresponding array axis.
        signs (dict[int, jax.Array] | None): Optional mapping physical-axis → broadcastable sign
            array applied to the mirror image (defaults to ``+1``).

    Returns:
        jax.Array: The unfolded array.
    """
    _check_has_symmetry(symmetry)
    for a in range(3):
        if symmetry[a] == 0:
            continue
        ax = spatial_axes[a]
        sign = 1.0 if signs is None else signs.get(a, 1.0)
        mirror = jnp.flip(arr, axis=ax) * sign
        arr = jnp.concatenate([mirror, arr], axis=ax)
    return arr


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
        return {"poynting_flux": unfold_array(arr, touched, (2, 3, 4), signs)}
    # (T, nx, ny, nz): single component along the propagation axis.
    p = detector.propagation_axis
    signs = {a: jnp.asarray(float(_poynting_parity(p, a, touched[a]))) for a in touched_axes}
    return {"poynting_flux": unfold_array(arr, touched, (1, 2, 3), signs)}


def _unfold_energy_slices(state: dict[str, jax.Array], touched: tuple[int, int, int]):
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
            out[key] = unfold_array(arr, (sub[0], sub[1], sub[2]), (spatial_axes[0], spatial_axes[1], spatial_axes[2]))
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
        return {"phasor": unfold_array(arr, touched, (3, 4, 5), signs)}

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
        return {"fields": unfold_array(arr, touched, (2, 3, 4), signs)}

    if isinstance(detector, EnergyDetector):
        if detector.as_slices:
            return _unfold_energy_slices(state, touched)
        if detector.reduce_volume:  # summed energy; energy density is even -> x2 per plane
            return {"energy": state["energy"] * (2**count)}
        arr = state["energy"]  # (T, nx, ny, nz)
        return {"energy": unfold_array(arr, touched, (1, 2, 3))}

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
        det_slice = detector._grid_slice_tuple
        # Symmetric axes whose reduced slice starts at the symmetry plane (min edge, index 0).
        touched: tuple[int, int, int] = (
            config.symmetry[0] if (config.symmetry[0] != 0 and det_slice[0][0] == 0) else 0,
            config.symmetry[1] if (config.symmetry[1] != 0 and det_slice[1][0] == 0) else 0,
            config.symmetry[2] if (config.symmetry[2] != 0 and det_slice[2][0] == 0) else 0,
        )
        count = sum(1 for a in range(3) if touched[a] != 0)
        if count == 0:
            new_states[name] = state
            continue
        new_states[name] = _unfold_one_detector(detector, state, touched, count)

    return arrays.aset("detector_states", new_states)
