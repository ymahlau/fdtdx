from __future__ import annotations

import pathlib
import tempfile
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import gdstk
import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.colors import XKCD_LIGHT_GREY, Color
from fdtdx.core.axis import get_transverse_axes
from fdtdx.core.grid import multi_polygons_to_mask, polygon_to_mask_at_points
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.materials import Material, compute_ordered_names
from fdtdx.objects.detectors.mode import ModeOverlapDetector
from fdtdx.objects.object import RealCoordinateConstraint
from fdtdx.objects.sources.mode import ModePlaneSource
from fdtdx.objects.static_material.static import SimulationVolume, StaticMultiMaterialObject


@dataclass(frozen=True)
class GDSLayerSpec:
    """Specification for a single GDS layer to be imported as a simulation object."""

    #: GDS layer number.
    gds_layer: int
    #: Key into the materials dictionary to use for this layer.
    material_name: str
    #: Layer thickness in metres.
    thickness: float
    #: GDS datatype number (default 0).
    gds_datatype: int = 0
    #: Distance from the simulation volume bottom face to the base of this layer, in metres.
    z_base: float = 0.0
    #: Optional name for the resulting object. Auto-generates ``"gds_{layer}_{datatype}"`` if None.
    name: str | None = None
    #: Tuple of (layer, datatype) pairs whose polygons are subtracted from this layer via
    #: a boolean NOT operation before voxelization. Useful for etched features such as via holes.
    etch_by: tuple[tuple[int, int], ...] = ()
    #: Sidewall angle in **degrees**, measured between the sidewall and the substrate plane, the way
    #: foundry PDKs specify it.  ``90.0`` (default) is a perfectly vertical wall.  An angle ``< 90``
    #: tilts the wall so the cross-section *shrinks* toward the top (regular trapezoid / positive-resist
    #: etch, e.g. 89 deg); an angle ``> 90`` makes it *grow* toward the top (re-entrant / undercut
    #: profile).  Must satisfy ``0 < sidewall_angle < 180``.  (Relation to the Tidy3D ``PolySlab``
    #: convention: ``polyslab_angle_rad = deg2rad(90 - sidewall_angle)``.)
    #:
    #: .. note::
    #:    For ``sidewall_angle != 90`` the trapezoidal profile is **staircased** on the z-grid: each
    #:    z-slice is eroded/dilated to a whole number of cells, so the wall is approximated by discrete
    #:    steps rather than a continuous slope.  This is therefore less accurate than Tidy3D's analytic
    #:    ``PolySlab`` (which represents the slanted face exactly).  A sub-cell fill-fraction treatment
    #:    that would remove the staircasing is tracked as a follow-up in issue #373.
    sidewall_angle: float = 90.0
    #: Which face keeps the nominal polygon footprint when ``sidewall_angle != 90``.
    #: ``"bottom"`` (default) keeps the base footprint and tapers the top inward for an angle ``< 90``;
    #: ``"top"`` keeps the top footprint; ``"middle"`` keeps the mid-height footprint and splits the
    #: taper symmetrically. Mirrors ``PolySlab.reference_plane``.
    reference_plane: Literal["bottom", "middle", "top"] = "bottom"
    color: Color | None = XKCD_LIGHT_GREY


@dataclass(frozen=True)
class GDSPortSpec:
    """Specification for a GDS port marker layer used to auto-generate sources or detectors.

    A port marker is a polygon (typically a thin rectangle) on a dedicated GDS layer.
    Its centroid determines the x/y position of the source or detector plane inside the
    simulation.  The source/detector is made 1 grid cell thick along ``propagation_axis``
    and spans the full simulation cross-section on the remaining two axes.
    """

    #: GDS layer containing the port marker polygons.
    gds_layer: int
    #: GDS datatype of the port markers (default 0).
    gds_datatype: int = 0
    #: Simulation axis along which the mode propagates (0=x, 1=y).
    #: Must be 0 or 1; the GDS layout encodes x/y positions only.
    propagation_axis: int = 0
    #: Prefix for generated object names. Objects are named ``"{name_prefix}_{index}"``.
    name_prefix: str = "port"


@autoinit
class GDSLayerObject(StaticMultiMaterialObject):
    """A simulation object built from a set of GDS polygons extruded along one axis.

    Each instance represents one GDS layer (layer/datatype pair) extruded uniformly
    along ``axis``. The cross-sectional shape is described by ``polygons``, given in
    GDS coordinate space (metres). The mapping from GDS space to the local grid is
    controlled by ``gds_center``, which gives the GDS coordinate that coincides
    with the x/y centre of the placed object. For example,
    ``gds_center=(0.0, 0.0)`` maps the GDS origin to the object's centre, while
    ``gds_center=(500e-9, 0.0)`` shifts the layout 500 nm to the left.
    """

    #: Sequence of (N, 2) vertex arrays for each polygon, in GDS metres.
    polygons: Sequence[np.ndarray] = frozen_field()

    #: GDS coordinate (horizontal, vertical) in metres that coincides with the x/y centre
    #: of the placed object.
    gds_center: tuple[float, float] = frozen_field()

    #: Key into the materials dictionary used for this object.
    material_name: str = frozen_field()

    #: The extrusion axis (0=x, 1=y, 2=z).
    axis: int = frozen_field()

    #: Extrusion thickness in metres.
    thickness: float = frozen_field()

    #: Sidewall angle in **degrees** between the wall and the substrate (foundry convention).
    #: ``90.0`` extrudes a vertical wall; other values produce a trapezoidal cross-section by eroding
    #: (angle ``< 90``) or dilating (angle ``> 90``) each z-slice laterally by an offset
    #: ``offset(z) = (z - z_ref) * tan(90deg - sidewall_angle)`` measured from ``reference_plane``.
    sidewall_angle: float = frozen_field(default=90.0)

    #: Face that keeps the nominal footprint: ``"bottom"``, ``"middle"`` or ``"top"``.
    reference_plane: Literal["bottom", "middle", "top"] = frozen_field(default="bottom")

    def __post_init__(self):
        if not 0.0 < self.sidewall_angle < 180.0:
            raise ValueError(f"sidewall_angle must lie in (0, 180) degrees (90 = vertical), got {self.sidewall_angle}.")
        if self.reference_plane not in ("bottom", "middle", "top"):
            raise ValueError(f"reference_plane must be one of 'bottom', 'middle', 'top'; got {self.reference_plane!r}.")

    @property
    def horizontal_axis(self) -> int:
        """Cross-section axis that is not the extrusion axis and not the vertical axis."""
        return get_transverse_axes(self.axis)[0]

    @property
    def vertical_axis(self) -> int:
        """Second cross-section axis perpendicular to the extrusion axis."""
        return get_transverse_axes(self.axis)[1]

    def get_geometry_size_hint(self) -> tuple[float | None, float | None, float | None]:
        hints: list[float | None] = [None, None, None]
        hints[self.axis] = self.thickness
        return (hints[0], hints[1], hints[2])

    def get_voxel_mask_for_shape(self) -> jax.Array:
        """Compute a 3-D boolean mask for the extruded polygon shape.

        When ``sidewall_angle`` is 90 deg the polygon footprint is extruded vertically and every
        z-slice is identical.  Otherwise each z-slice is offset laterally by
        ``(z_center - z_ref) * tan(90deg - sidewall_angle)`` (erosion for an angle < 90, dilation for
        an angle > 90), giving a trapezoidal cross-section that approximates ``Tidy3D.PolySlab`` with
        the equivalent angle / ``reference_plane``. The per-slice offset is applied in physical
        (metre) units via a Euclidean distance transform; on a non-uniform cross-section grid it uses a
        constant per-axis pitch (the minimum cell width), which is an approximation.

        Note:
            Only ``axis=2`` (z-extrusion) is supported.  GDS layouts encode x/y
            coordinates only, so extruding along x or y would require a z-coordinate
            that does not exist in the GDS file.

        Returns:
            jax.Array: Boolean array of shape ``self.grid_shape``.

        Raises:
            ValueError: If ``self.axis != 2``.
        """
        if self.axis != 2:
            raise ValueError(
                f"GDSLayerObject.get_voxel_mask_for_shape only supports axis=2 (z-extrusion); "
                f"got axis={self.axis}. GDS layouts encode x/y coordinates only."
            )

        mask_2d = self._base_mask_2d()
        extrusion_height = self.grid_shape[self.axis]

        if self.sidewall_angle == 90.0:
            mask = jnp.repeat(
                jnp.expand_dims(jnp.asarray(mask_2d, dtype=jnp.bool_), axis=self.axis),
                repeats=extrusion_height,
                axis=self.axis,
            )
            return mask

        mask_3d = self._extrude_with_sidewall(mask_2d, extrusion_height)
        return jnp.asarray(mask_3d, dtype=jnp.bool_)

    def _base_mask_2d(self) -> np.ndarray:
        """Rasterize the (vertical-extrusion) polygon footprint to a 2-D boolean mask.

        Returns:
            np.ndarray: Boolean array of shape ``(n_horizontal, n_vertical)``.
        """
        n_h = self.grid_shape[self.horizontal_axis]
        n_v = self.grid_shape[self.vertical_axis]

        real_h = self.real_shape[self.horizontal_axis]
        real_v = self.real_shape[self.vertical_axis]

        # Origin is now at the center of the simulation volume
        origin_h = self.gds_center[0]
        origin_v = self.gds_center[1]

        local_polygons = [poly - np.array([origin_h, origin_v]) for poly in self.polygons]

        grid = self._config.resolved_grid
        if grid is None:
            res = self._config.uniform_spacing()
            half = res / 2.0
            # Use cell centers boundaries
            mask_2d = multi_polygons_to_mask(
                boundary=(-real_h / 2 + half, -real_v / 2 + half, real_h / 2 - half, real_v / 2 - half),
                resolution=res,
                polygon_list=local_polygons,
            )
        else:
            h_lower, h_upper = self.grid_slice_tuple[self.horizontal_axis]
            v_lower, v_upper = self.grid_slice_tuple[self.vertical_axis]
            h_edges = np.asarray(grid.edges(self.horizontal_axis))
            v_edges = np.asarray(grid.edges(self.vertical_axis))
            # local_polygons are in gds_center-relative space (poly - gds_center).
            # gds_center maps to the object's physical centre in the simulation, so
            # cell centres must be expressed relative to the object's slice centre too.
            h_obj_center = 0.5 * (h_edges[h_lower] + h_edges[h_upper])
            v_obj_center = 0.5 * (v_edges[v_lower] + v_edges[v_upper])
            h_centers = 0.5 * (h_edges[h_lower:h_upper] + h_edges[h_lower + 1 : h_upper + 1]) - h_obj_center
            v_centers = 0.5 * (v_edges[v_lower:v_upper] + v_edges[v_lower + 1 : v_upper + 1]) - v_obj_center
            if len(local_polygons) == 0:
                mask_2d = np.zeros((n_h, n_v), dtype=bool)
            else:
                masks = [
                    polygon_to_mask_at_points(x_coords=h_centers, y_coords=v_centers, polygon_vertices=poly)
                    for poly in local_polygons
                ]
                mask_2d = np.any(np.stack(masks, axis=0), axis=0)
        return np.asarray(mask_2d, dtype=bool)

    def _z_centers(self, extrusion_height: int) -> np.ndarray:
        """Physical z cell-center coordinates relative to the layer base, one per z-slice."""
        grid = self._config.resolved_grid
        if grid is None:
            res = self._config.uniform_spacing()
            return (np.arange(extrusion_height) + 0.5) * res
        z_lower, z_upper = self.grid_slice_tuple[self.axis]
        z_edges = np.asarray(grid.edges(self.axis))
        return 0.5 * (z_edges[z_lower:z_upper] + z_edges[z_lower + 1 : z_upper + 1]) - z_edges[z_lower]

    def _extrude_with_sidewall(self, mask_2d: np.ndarray, extrusion_height: int) -> np.ndarray:
        """Stack the 2-D footprint into a trapezoidal 3-D mask using the sidewall model.

        ``sidewall_angle`` is validated in ``__post_init__`` (must be in ``(0, 180)`` degrees).
        """
        z_centers = self._z_centers(extrusion_height)
        if z_centers.size == 0:  # zero-height layer: nothing to extrude (parity with the 90deg fast path)
            empty_shape = list(mask_2d.shape)
            empty_shape.insert(self.axis, 0)
            return np.zeros(empty_shape, dtype=bool)
        if self.reference_plane == "bottom":
            z_ref = float(z_centers[0])
        elif self.reference_plane == "top":
            z_ref = float(z_centers[-1])
        else:  # middle: mid-plane of the resolved slices (offset 0 for a single-slice layer)
            z_ref = 0.5 * (float(z_centers[0]) + float(z_centers[-1]))

        # Physical cell pitch on the two cross-section axes (use the min spacing for the distance
        # transform sampling; uniform grids are exact, non-uniform grids are approximated).
        grid = self._config.resolved_grid
        if grid is None:
            pitch_h = pitch_v = float(self._config.uniform_spacing())
        else:
            pitch_h = float(np.min(grid.cell_widths(self.horizontal_axis)))
            pitch_v = float(np.min(grid.cell_widths(self.vertical_axis)))

        tan = float(np.tan(np.deg2rad(90.0 - self.sidewall_angle)))
        slices = []
        for z in z_centers:
            offset = (float(z) - z_ref) * tan  # angle<90 -> erode (shrink), angle>90 -> dilate (grow)
            slices.append(_offset_mask(mask_2d, offset, pitch_h, pitch_v))
        return np.stack(slices, axis=self.axis)

    def get_material_mapping(self) -> jax.Array:
        """Return an integer array filled with the index of ``material_name`` in the sorted material list.

        Returns:
            jax.Array: Integer array of shape ``self.grid_shape`` where every voxel has
            the same value — the position of ``material_name`` in the sorted material name list.
        """
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        return jnp.ones(self.grid_shape, dtype=jnp.int32) * idx


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _offset_mask(mask_2d: np.ndarray, offset: float, pitch_h: float, pitch_v: float) -> np.ndarray:
    """Erode or dilate a boolean mask by a physical lateral ``offset`` (metres).

    The offset is applied with a Euclidean distance transform sampled at the (possibly anisotropic)
    grid pitch, so a single fractional/grid-cell offset is handled uniformly and the result matches
    a constant inward (``offset > 0``) or outward (``offset < 0``) normal displacement of the polygon
    boundary — the discrete analogue of ``Tidy3D.PolySlab``'s linear sidewall.

    Args:
        mask_2d: Boolean footprint mask of shape ``(n_h, n_v)``.
        offset: Inward erosion distance in metres (negative dilates outward).
        pitch_h: Cell pitch along the horizontal axis in metres.
        pitch_v: Cell pitch along the vertical axis in metres.

    Returns:
        np.ndarray: Boolean mask of the same shape, eroded/dilated by ``offset``.
    """
    from scipy import ndimage

    if abs(offset) < 1e-15 or not mask_2d.any():
        # No taper here, or nothing to taper. A purely-dilated empty mask stays empty.
        return mask_2d.copy()

    sampling = (pitch_h, pitch_v)
    if offset > 0:
        # Distance (metres) from each solid cell to the nearest void cell; keep cells deeper than offset.
        dist_in = ndimage.distance_transform_edt(mask_2d, sampling=sampling)
        return dist_in > offset
    # Dilation: distance from each void cell to the nearest solid cell; add cells within |offset|.
    dist_out = ndimage.distance_transform_edt(~mask_2d, sampling=sampling)
    return mask_2d | (dist_out <= -offset)


def _load_gds_cell(
    gds_source: str | pathlib.Path | gdstk.Library,
    cell_name: str,
    flatten: bool,
) -> tuple[gdstk.Library, gdstk.Cell]:
    """Load a gdstk Library and find the named cell.

    Note:
        When *flatten* is ``True`` and *gds_source* is an already-loaded
        :class:`gdstk.Library`, the cell is flattened **in place** inside the
        caller's library.  Pass a file path to avoid mutating external state.
    """
    if isinstance(gds_source, str | pathlib.Path):
        lib = gdstk.read_gds(str(gds_source))
    else:
        lib = gds_source
    cell = next((c for c in lib.cells if isinstance(c, gdstk.Cell) and c.name == cell_name), None)
    if cell is None:
        raise ValueError(f"Cell '{cell_name}' not found in GDS library")
    if flatten:
        cell.flatten()
    return lib, cell


def _gds_to_sim(gds_val: float, gds_center_val: float) -> float:
    """Convert a GDS coordinate (metres) to a simulation real-space coordinate (metres).

    Simulation real-space coordinates are now measured from the center of the simulation volume.
    """
    return gds_val - gds_center_val


@dataclass(frozen=True)
class _PortEntry:
    """Placement data for one port polygon, computed by _iter_port_placements.

    Collects the pre-computed shapes, axes, and simulation-space position so
    that sources_from_gds_ports / detectors_from_gds_ports can construct objects
    and constraints without re-reading the GDS geometry.
    """

    name: str  # generated object name, e.g. "port_0"
    dir_val: Literal["+", "-"]  # propagation direction
    pgshape: tuple[int | None, int | None, int | None]  # partial_grid_shape (1 on prop_axis)
    prshape: tuple[float | None, float | None, float | None]  # partial_real_shape (width on trans_axis)
    prop_axis: int  # simulation propagation axis (0 or 1)
    trans_axis: int  # simulation transverse axis (the other of 0/1)
    sim_prop_pos: float  # left-face position along prop_axis in simulation real-space (metres)


def _iter_port_placements(
    lib: gdstk.Library,
    cell: gdstk.Cell,
    port_specs: list[GDSPortSpec],
    simulation_volume: SimulationVolume,
    gds_center: tuple[float, float],
    direction: Literal["+", "-"] | dict[str, Literal["+", "-"]],
) -> Generator[_PortEntry, None, None]:
    """Validate port specs and yield placement data for each port polygon."""
    for spec in port_specs:
        if spec.propagation_axis not in (0, 1):
            raise ValueError(
                f"propagation_axis={spec.propagation_axis} is not supported; "
                "GDS port markers encode x/y positions only - use 0 (x) or 1 (y)."
            )

        trans_axis = 1 if spec.propagation_axis == 0 else 0
        vol_size = simulation_volume.partial_real_shape[spec.propagation_axis]
        if vol_size is None:
            raise ValueError(
                f"simulation_volume.partial_real_shape[{spec.propagation_axis}] must be "
                "set to compute port positions from GDS coordinates."
            )

        matching = [p for p in cell.polygons if p.layer == spec.gds_layer and p.datatype == spec.gds_datatype]

        # Sort by (x, y) centroid so port_0 / port_1 / ... are always deterministic.
        poly_data = sorted(
            [(np.array(p.points).mean(axis=0) * lib.unit, p) for p in matching],
            key=lambda item: (float(item[0][0]), float(item[0][1])),
        )

        for i, (centroid, poly) in enumerate(poly_data):
            name = f"{spec.name_prefix}_{i}"

            gds_prop_val = centroid[spec.propagation_axis]
            sim_prop_pos = _gds_to_sim(gds_prop_val, gds_center[spec.propagation_axis])

            # Transverse width from the actual marker polygon — avoids spanning
            # neighbouring waveguides and corrupting mode-overlap integrals.
            points = np.array(poly.points) * lib.unit
            transverse_width = float(np.max(points[:, trans_axis]) - np.min(points[:, trans_axis]))

            pgshape_list: list[int | None] = [None, None, None]
            pgshape_list[spec.propagation_axis] = 1
            prshape_list: list[float | None] = [None, None, None]
            prshape_list[trans_axis] = transverse_width

            dir_val: Literal["+", "-"] = direction.get(name, "+") if isinstance(direction, dict) else direction

            yield _PortEntry(
                name=name,
                dir_val=dir_val,
                pgshape=(pgshape_list[0], pgshape_list[1], pgshape_list[2]),
                prshape=(prshape_list[0], prshape_list[1], prshape_list[2]),
                prop_axis=spec.propagation_axis,
                trans_axis=trans_axis,
                sim_prop_pos=sim_prop_pos,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gds_layer_stack(
    gds_source: str | pathlib.Path | gdstk.Library,
    cell_name: str,
    layers: list[GDSLayerSpec],
    materials: dict[str, Material],
    simulation_volume: SimulationVolume,
    gds_center: tuple[float, float],
    flatten: bool = True,
) -> tuple[list[GDSLayerObject], list[Any]]:
    """Build simulation objects from a GDS file according to a layer stack specification.

    For each :class:`GDSLayerSpec`, polygons are extracted from the named GDS cell,
    optionally etched by other layers, converted to metres, and wrapped in a
    :class:`GDSLayerObject`.  Layers are always extruded along z (axis 2) — GDS encodes
    only x/y polygon geometry, so z is the only axis that can be inferred from the file.
    For non-z extrusion (e.g. cross-section simulations), construct
    :class:`GDSLayerObject` instances directly with the desired ``axis``.  Two
    constraints are generated per object:

    * A position constraint aligning the object's bottom face (z) with the simulation
      volume's bottom face offset by ``spec.z_base``.
    * A size constraint matching the simulation volume's extent in the x/y axes.

    Args:
        gds_source: Path to a ``.gds`` file or an already-loaded :class:`gdstk.Library`.
        cell_name: Name of the GDS cell to read polygons from.
        layers: Ordered list of :class:`GDSLayerSpec` objects.
        materials: Materials dictionary forwarded to every :class:`GDSLayerObject`.
        simulation_volume: Used for size/position constraints.
        gds_center: GDS coordinate (in metres) that maps to the x/y centre of the
            simulation volume.
        flatten: Flatten sub-cell references before reading polygons. Defaults to ``True``.

    Returns:
        ``(objects, constraints)`` - one :class:`GDSLayerObject` and two constraints per
        layer spec.

    Raises:
        ValueError: If *cell_name* is not found in the library.
                    If *layers* is empty
    """
    if not layers:
        raise ValueError("*layers* list is empty, no objects to create.")

    lib, cell = _load_gds_cell(gds_source, cell_name, flatten)

    objects: list[GDSLayerObject] = []
    constraints: list[Any] = []

    for spec in layers:
        matching = [p for p in cell.polygons if p.layer == spec.gds_layer and p.datatype == spec.gds_datatype]

        if spec.etch_by and matching:
            etch_polys = [
                p
                for etch_layer, etch_dtype in spec.etch_by
                for p in cell.polygons
                if p.layer == etch_layer and p.datatype == etch_dtype
            ]
            if etch_polys:
                matching = gdstk.boolean(matching, etch_polys, "not")

        polygons = [np.array(p.points) * lib.unit for p in matching]

        name = spec.name if spec.name is not None else f"gds_{spec.gds_layer}_{spec.gds_datatype}"

        obj = GDSLayerObject(
            name=name,
            materials=materials,
            color=spec.color,
            polygons=polygons,
            gds_center=gds_center,
            material_name=spec.material_name,
            axis=2,
            thickness=spec.thickness,
            sidewall_angle=spec.sidewall_angle,
            reference_plane=spec.reference_plane,
            partial_real_shape=(None, None, spec.thickness),
        )

        z_constraint = obj.place_relative_to(
            other=simulation_volume,
            axes=(2,),
            own_positions=(-1.0,),
            other_positions=(-1.0,),
            margins=(spec.z_base,),
        )
        xy_constraint = obj.same_size(other=simulation_volume, axes=(0, 1))

        objects.append(obj)
        constraints.append(z_constraint)
        constraints.append(xy_constraint)

    return objects, constraints


def gds_layer_stack_from_component(
    component: Any,
    layers: list[GDSLayerSpec],
    materials: dict[str, Material],
    simulation_volume: SimulationVolume,
    gds_center: tuple[float, float],
    cell_name: str | None = None,
    flatten: bool = True,
) -> tuple[list[GDSLayerObject], list[Any]]:
    """Build a layer stack from a gdsfactory ``Component``.

    This is a thin wrapper around :func:`gds_layer_stack` that accepts a
    `gdsfactory <https://gdsfactory.github.io/gdsfactory/>`_ ``Component`` object
    instead of a GDS file path.  gdsfactory is **not** a required dependency of
    fdtdx; it must be installed separately (``pip install gdsfactory``).

    The component is exported to a temporary GDS file, which is read back via
    :func:`gdstk.read_gds`.  This approach is version-agnostic and works with all
    current gdsfactory releases.

    Args:
        component: A ``gdsfactory.Component`` instance.
        layers: Layer specifications forwarded to :func:`gds_layer_stack`.
        materials: Materials dictionary forwarded to every :class:`GDSLayerObject`.
        simulation_volume: Used for size/position constraints.
        gds_center: GDS coordinate (in metres) that maps to the x/y centre of the
            simulation volume.
        cell_name: GDS cell name to read.  Defaults to ``component.name``.
        flatten: Flatten sub-cell references before reading polygons.

    Returns:
        ``(objects, constraints)`` - same as :func:`gds_layer_stack`.

    Raises:
        ImportError: If gdsfactory is not installed.
        ValueError: If the resolved cell name is not found in the exported GDS.
    """
    try:
        import gdsfactory  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "gdsfactory is required for gds_layer_stack_from_component. Install it with: pip install gdsfactory"
        ) from exc

    name = cell_name if cell_name is not None else component.name

    # delete=False so the file is closed (and releasable on Windows) before write_gds opens it.
    with tempfile.NamedTemporaryFile(suffix=".gds", delete=False) as f:
        tmp = pathlib.Path(f.name)
    try:
        component.write_gds(str(tmp))
        lib = gdstk.read_gds(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)

    return gds_layer_stack(lib, name, layers, materials, simulation_volume, gds_center, flatten)


def sources_from_gds_ports(
    gds_source: str | pathlib.Path | gdstk.Library,
    cell_name: str,
    port_specs: list[GDSPortSpec],
    wave_character: WaveCharacter,
    simulation_volume: SimulationVolume,
    gds_center: tuple[float, float],
    direction: Literal["+", "-"] | dict[str, Literal["+", "-"]] = "+",
    mode_index: int = 0,
    filter_pol: Literal["te", "tm"] | None = None,
    height_axis: int = 2,
    flatten: bool = True,
) -> tuple[list[ModePlaneSource], list[Any]]:
    """Create :class:`~fdtdx.objects.sources.mode.ModePlaneSource` objects from GDS port markers.

    Each polygon on a port marker layer becomes one source.  The polygon's centroid
    determines the position of the source's left face along ``propagation_axis``; the
    source's width is set to match the port marker polygon's width on the transverse axis.

    Args:
        gds_source: Path to a ``.gds`` file or an already-loaded :class:`gdstk.Library`.
        cell_name: GDS cell containing the port marker polygons.
        port_specs: List of :class:`GDSPortSpec` objects (one per port layer).
        wave_character: Wavelength / frequency character forwarded to every source.
        simulation_volume: Reference object for cross-section size/position constraints.
            Its ``partial_real_shape`` must be set on ``propagation_axis``.
        gds_center: GDS coordinate (in metres) mapped to the x/y centre of the simulation
            volume.
        direction: Propagation direction passed to each source (``"+"`` or ``"-"``). Can also be a dict mapping port names to directions.
        mode_index: Waveguide mode index (default 0 = fundamental).
        filter_pol: Optional polarisation filter (``"te"``, ``"tm"``, or ``None``).
        height_axis: Simulation axis treated as the out-of-plane height; sources are made
            to span the full simulation extent on this axis (default 2 = z).  This assumes
            a single vertical stack — sources run from one face of the simulation volume to
            the other along ``height_axis``.
        flatten: Flatten sub-cell references before reading polygons.

    Returns:
        ``(sources, constraints)`` — four constraints per source: propagation position,
        transverse center, full-height size, and vertical center.

    Raises:
        ValueError: If ``propagation_axis`` is not 0 or 1, or if
            ``simulation_volume.partial_real_shape`` is ``None`` on ``propagation_axis``.
    """
    lib, cell = _load_gds_cell(gds_source, cell_name, flatten)
    sources: list[ModePlaneSource] = []
    constraints: list[Any] = []

    for entry in _iter_port_placements(lib, cell, port_specs, simulation_volume, gds_center, direction):
        src = ModePlaneSource(
            name=entry.name,
            wave_character=wave_character,
            direction=entry.dir_val,
            mode_index=mode_index,
            filter_pol=filter_pol,
            partial_grid_shape=entry.pgshape,
            partial_real_shape=entry.prshape,
        )
        prop_constraint = RealCoordinateConstraint(
            object=entry.name,
            axes=(entry.prop_axis,),
            sides=("-",),
            coordinates=(entry.sim_prop_pos,),
        )
        trans_center_constraint = src.place_at_center(simulation_volume, axes=(entry.trans_axis,))
        height_constraint = src.same_size(simulation_volume, axes=(height_axis,))
        vert_center_constraint = src.place_at_center(simulation_volume, axes=(height_axis,))
        sources.append(src)
        constraints.extend([prop_constraint, trans_center_constraint, height_constraint, vert_center_constraint])

    return sources, constraints


def detectors_from_gds_ports(
    gds_source: str | pathlib.Path | gdstk.Library,
    cell_name: str,
    port_specs: list[GDSPortSpec],
    wave_characters: Sequence[WaveCharacter],
    simulation_volume: SimulationVolume,
    gds_center: tuple[float, float],
    direction: Literal["+", "-"] | dict[str, Literal["+", "-"]] = "+",
    mode_index: int = 0,
    filter_pol: Literal["te", "tm"] | None = None,
    height_axis: int = 2,
    flatten: bool = True,
) -> tuple[list[ModeOverlapDetector], list[Any]]:
    """Create :class:`~fdtdx.objects.detectors.mode.ModeOverlapDetector` objects from GDS port markers.

    Mirrors :func:`sources_from_gds_ports` exactly but produces
    :class:`~fdtdx.objects.detectors.mode.ModeOverlapDetector` objects.  ``wave_characters``
    is a *sequence* to match the detector constructor signature.

    See :func:`sources_from_gds_ports` for full documentation of the constraint layout,
    ordering guarantee, and ``direction`` dict usage.

    Args:
        gds_source: Path to a ``.gds`` file or an already-loaded :class:`gdstk.Library`.
        cell_name: GDS cell containing the port marker polygons.
        port_specs: List of :class:`GDSPortSpec` objects.
        wave_characters: Sequence of wavelength characters forwarded to each detector.
        simulation_volume: Reference object for size/position constraints.
            Its ``partial_real_shape`` must be set on ``propagation_axis``.
        gds_center: GDS coordinate (in metres) mapped to the x/y centre of the
            simulation volume.
        direction: Propagation direction (``"+"`` or ``"-"``), applied to all detectors.
            Pass a ``dict`` mapping port names to individual directions when needed.
        mode_index: Waveguide mode index.
        filter_pol: Optional polarisation filter.
        height_axis: Simulation axis treated as the out-of-plane height; detectors are
            made to span the full simulation extent on this axis (default 2 = z).  This
            assumes a single vertical stack — detectors run from one face of the simulation
            volume to the other along ``height_axis``.
        flatten: Flatten sub-cell references before reading polygons.

    Returns:
        ``(detectors, constraints)`` — four constraints per detector: propagation position,
        transverse center, full-height size, and vertical center.

    Raises:
        ValueError: If ``propagation_axis`` is not 0 or 1, or if
            ``simulation_volume.partial_real_shape`` is ``None`` on ``propagation_axis``.
    """
    lib, cell = _load_gds_cell(gds_source, cell_name, flatten)
    detectors: list[ModeOverlapDetector] = []
    constraints: list[Any] = []

    for entry in _iter_port_placements(lib, cell, port_specs, simulation_volume, gds_center, direction):
        det = ModeOverlapDetector(
            name=entry.name,
            wave_characters=tuple(wave_characters),
            direction=entry.dir_val,
            mode_index=mode_index,
            filter_pol=filter_pol,
            partial_grid_shape=entry.pgshape,
            partial_real_shape=entry.prshape,
        )
        prop_constraint = RealCoordinateConstraint(
            object=entry.name,
            axes=(entry.prop_axis,),
            sides=("-",),
            coordinates=(entry.sim_prop_pos,),
        )
        trans_center_constraint = det.place_at_center(simulation_volume, axes=(entry.trans_axis,))
        height_constraint = det.same_size(simulation_volume, axes=(height_axis,))
        vert_center_constraint = det.place_at_center(simulation_volume, axes=(height_axis,))
        detectors.append(det)
        constraints.extend([prop_constraint, trans_center_constraint, height_constraint, vert_center_constraint])

    return detectors, constraints
