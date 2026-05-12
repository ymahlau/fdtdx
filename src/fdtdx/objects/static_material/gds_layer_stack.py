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

from fdtdx.core.grid import polygon_to_mask_at_points, polygons_to_mask
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.materials import Material, compute_ordered_names
from fdtdx.objects.detectors.mode import ModeOverlapDetector
from fdtdx.objects.object import RealCoordinateConstraint
from fdtdx.objects.sources.mode import ModePlaneSource
from fdtdx.objects.static_material.static import SimulationVolume, StaticMultiMaterialObject


@dataclass(frozen=True)
class GDSLayerSpec:
    """Specification for a single GDS layer to be imported as a simulation object.

    Attributes:
        gds_layer: GDS layer number.
        material_name: Key into the materials dictionary to use for this layer.
        thickness: Layer thickness in metres.
        gds_datatype: GDS datatype number (default 0).
        z_base: Distance from the simulation volume bottom face to the base of this layer, in metres.
        name: Optional name for the resulting object. Auto-generates "gds_{layer}_{datatype}" if None.
        etch_by: Tuple of (layer, datatype) pairs whose polygons are subtracted from this layer via
            a boolean NOT operation before voxelization. Useful for etched features such as via holes.
    """

    gds_layer: int
    material_name: str
    thickness: float
    gds_datatype: int = 0
    z_base: float = 0.0
    name: str | None = None
    etch_by: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class GDSPortSpec:
    """Specification for a GDS port marker layer used to auto-generate sources or detectors.

    A port marker is a polygon (typically a thin rectangle) on a dedicated GDS layer.
    Its centroid determines the x/y position of the source or detector plane inside the
    simulation.  The source/detector is made 1 grid cell thick along ``propagation_axis``
    and spans the full simulation cross-section on the remaining two axes.

    Attributes:
        gds_layer: GDS layer containing the port marker polygons.
        gds_datatype: GDS datatype of the port markers (default 0).
        propagation_axis: Simulation axis along which the mode propagates (0=x, 1=y).
            Must be 0 or 1; the GDS layout encodes x/y positions only.
        name_prefix: Prefix for generated object names. Objects are named
            ``"{name_prefix}_{index}"``.
    """

    gds_layer: int
    gds_datatype: int = 0
    propagation_axis: int = 0
    name_prefix: str = "port"


@autoinit
class GDSLayerObject(StaticMultiMaterialObject):
    """A simulation object built from a set of GDS polygons extruded along one axis.

    Each instance represents one GDS layer (layer/datatype pair) extruded uniformly
    along ``axis``. The cross-sectional shape is described by ``polygons``, given in
    GDS coordinate space (metres). The mapping from GDS space to the local grid is
    controlled by ``gds_center``, which gives the GDS coordinate that coincides with
    the x/y center of the simulation object.
    """

    #: Sequence of (N, 2) vertex arrays for each polygon, in GDS metres.
    polygons: Sequence[np.ndarray] = frozen_field()

    #: GDS coordinate (horizontal, vertical) of the simulation x/y center.
    gds_center: tuple[float, float] = frozen_field()

    #: Key into the materials dictionary used for this object.
    material_name: str = frozen_field()

    #: The extrusion axis (0=x, 1=y, 2=z).
    axis: int = frozen_field()

    #: Extrusion thickness in metres.
    thickness: float = frozen_field()

    @property
    def horizontal_axis(self) -> int:
        """Cross-section axis that is not the extrusion axis and not the vertical axis.

        Returns:
            int: 1 (y) when extrusion axis is 0 (x), otherwise 0 (x).
        """
        return 1 if self.axis == 0 else 0

    @property
    def vertical_axis(self) -> int:
        """Second cross-section axis perpendicular to the extrusion axis.

        Returns:
            int: 1 (y) when extrusion axis is 2 (z), otherwise 2 (z).
        """
        return 1 if self.axis == 2 else 2

    def get_geometry_size_hint(self) -> tuple[float | None, float | None, float | None]:
        hints: list[float | None] = [None, None, None]
        hints[self.axis] = self.thickness
        return (hints[0], hints[1], hints[2])

    def get_voxel_mask_for_shape(self) -> jax.Array:
        """Compute a 3-D boolean mask for the extruded polygon shape.

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

        n_h = self.grid_shape[self.horizontal_axis]
        n_v = self.grid_shape[self.vertical_axis]

        real_h = self.real_shape[self.horizontal_axis]
        real_v = self.real_shape[self.vertical_axis]

        origin_h = self.gds_center[0] - real_h / 2
        origin_v = self.gds_center[1] - real_v / 2

        local_polygons = [poly - np.array([origin_h, origin_v]) for poly in self.polygons]

        grid = self._config.resolved_grid
        if grid is None:
            res = self._config.uniform_spacing()
            half_res = 0.5 * res
            mask_2d = polygons_to_mask(
                boundary=(half_res, half_res, (n_h - 0.5) * res, (n_v - 0.5) * res),
                resolution=res,
                polygon_list=local_polygons,
            )
        else:
            h_lower, h_upper = self.grid_slice_tuple[self.horizontal_axis]
            v_lower, v_upper = self.grid_slice_tuple[self.vertical_axis]
            h_edges = np.asarray(grid.edges(self.horizontal_axis))
            v_edges = np.asarray(grid.edges(self.vertical_axis))
            h_centers = 0.5 * (h_edges[h_lower:h_upper] + h_edges[h_lower + 1 : h_upper + 1]) - h_edges[h_lower]
            v_centers = 0.5 * (v_edges[v_lower:v_upper] + v_edges[v_lower + 1 : v_upper + 1]) - v_edges[v_lower]
            if len(local_polygons) == 0:
                mask_2d = np.zeros((n_h, n_v), dtype=bool)
            else:
                masks = [
                    polygon_to_mask_at_points(x_coords=h_centers, y_coords=v_centers, polygon_vertices=poly)
                    for poly in local_polygons
                ]
                mask_2d = np.any(np.stack(masks, axis=0), axis=0)

        extrusion_height = self.grid_shape[self.axis]
        mask = jnp.repeat(
            jnp.expand_dims(jnp.asarray(mask_2d, dtype=jnp.bool_), axis=self.axis),
            repeats=extrusion_height,
            axis=self.axis,
        )
        return mask

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


def _gds_to_sim(gds_val: float, gds_center_val: float, vol_size: float) -> float:
    """Convert a GDS coordinate (metres) to a simulation real-space coordinate (metres).

    Simulation real-space coordinates are measured from the lower-left corner of the
    simulation volume (grid index 0).
    """
    return gds_val - gds_center_val + vol_size / 2


@dataclass(frozen=True)
class _PortEntry:
    name: str
    dir_val: Literal["+", "-"]
    pgshape: tuple[int | None, int | None, int | None]
    prshape: tuple[float | None, float | None, float | None]
    prop_axis: int
    trans_axis: int
    sim_prop_pos: float


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
            sim_prop_pos = _gds_to_sim(gds_prop_val, gds_center[spec.propagation_axis], vol_size)

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
    """
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
            polygons=polygons,
            gds_center=gds_center,
            material_name=spec.material_name,
            axis=2,
            thickness=spec.thickness,
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
