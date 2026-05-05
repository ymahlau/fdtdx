from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

import gdstk
import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.grid import polygons_to_mask
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.materials import Material, compute_ordered_names
from fdtdx.objects.static_material.static import SimulationVolume, StaticMultiMaterialObject


@dataclass
class GDSLayerSpec:
    """Specification for a single GDS layer to be imported as a simulation object.

    Attributes:
        gds_layer: GDS layer number.
        gds_datatype: GDS datatype number (default 0).
        material_name: Key into the materials dictionary to use for this layer.
        thickness: Layer thickness in metres.
        z_base: Distance from the simulation volume bottom face to the base of this layer, in metres.
        name: Optional name for the resulting object. Auto-generates "gds_{layer}_{datatype}" if None.
    """

    gds_layer: int
    gds_datatype: int = 0
    material_name: str = ""
    thickness: float = 0.0
    z_base: float = 0.0
    name: str | None = None


@autoinit
class GDSLayerObject(StaticMultiMaterialObject):
    """A simulation object built from a set of GDS polygons extruded along one axis.

    Each instance represents one GDS layer (layer/datatype pair) extruded uniformly
    along ``axis``. The cross-sectional shape is described by ``polygons``, given in
    GDS coordinate space (metres). The mapping from GDS space to the local grid is
    controlled by ``gds_center``, which gives the GDS coordinate that coincides with
    the x/y center of the simulation object.
    """

    #: List of (N, 2) vertex arrays for each polygon, in GDS metres.
    polygons: list[np.ndarray] = frozen_field()

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
        """Return a per-axis size hint used during constraint resolution.

        The extrusion axis is constrained to ``thickness``; the two cross-section
        axes are unconstrained (None).

        Returns:
            tuple[float | None, float | None, float | None]: Size hints for axes 0, 1, 2.
        """
        hints: list[float | None] = [None, None, None]
        hints[self.axis] = self.thickness
        return (hints[0], hints[1], hints[2])

    def get_voxel_mask_for_shape(self) -> jax.Array:
        """Compute a 3-D boolean mask for the extruded polygon shape.

        The cross-sectional mask is computed from all GDS polygons via
        :func:`fdtdx.core.grid.polygons_to_mask`, then repeated along the
        extrusion axis to produce the full 3-D mask.

        Returns:
            jax.Array: Boolean array of shape ``self.grid_shape``.
        """
        n_h = self.grid_shape[self.horizontal_axis]
        n_v = self.grid_shape[self.vertical_axis]
        res = self._config.resolution
        half_res = 0.5 * res

        # Real-space size of the object in the cross-section plane.
        real_h = self.real_shape[self.horizontal_axis]
        real_v = self.real_shape[self.vertical_axis]

        # GDS coordinate of the lower-left corner of this object.
        origin_h = self.gds_center[0] - real_h / 2
        origin_v = self.gds_center[1] - real_v / 2

        # Shift polygons from GDS coords to local (object-relative) grid coords.
        local_polygons = [poly - np.array([origin_h, origin_v]) for poly in self.polygons]

        mask_2d = polygons_to_mask(
            boundary=(half_res, half_res, (n_h - 0.5) * res, (n_v - 0.5) * res),
            resolution=res,
            polygon_list=local_polygons,
        )

        # Extrude the 2-D mask along the extrusion axis.
        extrusion_height = self.grid_shape[self.axis]
        mask = jnp.repeat(
            jnp.expand_dims(jnp.asarray(mask_2d, dtype=jnp.bool), axis=self.axis),
            repeats=extrusion_height,
            axis=self.axis,
        )
        return mask

    def get_material_mapping(self) -> jax.Array:
        """Return an index array mapping every voxel to a material index.

        The material index is determined by the position of ``material_name``
        in the ordered names list computed from ``self.materials``.

        Returns:
            jax.Array: Integer array of shape ``self.grid_shape`` filled with
            the index of ``material_name``.
        """
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        return jnp.ones(self.grid_shape, dtype=jnp.int32) * idx


def gds_layer_stack(
    gds_source: str | pathlib.Path | gdstk.Library,
    cell_name: str,
    layers: list[GDSLayerSpec],
    materials: dict[str, Material],
    simulation_volume: SimulationVolume,
    gds_center: tuple[float, float] = (0.0, 0.0),
    extrusion_axis: int = 2,
    flatten: bool = True,
) -> tuple[list[GDSLayerObject], list[Any]]:
    """Build a list of :class:`GDSLayerObject` instances from a GDS file and layer specs.

    For each :class:`GDSLayerSpec`, polygons are extracted from the named GDS cell,
    converted to metres, and wrapped in a :class:`GDSLayerObject`. Two constraints are
    generated per object:

    * A position constraint that aligns the object's bottom face (along
      ``extrusion_axis``) with the simulation volume's bottom face offset by
      ``spec.z_base``.
    * A size constraint that matches the simulation volume's extent in the two
      cross-section axes.

    Args:
        gds_source: Path to a ``.gds`` file (str or :class:`pathlib.Path`) or an
            already-loaded :class:`gdstk.Library`.
        cell_name: Name of the GDS cell to read polygons from.
        layers: Ordered list of :class:`GDSLayerSpec` objects, one per GDS layer.
        materials: Shared materials dictionary forwarded to every
            :class:`GDSLayerObject`.
        simulation_volume: The :class:`SimulationVolume` used for size/position
            constraints.
        gds_center: GDS coordinate (horizontal, vertical) that maps to the x/y
            center of the simulation volume. Defaults to ``(0.0, 0.0)``.
        extrusion_axis: Axis along which layers are extruded (default 2 = z).
        flatten: If ``True``, call :meth:`gdstk.Cell.flatten` before reading
            polygons so that sub-cell references are included. Defaults to ``True``.

    Returns:
        A 2-tuple ``(objects, constraints)`` where *objects* is a list of
        :class:`GDSLayerObject` and *constraints* is a flat list of positional and
        size constraints (one pair per layer spec).

    Raises:
        ValueError: If *gds_source* is a file path but cannot be read, or if
            *cell_name* is not found in the library.
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

    cross_axes = tuple(a for a in range(3) if a != extrusion_axis)

    objects: list[GDSLayerObject] = []
    constraints: list[Any] = []

    for spec in layers:
        matching = [p for p in cell.polygons if p.layer == spec.gds_layer and p.datatype == spec.gds_datatype]
        polygons = [np.array(p.points) * lib.unit for p in matching]

        name = spec.name if spec.name is not None else f"gds_{spec.gds_layer}_{spec.gds_datatype}"

        extrusion_shape: list[float | None] = [None, None, None]
        extrusion_shape[extrusion_axis] = spec.thickness

        obj = GDSLayerObject(
            name=name,
            materials=materials,
            polygons=polygons,
            gds_center=gds_center,
            material_name=spec.material_name,
            axis=extrusion_axis,
            thickness=spec.thickness,
            partial_real_shape=tuple(extrusion_shape),  # type: ignore[arg-type]
        )

        # Align bottom face of this layer to simulation_volume bottom + z_base offset.
        z_constraint = obj.place_relative_to(
            other=simulation_volume,
            axes=(extrusion_axis,),
            own_positions=(-1.0,),
            other_positions=(-1.0,),
            margins=(spec.z_base,),
        )

        # Match the simulation volume's extent in the cross-section axes.
        xy_constraint = obj.same_size(
            other=simulation_volume,
            axes=cross_axes,
        )

        objects.append(obj)
        constraints.append(z_constraint)
        constraints.append(xy_constraint)

    return objects, constraints
