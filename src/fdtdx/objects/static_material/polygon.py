import pathlib

import gdstk
import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.axis import get_transverse_axes
from fdtdx.core.grid import polygon_to_mask_at_points
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.materials import compute_ordered_names
from fdtdx.objects.static_material.static import StaticMultiMaterialObject


@autoinit
class ExtrudedPolygon(StaticMultiMaterialObject):
    """A polygon object specified by a list of vertices.

    The vertices must be given in a coordinate system centered at the origin, i.e. (0, 0)
    corresponds to the center of the object's bounding box. The polygon is placed so that
    its center coincides with the center of the grid region allocated to this object.

    The cross-section size is automatically inferred from the vertex bounding box for the
    two axes perpendicular to ``axis``, so ``partial_real_shape`` does not need to be
    specified for those axes.  The extrusion axis size must still be determined by a
    constraint or an explicit ``partial_real_shape`` entry.
    """

    #: Name of the material in the materials dictionary to be used for the object
    material_name: str = frozen_field()

    #: The extrusion axis.
    axis: int = frozen_field()

    #: numpy array of shape (N, 2) with vertices in metrical units (meter), centered at origin.
    vertices: np.ndarray = frozen_field()

    def __post_init__(self):
        w = float(self.vertices[:, 0].max() - self.vertices[:, 0].min())
        h = float(self.vertices[:, 1].max() - self.vertices[:, 1].min())
        real_shape = list(self.partial_real_shape)
        grid_shape = list(self.partial_grid_shape)
        for ax, size in ((self.horizontal_axis, w), (self.vertical_axis, h)):
            if real_shape[ax] is not None and real_shape[ax] != size:
                raise Exception(
                    f"ExtrudedPolygon {self.name}: partial_real_shape for axis {ax} is derived from the "
                    f"vertex bounding box ({size:.3e} m). Do not specify it explicitly."
                )
            if grid_shape[ax] is not None:
                raise Exception(
                    f"ExtrudedPolygon {self.name}: partial_grid_shape for axis {ax} is derived from the "
                    f"vertex bounding box. Do not specify it explicitly."
                )
            real_shape[ax] = size
        object.__setattr__(self, "partial_real_shape", tuple(real_shape))

    @property
    def horizontal_axis(self) -> int:
        """Gets the horizontal axis perpendicular to the fiber axis."""
        return get_transverse_axes(self.axis)[0]

    @property
    def vertical_axis(self) -> int:
        """Gets the vertical axis perpendicular to the fiber axis."""
        return get_transverse_axes(self.axis)[1]

    def _axis_sample_points(self, axis: int) -> tuple[np.ndarray, float]:
        """Cell centres of this object along ``axis``, measured from the start of the *unclipped* object.

        With mirror symmetry (``config.symmetry``) an object that crosses a symmetry plane is clipped to
        the kept half, but the polygon is defined on the full object. The cells that were cut off at the
        lower end are the mirror image of the first kept cells, so the kept cells start that many cells
        (``lost``) after the beginning of the full object. Without clipping ``lost`` is zero.

        Returns:
            tuple[np.ndarray, float]: the cell-centre coordinates in metres, and the extent of the full
            (unclipped) object along ``axis`` in metres.
        """
        lower, upper = self.grid_slice_tuple[axis]
        lost = lower - self.unreduced_grid_slice_tuple[axis][0]

        grid = self._config.resolved_grid
        if grid is None:
            spacing = self._config.uniform_spacing()
            centers = (np.arange(upper - lower) + lost + 0.5) * spacing
            return centers, (upper - lower + lost) * spacing

        edges = np.asarray(grid.edges(axis))
        shift = edges[lower + lost] - edges[lower]  # extent of the lost cells (mirror of the first kept cells)
        centers = 0.5 * (edges[lower:upper] + edges[lower + 1 : upper + 1]) - edges[lower] + shift
        return centers, float(edges[upper] - edges[lower] + shift)

    def get_voxel_mask_for_shape(self) -> jax.Array:
        h_centers, h_extent = self._axis_sample_points(self.horizontal_axis)
        v_centers, v_extent = self._axis_sample_points(self.vertical_axis)

        # Shift vertices from object-center coords to local grid coords.
        grid_vertices = self.vertices + np.array([0.5 * h_extent, 0.5 * v_extent])

        mask_2d = polygon_to_mask_at_points(
            x_coords=h_centers,
            y_coords=v_centers,
            polygon_vertices=grid_vertices,
        )
        extrusion_height = self.grid_shape[self.axis]
        mask = jnp.repeat(
            jnp.expand_dims(jnp.asarray(mask_2d, dtype=jnp.bool), axis=self.axis),
            repeats=extrusion_height,
            axis=self.axis,
        )

        return mask

    def get_material_mapping(
        self,
    ) -> jax.Array:
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        arr = jnp.ones(self.grid_shape, dtype=jnp.int32) * idx
        return arr


def extruded_polygon_from_gds(
    lib: gdstk.Library,
    cell_name: str,
    layer: int,
    datatype: int = 0,
    polygon_index: int = 0,
    **kwargs,
) -> ExtrudedPolygon:
    """Create an ExtrudedPolygon from a polygon in an already-loaded gdstk Library.

    Args:
        lib: An already-loaded gdstk Library.
        cell_name: Name of the GDS cell containing the polygon.
        layer: GDS layer number to read.
        datatype: GDS datatype (default 0).
        polygon_index: Which polygon to use when multiple exist on the layer (default 0).
        **kwargs: Forwarded to ExtrudedPolygon (axis, material_name, materials, …).

    Returns:
        ExtrudedPolygon with vertices centered around the origin in metres.

    Raises:
        ValueError: If the cell or layer/datatype combination is not found.
        IndexError: If polygon_index is out of range.
    """
    cell = next((c for c in lib.cells if isinstance(c, gdstk.Cell) and c.name == cell_name), None)
    if cell is None:
        raise ValueError(f"Cell '{cell_name}' not found in library")

    matching = [p for p in cell.polygons if p.layer == layer and p.datatype == datatype]
    if not matching:
        raise ValueError(f"No polygons on layer={layer}, datatype={datatype} in cell '{cell_name}'")
    if polygon_index >= len(matching):
        raise IndexError(
            f"polygon_index={polygon_index} out of range; found {len(matching)} polygon(s) on layer={layer}"
        )

    polygon = matching[polygon_index]
    vertices_m = np.array(polygon.points) * lib.unit  # library units → metres

    # centre vertices around origin (ExtrudedPolygon convention)
    centre = 0.5 * (vertices_m.min(axis=0) + vertices_m.max(axis=0))
    centred = vertices_m - centre

    return ExtrudedPolygon(vertices=centred, **kwargs)


def extruded_polygon_from_gds_path(
    gds_file: str | pathlib.Path,
    cell_name: str,
    layer: int,
    datatype: int = 0,
    polygon_index: int = 0,
    **kwargs,
) -> ExtrudedPolygon:
    """Create an ExtrudedPolygon from a polygon in a GDS file.

    Args:
        gds_file: Path to the .gds file.
        cell_name: Name of the GDS cell containing the polygon.
        layer: GDS layer number to read.
        datatype: GDS datatype (default 0).
        polygon_index: Which polygon to use when multiple exist on the layer (default 0).
        **kwargs: Forwarded to ExtrudedPolygon (axis, material_name, materials, …).

    Returns:
        ExtrudedPolygon with vertices centered around the origin in metres.

    Raises:
        ValueError: If the cell or layer/datatype combination is not found.
        IndexError: If polygon_index is out of range.
    """
    lib = gdstk.read_gds(str(gds_file))
    return extruded_polygon_from_gds(lib, cell_name, layer, datatype, polygon_index, **kwargs)
