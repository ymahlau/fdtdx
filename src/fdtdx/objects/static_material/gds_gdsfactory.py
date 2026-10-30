from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Sequence
from typing import Any, Literal

from loguru import logger

from fdtdx.materials import Material
from fdtdx.objects.static_material.gds_layer_stack import (
    GDSLayerObject,
    GDSLayerSpec,
    gds_layer_stack_from_component,
)
from fdtdx.objects.static_material.static import SimulationVolume

#: One micrometre in metres. gdsfactory gives all lengths in micrometres.
_UM = 1e-6
#: A level that is thinner than this value (in metres) after clipping is skipped.
_MIN_THICKNESS = 1e-12
#: Default value of ``LayerLevel.mesh_order`` in gdsfactory.
_DEFAULT_MESH_ORDER = 3
#: Values of ``LayerLevel.width_to_z`` that have an equivalent ``GDSLayerSpec.reference_plane``.
_REFERENCE_PLANES: dict[float, Literal["bottom", "middle", "top"]] = {0.0: "bottom", 0.5: "middle", 1.0: "top"}


def _warn_level(level_name: str, message: str) -> None:
    """Show a ``UserWarning`` about one LayerStack level at the code that called the public function."""
    warnings.warn(f"LayerStack level '{level_name}': {message}", UserWarning, stacklevel=4)


def _mesh_order(level: Any) -> int:
    """Return the mesh order of a level. As in gdsfactory, a value in ``level.info`` overrides the field."""
    info = getattr(level, "info", None) or {}
    if "mesh_order" in info:
        return int(info["mesh_order"])
    return int(getattr(level, "mesh_order", _DEFAULT_MESH_ORDER))


def _is_zero_bias(bias: Any) -> bool:
    """Return True if ``LayerLevel.bias`` does not change the mask."""
    if bias is None:
        return True
    if isinstance(bias, int | float):
        return bias == 0
    return all(b == 0 for b in bias)


def _is_zero_z_to_bias(z_to_bias: Any) -> bool:
    """Return True if ``LayerLevel.z_to_bias`` does not change the mask at any height."""
    if z_to_bias is None:
        return True
    _, biases = z_to_bias
    return all(b == 0 for b in biases)


def _layer_stack_to_specs(
    layer_stack: Any,
    volume_bottom_z: float,
    volume_height: float,
    name_prefix: str = "",
    placement_order_base: int | None = None,
    subpixel_smoothing: bool = False,
) -> tuple[list[GDSLayerSpec], dict[str, str]]:
    """Convert the levels of a gdsfactory ``LayerStack`` to :class:`GDSLayerSpec` objects.

    This function does not import gdsfactory. It reads only the attributes of the levels. Level number ``i``
    in ``layer_stack.layers`` gets the GDS layer ``(i, 0)``. The caller must write the geometry of that level
    to this layer.

    Args:
        layer_stack: A gdsfactory ``LayerStack``, or an object with a ``layers`` dict of level-like objects.
        volume_bottom_z: z coordinate in metres, in the frame of the LayerStack, of the bottom face of the
            simulation volume.
        volume_height: Height of the simulation volume in metres. The levels are clipped to the volume.
        name_prefix: Prefix for the object names.
        placement_order_base: Placement order of the level with the lowest priority. If None, the levels get
            the placement orders ``-N, ..., -1``.
        subpixel_smoothing: Value of ``GDSLayerSpec.subpixel_smoothing`` for all levels.

    Returns:
        ``(specs, dropped)``. ``specs`` is sorted by priority: the last spec wins where specs overlap.
        ``dropped`` maps the name of each skipped level to the reason.

    Raises:
        ValueError: If the sidewall angle of a level is not in the range that fdtdx supports.
    """
    kept: list[tuple[int, int, GDSLayerSpec]] = []
    dropped: dict[str, str] = {}
    for index, (level_name, level) in enumerate(layer_stack.layers.items()):
        if level.thickness == 0:
            dropped[level_name] = "zero thickness"
            continue
        if level.material is None:
            _warn_level(level_name, "no material is set. The level is skipped.")
            dropped[level_name] = "no material"
            continue

        # A negative thickness extrudes downward from zmin.
        z_start = level.zmin * _UM - volume_bottom_z
        z_end = (level.zmin + level.thickness) * _UM - volume_bottom_z
        lower, upper = min(z_start, z_end), max(z_start, z_end)
        z_base, z_top = max(lower, 0.0), min(upper, volume_height)
        if z_top - z_base < _MIN_THICKNESS:
            dropped[level_name] = (
                f"outside of the simulation volume (level from z={lower:.4g} m to z={upper:.4g} m, "
                f"volume from z=0 m to z={volume_height:.4g} m)"
            )
            continue

        # gdsfactory measures the sidewall angle from the normal (0 = vertical). fdtdx measures it from the
        # substrate (90 = vertical). A downward extrusion mirrors the profile and the reference face.
        downward = level.thickness < 0
        sidewall_angle = 90.0 + level.sidewall_angle if downward else 90.0 - level.sidewall_angle
        if not 0.0 < sidewall_angle < 180.0:
            raise ValueError(
                f"LayerStack level '{level_name}': sidewall_angle={level.sidewall_angle} deg (from the normal) "
                f"gives {sidewall_angle} deg (from the substrate) in fdtdx. This value must be in (0, 180)."
            )
        width_to_z = 1.0 - level.width_to_z if downward else level.width_to_z
        nearest = min(_REFERENCE_PLANES, key=lambda w: abs(w - width_to_z))
        reference_plane = _REFERENCE_PLANES[nearest]
        if sidewall_angle != 90.0:
            if abs(nearest - width_to_z) > 1e-9:
                _warn_level(
                    level_name,
                    f"width_to_z={level.width_to_z} is not supported. fdtdx uses the {reference_plane} face "
                    "as the sidewall reference.",
                )
            bottom_clipped, top_clipped = z_base > lower, z_top < upper
            reference_clipped = {
                "bottom": bottom_clipped,
                "middle": bottom_clipped or top_clipped,
                "top": top_clipped,
            }[reference_plane]
            if reference_clipped:
                _warn_level(
                    level_name,
                    f"the simulation volume clips the sidewall reference face ({reference_plane}). "
                    "The footprint of the clipped level is not exact.",
                )
        if not _is_zero_bias(getattr(level, "bias", None)):
            _warn_level(level_name, "bias is not supported and is ignored.")
        if not _is_zero_z_to_bias(getattr(level, "z_to_bias", None)):
            _warn_level(level_name, "z_to_bias is not supported and is ignored.")

        spec = GDSLayerSpec(
            gds_layer=index,
            material_name=level.material,
            thickness=z_top - z_base,
            z_base=z_base,
            name=name_prefix + level_name,
            sidewall_angle=sidewall_angle,
            reference_plane=reference_plane,
            subpixel_smoothing=subpixel_smoothing,
        )
        kept.append((_mesh_order(level), index, spec))

    # In gdsfactory, the level with the lower mesh order wins overlaps. In fdtdx, the object with the higher
    # placement order is applied later and wins. For equal mesh orders, the later level in the stack wins.
    kept.sort(key=lambda item: (-item[0], item[1]))
    base = -len(kept) if placement_order_base is None else placement_order_base
    specs = [dataclasses.replace(spec, placement_order=base + rank) for rank, (_, _, spec) in enumerate(kept)]
    return specs, dropped


def gds_layer_stack_from_gdsfactory(
    component: Any,
    materials: dict[str, Material],
    simulation_volume: SimulationVolume,
    volume_bottom_z: float,
    layer_stack: Any | None = None,
    gds_center: tuple[float, float] | None = None,
    fill_levels: Sequence[str] = (),
    name_prefix: str = "",
    placement_order_base: int | None = None,
    subpixel_smoothing: bool = False,
) -> tuple[list[GDSLayerObject], list[Any]]:
    """Build simulation objects from a gdsfactory ``Component`` and a gdsfactory ``LayerStack``.

    Each level of the ``LayerStack`` becomes one :class:`GDSLayerObject` that is extruded along z. You do
    not write :class:`GDSLayerSpec` objects. gdsfactory is not a required dependency of fdtdx. Install it
    separately (``pip install gdsfactory``). A gdsfactory PDK must be active, because gdsfactory uses it to
    find the layers.

    The function converts the levels as follows:

    * Geometry: ``level.layer.get_shapes(component)`` gives the polygons of each level. This includes
      derived layers (boolean operations) and sized layers. Sub-cells are flattened. Two levels on the same
      GDS layer keep their own geometry.
    * z position: ``zmin`` and ``thickness`` (micrometres) become ``z_base`` and ``thickness`` (metres)
      relative to the bottom face of ``simulation_volume``. The parts of a level outside of the volume are
      removed. A level that is fully outside of the volume is skipped.
    * Sidewall: gdsfactory measures ``sidewall_angle`` from the normal (0 is vertical). fdtdx measures it
      from the substrate (90 is vertical). Thus the fdtdx angle is ``90 - sidewall_angle``. The
      ``width_to_z`` values 0, 0.5 and 1 become the reference planes ``"bottom"``, ``"middle"`` and
      ``"top"``.
    * Overlaps: in gdsfactory, the level with the lower ``mesh_order`` wins. Thus the level with the lowest
      ``mesh_order`` gets the highest ``placement_order``. By default, the levels get the placement orders
      ``-N, ..., -1``. Thus your own static objects with the default placement order 0 are applied after
      the stack and win.
    * Levels without polygons in the component are skipped, for example metal levels that the component
      does not use. Thus ``materials`` must only contain the materials of the levels that are used.

    Background levels such as BOX, cladding or substrate are usually on a layer that components do not draw
    (for example ``WAFER`` in the generic PDK). Give their names in ``fill_levels`` to extrude them as full
    slabs. Levels with ``background=True`` are also filled. A fill level gets ``invert=True`` (see
    :attr:`~fdtdx.objects.static_material.gds_layer_stack.GDSLayerObject.invert`). Thus it covers the full
    simulation volume for every volume size, not only the bounding box of the component. The polygons on the ``background_exclude_layers`` of a level are
    removed from its fill.

    fdtdx does not support ``bias``, ``z_to_bias`` and ``width_to_z`` values other than 0, 0.5 and 1. A
    ``UserWarning`` shows for each level that uses them, and the level is extruded without them. Process
    variations and tolerances are ignored.

    Without sub-pixel smoothing, the faces of a layer snap to the z grid. For layers that are thinner than a
    few grid cells, set ``subpixel_smoothing=True``.

    Example::

        import gdsfactory as gf
        import fdtdx

        gf.gpdk.PDK.activate()
        component = gf.components.straight(length=10)
        layer_stack = gf.get_active_pdk().layer_stack.filtered(["box", "core", "clad"])
        materials = {
            "si": fdtdx.Material(permittivity=3.48**2),
            "sio2": fdtdx.Material(permittivity=1.44**2),
        }
        volume = fdtdx.SimulationVolume(partial_real_shape=(12e-6, 3e-6, 2e-6))
        objects, constraints = fdtdx.gds_layer_stack_from_gdsfactory(
            component,
            materials,
            volume,
            volume_bottom_z=-1e-6,  # the volume starts 1 um below z = 0 of the LayerStack
            layer_stack=layer_stack,
            fill_levels=["box", "clad"],
        )

    Args:
        component: A ``gdsfactory.Component``.
        materials: Materials dictionary. Its keys are the ``material`` names of the LayerStack levels.
        simulation_volume: Used for size and position constraints. Its ``partial_real_shape[2]`` must be set.
        volume_bottom_z: z coordinate in metres, in the frame of the LayerStack, of the bottom face of the
            simulation volume. Example: for ``volume_bottom_z=-1e-6``, a level with ``zmin=0`` starts 1 um
            above the bottom face of the volume.
        layer_stack: A ``gdsfactory.technology.LayerStack``. If None, the layer stack of the active PDK is
            used.
        gds_center: GDS coordinate (in metres) that maps to the x/y centre of the simulation volume. If None,
            the centre of the bounding box of the component is used.
        fill_levels: Names of the levels to extrude as full slabs across the simulation volume.
        name_prefix: Prefix for the object names. The object of a level gets the name
            ``name_prefix + level_name``. Use different prefixes to put two stacks into one simulation.
        placement_order_base: Placement order of the level with the lowest priority. The other levels get the
            next higher values. If None, the levels get the placement orders ``-N, ..., -1``.
        subpixel_smoothing: Enable sub-pixel smoothing for all levels.

    Returns:
        ``(objects, constraints)`` - same as :func:`gds_layer_stack`.

    Raises:
        ImportError: If gdsfactory is not installed.
        ValueError: If no layer stack is available, if ``simulation_volume.partial_real_shape[2]`` is not
            set, if ``fill_levels`` contains an unknown level name, if no level gives an object, or if
            ``materials`` does not contain a material that a level uses.
    """
    try:
        import gdsfactory as gf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "gdsfactory is required for gds_layer_stack_from_gdsfactory. Install it with: pip install gdsfactory"
        ) from exc

    if layer_stack is None:
        try:
            layer_stack = gf.get_active_pdk().layer_stack
        except ValueError as exc:
            raise ValueError(
                "No layer_stack is given and no gdsfactory PDK is active. Pass layer_stack or activate a PDK."
            ) from exc
        if layer_stack is None:
            raise ValueError("The active gdsfactory PDK has no layer stack. Pass layer_stack.")

    volume_height = simulation_volume.partial_real_shape[2]
    if volume_height is None:
        raise ValueError(
            "simulation_volume.partial_real_shape[2] must be set to place the LayerStack levels and to clip "
            "them to the simulation volume."
        )

    level_names = list(layer_stack.layers)
    unknown = [name for name in fill_levels if name not in layer_stack.layers]
    if unknown:
        raise ValueError(f"Unknown fill_levels {unknown}. Valid level names: {level_names}.")

    specs, dropped = _layer_stack_to_specs(
        layer_stack,
        volume_bottom_z=volume_bottom_z,
        volume_height=volume_height,
        name_prefix=name_prefix,
        placement_order_base=placement_order_base,
        subpixel_smoothing=subpixel_smoothing,
    )

    if gds_center is None:
        center = component.dbbox().center()
        gds_center = (center.x * _UM, center.y * _UM)

    # Write the geometry of each level to its own GDS layer in a new component, which is then exported.
    extrusion = gf.Component(kcl=component.kcl)
    try:
        kept: list[GDSLayerSpec] = []
        for spec in specs:
            level_name = level_names[spec.gds_layer]
            level = layer_stack.layers[level_name]
            if level_name in fill_levels or getattr(level, "background", False):
                # An inverted layer fills the full object. Its polygons are only the excluded areas.
                spec = dataclasses.replace(spec, invert=True)
                region = gf.kdb.Region()
                for excluded_layer in getattr(level, "background_exclude_layers", ()):
                    region |= gf.technology.LogicalLayer(layer=excluded_layer).get_shapes(component)
            else:
                region = level.layer.get_shapes(component)
                if region.is_empty():
                    dropped[level_name] = "no polygons in the component"
                    continue
            extrusion.shapes(extrusion.kcl.layout.layer(spec.gds_layer, 0)).insert(region)
            kept.append(spec)

        for level_name, reason in dropped.items():
            logger.info(f"LayerStack level '{level_name}' is skipped: {reason}.")
        if not kept:
            reasons = "; ".join(f"'{name}': {reason}" for name, reason in dropped.items())
            raise ValueError(f"No LayerStack level gives a simulation object. Reasons: {reasons}.")

        missing: dict[str, list[str]] = {}
        for spec in kept:
            if spec.material_name not in materials:
                missing.setdefault(spec.material_name, []).append(level_names[spec.gds_layer])
        if missing:
            details = "; ".join(f"'{material}' (levels {levels})" for material, levels in missing.items())
            raise ValueError(
                f"The materials dict has no entry for these LayerStack materials: {details}. "
                f"Available keys: {sorted(materials)}."
            )

        return gds_layer_stack_from_component(
            component=extrusion,
            layers=kept,
            materials=materials,
            simulation_volume=simulation_volume,
            gds_center=gds_center,
            cell_name=extrusion.name,
        )
    finally:
        extrusion.delete()
