"""Unit tests for objects/static_material/gds_gdsfactory.py.

The tests of ``_layer_stack_to_specs`` use fake levels and run without gdsfactory. The tests in
``TestWithGdsfactory`` are skipped when gdsfactory is not installed. Run them locally with:

    uv run --with gdsfactory python -m pytest tests/unit/objects/static_material/test_gds_gdsfactory.py
"""

import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from fdtdx.core.grid import polygon_to_mask_at_points
from fdtdx.materials import Material
from fdtdx.objects.static_material.gds_gdsfactory import _layer_stack_to_specs, gds_layer_stack_from_gdsfactory
from fdtdx.objects.static_material.gds_layer_stack import GDSLayerObject
from fdtdx.objects.static_material.static import SimulationVolume

# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


def _level(**kwargs):
    """Fake gdsfactory LayerLevel with the gdsfactory defaults."""
    attrs = dict(
        layer=None,
        thickness=0.22,
        zmin=0.0,
        sidewall_angle=0.0,
        width_to_z=0.0,
        z_to_bias=None,
        bias=None,
        mesh_order=3,
        material="si",
        background=False,
        background_exclude_layers=(),
        info={},
    )
    attrs.update(kwargs)
    return SimpleNamespace(**attrs)


def _stack(**levels):
    """Fake gdsfactory LayerStack."""
    return SimpleNamespace(layers=dict(levels))


def _convert(stack, volume_bottom_z=0.0, volume_height=2e-6, **kwargs):
    """Run the conversion and return ``({name: spec}, specs, dropped)``."""
    specs, dropped = _layer_stack_to_specs(stack, volume_bottom_z, volume_height, **kwargs)
    return {spec.name: spec for spec in specs}, specs, dropped


def _no_warnings(stack, **kwargs):
    """Run the conversion and fail if it shows a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return _convert(stack, **kwargs)


def _mask_at(polygons, x, y):
    """True if the point (x, y) in metres is inside one of the polygons."""
    return any(bool(polygon_to_mask_at_points(np.array([x]), np.array([y]), polygon)[0, 0]) for polygon in polygons)


def _area(polygons):
    """Total area of the polygons (shoelace formula)."""
    total = 0.0
    for p in polygons:
        x, y = p[:, 0], p[:, 1]
        total += 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return total


# ---------------------------------------------------------------------------
# z placement and units
# ---------------------------------------------------------------------------


class TestPlacement:
    def test_unit_conversion(self):
        """Micrometres become metres, and zmin is measured from the bottom face of the volume."""
        by_name, _, _ = _no_warnings(_stack(core=_level(thickness=0.22, zmin=0.0)), volume_bottom_z=-1e-6)
        assert by_name["core"].thickness == pytest.approx(2.2e-7)
        assert by_name["core"].z_base == pytest.approx(1e-6)

    def test_material_and_gds_layer(self):
        """The material name is copied and level number i gets the GDS layer (i, 0)."""
        stack = _stack(box=_level(material="sio2", zmin=-0.5, thickness=0.5), core=_level())
        by_name, _, _ = _no_warnings(stack, volume_bottom_z=-1e-6)
        assert by_name["box"].material_name == "sio2"
        assert (by_name["box"].gds_layer, by_name["box"].gds_datatype) == (0, 0)
        assert (by_name["core"].gds_layer, by_name["core"].gds_datatype) == (1, 0)

    def test_name_prefix(self):
        by_name, _, _ = _no_warnings(_stack(core=_level()), name_prefix="wg1_")
        assert list(by_name) == ["wg1_core"]

    def test_negative_thickness_extrudes_downward(self):
        """A negative thickness gives a level from zmin + thickness to zmin."""
        by_name, _, _ = _no_warnings(_stack(undercut=_level(zmin=0.0, thickness=-0.5)), volume_bottom_z=-1e-6)
        assert by_name["undercut"].z_base == pytest.approx(0.5e-6)
        assert by_name["undercut"].thickness == pytest.approx(0.5e-6)

    def test_subpixel_smoothing_forwarded(self):
        by_name, _, _ = _no_warnings(_stack(core=_level()), subpixel_smoothing=True)
        assert by_name["core"].subpixel_smoothing is True


# ---------------------------------------------------------------------------
# Clipping to the simulation volume
# ---------------------------------------------------------------------------


class TestClipping:
    @pytest.mark.parametrize(
        "zmin, thickness",
        [
            (-3.0, 1.0),  # fully below the volume
            (2.5, 1.0),  # fully above the volume
            (-1.0, 1.0),  # top face touches the bottom face of the volume
            (2.0, 1.0),  # bottom face touches the top face of the volume
        ],
    )
    def test_level_outside_is_dropped(self, zmin, thickness):
        by_name, _, dropped = _no_warnings(_stack(sub=_level(zmin=zmin, thickness=thickness)))
        assert by_name == {}
        assert "outside" in dropped["sub"]

    def test_straddles_bottom(self):
        by_name, _, _ = _no_warnings(_stack(box=_level(zmin=-3.0, thickness=3.5)))
        assert by_name["box"].z_base == 0.0
        assert by_name["box"].thickness == pytest.approx(0.5e-6)

    def test_straddles_top(self):
        by_name, _, _ = _no_warnings(_stack(clad=_level(zmin=1.5, thickness=3.0)))
        assert by_name["clad"].z_base == pytest.approx(1.5e-6)
        assert by_name["clad"].thickness == pytest.approx(0.5e-6)

    def test_exactly_fills_volume(self):
        by_name, _, _ = _no_warnings(_stack(clad=_level(zmin=0.0, thickness=2.0)))
        assert by_name["clad"].z_base == 0.0
        assert by_name["clad"].thickness == pytest.approx(2e-6)

    def test_angled_wall_clipped_at_reference_face_warns(self):
        """Clipping the reference face of an angled wall changes the footprint, which is reported."""
        stack = _stack(core=_level(zmin=-0.5, thickness=1.0, sidewall_angle=10.0, width_to_z=0.0))
        with pytest.warns(UserWarning, match="reference face"):
            by_name, _, _ = _convert(stack)
        assert by_name["core"].thickness == pytest.approx(0.5e-6)

    def test_angled_wall_clipped_at_other_face_is_silent(self):
        """Clipping the face opposite to the reference face keeps the footprint exact."""
        stack = _stack(core=_level(zmin=1.5, thickness=1.0, sidewall_angle=10.0, width_to_z=0.0))
        by_name, _, _ = _no_warnings(stack)
        assert by_name["core"].reference_plane == "bottom"


# ---------------------------------------------------------------------------
# Sidewall angle and reference plane
# ---------------------------------------------------------------------------


class TestSidewall:
    @pytest.mark.parametrize("gf_angle, fdtdx_angle", [(0.0, 90.0), (10.0, 80.0), (-5.0, 95.0)])
    def test_angle_conversion(self, gf_angle, fdtdx_angle):
        by_name, _, _ = _no_warnings(_stack(core=_level(sidewall_angle=gf_angle)))
        assert by_name["core"].sidewall_angle == pytest.approx(fdtdx_angle)

    def test_angle_out_of_range_raises(self):
        with pytest.raises(ValueError, match="core"):
            _convert(_stack(core=_level(sidewall_angle=90.0)))

    @pytest.mark.parametrize("width_to_z, plane", [(0.0, "bottom"), (0.5, "middle"), (1.0, "top")])
    def test_reference_plane(self, width_to_z, plane):
        by_name, _, _ = _no_warnings(_stack(core=_level(sidewall_angle=10.0, width_to_z=width_to_z)))
        assert by_name["core"].reference_plane == plane

    def test_negative_thickness_mirrors_angle_and_reference(self):
        """A downward extrusion that shrinks toward its end grows toward the top in fdtdx."""
        stack = _stack(via=_level(zmin=1.0, thickness=-0.5, sidewall_angle=10.0, width_to_z=0.0))
        by_name, _, _ = _no_warnings(stack)
        assert by_name["via"].sidewall_angle == pytest.approx(100.0)
        assert by_name["via"].reference_plane == "top"

    def test_unsupported_width_to_z_warns_and_snaps(self):
        with pytest.warns(UserWarning, match="width_to_z"):
            by_name, _, _ = _convert(_stack(core=_level(sidewall_angle=10.0, width_to_z=0.3)))
        assert by_name["core"].reference_plane == "middle"

    def test_unsupported_width_to_z_with_vertical_wall_is_silent(self):
        _no_warnings(_stack(core=_level(sidewall_angle=0.0, width_to_z=0.3)))


# ---------------------------------------------------------------------------
# mesh_order -> placement_order
# ---------------------------------------------------------------------------


class TestPlacementOrder:
    def test_lower_mesh_order_wins(self):
        """The level with the lowest mesh order is last and has the highest placement order."""
        stack = _stack(core=_level(mesh_order=1), clad=_level(material="sio2", mesh_order=2))
        _, specs, _ = _no_warnings(stack)
        assert [s.name for s in specs] == ["clad", "core"]
        assert [s.placement_order for s in specs] == [-2, -1]

    def test_ties_use_stack_order(self):
        """For equal mesh orders, the later level in the stack wins."""
        stack = _stack(a=_level(), b=_level(), c=_level())
        _, specs, _ = _no_warnings(stack)
        assert [s.name for s in specs] == ["a", "b", "c"]
        assert [s.placement_order for s in specs] == [-3, -2, -1]

    def test_info_mesh_order_overrides_field(self):
        stack = _stack(core=_level(mesh_order=1), clad=_level(mesh_order=2, info={"mesh_order": 0}))
        _, specs, _ = _no_warnings(stack)
        assert [s.name for s in specs] == ["core", "clad"]

    def test_explicit_base(self):
        stack = _stack(core=_level(mesh_order=1), clad=_level(mesh_order=2))
        _, specs, _ = _no_warnings(stack, placement_order_base=10)
        assert [s.placement_order for s in specs] == [10, 11]


# ---------------------------------------------------------------------------
# Skipped levels and unsupported features
# ---------------------------------------------------------------------------


class TestSkippedAndUnsupported:
    def test_zero_thickness_is_skipped_silently(self):
        by_name, _, dropped = _no_warnings(_stack(marker=_level(thickness=0.0)))
        assert by_name == {}
        assert dropped["marker"] == "zero thickness"

    def test_no_material_is_skipped_with_warning(self):
        with pytest.warns(UserWarning, match="no material"):
            by_name, _, dropped = _convert(_stack(core=_level(material=None)))
        assert by_name == {}
        assert dropped["core"] == "no material"

    @pytest.mark.parametrize("bias", [None, 0, 0.0, (0.0, 0.0)])
    def test_zero_bias_is_silent(self, bias):
        _no_warnings(_stack(core=_level(bias=bias)))

    @pytest.mark.parametrize("z_to_bias", [None, ([0.0, 1.0], [0.0, 0.0])])
    def test_zero_z_to_bias_is_silent(self, z_to_bias):
        _no_warnings(_stack(core=_level(z_to_bias=z_to_bias)))

    def test_bias_warns(self):
        with pytest.warns(UserWarning, match="bias is not supported"):
            by_name, _, _ = _convert(_stack(core=_level(bias=0.01)))
        assert "core" in by_name

    def test_z_to_bias_warns(self):
        with pytest.warns(UserWarning, match="z_to_bias is not supported"):
            by_name, _, _ = _convert(_stack(core=_level(z_to_bias=([0.0, 1.0], [0.0, -0.1]))))
        assert "core" in by_name


# ---------------------------------------------------------------------------
# Public function without gdsfactory
# ---------------------------------------------------------------------------


class TestWithoutGdsfactory:
    def test_requires_gdsfactory(self):
        """Without gdsfactory installed, ImportError is raised with a helpful message."""
        volume = SimulationVolume(partial_real_shape=(1e-6, 1e-6, 1e-6))
        with patch.dict("sys.modules", {"gdsfactory": None}):
            with pytest.raises(ImportError, match="gdsfactory"):
                gds_layer_stack_from_gdsfactory(
                    component=object(),
                    materials={},
                    simulation_volume=volume,
                    volume_bottom_z=0.0,
                    layer_stack=_stack(core=_level()),
                )


# ---------------------------------------------------------------------------
# Public function with gdsfactory (skipped when gdsfactory is not installed)
# ---------------------------------------------------------------------------


_MATERIALS = {"si": Material(permittivity=3.48**2), "sio2": Material(permittivity=1.44**2)}


@pytest.fixture
def gf():
    gf = pytest.importorskip("gdsfactory")
    gf.gpdk.PDK.activate()
    return gf


@pytest.fixture
def volume():
    return SimulationVolume(partial_real_shape=(12e-6, 2e-6, 2e-6))


class TestWithGdsfactory:
    def test_straight_with_generic_stack(self, gf, volume):
        """The generic PDK stack gives box, core and clad objects with the expected z extents."""
        stack = gf.get_active_pdk().layer_stack.filtered(["box", "core", "clad"])
        objects, constraints = gds_layer_stack_from_gdsfactory(
            gf.components.straight(length=10),
            _MATERIALS,
            volume,
            volume_bottom_z=-1e-6,
            layer_stack=stack,
            fill_levels=["box", "clad"],
        )
        by_name = {o.name: o for o in objects}
        assert set(by_name) == {"box", "core", "clad"}
        assert len(constraints) == 6
        assert all(isinstance(o, GDSLayerObject) for o in objects)
        # mesh orders in the generic PDK: clad 10, box 9, core 2 -> core wins
        assert [o.name for o in sorted(objects, key=lambda o: o.placement_order)] == ["clad", "box", "core"]
        assert by_name["core"].thickness == pytest.approx(0.22e-6)
        assert by_name["core"].material_name == "si"
        assert by_name["core"].sidewall_angle == pytest.approx(80.0)
        assert by_name["core"].reference_plane == "middle"
        assert by_name["box"].thickness == pytest.approx(1e-6)  # clipped from 3 um
        assert by_name["clad"].thickness == pytest.approx(1e-6)  # clipped from 3 um
        # the waveguide is 0.5 um wide
        assert _mask_at(by_name["core"].polygons, 5e-6, 0.0)
        assert not _mask_at(by_name["core"].polygons, 5e-6, 0.5e-6)
        # fill levels are inverted and have no excluded areas, thus they fill the full volume
        assert not by_name["core"].invert
        for name in ("box", "clad"):
            assert by_name[name].invert
            assert len(by_name[name].polygons) == 0

    def test_default_layer_stack_and_gds_center(self, gf, volume):
        """Without layer_stack, the PDK stack is used. Levels without polygons need no material."""
        objects, _ = gds_layer_stack_from_gdsfactory(
            gf.components.straight(length=10),
            _MATERIALS,
            volume,
            volume_bottom_z=-1e-6,
            fill_levels=["box", "clad"],
        )
        assert {o.name for o in objects} == {"box", "core", "clad"}
        assert objects[0].gds_center == pytest.approx((5e-6, 0.0))

    def test_derived_layer(self, gf, volume):
        """The generic core level is WG - DEEP_ETCH - SHALLOW_ETCH; the etched area is removed."""
        c = gf.Component()
        c.add_polygon([(-2, -1), (2, -1), (2, 1), (-2, 1)], layer="WG")
        c.add_polygon([(-0.5, -1), (0.5, -1), (0.5, 1), (-0.5, 1)], layer="DEEP_ETCH")
        stack = gf.get_active_pdk().layer_stack.filtered(["core"])
        objects, _ = gds_layer_stack_from_gdsfactory(c, _MATERIALS, volume, volume_bottom_z=-1e-6, layer_stack=stack)
        assert _area(objects[0].polygons) == pytest.approx(6e-12)
        assert not _mask_at(objects[0].polygons, 0.0, 0.0)
        assert _mask_at(objects[0].polygons, 1.5e-6, 0.0)

    def test_hierarchical_component_is_flattened(self, gf, volume):
        c = gf.components.mzi()
        stack = gf.get_active_pdk().layer_stack.filtered(["core"])
        objects, _ = gds_layer_stack_from_gdsfactory(c, _MATERIALS, volume, volume_bottom_z=-1e-6, layer_stack=stack)
        expected = stack.layers["core"].layer.get_shapes(c).area() * c.kcl.dbu**2 * 1e-12
        assert _area(objects[0].polygons) == pytest.approx(expected, rel=1e-6)

    def test_polygon_with_hole(self, gf, volume):
        c = gf.components.ring(radius=5, width=0.5)
        stack = gf.get_active_pdk().layer_stack.filtered(["core"])
        objects, _ = gds_layer_stack_from_gdsfactory(c, _MATERIALS, volume, volume_bottom_z=-1e-6, layer_stack=stack)
        assert _mask_at(objects[0].polygons, 5e-6, 0.0)
        assert not _mask_at(objects[0].polygons, 0.0, 0.0)

    def test_levels_on_the_same_layer_keep_their_geometry(self, gf, volume):
        """A rib waveguide: a full-height core and a thin slab, both on the WG layer."""
        from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer

        stack = LayerStack(
            layers={
                "core": LayerLevel(layer=LogicalLayer(layer=(1, 0)), thickness=0.22, zmin=0.0, material="si"),
                "slab": LayerLevel(layer=LogicalLayer(layer=(1, 0)), thickness=0.09, zmin=0.0, material="si"),
            }
        )
        c = gf.components.straight(length=10)
        objects, _ = gds_layer_stack_from_gdsfactory(c, _MATERIALS, volume, volume_bottom_z=-1e-6, layer_stack=stack)
        by_name = {o.name: o for o in objects}
        assert by_name["core"].thickness == pytest.approx(0.22e-6)
        assert by_name["slab"].thickness == pytest.approx(0.09e-6)
        assert _area(by_name["core"].polygons) == pytest.approx(_area(by_name["slab"].polygons))

    def test_background_level_with_exclusion(self, gf, volume):
        from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer

        if "background" not in LayerLevel.model_fields:
            pytest.skip("this gdsfactory version has no LayerLevel.background")
        stack = LayerStack(
            layers={
                "clad": LayerLevel(
                    layer=LogicalLayer(layer=(999, 0)),
                    thickness=1.0,
                    zmin=0.0,
                    material="sio2",
                    background=True,
                    background_exclude_layers=((2, 0),),
                ),
            }
        )
        c = gf.Component()
        c.add_polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)], layer=(2, 0))
        objects, _ = gds_layer_stack_from_gdsfactory(
            c, _MATERIALS, volume, volume_bottom_z=0.0, layer_stack=stack, gds_center=(0.0, 0.0)
        )
        # the level is inverted: its polygons are the excluded areas
        assert objects[0].invert
        assert _mask_at(objects[0].polygons, 0.0, 0.0)
        assert not _mask_at(objects[0].polygons, 3e-6, 0.0)

    def test_name_prefix(self, gf, volume):
        stack = gf.get_active_pdk().layer_stack.filtered(["core"])
        objects, _ = gds_layer_stack_from_gdsfactory(
            gf.components.straight(), _MATERIALS, volume, volume_bottom_z=-1e-6, layer_stack=stack, name_prefix="a_"
        )
        assert objects[0].name == "a_core"

    def test_missing_materials_listed_in_one_error(self, gf, volume):
        from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer

        stack = LayerStack(
            layers={
                "core": LayerLevel(layer=LogicalLayer(layer=(1, 0)), thickness=0.2, zmin=0.0, material="gaas"),
                "clad": LayerLevel(layer=LogicalLayer(layer=(1, 0)), thickness=0.5, zmin=0.0, material="sin"),
            }
        )
        with pytest.raises(ValueError, match="gaas") as info:
            gds_layer_stack_from_gdsfactory(
                gf.components.straight(), _MATERIALS, volume, volume_bottom_z=0.0, layer_stack=stack
            )
        assert "sin" in str(info.value)
        assert "Available keys" in str(info.value)

    def test_unknown_fill_level_raises(self, gf, volume):
        with pytest.raises(ValueError, match="Unknown fill_levels"):
            gds_layer_stack_from_gdsfactory(
                gf.components.straight(), _MATERIALS, volume, volume_bottom_z=0.0, fill_levels=["nope"]
            )

    def test_all_levels_dropped_raises(self, gf, volume):
        """box and clad are on WAFER, which the component does not draw. Without a fill, nothing remains."""
        stack = gf.get_active_pdk().layer_stack.filtered(["box", "clad"])
        with pytest.raises(ValueError, match="no polygons in the component"):
            gds_layer_stack_from_gdsfactory(
                gf.components.straight(), _MATERIALS, volume, volume_bottom_z=-1e-6, layer_stack=stack
            )

    def test_volume_height_required(self, gf):
        with pytest.raises(ValueError, match=r"partial_real_shape\[2\]"):
            gds_layer_stack_from_gdsfactory(
                gf.components.straight(), _MATERIALS, SimulationVolume(), volume_bottom_z=0.0
            )
