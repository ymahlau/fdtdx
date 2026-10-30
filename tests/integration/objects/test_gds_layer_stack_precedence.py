"""Overlap precedence of gdsfactory LayerStack levels after the conversion to GDS layer specs.

The tests do not need gdsfactory. They convert fake LayerStack levels with ``_layer_stack_to_specs``, place
the objects with ``place_objects`` and read the inverse permittivity where the core and the cladding overlap.

Volume: 1 um x 1 um x 1 um at 100 nm resolution -> 10 x 10 x 10 grid cells.
"""

import dataclasses
from types import SimpleNamespace

import gdstk
import jax
import pytest

import fdtdx
from fdtdx.objects.static_material.gds_gdsfactory import _layer_stack_to_specs

SPACING = 100e-9
SIDE = 1e-6
EPS_SI = 12.25
EPS_SIO2 = 2.25
MATERIALS = {"si": fdtdx.Material(permittivity=EPS_SI), "sio2": fdtdx.Material(permittivity=EPS_SIO2)}
OVERLAP_VOXEL = (0, 5, 5, 4)  # inside the core (z cells 4..5) and inside the cladding (z cells 1..6)
CLAD_ONLY_VOXEL = (0, 5, 5, 2)  # inside the cladding, below the core


def _level(material, zmin, thickness, mesh_order):
    """Fake gdsfactory LayerLevel with a vertical sidewall."""
    return SimpleNamespace(
        layer=None,
        thickness=thickness,
        zmin=zmin,
        sidewall_angle=0.0,
        width_to_z=0.0,
        z_to_bias=None,
        bias=None,
        mesh_order=mesh_order,
        material=material,
        info={},
    )


def _inv_permittivities(core_mesh_order, clad_mesh_order, with_user_object=False, inverted_clad=False):
    stack = SimpleNamespace(
        layers={
            "core": _level("si", zmin=0.0, thickness=0.2, mesh_order=core_mesh_order),
            "clad": _level("sio2", zmin=-0.3, thickness=0.6, mesh_order=clad_mesh_order),
        }
    )
    specs, _ = _layer_stack_to_specs(stack, volume_bottom_z=-0.4e-6, volume_height=SIDE)
    if inverted_clad:
        specs = [dataclasses.replace(s, invert=True) if s.name == "clad" else s for s in specs]

    # Level number i is on GDS layer (i, 0): a small core square and a cladding square that fills the volume.
    # An inverted cladding has no polygons and fills the volume without them.
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("TOP")
    cell.add(gdstk.rectangle((-0.2, -0.2), (0.2, 0.2), layer=0))
    if not inverted_clad:
        cell.add(gdstk.rectangle((-0.5, -0.5), (0.5, 0.5), layer=1))

    volume = fdtdx.SimulationVolume(partial_real_shape=(SIDE, SIDE, SIDE))
    objects, constraints = fdtdx.gds_layer_stack(lib, "TOP", specs, MATERIALS, volume, gds_center=(0.0, 0.0))
    if with_user_object:
        # An air block with the default placement order 0 that covers the full volume.
        air = fdtdx.UniformMaterialObject(
            name="user_air",
            partial_real_shape=(SIDE, SIDE, SIDE),
            material=fdtdx.Material(permittivity=1.0),
        )
        objects = [*objects, air]
        constraints = [*constraints, air.place_at_center(volume)]

    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=SPACING), time=10e-15, backend="cpu")
    _, arrays, _, _, _ = fdtdx.place_objects([volume, *objects], config, constraints, jax.random.PRNGKey(0))
    return arrays.inv_permittivities


def test_lower_mesh_order_core_wins():
    inv_eps = _inv_permittivities(core_mesh_order=1, clad_mesh_order=2)
    assert float(inv_eps[OVERLAP_VOXEL]) == pytest.approx(1 / EPS_SI, rel=1e-6)
    assert float(inv_eps[CLAD_ONLY_VOXEL]) == pytest.approx(1 / EPS_SIO2, rel=1e-6)


def test_lower_mesh_order_clad_wins():
    inv_eps = _inv_permittivities(core_mesh_order=2, clad_mesh_order=1)
    assert float(inv_eps[OVERLAP_VOXEL]) == pytest.approx(1 / EPS_SIO2, rel=1e-6)


def test_user_object_with_default_order_wins_over_stack():
    """The stack gets negative placement orders, so a user object with the default order 0 is applied last."""
    inv_eps = _inv_permittivities(core_mesh_order=1, clad_mesh_order=2, with_user_object=True)
    assert float(inv_eps[OVERLAP_VOXEL]) == pytest.approx(1.0, rel=1e-6)
    assert float(inv_eps[CLAD_ONLY_VOXEL]) == pytest.approx(1.0, rel=1e-6)


def test_inverted_cladding_fills_volume():
    """An inverted cladding without polygons fills the full volume, and the core still wins the overlap."""
    inv_eps = _inv_permittivities(core_mesh_order=1, clad_mesh_order=2, inverted_clad=True)
    assert float(inv_eps[OVERLAP_VOXEL]) == pytest.approx(1 / EPS_SI, rel=1e-6)
    for voxel in (CLAD_ONLY_VOXEL, (0, 0, 0, 3), (0, 9, 9, 3), (0, 0, 9, 6)):
        assert float(inv_eps[voxel]) == pytest.approx(1 / EPS_SIO2, rel=1e-6)
    assert float(inv_eps[0, 5, 5, 8]) == pytest.approx(1.0, rel=1e-6)  # above the cladding
