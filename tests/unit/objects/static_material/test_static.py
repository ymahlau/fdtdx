"""Unit tests for objects/static_material/static.py.

Tests UniformMaterialObject, StaticMultiMaterialObject (via concrete subclass),
and SimulationVolume.
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.colors import XKCD_LIGHT_GREY
from fdtdx.core.jax.pytrees import autoinit
from fdtdx.materials import Material
from fdtdx.objects.static_material.static import (
    SimulationVolume,
    StaticMultiMaterialObject,
    UniformMaterialObject,
)

# ---------------------------------------------------------------------------
# UniformMaterialObject
# ---------------------------------------------------------------------------


class TestUniformMaterialObject:
    def test_construction_with_material(self):
        mat = Material(permittivity=2.25)
        obj = UniformMaterialObject(material=mat)
        assert obj.material is mat

    def test_default_color_is_light_grey(self):
        mat = Material(permittivity=1.0)
        obj = UniformMaterialObject(material=mat)
        assert obj.color == XKCD_LIGHT_GREY

    def test_color_overridable(self):
        from fdtdx.colors import Color

        mat = Material(permittivity=1.0)
        custom_color = Color(r=0.5, g=0.5, b=0.5)
        obj = UniformMaterialObject(material=mat, color=custom_color)
        assert obj.color == custom_color

    def test_is_simulation_object(self):
        from fdtdx.objects.object import SimulationObject

        mat = Material(permittivity=1.0)
        obj = UniformMaterialObject(material=mat)
        assert isinstance(obj, SimulationObject)

    def test_placement_order_default(self):
        mat = Material(permittivity=1.0)
        obj = UniformMaterialObject(material=mat)
        # OrderableObject default is 0
        assert obj.placement_order == 0


# ---------------------------------------------------------------------------
# StaticMultiMaterialObject – via minimal concrete subclass
# ---------------------------------------------------------------------------


@autoinit
class _MinimalMultiMaterial(StaticMultiMaterialObject):
    """Minimal concrete StaticMultiMaterialObject for testing the base class."""

    def get_voxel_mask_for_shape(self) -> jax.Array:
        return jnp.ones(self.grid_shape, dtype=jnp.bool_)

    def get_material_mapping(self) -> jax.Array:
        return jnp.zeros(self.grid_shape, dtype=jnp.int32)


class TestStaticMultiMaterialObject:
    @pytest.fixture
    def config(self):
        from fdtdx.config import SimulationConfig

        return SimulationConfig(
            time=100e-15,
            resolution=50e-9,
            backend="cpu",
            dtype=jnp.float32,
            gradient_config=None,
        )

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(0)

    def test_construction_with_materials(self):
        mats = {"air": Material(permittivity=1.0), "si": Material(permittivity=12.25)}
        obj = _MinimalMultiMaterial(materials=mats)
        assert obj.materials == mats

    def test_default_color_is_light_grey(self):
        mats = {"air": Material(permittivity=1.0)}
        obj = _MinimalMultiMaterial(materials=mats)
        assert obj.color == XKCD_LIGHT_GREY

    def test_concrete_get_voxel_mask(self, config, key):
        mats = {"air": Material(permittivity=1.0), "si": Material(permittivity=12.25)}
        obj = _MinimalMultiMaterial(materials=mats)
        placed = obj.place_on_grid(((0, 4), (0, 5), (0, 6)), config, key)
        mask = placed.get_voxel_mask_for_shape()
        assert mask.shape == (4, 5, 6)
        assert bool(jnp.all(mask))

    def test_concrete_get_material_mapping(self, config, key):
        mats = {"air": Material(permittivity=1.0), "si": Material(permittivity=12.25)}
        obj = _MinimalMultiMaterial(materials=mats)
        placed = obj.place_on_grid(((0, 4), (0, 5), (0, 6)), config, key)
        mapping = placed.get_material_mapping()
        assert mapping.shape == (4, 5, 6)
        assert bool(jnp.all(mapping == 0))

    def test_abstract_methods_required(self):
        """StaticMultiMaterialObject cannot be instantiated without implementations."""

        @autoinit
        class _IncompleteObject(StaticMultiMaterialObject):
            pass

        # _IncompleteObject has unimplemented abstract methods; instantiation
        # should raise TypeError
        with pytest.raises(TypeError):
            _IncompleteObject(materials={"air": Material(permittivity=1.0)})


# ---------------------------------------------------------------------------
# SimulationVolume
# ---------------------------------------------------------------------------


class TestSimulationVolume:
    def test_default_placement_order(self):
        vol = SimulationVolume()
        assert vol.placement_order == -1000

    def test_default_material_is_vacuum(self):
        vol = SimulationVolume()
        mat = vol.material
        # vacuum: permittivity = (1,0,0,0,1,0,0,0,1)
        assert mat.permittivity[0] == pytest.approx(1.0)
        assert mat.permittivity[4] == pytest.approx(1.0)
        assert mat.permittivity[8] == pytest.approx(1.0)

    def test_default_permeability_is_vacuum(self):
        vol = SimulationVolume()
        mat = vol.material
        assert mat.permeability[0] == pytest.approx(1.0)

    def test_custom_material_overrides_default(self):
        mat = Material(permittivity=2.25)
        vol = SimulationVolume(material=mat)
        assert vol.material.permittivity[0] == pytest.approx(2.25)

    def test_is_uniform_material_object(self):
        vol = SimulationVolume()
        assert isinstance(vol, UniformMaterialObject)

    def test_name_auto_assigned(self):
        vol = SimulationVolume()
        assert vol.name is not None
        assert isinstance(vol.name, str)
