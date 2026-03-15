"""Unit tests for objects/device/device.py.

Tests Device abstract base class properties and methods via a minimal
concrete subclass. Complex transform chains are tested through the simple
ClosestIndex transform.
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit
from fdtdx.materials import Material
from fdtdx.objects.device.device import Device
from fdtdx.objects.device.parameters.discretization import ClosestIndex
from fdtdx.typing import ParameterType

# ---------------------------------------------------------------------------
# Minimal concrete Device for testing
# ---------------------------------------------------------------------------


@autoinit
class _ConcreteDevice(Device):
    """Minimal concrete Device subclass for unit testing."""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return SimulationConfig(
        time=100e-15,
        resolution=50e-9,
        backend="cpu",
        dtype=jnp.float32,
        gradient_config=None,
    )


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def two_materials():
    return {
        "air": Material(permittivity=1.0),
        "si": Material(permittivity=12.25),
    }


@pytest.fixture
def three_materials():
    return {
        "air": Material(permittivity=1.0),
        "sio2": Material(permittivity=2.25),
        "si": Material(permittivity=12.25),
    }


def _make_device(materials, param_transforms=None, voxel_grid=(1, 1, 1)):
    if param_transforms is None:
        param_transforms = []
    return _ConcreteDevice(
        materials=materials,
        param_transforms=param_transforms,
        partial_voxel_grid_shape=voxel_grid,
    )


def _place_device(device, config, key, grid_slices=((0, 10), (0, 10), (0, 10))):
    return device.place_on_grid(grid_slice_tuple=grid_slices, config=config, key=key)


# ---------------------------------------------------------------------------
# output_type (does not require placement)
# ---------------------------------------------------------------------------


class TestOutputType:
    def test_no_transforms_returns_continuous(self, two_materials):
        device = _make_device(two_materials, param_transforms=[])
        assert device.output_type == ParameterType.CONTINUOUS

    def test_with_closest_index_returns_binary_for_two_materials(self, config, key, two_materials):
        transform = ClosestIndex()
        device = _make_device(two_materials, param_transforms=[transform])
        placed = _place_device(device, config, key)
        assert placed.output_type == ParameterType.BINARY

    def test_with_closest_index_returns_discrete_for_three_materials(self, config, key, three_materials):
        transform = ClosestIndex()
        device = _make_device(three_materials, param_transforms=[transform], voxel_grid=(1, 1, 1))
        placed = _place_device(device, config, key)
        assert placed.output_type == ParameterType.DISCRETE


# ---------------------------------------------------------------------------
# single_voxel_grid_shape – raises before placement
# ---------------------------------------------------------------------------


class TestSingleVoxelGridShapeNotInitialized:
    def test_raises_before_placement(self, two_materials):
        device = _make_device(two_materials)
        with pytest.raises(Exception, match="not initialized"):
            _ = device.single_voxel_grid_shape


# ---------------------------------------------------------------------------
# place_on_grid
# ---------------------------------------------------------------------------


class TestPlaceOnGrid:
    def test_single_voxel_grid_shape_after_placement(self, config, key, two_materials):
        device = _make_device(two_materials, voxel_grid=(2, 2, 2))
        placed = _place_device(device, config, key, ((0, 10), (0, 10), (0, 10)))
        assert placed.single_voxel_grid_shape == (2, 2, 2)

    def test_matrix_voxel_grid_shape_after_placement(self, config, key, two_materials):
        device = _make_device(two_materials, voxel_grid=(2, 2, 2))
        placed = _place_device(device, config, key, ((0, 10), (0, 10), (0, 10)))
        assert placed.matrix_voxel_grid_shape == (5, 5, 5)

    def test_single_voxel_real_shape(self, config, key, two_materials):
        device = _make_device(two_materials, voxel_grid=(2, 2, 2))
        placed = _place_device(device, config, key)
        res = config.resolution
        assert placed.single_voxel_real_shape == pytest.approx((2 * res, 2 * res, 2 * res))

    def test_voxel_size_from_real_shape(self, config, key, two_materials):
        """partial_voxel_real_shape should convert to grid units."""
        device = _ConcreteDevice(
            materials=two_materials,
            param_transforms=[],
            partial_voxel_real_shape=(100e-9, 100e-9, 100e-9),  # 2 voxels at 50nm
        )
        placed = _place_device(device, config, key, ((0, 10), (0, 10), (0, 10)))
        assert placed.single_voxel_grid_shape == (2, 2, 2)

    def test_overspecified_voxel_raises(self, config, key, two_materials):
        """Providing both grid and real shape for the same axis should raise."""
        device = _ConcreteDevice(
            materials=two_materials,
            param_transforms=[],
            partial_voxel_grid_shape=(2, None, None),
            partial_voxel_real_shape=(None, 100e-9, None),
        )
        # axis 0 and 1 are each singly specified, axis 2 is not specified at all
        with pytest.raises(Exception):
            _place_device(device, config, key)

    def test_no_voxel_spec_raises(self, config, key, two_materials):
        """No voxel shape spec at all should raise."""
        device = _ConcreteDevice(
            materials=two_materials,
            param_transforms=[],
            # partial_voxel_grid_shape and partial_voxel_real_shape both default to UNDEFINED
        )
        with pytest.raises(Exception):
            _place_device(device, config, key)

    def test_continuous_output_with_three_materials_raises(self, config, key, three_materials):
        """Continuous output (no transforms) requires exactly 2 materials."""
        device = _make_device(three_materials, param_transforms=[])
        with pytest.raises(Exception, match="exactly two materials"):
            _place_device(device, config, key)


# ---------------------------------------------------------------------------
# init_params
# ---------------------------------------------------------------------------


class TestInitParams:
    def test_returns_array_with_matrix_shape_no_transforms(self, config, key, two_materials):
        device = _make_device(two_materials, voxel_grid=(2, 2, 2))
        placed = _place_device(device, config, key, ((0, 10), (0, 10), (0, 10)))
        params = placed.init_params(key)
        assert params.shape == (5, 5, 5)

    def test_params_in_unit_interval(self, config, key, two_materials):
        device = _make_device(two_materials, voxel_grid=(1, 1, 1))
        placed = _place_device(device, config, key)
        params = placed.init_params(key)
        assert float(jnp.min(params)) >= 0.0
        assert float(jnp.max(params)) <= 1.0

    def test_params_are_float32(self, config, key, two_materials):
        device = _make_device(two_materials, voxel_grid=(1, 1, 1))
        placed = _place_device(device, config, key)
        params = placed.init_params(key)
        assert params.dtype == jnp.float32


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------


class TestCall:
    def test_no_transforms_returns_input_unchanged(self, config, key, two_materials):
        device = _make_device(two_materials, voxel_grid=(1, 1, 1))
        placed = _place_device(device, config, key)
        params = jnp.ones((10, 10, 10), dtype=jnp.float32) * 0.5
        result = placed(params)
        assert result.shape == params.shape
        assert jnp.allclose(result, params)

    def test_with_dict_input(self, config, key, two_materials):
        device = _make_device(two_materials, voxel_grid=(1, 1, 1))
        placed = _place_device(device, config, key)
        params = {"params": jnp.ones((10, 10, 10), dtype=jnp.float32) * 0.5}
        result = placed(params)
        assert result.shape == (10, 10, 10)

    def test_expand_to_sim_grid(self, config, key, two_materials):
        device = _make_device(two_materials, voxel_grid=(2, 2, 2))
        placed = _place_device(device, config, key, ((0, 10), (0, 10), (0, 10)))
        # matrix shape is (5, 5, 5), expanded should be (10, 10, 10)
        params = jnp.ones((5, 5, 5), dtype=jnp.float32) * 0.5
        result = placed(params, expand_to_sim_grid=True)
        assert result.shape == (10, 10, 10)

    def test_with_closest_index_transform(self, config, key, two_materials):
        """ClosestIndex maps continuous→binary."""
        transform = ClosestIndex()
        device = _make_device(two_materials, param_transforms=[transform])
        placed = _place_device(device, config, key)
        params = placed.init_params(key)
        result = placed(params)
        # binary: all values should be 0 or 1
        assert jnp.all((result == 0) | (result == 1))
