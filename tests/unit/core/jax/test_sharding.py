"""Unit tests for fdtdx.core.jax.sharding module."""

from unittest import mock
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest

import fdtdx.core.jax.sharding as sharding_module
from fdtdx.constants import SHARD_STR
from fdtdx.core.jax.sharding import (
    create_named_sharded_matrix,
    get_dtype_bytes,
    get_named_sharding_from_shape,
    pretty_print_sharding,
    sharding_preserving_add,
    sharding_preserving_set,
)

CPU_DEVICES = jax.devices("cpu")


# ---- get_dtype_bytes ----


class TestGetDtypeBytes:
    """Tests for the get_dtype_bytes helper."""

    @pytest.mark.parametrize(
        "dtype, expected",
        [
            (jnp.float32, 4),
            (jnp.float64, 8),
            (jnp.int32, 4),
            (jnp.int64, 8),
            (jnp.float16, 2),
            (jnp.bfloat16, 2),
            (jnp.complex64, 8),
            (jnp.bool_, 1),
            (jnp.uint8, 1),
        ],
    )
    def test_returns_correct_byte_size(self, dtype, expected):
        assert get_dtype_bytes(dtype) == expected


# ---- _local_shape_from_global_index ----


class TestLocalShapeFromGlobalIndex:
    """Tests for deriving local buffer shapes from JAX's global shard indices."""

    @pytest.mark.parametrize(
        "shape, global_index, expected",
        [
            ((64, 8), (slice(16, 32), slice(None)), (16, 8)),
            ((3, 64, 8), (slice(None), slice(48, 64), slice(None)), (3, 16, 8)),
            ((10,), (slice(1, 9, 2),), (4,)),
        ],
    )
    def test_derives_shape_from_slices(self, shape, global_index, expected):
        """Derive each local shard dimension from its global slice."""
        assert sharding_module._local_shape_from_global_index(shape, global_index) == expected

    @pytest.mark.parametrize("global_index", [None, (slice(None),), (0, slice(None))])
    def test_rejects_invalid_indices(self, global_index):
        """Reject missing, incomplete, and non-slice shard indices."""
        with pytest.raises(ValueError, match=r"Invalid shard index|Expected a slice"):
            sharding_module._local_shape_from_global_index((4, 6), global_index)


# ---- pretty_print_sharding ----


class TestPrettyPrintSharding:
    """Tests for the pretty_print_sharding helper.

    PositionalSharding and SingleDeviceSharding are deprecated in recent JAX,
    so we patch them as fake classes for isinstance checks to work.
    """

    # Fake classes for deprecated sharding types
    class _FakePositionalSharding:
        pass

    class _FakeSingleDeviceSharding:
        pass

    @pytest.fixture(autouse=True)
    def _patch_deprecated_shardings(self):
        """Patch deprecated sharding types so isinstance checks don't raise."""
        with (
            mock.patch.object(jax.sharding, "PositionalSharding", self._FakePositionalSharding, create=True),
            mock.patch.object(jax.sharding, "SingleDeviceSharding", self._FakeSingleDeviceSharding, create=True),
        ):
            yield

    def test_named_sharding(self):
        sharding = get_named_sharding_from_shape((10, 20), sharding_axis=0)
        result = pretty_print_sharding(sharding)
        assert result.startswith("NamedSharding(")
        # JAX renders PartitionSpec as either "PartitionSpec(...)" or "P(...)"
        assert "PartitionSpec" in result or "P(" in result

    def test_single_device_sharding(self):
        obj = self._FakeSingleDeviceSharding()
        obj._device = "cpu:0"
        result = pretty_print_sharding(obj)
        assert result == "SingleDeviceSharding(cpu:0)"

    def test_unknown_sharding_type_falls_back_to_str(self):
        class UnknownSharding:
            def __str__(self):
                return "UnknownSharding(custom)"

        result = pretty_print_sharding(UnknownSharding())
        assert result == "UnknownSharding(custom)"


# ---- get_named_sharding_from_shape ----


class TestGetNamedShardingFromShape:
    """Tests for the get_named_sharding_from_shape function."""

    def test_returns_named_sharding(self):
        result = get_named_sharding_from_shape((10, 20, 30), sharding_axis=0)
        assert isinstance(result, jax.sharding.NamedSharding)

    def test_partition_spec_shards_correct_axis(self):
        result = get_named_sharding_from_shape((10, 20, 30), sharding_axis=1)
        spec = result.spec
        assert spec[0] is None
        assert spec[1] == SHARD_STR
        assert spec[2] is None

    def test_partition_spec_first_axis(self):
        result = get_named_sharding_from_shape((10, 20), sharding_axis=0)
        spec = result.spec
        assert spec[0] == SHARD_STR
        assert spec[1] is None

    def test_mesh_has_shard_axis_name(self):
        result = get_named_sharding_from_shape((10, 20), sharding_axis=0)
        assert SHARD_STR in result.mesh.axis_names

    def test_mesh_device_count_matches_available(self):
        result = get_named_sharding_from_shape((10, 20), sharding_axis=0)
        num_devices = len(jax.devices())
        assert result.mesh.devices.shape == (num_devices,)


# ---- create_named_sharded_matrix ----


class TestCreateNamedShardedMatrix:
    """Tests for the create_named_sharded_matrix function."""

    @pytest.fixture(autouse=True)
    def _force_cpu(self):
        """Ensure both jax.devices() and jax.devices(backend=...) return CPU
        so sharding mesh and array placement are on the same device."""
        with mock.patch.object(jax, "devices", return_value=CPU_DEVICES):
            yield

    def test_uses_requested_backend(self):
        """Use the requested backend for validation and mesh construction."""
        with mock.patch.object(jax, "devices", return_value=CPU_DEVICES) as devices:
            create_named_sharded_matrix(shape=(4, 6), value=1.0, sharding_axis=0, dtype=jnp.float32, backend="cpu")

        assert devices.call_args_list == [mock.call(backend="cpu"), mock.call(backend="cpu")]

    def test_returns_jax_array(self):
        result = create_named_sharded_matrix(shape=(4, 6), value=1.0, sharding_axis=0, dtype=jnp.float32, backend="cpu")
        assert isinstance(result, jax.Array)

    def test_correct_shape(self):
        shape = (4, 8, 2)
        result = create_named_sharded_matrix(shape=shape, value=1.0, sharding_axis=0, dtype=jnp.float32, backend="cpu")
        assert result.shape == shape

    def test_correct_dtype(self):
        result = create_named_sharded_matrix(shape=(4, 6), value=1.0, sharding_axis=0, dtype=jnp.float32, backend="cpu")
        assert result.dtype == jnp.float32

    def test_filled_with_value(self):
        result = create_named_sharded_matrix(shape=(4, 6), value=3.5, sharding_axis=0, dtype=jnp.float32, backend="cpu")
        assert jnp.allclose(result, 3.5)

    def test_raises_on_indivisible_sharding_axis(self):
        # Mock 2 CPU devices to trigger the divisibility check
        cpu = CPU_DEVICES[0]
        fake_devices = [cpu, cpu]
        fake_sharding = MagicMock()
        with (
            mock.patch.object(jax, "devices", return_value=fake_devices),
            mock.patch(
                "fdtdx.core.jax.sharding.get_named_sharding_from_shape",
                return_value=fake_sharding,
            ),
        ):
            with pytest.raises(ValueError, match="divisible by num_devices"):
                create_named_sharded_matrix(
                    shape=(3, 5),
                    value=1.0,
                    sharding_axis=0,
                    dtype=jnp.float32,
                    backend="cpu",
                )

    def test_constructs_only_addressable_device_arrays(self):
        """Construct input arrays only for locally addressable devices."""
        cpu = CPU_DEVICES[0]
        global_devices = [cpu] * 4
        shape = (8, 6)
        fake_sharding = MagicMock()
        fake_sharding.addressable_devices_indices_map.return_value = {
            cpu: (slice(4, 6), slice(None)),
        }
        expected = object()

        with (
            mock.patch.object(jax, "devices", return_value=global_devices),
            mock.patch(
                "fdtdx.core.jax.sharding.get_named_sharding_from_shape",
                return_value=fake_sharding,
            ),
            mock.patch.object(jax, "make_array_from_single_device_arrays", return_value=expected) as make_array,
        ):
            result = create_named_sharded_matrix(
                shape=shape,
                value=2.0,
                sharding_axis=0,
                dtype=jnp.float32,
                backend="cpu",
            )

        assert result is expected
        fake_sharding.addressable_devices_indices_map.assert_called_once_with(shape)
        matrices = make_array.call_args.args[2]
        assert len(matrices) == len(fake_sharding.addressable_devices_indices_map.return_value)
        assert matrices[0].shape == (2, 6)
        assert jnp.allclose(matrices[0], 2.0)

    def test_rejects_mapped_local_shape_mismatch(self):
        """Reject addressable mappings that violate the even-shard contract."""
        cpu = CPU_DEVICES[0]
        fake_sharding = MagicMock()
        fake_sharding.addressable_devices_indices_map.return_value = {
            cpu: (slice(0, 3), slice(None)),
        }

        with (
            mock.patch.object(jax, "devices", return_value=[cpu, cpu]),
            mock.patch(
                "fdtdx.core.jax.sharding.get_named_sharding_from_shape",
                return_value=fake_sharding,
            ),
            pytest.raises(ValueError, match="Mapped local shard shape"),
        ):
            create_named_sharded_matrix(
                shape=(4, 6),
                value=1.0,
                sharding_axis=0,
                dtype=jnp.float32,
                backend="cpu",
            )

    def test_counter_increments(self):
        old_counter = sharding_module.counter
        create_named_sharded_matrix(shape=(2, 4), value=1.0, sharding_axis=0, dtype=jnp.float32, backend="cpu")
        # 2*4 elements * 4 bytes (float32) = 32
        assert sharding_module.counter == old_counter + 32

    def test_sharding_axis_fallback_when_dim_is_one(self):
        # When shape[sharding_axis] == 1, it should pick the first axis with dim != 1
        result = create_named_sharded_matrix(
            shape=(1, 4, 6), value=2.0, sharding_axis=0, dtype=jnp.float32, backend="cpu"
        )
        assert result.shape == (1, 4, 6)
        assert jnp.allclose(result, 2.0)
        # Sharding should fall back to axis=1 (first axis with dim != 1)
        assert isinstance(result.sharding, jax.sharding.NamedSharding)
        spec = result.sharding.spec
        # axis 0 is None (dim=1, not sharded), axis 1 is sharded (dim=4 != 1)
        assert spec[0] is None
        assert spec[1] is not None

    def test_has_named_sharding(self):
        result = create_named_sharded_matrix(shape=(4, 6), value=1.0, sharding_axis=0, dtype=jnp.float32, backend="cpu")
        assert isinstance(result.sharding, jax.sharding.NamedSharding)


class TestShardingPreservingIndexedUpdate:
    """Tests indexed updates that keep the destination's sharding contract."""

    @pytest.mark.parametrize(
        ("update", "initial_value", "update_value", "expected_value"),
        [
            (sharding_preserving_set, 0.0, 2.0, 2.0),
            (sharding_preserving_add, 1.0, 2.0, 3.0),
        ],
    )
    def test_single_device_update_bypasses_jit(
        self,
        update,
        initial_value,
        update_value,
        expected_value,
    ):
        """Avoid per-object JIT compilation when the array uses one device."""
        array = jnp.full((2, 4), initial_value, dtype=jnp.float32)
        assert len(array.devices()) == 1

        with mock.patch.object(jax, "jit") as jit:
            result = update(array, (slice(None), slice(None)), update_value)

        jit.assert_not_called()
        assert jnp.allclose(result, expected_value)

    @pytest.mark.parametrize("is_fully_addressable", [True, False])
    def test_multi_device_update_uses_sharded_jit(self, is_fully_addressable):
        """Retain the constrained donated JIT path for every multi-device array."""
        array = MagicMock()
        array.is_fully_addressable = is_fully_addressable
        array.devices.return_value = (object(), object())
        array.sharding = object()
        expected = object()
        sharded_update = MagicMock(return_value=expected)

        with mock.patch.object(jax, "jit", return_value=sharded_update) as jit:
            result = sharding_module._sharding_preserving_indexed_update(
                array,
                slice(None),
                2.0,
                operation="set",
            )

        assert result is expected
        jit.assert_called_once()
        assert jit.call_args.kwargs == {
            "out_shardings": array.sharding,
            "donate_argnums": (0,),
        }
        sharded_update.assert_called_once_with(array, 2.0)

    @pytest.mark.parametrize(
        ("update", "initial_value", "update_value", "expected_value"),
        [
            (sharding_preserving_set, 0.0, 2.0, 2.0),
            (sharding_preserving_add, 1.0, 2.0, 3.0),
        ],
    )
    def test_full_domain_update_preserves_named_sharding(
        self,
        update,
        initial_value,
        update_value,
        expected_value,
    ):
        """Preserve NamedSharding and values for full-domain indexed updates."""
        device_count = len(CPU_DEVICES)
        shape = (1, 2 * device_count, 4, 4)
        array = create_named_sharded_matrix(
            shape=shape,
            value=initial_value,
            sharding_axis=1,
            dtype=jnp.float32,
            backend="cpu",
        )
        original_sharding = array.sharding
        assert array.is_fully_addressable
        assert len(array.devices()) == 1

        result = update(
            array,
            (slice(None),) * len(shape),
            jnp.asarray([[[[update_value]]]], dtype=jnp.float32),
        )

        assert result.sharding == original_sharding
        assert result.sharding.spec == jax.sharding.PartitionSpec(None, SHARD_STR, None, None)
        assert {shard.data.shape for shard in result.addressable_shards} == {(1, 2, 4, 4)}
        assert jnp.allclose(result, expected_value)

    def test_rejects_unsupported_operation(self):
        """Reject indexed-update operations outside the supported set/add contract."""
        with pytest.raises(ValueError, match="Unsupported indexed update operation"):
            sharding_module._sharding_preserving_indexed_update(
                jnp.ones((1,)),
                slice(None),
                1.0,
                operation="multiply",
            )
