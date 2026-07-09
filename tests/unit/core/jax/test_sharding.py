"""Unit tests for fdtdx.core.jax.sharding module."""

import os
import subprocess
import sys
import textwrap
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
    pin_to_single_device,
    pretty_print_sharding,
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


# ---- pin_to_single_device ----


def _run_in_subprocess_with_devices(num_devices: int, body: str) -> None:
    """Run ``body`` in a fresh interpreter with ``num_devices`` virtual CPU devices.

    ``jax.devices()`` reflects the platform's device count only at JAX's first backend
    initialisation, which has already happened (with a single forced CPU device, see
    ``tests/conftest.py``) by the time this test module runs. A real multi-device mesh can
    therefore only be exercised in a separate process started with
    ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` set beforehand. Mocking
    ``jax.devices()`` to return the same physical device twice is not a substitute here:
    unlike the mesh-shape/spec checks elsewhere in this file, ``pin_to_single_device`` calls
    ``jax.lax.with_sharding_constraint`` eagerly, which raises ValueError ("the input arrays
    must be from distinct devices") when the mesh contains a duplicated physical device.
    """
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={num_devices}",
    }
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


class TestPinToSingleDevice:
    """Tests for the pin_to_single_device helper."""

    def test_single_device_is_noop_identity(self):
        # The default test session forces a single CPU device (see tests/conftest.py).
        assert len(jax.devices()) == 1
        arr = jnp.arange(12.0).reshape(3, 4)
        result = pin_to_single_device(arr)
        assert result is arr

    def test_multi_device_preserves_shape_dtype_and_values(self):
        _run_in_subprocess_with_devices(
            2,
            """
            import jax, jax.numpy as jnp
            from fdtdx.core.jax.sharding import pin_to_single_device

            assert len(jax.devices()) == 2
            arr = jnp.arange(24.0, dtype=jnp.float32).reshape(4, 6)
            out = pin_to_single_device(arr)
            assert out.shape == arr.shape
            assert out.dtype == arr.dtype
            assert bool(jnp.array_equal(out, arr))
            """,
        )

    def test_multi_device_result_is_replicated_named_sharding(self):
        _run_in_subprocess_with_devices(
            2,
            """
            import jax, jax.numpy as jnp
            from fdtdx.core.jax.sharding import pin_to_single_device

            out = pin_to_single_device(jnp.zeros((4, 6), dtype=jnp.float32))
            assert isinstance(out.sharding, jax.sharding.NamedSharding)
            # Fully replicated: an empty PartitionSpec, not sharded along any axis.
            assert tuple(out.sharding.spec) == ()
            assert "shard" in out.sharding.mesh.axis_names
            """,
        )

    def test_multi_device_compatible_with_sliced_x_sharded_parent(self):
        """Regression test for the exact usage in ModePlaneSource / ModeOverlapDetector:
        pinning a plane sliced out of an x-sharded array inside a jit must not raise JAX's
        "Received incompatible devices for jitted computation" error, which is what happens
        if the sliced plane were instead placed on a plain SingleDeviceSharding (device 0)
        while the surrounding computation is compiled over the full device mesh.
        """
        _run_in_subprocess_with_devices(
            2,
            """
            import jax, jax.numpy as jnp
            from fdtdx.core.jax.sharding import create_named_sharded_matrix, pin_to_single_device

            # (component, Nx, Ny, Nz) permittivity-like array, x-sharded on axis=1.
            arr = create_named_sharded_matrix(
                (3, 8, 6, 6), value=2.0, dtype=jnp.float32, sharding_axis=1, backend="cpu"
            )

            @jax.jit
            def slice_and_pin(a):
                plane = a[:, 2:3, :, :]
                return pin_to_single_device(plane)

            out = slice_and_pin(arr)
            assert out.shape == (3, 1, 6, 6)
            assert bool(jnp.allclose(out, 2.0))
            """,
        )
