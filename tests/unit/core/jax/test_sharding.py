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
    _replicated_mesh_sharding,
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


# ---- _replicated_mesh_sharding ----


class TestReplicatedMeshSharding:
    """Tests for the _replicated_mesh_sharding helper, split out of pin_to_single_device so this
    (pure, cheap) mesh/sharding construction can be covered in-process -- unlike pin_to_single_device's
    own data placement, it does not require genuinely distinct physical devices, so a MOCKED duplicated
    device list is a faithful, subprocess-free test (the corresponding data placement is exercised
    separately by TestPinToSingleDevice's subprocess-based multi-device tests below)."""

    def test_single_device_replicated_spec(self):
        cpu = CPU_DEVICES[0]
        result = _replicated_mesh_sharding([cpu])
        assert isinstance(result, jax.sharding.NamedSharding)
        assert tuple(result.spec) == ()
        assert result.mesh.devices.shape == (1,)

    def test_duplicated_device_list_replicated_spec(self):
        # A duplicated physical device (simulating >1 "device" without needing real distinct hardware)
        # is accepted by mesh/sharding CONSTRUCTION -- only the actual data placement in
        # pin_to_single_device requires genuine distinctness (see module docstring above).
        cpu = CPU_DEVICES[0]
        result = _replicated_mesh_sharding([cpu, cpu])
        assert isinstance(result, jax.sharding.NamedSharding)
        assert tuple(result.spec) == ()
        assert result.mesh.devices.shape == (2,)
        assert SHARD_STR in result.mesh.axis_names

    def test_result_usable_as_pin_to_single_device_input(self):
        """pin_to_single_device calls this helper directly with jax.devices() -- confirm the two agree
        on shape/axis-name conventions for a real (single, non-mocked) device list."""
        expected = _replicated_mesh_sharding(jax.devices())
        assert isinstance(expected, jax.sharding.NamedSharding)
        assert expected.mesh.devices.shape == (len(jax.devices()),)


# ---- pin_to_single_device ----


def _run_in_subprocess_with_devices(num_devices: int, body: str) -> None:
    """Run ``body`` in a fresh interpreter with ``num_devices`` virtual CPU devices.

    ``jax.devices()`` reflects the platform's device count only at JAX's first backend
    initialisation, which has already happened (with a single forced CPU device, see
    ``tests/conftest.py``) by the time this test module runs. A real multi-device mesh can
    therefore only be exercised in a separate process started with
    ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` set beforehand. Mocking
    ``jax.devices()`` to return the same physical device twice is not a substitute here:
    unlike the mesh-shape/spec checks elsewhere in this file, ``pin_to_single_device`` places
    data onto a real device mesh eagerly, which requires genuinely distinct physical devices
    (a mesh built from a duplicated device raises at construction/placement time).
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

    def test_multi_device_compatible_with_pure_callback_output(self):
        """Regression test for compute_mode's usage: pinning the RAW OUTPUT of a
        jax.pure_callback (as returned by the Tidy3D mode solver) must not raise "Cannot
        convert GSPMDSharding {maximal device=0} into SdyArray". A pure_callback's result
        carries a maximal/single-device GSPMDSharding -- a fundamentally different
        representation than the ordinary NamedSharding case covered by the sliced-input
        test above, which XLA's Shardy partitioner cannot convert via
        jax.lax.with_sharding_constraint (this failed before this fix; device_put succeeds
        because it performs the placement directly instead of a Shardy-mediated re-shard).
        """
        _run_in_subprocess_with_devices(
            2,
            """
            import numpy as np
            import jax, jax.numpy as jnp
            from fdtdx.core.jax.sharding import create_named_sharded_matrix, pin_to_single_device

            arr = create_named_sharded_matrix(
                (3, 8, 6, 6), value=2.0, dtype=jnp.float32, sharding_axis=1, backend="cpu"
            )

            def host_fn(x):
                # Stand-in for the Tidy3D mode-solver call in compute_mode: an arbitrary
                # host-side computation returning a NEW array (not just a passthrough).
                return np.asarray(x) * 3.0

            @jax.jit
            def slice_callback_and_pin(a):
                plane = a[:, 2:3, :, :]
                raw = jax.pure_callback(
                    host_fn, jax.ShapeDtypeStruct(plane.shape, plane.dtype), plane
                )
                pinned = pin_to_single_device(raw)
                # Mirrors compute_mode's jnp.expand_dims(mode_E_raw, ...) right after pinning --
                # the exact next op that raised before this fix.
                return jnp.expand_dims(pinned, axis=0)

            out = slice_callback_and_pin(arr)
            assert out.shape == (1, 3, 1, 6, 6)
            assert bool(jnp.allclose(out, 6.0))
            """,
        )
