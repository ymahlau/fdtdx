"""Run the multi-process sharding regression tests on a TPU VM slice.

Invoke this script simultaneously from the repository root on every TPU
worker. It initializes JAX distributed and invokes pytest in the same
interpreter so all integration checks share the initialized runtime.
"""

import json
from pathlib import Path

import jax


def main() -> None:
    """Initialize distributed JAX and run the multi-host integration tests."""
    jax.distributed.initialize()
    if jax.default_backend() != "tpu":
        raise RuntimeError(f"Expected TPU backend, got {jax.default_backend()!r}")

    import pytest

    import fdtdx

    process_count = jax.process_count()
    global_device_count = jax.device_count()
    local_device_count = jax.local_device_count()

    assert process_count > 1
    assert global_device_count > local_device_count

    test_path = Path(__file__).parents[1] / "tests" / "integration" / "core" / "jax" / "test_multihost_sharding.py"
    exit_code = pytest.main(["-q", "-p", "no:cacheprovider", str(test_path)])
    if exit_code != pytest.ExitCode.OK:
        raise RuntimeError(f"Multi-host sharding regression failed with {exit_code=}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "process_index": jax.process_index(),
                "process_count": process_count,
                "global_device_count": global_device_count,
                "local_device_count": local_device_count,
                "tests": 2,
                "fdtdx_file": fdtdx.__file__,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
