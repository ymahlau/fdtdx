"""Root conftest.py - force JAX to use CPU only for all test suites.

Without this, ``jax.devices()`` (no backend argument) may return CUDA devices
once the GPU backend is initialised by any ``jnp`` operation.
``create_named_sharded_matrix`` uses ``jax.devices()`` for the sharding mesh
but ``jax.devices(backend="cpu")`` for the buffers, which causes a
CPU-buffer / CUDA-sharding mismatch when the unit and integration suites are
run in the same process.

Setting ``JAX_PLATFORMS=cpu`` before JAX is first imported keeps the device
list consistent across the entire test session.
"""

import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

# Persist compiled JAX kernels across test runs so simulation tests
# pay the JIT compilation cost only once per unique kernel.
_CACHE_DIR = str(Path(__file__).parent.parent / ".jax_cache")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _CACHE_DIR)
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
# Cache all kernels, not just those exceeding the default 1s threshold.
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
