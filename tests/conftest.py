"""Root conftest.py – force JAX to use CPU only for all test suites.

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

os.environ["JAX_PLATFORMS"] = "cpu"
