"""Shared fixtures for detector tests."""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig


@pytest.fixture
def simulation_config():
    """Create a minimal SimulationConfig for testing."""
    return SimulationConfig(
        time=1e-12,  # 1 picosecond
        resolution=1e-7,  # 100 nm
        backend="cpu",
    )


@pytest.fixture
def small_grid_slice():
    """A small 8x8x8 grid slice for testing."""
    return ((0, 8), (0, 8), (0, 8))


@pytest.fixture
def plane_grid_slice():
    """A 2D plane slice (8x8x1) for testing plane detectors."""
    return ((0, 8), (0, 8), (0, 1))


@pytest.fixture
def line_grid_slice():
    """A 1D line slice (8x1x1) for testing line detectors."""
    return ((0, 8), (0, 1), (0, 1))


@pytest.fixture
def point_grid_slice():
    """A point slice (1x1x1) for testing point detectors."""
    return ((0, 1), (0, 1), (0, 1))


@pytest.fixture
def random_key():
    """Create a JAX random key for testing."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def constant_E_field():
    """Create a constant E field (3, 8, 8, 8)."""
    return jnp.ones((3, 8, 8, 8), dtype=jnp.float32)


@pytest.fixture
def constant_H_field():
    """Create a constant H field (3, 8, 8, 8)."""
    return jnp.ones((3, 8, 8, 8), dtype=jnp.float32) * 0.5


@pytest.fixture
def sinusoidal_E_field():
    """Create a sinusoidal E field pattern."""
    x = jnp.linspace(0, 2 * jnp.pi, 8)
    y = jnp.linspace(0, 2 * jnp.pi, 8)
    z = jnp.linspace(0, 2 * jnp.pi, 8)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    Ex = jnp.sin(X) * jnp.cos(Y)
    Ey = jnp.sin(Y) * jnp.cos(Z)
    Ez = jnp.sin(Z) * jnp.cos(X)

    return jnp.stack([Ex, Ey, Ez], axis=0).astype(jnp.float32)


@pytest.fixture
def sinusoidal_H_field():
    """Create a sinusoidal H field pattern."""
    x = jnp.linspace(0, 2 * jnp.pi, 8)
    y = jnp.linspace(0, 2 * jnp.pi, 8)
    z = jnp.linspace(0, 2 * jnp.pi, 8)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    Hx = jnp.cos(X) * jnp.sin(Y) * 0.5
    Hy = jnp.cos(Y) * jnp.sin(Z) * 0.5
    Hz = jnp.cos(Z) * jnp.sin(X) * 0.5

    return jnp.stack([Hx, Hy, Hz], axis=0).astype(jnp.float32)


@pytest.fixture
def inv_permittivity():
    """Create inverse permittivity array (vacuum = 1.0)."""
    return jnp.ones((3, 8, 8, 8), dtype=jnp.float32)


@pytest.fixture
def inv_permeability():
    """Create inverse permeability (vacuum = 1.0)."""
    return 1.0
