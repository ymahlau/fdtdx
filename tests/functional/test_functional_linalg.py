
import jax
import jax.numpy as jnp

from fdtdx.units.typing import SI
from fdtdx.units.unitful import Unit, Unitful


def test_norm_unitful_vector_magnitude():
    """Test norm with Unitful object containing velocity vectors"""
    # Create velocity unit: m/s with scale=2 (hundred m/s)
    velocity_unit = Unit(scale=2, dim={SI.m: 1, SI.s: -1})
    # 2D velocity vectors in a matrix (each row is a velocity vector)
    velocities = Unitful(val=jnp.array([
        [3.0, 4.0, 0.0],     # velocity vector 1: magnitude = 5
        [0.0, 6.0, 8.0],     # velocity vector 2: magnitude = 10
        [-5.0, 0.0, 12.0]    # velocity vector 3: magnitude = 13
    ]), unit=velocity_unit)
    
    # Calculate L2 norm along axis 1 (magnitude of each velocity vector)
    result = jnp.linalg.norm(velocities, axis=1)  # type: ignore
    
    assert isinstance(result, Unitful)
    # Expected magnitudes with scale factor (10^2 = 100)
    expected_vals = jnp.array([5.0, 10.0, 13.0]) * 100
    assert jnp.allclose(result.value(), expected_vals)
    # Units should be preserved (velocity magnitude has same units as velocity)
    assert result.unit.dim == {SI.m: 1, SI.s: -1}
    # Result should be 1D array with shape (3,)
    assert result.shape == (3,)


def test_norm_jax_array_frobenius():
    """Test norm with regular JAX array using Frobenius norm"""
    # Create a 3x3 matrix
    matrix = jnp.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    # Calculate Frobenius norm (default for matrices)
    result = jnp.linalg.norm(matrix)  # type: ignore
    
    # Frobenius norm = sqrt(sum of squares of all elements)
    # = sqrt(1^2 + 2^2 + ... + 9^2) = sqrt(285) ≈ 16.882
    expected = jnp.sqrt(jnp.sum(matrix**2))
    
    assert jnp.allclose(result, expected)
    # Should return a regular JAX scalar, not a Unitful object
    assert isinstance(result, jax.Array)
    assert not isinstance(result, Unitful)
    # Result should be a scalar (0-dimensional array)
    assert result.shape == ()