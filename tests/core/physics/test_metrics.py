"""Test cases for electromagnetic field metrics and normalization utilities."""

import jax.numpy as jnp

from fdtdx.core.physics.metrics import (
    compute_energy,
    compute_poynting_flux,
    normalize_by_energy,
    normalize_by_poynting_flux,
)


def test_compute_energy_basic():
    """Test basic energy computation with simple field configuration."""
    # Create simple uniform fields
    E = jnp.ones((3, 4, 4, 4))
    H = jnp.ones((3, 4, 4, 4))
    inv_permittivity = 1.0
    inv_permeability = 1.0

    energy = compute_energy(E, H, inv_permittivity, inv_permeability)

    # Energy should be 0.5 * (1/inv_eps) * |E|^2 + 0.5 * (1/inv_mu) * |H|^2
    # For uniform fields: |E|^2 = 3, |H|^2 = 3
    expected_energy = 0.5 * (1 / inv_permittivity) * 3 + 0.5 * (1 / inv_permeability) * 3
    expected_energy = 3.0

    assert energy.shape == (4, 4, 4)
    assert jnp.allclose(energy, expected_energy)


def test_compute_energy_complex_fields():
    """Test energy computation with complex fields."""
    # Create complex fields
    E = jnp.array([[[1 + 1j]], [[2 + 0j]], [[0 + 2j]]])
    H = jnp.array([[[1 - 1j]], [[1 + 1j]], [[2 + 0j]]])
    inv_permittivity = 2.0
    inv_permeability = 0.5

    energy = compute_energy(E, H, inv_permittivity, inv_permeability)

    # |E|^2 = |1+1j|^2 + |2+0j|^2 + |0+2j|^2 = 2 + 4 + 4 = 10
    # |H|^2 = |1-1j|^2 + |1+1j|^2 + |2+0j|^2 = 2 + 2 + 4 = 8
    expected_energy = 0.5 * (1 / 2.0) * 10 + 0.5 * (1 / 0.5) * 8
    expected_energy = 2.5 + 8.0

    assert jnp.allclose(energy, expected_energy)


def test_compute_energy_spatially_varying_materials():
    """Test energy computation with spatially varying material properties."""
    E = jnp.ones((3, 2, 2, 2))
    H = jnp.ones((3, 2, 2, 2))

    # Create spatially varying permittivity and permeability
    inv_permittivity = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    inv_permeability = jnp.array([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]])

    energy = compute_energy(E, H, inv_permittivity, inv_permeability)

    # For uniform fields with |E|^2 = |H|^2 = 3
    expected_energy = 0.5 * (1 / inv_permittivity) * 3 + 0.5 * (1 / inv_permeability) * 3

    assert energy.shape == (2, 2, 2)
    assert jnp.allclose(energy, expected_energy)


def test_compute_energy_different_axis():
    """Test energy computation with different axis parameter."""
    # Shape (4, 3, 2, 2) - axis 1 contains field components
    E = jnp.ones((4, 3, 2, 2))
    H = jnp.ones((4, 3, 2, 2))
    inv_permittivity = 1.0
    inv_permeability = 1.0

    energy = compute_energy(E, H, inv_permittivity, inv_permeability, axis=1)

    assert energy.shape == (4, 2, 2)
    assert jnp.allclose(energy, 3.0)


def test_normalize_by_energy_basic():
    """Test basic field normalization by energy."""
    E = jnp.array([[[2.0]], [[0.0]], [[0.0]]])
    H = jnp.array([[[0.0]], [[2.0]], [[0.0]]])
    inv_permittivity = 1.0
    inv_permeability = 1.0

    E_norm, H_norm = normalize_by_energy(E, H, inv_permittivity, inv_permeability)

    # Check that normalized fields have unit total energy
    total_energy = compute_energy(E_norm, H_norm, inv_permittivity, inv_permeability)
    assert jnp.allclose(jnp.sum(total_energy), 1.0)


def test_normalize_by_energy_preservation():
    """Test that normalization preserves field ratios."""
    E = jnp.array([[[1.0]], [[2.0]], [[3.0]]])
    H = jnp.array([[[4.0]], [[5.0]], [[6.0]]])
    inv_permittivity = 1.0
    inv_permeability = 1.0

    E_norm, H_norm = normalize_by_energy(E, H, inv_permittivity, inv_permeability)

    # Check that ratios are preserved
    assert jnp.allclose(E_norm[1] / E_norm[0], E[1] / E[0])
    assert jnp.allclose(H_norm[2] / H_norm[1], H[2] / H[1])


def test_compute_poynting_flux_basic():
    """Test basic Poynting vector computation."""
    # Create perpendicular E and H fields
    E = jnp.array([[[1.0]], [[0.0]], [[0.0]]])  # E in x-direction
    H = jnp.array([[[0.0]], [[1.0]], [[0.0]]])  # H in y-direction

    S = compute_poynting_flux(E, H)

    # Poynting vector should be in z-direction: E × H* = x × y = z
    expected_S = jnp.array([[[0.0]], [[0.0]], [[1.0]]])
    assert jnp.allclose(S, expected_S)


def test_compute_poynting_flux_complex_fields():
    """Test Poynting vector with complex fields."""
    # Create complex fields
    E = jnp.array([[[1 + 1j]], [[0.0]], [[0.0]]])
    H = jnp.array([[[0.0]], [[1 - 1j]], [[0.0]]])

    S = compute_poynting_flux(E, H)

    # E × H* = (1+1j) × (1+1j) = (1+1j)(1+1j) in z-direction
    expected_z = (1 + 1j) * (1 + 1j)
    expected_S = jnp.array([[[0.0]], [[0.0]], [[expected_z]]])

    assert jnp.allclose(S, expected_S)


def test_compute_poynting_flux_different_axis():
    """Test Poynting vector computation with different axis."""
    # Shape (2, 3, 2) - axis 1 contains field components
    E = jnp.ones((2, 3, 2))
    H = jnp.ones((2, 3, 2))

    S = compute_poynting_flux(E, H, axis=1)

    assert S.shape == (2, 3, 2)


def test_normalize_by_poynting_flux_basic():
    """Test basic normalization by Poynting flux."""
    # Create fields with known Poynting vector
    E = jnp.array([[[2.0]], [[0.0]], [[0.0]]])
    H = jnp.array([[[0.0]], [[2.0]], [[0.0]]])
    axis = 2  # z-direction

    E_norm, H_norm = normalize_by_poynting_flux(E, H, axis)

    # Check that normalized fields have unit power flow
    S_norm = jnp.cross(jnp.conj(E_norm), H_norm, axisa=0, axisb=0, axisc=0)
    power = jnp.abs(jnp.sum(0.5 * jnp.real(S_norm[axis])))

    assert jnp.allclose(power, 1.0)


def test_normalize_by_poynting_flux_preservation():
    """Test that normalization preserves field structure."""
    E = jnp.array([[[1.0]], [[2.0]], [[0.0]]])
    H = jnp.array([[[3.0]], [[4.0]], [[0.0]]])
    axis = 2

    E_norm, H_norm = normalize_by_poynting_flux(E, H, axis)

    # Check that field ratios are preserved
    assert jnp.allclose(E_norm[1] / E_norm[0], E[1] / E[0])
    assert jnp.allclose(H_norm[1] / H_norm[0], H[1] / H[0])
