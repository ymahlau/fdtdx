"""Test cases for electromagnetic field metrics and normalization utilities."""

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.physics.metrics import (
    compute_energy,
    compute_poynting_flux,
    normalize_by_energy,
    normalize_by_poynting_flux,
)
from fdtdx.units.composite import V, m, A, J, s
from fdtdx.units.typing import SI
from fdtdx.units.unitful import Unitful
from fdtdx.constants import eps0, mu0
import fdtdx.functional as ff


def test_compute_energy_basic():
    """Test basic energy computation with simple field configuration."""
    # Create simple uniform fields
    E = (V / m) * jnp.ones((3, 4, 4, 4)) * 1e8
    H = (A / m) * jnp.ones((3, 4, 4, 4))
    inv_permittivity = 1.0
    inv_permeability = 1.0

    energy = compute_energy(E, H, inv_permittivity, inv_permeability)

    expected_energy = 0.5 * eps0.value() * (1 / inv_permittivity) * 3 * 1e16 + 0.5 * mu0.value() * (1 / inv_permeability) * 3

    assert energy.shape == (4, 4, 4)
    assert energy.unit.dim == {SI.kg: 1, SI.m: -1, SI.s: -2} # J / m^3
    assert jnp.allclose(energy.value(), expected_energy)


def test_compute_energy_complex_fields():
    """Test energy computation with complex fields."""
    # Create complex fields
    E = (V / m) * jnp.array([[[1 + 1j]], [[2 + 0j]], [[0 + 2j]]])
    H = (A / m) * jnp.array([[[1 - 1j]], [[1 + 1j]], [[2 + 0j]]])
    inv_permittivity = 2.0
    inv_permeability = 0.5

    energy = compute_energy(E, H, inv_permittivity, inv_permeability)

    # |E|^2 = |1+1j|^2 + |2+0j|^2 + |0+2j|^2 = 2 + 4 + 4 = 10
    # |H|^2 = |1-1j|^2 + |1+1j|^2 + |2+0j|^2 = 2 + 2 + 4 = 8
    expected_energy = 0.5 * eps0.value() * (1 / 2.0) * 10 + mu0.value() * 0.5 * (1 / 0.5) * 8

    assert jnp.allclose(energy.value(), expected_energy)


def test_normalize_by_energy_basic():
    """Test basic field normalization by energy."""
    E = (V / m) * jnp.array([[[2.0]], [[0.0]], [[0.0]]]) * 1e8
    H = (A / m) * jnp.array([[[0.0]], [[2.0]], [[0.0]]])
    inv_permittivity = 1.0
    inv_permeability = 1.0
    resolution = 1 * m

    E_norm, H_norm = normalize_by_energy(E, H, inv_permittivity, inv_permeability, resolution)

    # Check that normalized fields have unit total energy
    total_energy = compute_energy(E_norm, H_norm, inv_permittivity, inv_permeability)
    assert jnp.allclose(ff.sum(total_energy*m**3).value(), 1.0)


def test_normalize_by_energy_preservation():
    """Test that normalization preserves field ratios."""
    E = (V / m) * jnp.array([[[1.0]], [[2.0]], [[3.0]]]) * 1e8
    H = (A / m) * jnp.array([[[4.0]], [[5.0]], [[6.0]]])
    inv_permittivity = 1.0
    inv_permeability = 1.0
    resolution = 1 * m

    E_norm, H_norm = normalize_by_energy(E, H, inv_permittivity, inv_permeability, resolution)

    # Check that ratios are preserved
    assert jnp.allclose((E_norm[1] / E_norm[0]).materialise(), (E[1] / E[0]).materialise())
    assert jnp.allclose((H_norm[2] / H_norm[1]).materialise(), (H[2] / H[1]).materialise())


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
