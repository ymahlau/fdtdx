"""Test cases for electromagnetic field metrics and normalization utilities."""

import jax.numpy as jnp
import pytest

from fdtdx.core.physics.metrics import (
    compute_energy,
    compute_poynting_flux,
    normalize_by_energy,
    normalize_by_poynting_flux,
)

pytestmark = pytest.mark.unit


def test_compute_energy_basic():
    """Test basic energy computation with simple field configuration."""
    E = jnp.ones((3, 4, 4, 4))
    H = jnp.ones((3, 4, 4, 4))
    inv_permittivity = 1.0
    inv_permeability = 1.0

    energy = compute_energy(E, H, inv_permittivity, inv_permeability)

    # For uniform fields: each component |E_i|^2=1, summed over 3 components = 3
    # energy = 0.5 * eps * 3 + 0.5 * mu * 3 = 1.5 + 1.5 = 3.0
    assert energy.shape == (4, 4, 4)
    assert jnp.allclose(energy, 3.0)


def test_compute_energy_basic_anisotropic():
    """Test energy computation with diagonal anisotropic permittivity (shape (3,...))."""
    E = jnp.ones((3, 4, 4, 4))
    H = jnp.ones((3, 4, 4, 4))
    inv_permittivity = jnp.stack(
        [
            jnp.full((4, 4, 4), 1.0),
            jnp.full((4, 4, 4), 0.5),
            jnp.full((4, 4, 4), 1.0),
        ]
    )
    inv_permeability = jnp.ones((3, 4, 4, 4))

    energy = compute_energy(E, H, inv_permittivity, inv_permeability)

    # eps_y = 1/0.5 = 2, others = 1
    # E energy = 0.5*(1*1 + 2*1 + 1*1) = 2.0
    # H energy = 0.5*(1*1 + 1*1 + 1*1) = 1.5
    assert energy.shape == (4, 4, 4)
    assert jnp.allclose(energy, 3.5)


def test_compute_energy_complex_fields():
    """Test energy computation with complex fields."""
    E = jnp.array([[[1 + 1j]], [[2 + 0j]], [[0 + 2j]]])
    H = jnp.array([[[1 - 1j]], [[1 + 1j]], [[2 + 0j]]])
    inv_permittivity = 2.0
    inv_permeability = 0.5

    energy = compute_energy(E, H, inv_permittivity, inv_permeability)

    # |E|^2 = 2 + 4 + 4 = 10, |H|^2 = 2 + 2 + 4 = 8
    # energy = 0.5*(1/2)*10 + 0.5*(1/0.5)*8 = 2.5 + 8.0 = 10.5
    assert jnp.allclose(energy, 10.5)


def test_compute_energy_complex_fields_anisotropic():
    """Test energy computation with complex fields and diagonal anisotropic permittivity."""
    E = jnp.array([[[1 + 1j]], [[2 + 0j]], [[0 + 2j]]])
    H = jnp.array([[[1 - 1j]], [[1 + 1j]], [[2 + 0j]]])
    inv_permittivity = jnp.stack(
        [
            jnp.full((1, 1), 1 / 2.0),
            jnp.full((1, 1), 1 / 1.5),
            jnp.full((1, 1), 1 / 2.0),
        ]
    )
    inv_permeability = 1 / 2.0

    energy = compute_energy(E, H, inv_permittivity, inv_permeability)

    # E energy = 0.5*(2.0*2 + 1.5*4 + 2.0*4) = 0.5*18 = 9.0
    # H energy = 0.5*2.0*8 = 8.0
    expected = 0.5 * (2.0 * 2.0 + 1.5 * 4 + 2.0 * 4.0) + 0.5 * 2.0 * 8
    assert jnp.allclose(energy, expected)


def test_compute_energy_spatially_varying_materials():
    """Test energy computation with spatially varying scalar material properties."""
    E = jnp.ones((3, 2, 2, 2))
    H = jnp.ones((3, 2, 2, 2))
    inv_permittivity = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    inv_permeability = jnp.array([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]])

    energy = compute_energy(E, H, inv_permittivity, inv_permeability)

    expected = 0.5 * (1 / inv_permittivity) * 3 + 0.5 * (1 / inv_permeability) * 3
    assert energy.shape == (2, 2, 2)
    assert jnp.allclose(energy, expected)


def test_compute_energy_spatially_varying_anisotropic_materials():
    """Test energy computation with spatially varying diagonal anisotropic materials."""
    E = jnp.ones((3, 2, 2, 2))
    H = jnp.ones((3, 2, 2, 2))
    inv_permittivity = jnp.stack(
        [
            jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            jnp.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]]),
            jnp.array([[[2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0]]]),
        ]
    )
    inv_permeability = jnp.array([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]])

    energy = compute_energy(E, H, inv_permittivity, inv_permeability)

    expected = (
        0.5 * (1 / inv_permittivity[0] + 1 / inv_permittivity[1] + 1 / inv_permittivity[2])
        + 0.5 * (1 / inv_permeability) * 3
    )
    assert energy.shape == (2, 2, 2)
    assert jnp.allclose(energy, expected)


def test_compute_energy_different_axis():
    """Test energy computation with different axis parameter."""
    E = jnp.ones((4, 3, 2, 2))
    H = jnp.ones((4, 3, 2, 2))

    energy = compute_energy(E, H, 1.0, 1.0, axis=1)

    assert energy.shape == (4, 2, 2)
    assert jnp.allclose(energy, 3.0)


def test_compute_energy_full_tensor_isotropic():
    """Test energy with 9-component tensor (full 3x3) that is actually isotropic."""
    # 9-component inv_permittivity representing identity matrix
    # expand_to_3x3 reshapes (9, nx, ny, nz) -> (3, 3, nx, ny, nz)
    nx, ny, nz = 2, 2, 2
    # Build a diagonal identity tensor: components [0,4,8] = 1.0, rest = 0
    inv_eps = jnp.zeros((9, nx, ny, nz))
    inv_eps = inv_eps.at[0].set(1.0)  # (0,0) component
    inv_eps = inv_eps.at[4].set(1.0)  # (1,1) component
    inv_eps = inv_eps.at[8].set(1.0)  # (2,2) component

    inv_mu = jnp.zeros((9, nx, ny, nz))
    inv_mu = inv_mu.at[0].set(1.0)
    inv_mu = inv_mu.at[4].set(1.0)
    inv_mu = inv_mu.at[8].set(1.0)

    E = jnp.ones((3, nx, ny, nz))
    H = jnp.ones((3, nx, ny, nz))

    energy = compute_energy(E, H, inv_eps, inv_mu)

    # Identity inv_eps/inv_mu -> eps=mu=I -> energy = 0.5*|E|^2 + 0.5*|H|^2 = 1.5+1.5=3.0
    assert energy.shape == (nx, ny, nz)
    assert jnp.allclose(energy, 3.0)


def test_compute_energy_full_tensor_diagonal():
    """Test energy with 9-component tensor that is diagonal but anisotropic."""
    nx, ny, nz = 2, 2, 2
    # inv_eps diagonal = [2, 0.5, 1] -> eps diagonal = [0.5, 2, 1]
    inv_eps = jnp.zeros((9, nx, ny, nz))
    inv_eps = inv_eps.at[0].set(2.0)  # inv_eps_xx
    inv_eps = inv_eps.at[4].set(0.5)  # inv_eps_yy
    inv_eps = inv_eps.at[8].set(1.0)  # inv_eps_zz

    inv_mu = jnp.zeros((9, nx, ny, nz))
    inv_mu = inv_mu.at[0].set(1.0)
    inv_mu = inv_mu.at[4].set(1.0)
    inv_mu = inv_mu.at[8].set(1.0)

    E = jnp.ones((3, nx, ny, nz))
    H = jnp.ones((3, nx, ny, nz))

    energy = compute_energy(E, H, inv_eps, inv_mu)

    # eps = diag(0.5, 2, 1), mu = I
    # E energy = 0.5 * (E^T eps E) = 0.5 * (0.5 + 2 + 1) = 1.75
    # H energy = 0.5 * (H^T mu H) = 0.5 * 3 = 1.5
    assert energy.shape == (nx, ny, nz)
    assert jnp.allclose(energy, 3.25)


def test_compute_energy_full_tensor_with_9_inv_mu():
    """Test energy when inv_permeability also has 9 components."""
    nx, ny, nz = 2, 2, 2
    inv_eps = jnp.zeros((9, nx, ny, nz))
    inv_eps = inv_eps.at[0].set(1.0)
    inv_eps = inv_eps.at[4].set(1.0)
    inv_eps = inv_eps.at[8].set(1.0)

    inv_mu = jnp.zeros((9, nx, ny, nz))
    inv_mu = inv_mu.at[0].set(2.0)  # inv_mu_xx
    inv_mu = inv_mu.at[4].set(2.0)  # inv_mu_yy
    inv_mu = inv_mu.at[8].set(2.0)  # inv_mu_zz

    E = jnp.ones((3, nx, ny, nz))
    H = jnp.ones((3, nx, ny, nz))

    energy = compute_energy(E, H, inv_eps, inv_mu)

    # eps = I, mu = diag(0.5, 0.5, 0.5)
    # E energy = 0.5 * 3 = 1.5
    # H energy = 0.5 * (0.5+0.5+0.5) = 0.75
    assert energy.shape == (nx, ny, nz)
    assert jnp.allclose(energy, 2.25)


def test_compute_energy_full_tensor_agrees_with_diagonal():
    """Verify full 3x3 tensor path gives same result as diagonal path for diagonal materials."""
    nx, ny, nz = 3, 3, 3
    E = jnp.array(
        [
            jnp.full((nx, ny, nz), 1.0),
            jnp.full((nx, ny, nz), 2.0),
            jnp.full((nx, ny, nz), 3.0),
        ]
    )
    H = jnp.array(
        [
            jnp.full((nx, ny, nz), 0.5),
            jnp.full((nx, ny, nz), 1.5),
            jnp.full((nx, ny, nz), 2.5),
        ]
    )

    # Diagonal inv_eps = [2, 0.5, 1]
    inv_eps_diag = jnp.stack(
        [
            jnp.full((nx, ny, nz), 2.0),
            jnp.full((nx, ny, nz), 0.5),
            jnp.full((nx, ny, nz), 1.0),
        ]
    )
    inv_mu_scalar = 1.0

    # Same as 9-component tensor
    inv_eps_full = jnp.zeros((9, nx, ny, nz))
    inv_eps_full = inv_eps_full.at[0].set(2.0)
    inv_eps_full = inv_eps_full.at[4].set(0.5)
    inv_eps_full = inv_eps_full.at[8].set(1.0)

    energy_diag = compute_energy(E, H, inv_eps_diag, inv_mu_scalar)
    energy_full = compute_energy(E, H, inv_eps_full, inv_mu_scalar)

    assert jnp.allclose(energy_diag, energy_full, atol=1e-5)


def test_normalize_by_energy_basic():
    """Test basic field normalization by energy."""
    E = jnp.array([[[2.0]], [[0.0]], [[0.0]]])
    H = jnp.array([[[0.0]], [[2.0]], [[0.0]]])

    E_norm, H_norm = normalize_by_energy(E, H, 1.0, 1.0)

    total_energy = compute_energy(E_norm, H_norm, 1.0, 1.0)
    assert jnp.allclose(jnp.sum(total_energy), 1.0)


def test_normalize_by_energy_preservation():
    """Test that normalization preserves field ratios."""
    E = jnp.array([[[1.0]], [[2.0]], [[3.0]]])
    H = jnp.array([[[4.0]], [[5.0]], [[6.0]]])

    E_norm, H_norm = normalize_by_energy(E, H, 1.0, 1.0)

    assert jnp.allclose(E_norm[1] / E_norm[0], E[1] / E[0])
    assert jnp.allclose(H_norm[2] / H_norm[1], H[2] / H[1])


def test_compute_poynting_flux_basic():
    """Test basic Poynting vector computation."""
    E = jnp.array([[[1.0]], [[0.0]], [[0.0]]])
    H = jnp.array([[[0.0]], [[1.0]], [[0.0]]])

    S = compute_poynting_flux(E, H)

    expected_S = jnp.array([[[0.0]], [[0.0]], [[1.0]]])
    assert jnp.allclose(S, expected_S)


def test_compute_poynting_flux_complex_fields():
    """Test Poynting vector with complex fields."""
    E = jnp.array([[[1 + 1j]], [[0.0]], [[0.0]]])
    H = jnp.array([[[0.0]], [[1 - 1j]], [[0.0]]])

    S = compute_poynting_flux(E, H)

    # E × conj(H): z-component = (1+1j)*(1+1j) = 2j
    expected_z = (1 + 1j) * (1 + 1j)
    expected_S = jnp.array([[[0.0]], [[0.0]], [[expected_z]]])
    assert jnp.allclose(S, expected_S)


def test_compute_poynting_flux_different_axis():
    """Test Poynting vector computation with different axis."""
    E = jnp.ones((2, 3, 2))
    H = jnp.ones((2, 3, 2))

    S = compute_poynting_flux(E, H, axis=1)

    assert S.shape == (2, 3, 2)


def test_normalize_by_poynting_flux_basic():
    """Test basic normalization by Poynting flux."""
    E = jnp.array([[[2.0]], [[0.0]], [[0.0]]])
    H = jnp.array([[[0.0]], [[2.0]], [[0.0]]])
    axis = 2

    E_norm, H_norm = normalize_by_poynting_flux(E, H, axis)

    S_norm = jnp.cross(jnp.conj(E_norm), H_norm, axisa=0, axisb=0, axisc=0)
    power = jnp.abs(jnp.sum(0.5 * jnp.real(S_norm[axis])))
    assert jnp.allclose(power, 1.0)


def test_normalize_by_poynting_flux_preservation():
    """Test that normalization preserves field structure."""
    E = jnp.array([[[1.0]], [[2.0]], [[0.0]]])
    H = jnp.array([[[3.0]], [[4.0]], [[0.0]]])
    axis = 2

    E_norm, H_norm = normalize_by_poynting_flux(E, H, axis)

    assert jnp.allclose(E_norm[1] / E_norm[0], E[1] / E[0])
    assert jnp.allclose(H_norm[1] / H_norm[0], H[1] / H[0])
