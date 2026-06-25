"""Unit tests for the near-to-far-field kernel (convention lock + sanity)."""

import jax.numpy as jnp
import numpy as np

from fdtdx import constants
from fdtdx.core.physics.farfield import (
    directivity_from_pattern,
    far_fields_from_NL,
    radiation_vectors,
    spherical_basis,
    surface_equivalent_currents,
)


def _plane_wave_surface(n_cells: int, dx: float):
    """Uniform +z plane wave on a z-normal plane: E=(E0,0,0), H=(0,E0,0) (eta0-normalized)."""
    e0 = 1.0
    shape = (3, n_cells, n_cells, 1)
    E = jnp.zeros(shape, dtype=jnp.complex64).at[0].set(e0)
    H = jnp.zeros(shape, dtype=jnp.complex64).at[1].set(e0)
    coords = (jnp.arange(n_cells) - n_cells / 2) * dx
    xx, yy = jnp.meshgrid(coords, coords, indexing="ij")
    positions = jnp.stack([xx, yy, jnp.zeros_like(xx)], axis=-1)[:, :, None, :]  # (n,n,1,3)
    area = dx * dx
    return E, H, positions, area


class TestSurfaceCurrents:
    def test_plane_wave_currents(self):
        # n_hat = +z. J = z x H = z x (0,E0,0) = -E0 x_hat ; M = -z x E = -z x (E0,0,0) = -E0 y_hat
        E, H, _, _ = _plane_wave_surface(4, 1.0)
        J, M = surface_equivalent_currents(E, H, normal_axis=2, outward_sign=1.0)
        assert np.allclose(J[0], -1.0) and np.allclose(J[1], 0.0) and np.allclose(J[2], 0.0)
        assert np.allclose(M[0], 0.0) and np.allclose(M[1], -1.0) and np.allclose(M[2], 0.0)


class TestPlaneWaveConventionLock:
    def test_lobe_peaks_at_normal_and_is_theta_polarized(self):
        wavelength = 1.0
        k = 2 * jnp.pi / wavelength
        # plane ~6 lambda wide so the main lobe is narrow
        E, H, positions, area = _plane_wave_surface(n_cells=60, dx=wavelength / 10)
        theta = jnp.linspace(0.0, jnp.radians(70.0), 36)
        phi = jnp.zeros_like(theta)
        r_hat, theta_hat, phi_hat = spherical_basis(theta, phi)
        N, L = radiation_vectors(E, H, positions, area, normal_axis=2, outward_sign=1.0, k=k, r_hat=r_hat)
        e_theta, e_phi = far_fields_from_NL(N, L, theta_hat, phi_hat, k=k, r=1.0, n_background=1.0)
        intensity = np.asarray(jnp.abs(e_theta) ** 2 + jnp.abs(e_phi) ** 2)

        # Forward (+z, theta=0) lobe dominates.
        assert int(np.argmax(intensity)) == 0
        assert intensity[0] > 50 * intensity[-1]
        # Polarisation: incident E_x -> theta-polarised at theta=0 (phi=0 plane), E_phi ~ 0.
        assert float(jnp.abs(e_phi[0])) < 1e-3 * float(jnp.abs(e_theta[0]))

    def test_tilted_beam_steers_lobe_to_incidence_angle(self):
        # A +x-tilted plane wave (phasor phase exp(+j k sin(theta0) x)) should peak near
        # theta=+theta0 at phi=0. This pins the sign of the radiation-integral kernel.
        wavelength = 1.0
        k = 2 * jnp.pi / wavelength
        theta0 = jnp.radians(20.0)
        n_cells, dx = 80, wavelength / 10
        e0 = 1.0
        coords = (jnp.arange(n_cells) - n_cells / 2) * dx
        ramp = jnp.exp(1j * k * jnp.sin(theta0) * coords)[:, None, None]  # phase along x
        shape = (3, n_cells, n_cells, 1)
        E = jnp.zeros(shape, dtype=jnp.complex64).at[0].set(e0 * ramp)
        H = jnp.zeros(shape, dtype=jnp.complex64).at[1].set(e0 * ramp)
        xx, yy = jnp.meshgrid(coords, coords, indexing="ij")
        positions = jnp.stack([xx, yy, jnp.zeros_like(xx)], axis=-1)[:, :, None, :]

        theta = jnp.linspace(jnp.radians(-50.0), jnp.radians(50.0), 101)
        phi = jnp.where(theta >= 0, 0.0, jnp.pi)  # negative theta -> phi=pi side
        r_hat, theta_hat, phi_hat = spherical_basis(jnp.abs(theta), phi)
        N, L = radiation_vectors(E, H, positions, dx * dx, normal_axis=2, outward_sign=1.0, k=k, r_hat=r_hat)
        e_theta, e_phi = far_fields_from_NL(N, L, theta_hat, phi_hat, k=k, r=1.0, n_background=1.0)
        intensity = np.asarray(jnp.abs(e_theta) ** 2 + jnp.abs(e_phi) ** 2)
        peak_theta = float(theta[int(np.argmax(intensity))])
        assert np.isclose(peak_theta, float(theta0), atol=np.radians(3.0))


class TestDirectivity:
    def test_isotropic_directivity_is_one(self):
        theta = jnp.linspace(0.0, jnp.pi, 91)
        phi = jnp.linspace(0.0, 2 * jnp.pi, 73)
        u = jnp.ones((theta.size, phi.size))
        d = directivity_from_pattern(u, theta, phi)
        assert np.allclose(np.asarray(d), 1.0, atol=1e-6)

    def test_dipole_directivity_is_1p5(self):
        theta = jnp.linspace(0.0, jnp.pi, 361)
        phi = jnp.linspace(0.0, 2 * jnp.pi, 5)
        u = (jnp.sin(theta) ** 2)[:, None] * jnp.ones_like(phi)[None, :]
        d = directivity_from_pattern(u, theta, phi)
        assert np.isclose(float(jnp.max(d)), 1.5, rtol=0.02)


def test_eta0_constant_sanity():
    assert np.isclose(constants.eta0, 376.730, atol=0.1)
