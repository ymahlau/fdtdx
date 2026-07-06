"""Tests for field-projection helper functions."""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.core.physics.field_projection import (
    cartesian_to_spherical_angles,
    edge_window_1d,
    spherical_basis_grid,
    spherical_basis_paired,
    subsample_indices,
    trapezoidal_weights_1d,
)

pytestmark = pytest.mark.unit


def test_spherical_basis_grid_matches_global_directions():
    theta = jnp.asarray([0.0, jnp.pi / 2])
    phi = jnp.asarray([0.0, jnp.pi / 2])

    radial, theta_hat, phi_hat = spherical_basis_grid(theta, phi)

    assert radial.shape == (3, 2, 2)
    assert jnp.allclose(radial[:, 0, 0], jnp.asarray([0.0, 0.0, 1.0]), atol=1e-6)
    assert jnp.allclose(radial[:, 1, 0], jnp.asarray([1.0, 0.0, 0.0]), atol=1e-6)
    assert jnp.allclose(theta_hat[:, 1, 0], jnp.asarray([0.0, 0.0, -1.0]), atol=1e-6)
    assert jnp.allclose(phi_hat[:, 1, 1], jnp.asarray([-1.0, 0.0, 0.0]), atol=1e-6)


def test_spherical_basis_paired_is_orthonormal():
    theta = jnp.asarray([0.2, 0.7, 1.1])
    phi = jnp.asarray([0.3, -0.4, 1.2])

    radial, theta_hat, phi_hat = spherical_basis_paired(theta, phi)

    assert radial.shape == (3, 3)
    assert jnp.allclose(jnp.sum(radial * theta_hat, axis=0), 0.0, atol=1e-6)
    assert jnp.allclose(jnp.sum(radial * phi_hat, axis=0), 0.0, atol=1e-6)
    assert jnp.allclose(jnp.sum(theta_hat * phi_hat, axis=0), 0.0, atol=1e-6)
    assert jnp.allclose(jnp.linalg.norm(radial, axis=0), 1.0, atol=1e-6)


def test_cartesian_to_spherical_angles_handles_origin_and_recovers_axes():
    points = jnp.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    radial_distance, theta, phi = cartesian_to_spherical_angles(points)

    assert jnp.allclose(radial_distance, jnp.asarray([0.0, 1.0, 1.0, 1.0]))
    assert jnp.all(jnp.isfinite(theta))
    assert jnp.all(jnp.isfinite(phi))
    assert jnp.allclose(theta[1:], jnp.asarray([0.0, jnp.pi / 2, jnp.pi / 2]), atol=1e-6)
    assert jnp.allclose(phi[1:], jnp.asarray([0.0, 0.0, jnp.pi / 2]), atol=1e-6)


def test_trapezoidal_weights_and_subsample_indices_keep_physical_endpoints():
    points = jnp.asarray([0.0, 0.25, 1.0, 2.0])

    weights = trapezoidal_weights_1d(points)
    indices = subsample_indices(num_points=6, interval=4)
    existing_endpoint_indices = subsample_indices(num_points=10, interval=3)

    assert jnp.allclose(weights, jnp.asarray([0.125, 0.5, 0.875, 0.5]))
    assert jnp.array_equal(trapezoidal_weights_1d(jnp.asarray([2.0])), jnp.asarray([1.0]))
    assert jnp.array_equal(indices, jnp.asarray([0, 4, 5]))
    assert jnp.array_equal(existing_endpoint_indices, jnp.asarray([0, 3, 6, 9]))


def test_edge_window_is_jittable_and_suppresses_only_edges():
    points = jnp.linspace(-1.0, 1.0, 5)

    eager = edge_window_1d(points, 0.8)
    jitted = jax.jit(edge_window_1d, static_argnames=("window_size",))(points, 0.8)

    assert jnp.allclose(jitted, eager)
    assert eager[2] == 1.0
    assert eager[0] < eager[1] < eager[2]
