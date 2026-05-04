import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import GridSpec
from fdtdx.core.misc import pad_fields
from fdtdx.core.physics.curl import curl_E, curl_H, interpolate_fields


def _make_config():
    return SimulationConfig(
        time=400e-15,
        resolution=1.0,
        courant_factor=0.99,
    )


def _make_nonuniform_config():
    grid = GridSpec(
        x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0, 10.0]),
        y_edges=jnp.asarray([0.0, 1.0, 2.5, 5.0, 9.0]),
        z_edges=jnp.asarray([0.0, 1.0, 4.0, 8.0, 13.0]),
    )
    return SimulationConfig(
        time=400e-15,
        resolution=1.0,
        grid=grid,
        courant_factor=0.99,
    )


# ──────────────────────────────────────────────────────────────
# interpolate_fields
# ──────────────────────────────────────────────────────────────


def test_interpolate_fields_basic():
    """Test basic interpolation with uniform fields (PEC boundaries)."""
    E_field = jnp.ones((3, 5, 5, 5))
    H_field = jnp.ones((3, 5, 5, 5)) * 0.5

    E_pad = pad_fields(E_field, (False, False, False))
    H_pad = pad_fields(H_field, (False, False, False))
    E_interp, H_interp = interpolate_fields(E_pad, H_pad)

    assert E_interp.shape == (3, 5, 5, 5)
    assert H_interp.shape == (3, 5, 5, 5)
    # E_z should remain unchanged (interpolation projects onto E_z)
    assert jnp.all(E_interp[2] == 1.0)
    assert jnp.all(jnp.isfinite(E_interp))
    assert jnp.all(jnp.isfinite(H_interp))


def test_interpolate_fields_periodic_boundaries():
    """Test interpolation with all-periodic boundary conditions."""
    x = jnp.linspace(0, 2 * jnp.pi, 6)
    y = jnp.linspace(0, 2 * jnp.pi, 6)
    z = jnp.linspace(0, 2 * jnp.pi, 6)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    E_field = jnp.stack([jnp.sin(X), jnp.cos(Y), jnp.sin(Z)], axis=0)
    H_field = jnp.stack([jnp.cos(X), jnp.sin(Y), jnp.cos(Z)], axis=0)

    E_pad = pad_fields(E_field, (True, True, True))
    H_pad = pad_fields(H_field, (True, True, True))
    E_interp, H_interp = interpolate_fields(E_pad, H_pad)

    assert E_interp.shape == (3, 6, 6, 6)
    assert H_interp.shape == (3, 6, 6, 6)
    assert jnp.all(jnp.isfinite(E_interp))
    assert jnp.all(jnp.isfinite(H_interp))


def test_interpolate_fields_mixed_boundaries():
    """Test interpolation with mixed periodic/PEC boundary conditions."""
    E_field = jnp.ones((3, 6, 6, 6))
    H_field = jnp.ones((3, 6, 6, 6)) * 2.0

    E_pad = pad_fields(E_field, (True, False, False))
    H_pad = pad_fields(H_field, (True, False, False))
    E_interp, H_interp = interpolate_fields(E_pad, H_pad)

    assert E_interp.shape == (3, 6, 6, 6)
    assert H_interp.shape == (3, 6, 6, 6)
    assert jnp.all(jnp.isfinite(E_interp))
    assert jnp.all(jnp.isfinite(H_interp))


def test_interpolate_fields_zero_fields():
    """Test interpolation with zero input fields."""
    E_field = jnp.zeros((3, 4, 4, 4))
    H_field = jnp.zeros((3, 4, 4, 4))

    E_pad = pad_fields(E_field, (False, False, False))
    H_pad = pad_fields(H_field, (False, False, False))
    E_interp, H_interp = interpolate_fields(E_pad, H_pad)

    assert E_interp.shape == (3, 4, 4, 4)
    assert H_interp.shape == (3, 4, 4, 4)
    assert jnp.allclose(E_interp, 0.0)
    assert jnp.allclose(H_interp, 0.0)


def test_interpolate_fields_nonuniform_center_to_edge_weights():
    """Distance-weighted interpolation recovers linear fields on stretched cells."""
    config = _make_nonuniform_config()
    nx, ny, nz = config.grid.shape
    x_centers = config.grid.centers(0)
    z_edges = config.grid.z_edges[:-1]
    Xc, _Y, Ze = jnp.meshgrid(x_centers, config.grid.y_edges[:-1], z_edges, indexing="ij")

    E = jnp.zeros((3, nx, ny, nz), dtype=jnp.float32)
    E = E.at[0].set(Xc + Ze)
    H = jnp.zeros((3, nx, ny, nz), dtype=jnp.float32)

    E_pad = pad_fields(E, (False, False, False))
    H_pad = pad_fields(H, (False, False, False))
    E_interp, _ = interpolate_fields(E_pad, H_pad, config=config)

    target_x_edges = config.grid.x_edges[:-1]
    target_z_centers = config.grid.centers(2)
    X_edge, _Y_edge, Z_center = jnp.meshgrid(target_x_edges, config.grid.y_edges[:-1], target_z_centers, indexing="ij")
    expected = X_edge + Z_center

    assert E_interp.shape == (3, nx, ny, nz)
    assert jnp.allclose(E_interp[0][1:, :, :-1], expected[1:, :, :-1], atol=1e-6)


# ──────────────────────────────────────────────────────────────
# curl_E
# ──────────────────────────────────────────────────────────────


def test_curl_E_linear_field():
    """Test curl_E with E_x=y, E_y=-x gives curl_z=-2."""
    nx, ny, nz = 6, 6, 6
    x = jnp.arange(nx)
    y = jnp.arange(ny)
    z = jnp.arange(nz)
    X, Y, _Z = jnp.meshgrid(x, y, z, indexing="ij")

    E = jnp.stack([Y.astype(float), -X.astype(float), jnp.zeros_like(X, dtype=float)], axis=0)
    E_pad = pad_fields(E, (True, True, True))
    psi_H = jnp.zeros((6, 6, 6, 6))

    curl_result, _ = curl_E(
        _make_config(),
        E_pad,
        psi_H,
        alpha=jnp.zeros((6, 6, 6, 6)),
        kappa=jnp.ones((6, 6, 6, 6)),
        sigma=jnp.zeros((6, 6, 6, 6)),
        simulate_boundaries=True,
    )

    assert curl_result.shape == (3, 6, 6, 6)
    assert jnp.allclose(curl_result[2][:-1, :-1], -2.0, atol=0.1)


def test_curl_E_zero_field():
    """Test curl_E with zero field and non-periodic boundaries."""
    E = jnp.zeros((3, 4, 4, 4))
    E_pad = pad_fields(E, (False, False, False))
    psi_H = jnp.zeros((6, 4, 4, 4))

    curl_result, _ = curl_E(
        _make_config(),
        E_pad,
        psi_H,
        alpha=jnp.zeros((6, 4, 4, 4)),
        kappa=jnp.ones((6, 4, 4, 4)),
        sigma=jnp.zeros((6, 4, 4, 4)),
        simulate_boundaries=True,
    )

    assert curl_result.shape == (3, 4, 4, 4)
    assert jnp.allclose(curl_result, 0.0)


def test_curl_E_no_boundaries():
    """Test curl_E with simulate_boundaries=False preserves psi unchanged."""
    n = 5
    E = jnp.ones((3, n, n, n)) * 0.5
    E_pad = pad_fields(E, (True, True, True))
    psi_H_init = jnp.ones((6, n, n, n)) * 0.3
    kappa = jnp.ones((6, n, n, n))
    sigma = jnp.ones((6, n, n, n)) * 0.1

    curl_result, psi_updated = curl_E(
        _make_config(),
        E_pad,
        psi_H_init,
        alpha=jnp.ones((6, n, n, n)) * 0.05,
        kappa=kappa,
        sigma=sigma,
        simulate_boundaries=False,
    )

    assert curl_result.shape == (3, n, n, n)
    assert jnp.all(jnp.isfinite(curl_result))
    # psi should remain unchanged when boundaries are not simulated
    assert jnp.allclose(psi_updated, psi_H_init)


def test_curl_E_pml_updates():
    """Test curl_E PML updates psi with non-trivial sigma/alpha."""
    n = 5
    # E_z with gradient in y → non-zero dyEz → psi_Hxy gets updated
    # E_x with gradient in y → non-zero dyEx → psi_Hzy gets updated
    ramp = jnp.linspace(0, 1, n)
    E = jnp.zeros((3, n, n, n))
    E = E.at[0].set(ramp.reshape(1, -1, 1))  # E_x varies in y
    E = E.at[2].set(ramp.reshape(1, -1, 1))  # E_z varies in y
    E_pad = pad_fields(E, (True, True, True))
    psi_H_init = jnp.zeros((6, n, n, n))
    sigma = jnp.ones((6, n, n, n)) * 0.5
    alpha = jnp.ones((6, n, n, n)) * 0.1

    curl_result, psi_updated = curl_E(
        _make_config(),
        E_pad,
        psi_H_init,
        alpha=alpha,
        kappa=jnp.ones((6, n, n, n)),
        sigma=sigma,
        simulate_boundaries=True,
    )

    assert curl_result.shape == (3, n, n, n)
    assert psi_updated.shape == (6, n, n, n)
    assert jnp.all(jnp.isfinite(curl_result))
    assert jnp.all(jnp.isfinite(psi_updated))
    # With non-zero sigma and gradient field, psi must change from zero
    assert not jnp.allclose(psi_updated, 0.0)


def test_curl_E_mixed_periodic():
    """Test curl_E with mixed periodic axes."""
    n = 6
    x = jnp.arange(n, dtype=float)
    X, Y, _Z = jnp.meshgrid(x, x, x, indexing="ij")
    E = jnp.stack([Y, -X, jnp.zeros_like(X)], axis=0)
    E_pad = pad_fields(E, (True, False, True))
    psi_H = jnp.zeros((6, n, n, n))

    curl_result, _ = curl_E(
        _make_config(),
        E_pad,
        psi_H,
        alpha=jnp.zeros((6, n, n, n)),
        kappa=jnp.ones((6, n, n, n)),
        sigma=jnp.zeros((6, n, n, n)),
        simulate_boundaries=True,
    )

    assert curl_result.shape == (3, n, n, n)
    assert jnp.all(jnp.isfinite(curl_result))
    # Interior z-component should still approximate -2
    assert jnp.allclose(curl_result[2][:-1, 1:-1, :-1], -2.0, atol=0.1)


def test_curl_E_nonuniform_metric_no_boundaries():
    """Local metric factors recover the physical curl of linear fields."""
    config = _make_nonuniform_config()
    nx, ny, nz = config.grid.shape
    x = config.grid.x_edges[:-1]
    y = config.grid.y_edges[:-1]
    X, Y, _Z = jnp.meshgrid(x, y, config.grid.z_edges[:-1], indexing="ij")

    E = jnp.stack([Y, -X, jnp.zeros((nx, ny, nz), dtype=jnp.float32)], axis=0)
    E_pad = pad_fields(E, (False, False, False))
    psi_H = jnp.zeros((6, nx, ny, nz))

    curl_result, _ = curl_E(
        config,
        E_pad,
        psi_H,
        alpha=jnp.zeros((6, nx, ny, nz)),
        kappa=jnp.ones((6, nx, ny, nz)),
        sigma=jnp.zeros((6, nx, ny, nz)),
        simulate_boundaries=False,
    )

    assert curl_result.shape == (3, nx, ny, nz)
    assert jnp.allclose(curl_result[2][:-1, :-1, :], -2.0, atol=1e-6)


def test_curl_E_nonuniform_quadratic_field_matches_local_physical_derivative():
    """Forward Yee derivatives use local stretched-cell widths, not index spacing."""
    config = _make_nonuniform_config()
    nx, ny, nz = config.grid.shape
    y = config.grid.y_edges[:-1]
    z = config.grid.z_edges[:-1]
    _X, Y, Z = jnp.meshgrid(config.grid.x_edges[:-1], y, z, indexing="ij")

    E = jnp.stack(
        [
            jnp.zeros((nx, ny, nz), dtype=jnp.float32),
            Z**2,
            Y**2,
        ],
        axis=0,
    )
    E_pad = pad_fields(E, (False, False, False))
    psi_H = jnp.zeros((6, nx, ny, nz))

    curl_result, _ = curl_E(
        config,
        E_pad,
        psi_H,
        alpha=jnp.zeros((6, nx, ny, nz)),
        kappa=jnp.ones((6, nx, ny, nz)),
        sigma=jnp.zeros((6, nx, ny, nz)),
        simulate_boundaries=False,
    )

    expected_y = y[:-1] + y[1:]
    expected_z = z[:-1] + z[1:]
    expected = expected_y[None, :, None] - expected_z[None, None, :]
    assert jnp.allclose(curl_result[0][:, :-1, :-1], expected, atol=1e-6)


def test_curl_E_nonuniform_pml_coefficients_use_time_step():
    """PML auxiliary coefficients do not require a uniform grid spacing."""
    config = _make_nonuniform_config()
    nx, ny, nz = config.grid.shape
    E = jnp.ones((3, nx, ny, nz), dtype=jnp.float32)
    E = E.at[2].set(jnp.arange(ny, dtype=jnp.float32).reshape(1, ny, 1))
    E_pad = pad_fields(E, (False, False, False))
    psi_H = jnp.zeros((6, nx, ny, nz))

    curl_result, psi_updated = curl_E(
        config,
        E_pad,
        psi_H,
        alpha=jnp.ones((6, nx, ny, nz)) * 0.05,
        kappa=jnp.ones((6, nx, ny, nz)),
        sigma=jnp.ones((6, nx, ny, nz)) * 0.1,
        simulate_boundaries=True,
    )

    assert jnp.all(jnp.isfinite(curl_result))
    assert jnp.all(jnp.isfinite(psi_updated))


# ──────────────────────────────────────────────────────────────
# curl_H
# ──────────────────────────────────────────────────────────────


def test_curl_H_linear_field():
    """Test curl_H with H_x=z, H_z=-x gives curl_y=2."""
    nx, ny, nz = 6, 6, 6
    x = jnp.arange(nx)
    y = jnp.arange(ny)
    z = jnp.arange(nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    H = jnp.stack([Z.astype(float), jnp.zeros_like(Y, dtype=float), -X.astype(float)], axis=0)
    H_pad = pad_fields(H, (True, True, True))
    psi_E = jnp.zeros((6, 6, 6, 6))

    curl_result, _ = curl_H(
        _make_config(),
        H_pad,
        psi_E,
        alpha=jnp.zeros((6, 6, 6, 6)),
        kappa=jnp.ones((6, 6, 6, 6)),
        sigma=jnp.zeros((6, 6, 6, 6)),
        simulate_boundaries=True,
    )

    assert curl_result.shape == (3, 6, 6, 6)
    assert jnp.allclose(curl_result[1][1:-1, 1:-1, 1:-1], 2.0, atol=0.1)


def test_curl_H_zero_field():
    """Test curl_H with zero field and non-periodic boundaries."""
    H = jnp.zeros((3, 4, 4, 4))
    H_pad = pad_fields(H, (False, False, False))
    psi_E = jnp.zeros((6, 4, 4, 4))

    curl_result, _ = curl_H(
        _make_config(),
        H_pad,
        psi_E,
        alpha=jnp.zeros((6, 4, 4, 4)),
        kappa=jnp.ones((6, 4, 4, 4)),
        sigma=jnp.zeros((6, 4, 4, 4)),
        simulate_boundaries=True,
    )

    assert curl_result.shape == (3, 4, 4, 4)
    assert jnp.allclose(curl_result, 0.0)


def test_curl_H_no_boundaries():
    """Test curl_H with simulate_boundaries=False preserves psi unchanged."""
    n = 5
    H = jnp.ones((3, n, n, n)) * 0.5
    H_pad = pad_fields(H, (True, True, True))
    psi_E_init = jnp.ones((6, n, n, n)) * 0.3
    sigma = jnp.ones((6, n, n, n)) * 0.1

    curl_result, psi_updated = curl_H(
        _make_config(),
        H_pad,
        psi_E_init,
        alpha=jnp.ones((6, n, n, n)) * 0.05,
        kappa=jnp.ones((6, n, n, n)),
        sigma=sigma,
        simulate_boundaries=False,
    )

    assert curl_result.shape == (3, n, n, n)
    assert jnp.all(jnp.isfinite(curl_result))
    # psi should remain unchanged when boundaries are not simulated
    assert jnp.allclose(psi_updated, psi_E_init)


def test_curl_H_pml_updates():
    """Test curl_H PML updates psi with non-trivial sigma/alpha."""
    n = 5
    H = jnp.ones((3, n, n, n))
    H = H.at[2].set(jnp.linspace(0, 1, n).reshape(1, -1, 1))  # gradient in y
    H_pad = pad_fields(H, (True, True, True))
    psi_E_init = jnp.zeros((6, n, n, n))
    sigma = jnp.ones((6, n, n, n)) * 0.5
    alpha = jnp.ones((6, n, n, n)) * 0.1

    curl_result, psi_updated = curl_H(
        _make_config(),
        H_pad,
        psi_E_init,
        alpha=alpha,
        kappa=jnp.ones((6, n, n, n)),
        sigma=sigma,
        simulate_boundaries=True,
    )

    assert curl_result.shape == (3, n, n, n)
    assert psi_updated.shape == (6, n, n, n)
    assert jnp.all(jnp.isfinite(curl_result))
    assert jnp.all(jnp.isfinite(psi_updated))
    # With non-zero sigma and a gradient field, psi must change from zero
    assert not jnp.allclose(psi_updated, 0.0)


def test_curl_H_mixed_periodic():
    """Test curl_H with mixed periodic axes."""
    n = 6
    x = jnp.arange(n, dtype=float)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    H = jnp.stack([Z, jnp.zeros_like(Y), -X], axis=0)
    H_pad = pad_fields(H, (False, True, False))
    psi_E = jnp.zeros((6, n, n, n))

    curl_result, _ = curl_H(
        _make_config(),
        H_pad,
        psi_E,
        alpha=jnp.zeros((6, n, n, n)),
        kappa=jnp.ones((6, n, n, n)),
        sigma=jnp.zeros((6, n, n, n)),
        simulate_boundaries=True,
    )

    assert curl_result.shape == (3, n, n, n)
    assert jnp.all(jnp.isfinite(curl_result))


def test_curl_H_nonuniform_metric_no_boundaries():
    """Backward-difference H curls use the local rectilinear metric."""
    config = _make_nonuniform_config()
    nx, ny, nz = config.grid.shape
    x = config.grid.x_edges[:-1]
    z = config.grid.z_edges[:-1]
    X, _Y, Z = jnp.meshgrid(x, config.grid.y_edges[:-1], z, indexing="ij")

    H = jnp.stack([Z, jnp.zeros((nx, ny, nz), dtype=jnp.float32), -X], axis=0)
    H_pad = pad_fields(H, (False, False, False))
    psi_E = jnp.zeros((6, nx, ny, nz))

    curl_result, _ = curl_H(
        config,
        H_pad,
        psi_E,
        alpha=jnp.zeros((6, nx, ny, nz)),
        kappa=jnp.ones((6, nx, ny, nz)),
        sigma=jnp.zeros((6, nx, ny, nz)),
        simulate_boundaries=False,
    )

    assert curl_result.shape == (3, nx, ny, nz)
    assert jnp.allclose(curl_result[1][1:, :, 1:], 2.0, atol=1e-6)


def test_curl_H_nonuniform_quadratic_field_matches_local_physical_derivative():
    """Backward Yee derivatives use previous stretched-cell widths."""
    config = _make_nonuniform_config()
    nx, ny, nz = config.grid.shape
    y = config.grid.y_edges[:-1]
    z = config.grid.z_edges[:-1]
    _X, Y, Z = jnp.meshgrid(config.grid.x_edges[:-1], y, z, indexing="ij")

    H = jnp.stack(
        [
            jnp.zeros((nx, ny, nz), dtype=jnp.float32),
            Z**2,
            Y**2,
        ],
        axis=0,
    )
    H_pad = pad_fields(H, (False, False, False))
    psi_E = jnp.zeros((6, nx, ny, nz))

    curl_result, _ = curl_H(
        config,
        H_pad,
        psi_E,
        alpha=jnp.zeros((6, nx, ny, nz)),
        kappa=jnp.ones((6, nx, ny, nz)),
        sigma=jnp.zeros((6, nx, ny, nz)),
        simulate_boundaries=False,
    )

    expected_y = y[1:] + y[:-1]
    expected_z = z[1:] + z[:-1]
    expected = expected_y[None, :, None] - expected_z[None, None, :]
    assert jnp.allclose(curl_result[0][:, 1:, 1:], expected, atol=1e-6)


def test_curl_H_nonuniform_pml_coefficients_use_time_step():
    """H-to-E PML auxiliary coefficients share the nonuniform-safe dt path."""
    config = _make_nonuniform_config()
    nx, ny, nz = config.grid.shape
    H = jnp.ones((3, nx, ny, nz), dtype=jnp.float32)
    H = H.at[2].set(jnp.arange(ny, dtype=jnp.float32).reshape(1, ny, 1))
    H_pad = pad_fields(H, (False, False, False))
    psi_E = jnp.zeros((6, nx, ny, nz))

    curl_result, psi_updated = curl_H(
        config,
        H_pad,
        psi_E,
        alpha=jnp.ones((6, nx, ny, nz)) * 0.05,
        kappa=jnp.ones((6, nx, ny, nz)),
        sigma=jnp.ones((6, nx, ny, nz)) * 0.1,
        simulate_boundaries=True,
    )

    assert jnp.all(jnp.isfinite(curl_result))
    assert jnp.all(jnp.isfinite(psi_updated))
