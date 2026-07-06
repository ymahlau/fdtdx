from unittest.mock import MagicMock

import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import RectilinearGrid, UniformGrid
from fdtdx.core.misc import pad_fields
from fdtdx.core.physics.curl import curl_E, curl_H, interpolate_fields


def _make_config():
    return SimulationConfig(
        time=400e-15,
        grid=UniformGrid(spacing=1.0),
        courant_factor=0.99,
    )


def _make_nonuniform_config():
    grid = RectilinearGrid(
        x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0, 10.0]),
        y_edges=jnp.asarray([0.0, 1.0, 2.5, 5.0, 9.0]),
        z_edges=jnp.asarray([0.0, 1.0, 4.0, 8.0, 13.0]),
    )
    return SimulationConfig(
        time=400e-15,
        grid=grid,
        courant_factor=0.99,
    )


def _mock_objects(pml_objects=None):
    """Create a mock ObjectContainer with an optional list of PML objects."""
    objects = MagicMock()
    objects.pml_objects = pml_objects or []
    return objects


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

    curl_result, psi_updated = curl_E(
        config=_make_config(),
        E_pad=E_pad,
        psi_H={},
        objects=_mock_objects(),
        simulate_boundaries=True,
    )

    assert curl_result.shape == (3, 6, 6, 6)
    assert jnp.allclose(curl_result[2][:-1, :-1], -2.0, atol=0.1)
    assert not psi_updated  # Dictionary should remain empty


def test_curl_E_zero_field():
    """Test curl_E with zero field and non-periodic boundaries."""
    E = jnp.zeros((3, 4, 4, 4))
    E_pad = pad_fields(E, (False, False, False))

    curl_result, psi_updated = curl_E(
        config=_make_config(),
        E_pad=E_pad,
        psi_H={},
        objects=_mock_objects(),
        simulate_boundaries=True,
    )

    assert curl_result.shape == (3, 4, 4, 4)
    assert jnp.allclose(curl_result, 0.0)
    assert not psi_updated


def test_curl_E_pml_integration():
    """Verify curl_E calls PML objects correctly and scatters corrections."""
    n = 5
    config = _make_config()
    E = jnp.zeros((3, n, n, n))
    E_pad = pad_fields(E, (False, False, False))

    slice_obj = (slice(None), slice(None), slice(1, 4))

    # Mock a PML object positioned on the x-axis (a=0)
    pml = MagicMock()
    pml.name = "pml_x"
    pml.axis = 0
    pml.grid_slice = slice_obj

    # Fake values returned by `step_cpml`
    corr_1 = jnp.ones((n, n, 3)) * 0.1
    corr_2 = jnp.ones((n, n, 3)) * 0.2
    psi_1_new = jnp.ones((n, n, 3)) * 0.3
    psi_2_new = jnp.ones((n, n, 3)) * 0.4
    pml.step_cpml.return_value = (corr_1, corr_2, psi_1_new, psi_2_new)

    psi_1_init = jnp.zeros((n, n, 3))
    psi_2_init = jnp.zeros((n, n, 3))
    psi_H = {"pml_x": (psi_1_init, psi_2_init)}

    objects = _mock_objects(pml_objects=[pml])

    curl_result, psi_updated = curl_E(
        config=config,
        E_pad=E_pad,
        psi_H=psi_H,
        objects=objects,
        simulate_boundaries=False,
    )

    # Verify dictionary updates
    assert "pml_x" in psi_updated
    assert jnp.allclose(psi_updated["pml_x"][0], psi_1_new)
    assert jnp.allclose(psi_updated["pml_x"][1], psi_2_new)

    # Verify `step_cpml` inputs via args AND kwargs
    pml.step_cpml.assert_called_once()
    args, kwargs = pml.step_cpml.call_args
    assert jnp.array_equal(args[2], psi_1_init)  # psi_1
    assert jnp.array_equal(args[3], psi_2_init)  # psi_2
    assert kwargs.get("is_curl_E") is True  # Checked via kwargs
    assert kwargs.get("simulate_boundaries") is False

    # Verify scatter update: a=0 -> i=1 (y), j=2 (z)
    assert jnp.allclose(curl_result[1][slice_obj], -0.1)
    assert jnp.allclose(curl_result[2][slice_obj], 0.2)


def test_curl_E_mixed_periodic():
    """Test curl_E with mixed periodic axes."""
    n = 6
    x = jnp.arange(n, dtype=float)
    X, Y, _Z = jnp.meshgrid(x, x, x, indexing="ij")
    E = jnp.stack([Y, -X, jnp.zeros_like(X)], axis=0)
    E_pad = pad_fields(E, (True, False, True))

    curl_result, _ = curl_E(
        config=_make_config(),
        E_pad=E_pad,
        psi_H={},
        objects=_mock_objects(),
        simulate_boundaries=True,
    )

    assert curl_result.shape == (3, n, n, n)
    assert jnp.all(jnp.isfinite(curl_result))
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

    curl_result, _ = curl_E(
        config=config,
        E_pad=E_pad,
        psi_H={},
        objects=_mock_objects(),
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

    curl_result, _ = curl_E(
        config=config,
        E_pad=E_pad,
        psi_H={},
        objects=_mock_objects(),
        simulate_boundaries=False,
    )

    expected_y = y[:-1] + y[1:]
    expected_z = z[:-1] + z[1:]
    expected = expected_y[None, :, None] - expected_z[None, None, :]
    assert jnp.allclose(curl_result[0][:, :-1, :-1], expected, atol=1e-6)


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

    curl_result, psi_updated = curl_H(
        config=_make_config(),
        H_pad=H_pad,
        psi_E={},
        objects=_mock_objects(),
        simulate_boundaries=True,
    )

    assert curl_result.shape == (3, 6, 6, 6)
    assert jnp.allclose(curl_result[1][1:-1, 1:-1, 1:-1], 2.0, atol=0.1)
    assert not psi_updated


def test_curl_H_zero_field():
    """Test curl_H with zero field and non-periodic boundaries."""
    H = jnp.zeros((3, 4, 4, 4))
    H_pad = pad_fields(H, (False, False, False))

    curl_result, psi_updated = curl_H(
        config=_make_config(),
        H_pad=H_pad,
        psi_E={},
        objects=_mock_objects(),
        simulate_boundaries=True,
    )

    assert curl_result.shape == (3, 4, 4, 4)
    assert jnp.allclose(curl_result, 0.0)
    assert not psi_updated


def test_curl_H_pml_integration():
    """Verify curl_H calls PML objects correctly and scatters corrections."""
    n = 5
    config = _make_config()
    H = jnp.zeros((3, n, n, n))
    H_pad = pad_fields(H, (False, False, False))

    slice_obj = (slice(None), slice(None), slice(1, 4))

    # Mock a PML object positioned on the y-axis (a=1)
    pml = MagicMock()
    pml.name = "pml_y"
    pml.axis = 1
    pml.grid_slice = slice_obj

    corr_1 = jnp.ones((n, n, 3)) * 0.5
    corr_2 = jnp.ones((n, n, 3)) * 0.6
    psi_1_new = jnp.ones((n, n, 3)) * 0.7
    psi_2_new = jnp.ones((n, n, 3)) * 0.8
    pml.step_cpml.return_value = (corr_1, corr_2, psi_1_new, psi_2_new)

    psi_1_init = jnp.zeros((n, n, 3))
    psi_2_init = jnp.zeros((n, n, 3))
    psi_E = {"pml_y": (psi_1_init, psi_2_init)}

    objects = _mock_objects(pml_objects=[pml])

    curl_result, psi_updated = curl_H(
        config=config,
        H_pad=H_pad,
        psi_E=psi_E,
        objects=objects,
        simulate_boundaries=True,
    )

    # Verify dictionary updates
    assert "pml_y" in psi_updated
    assert jnp.allclose(psi_updated["pml_y"][0], psi_1_new)
    assert jnp.allclose(psi_updated["pml_y"][1], psi_2_new)

    # Verify `step_cpml` inputs via args AND kwargs
    pml.step_cpml.assert_called_once()
    args, kwargs = pml.step_cpml.call_args
    assert jnp.array_equal(args[2], psi_1_init)
    assert jnp.array_equal(args[3], psi_2_init)
    assert kwargs.get("is_curl_E") is False  # Checked via kwargs
    assert kwargs.get("simulate_boundaries") is True

    # Verify scatter update: a=1 -> i=2 (z), j=0 (x)
    assert jnp.allclose(curl_result[2][slice_obj], -0.5)
    assert jnp.allclose(curl_result[0][slice_obj], 0.6)


def test_curl_H_mixed_periodic():
    """Test curl_H with mixed periodic axes."""
    n = 6
    x = jnp.arange(n, dtype=float)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    H = jnp.stack([Z, jnp.zeros_like(Y), -X], axis=0)
    H_pad = pad_fields(H, (False, True, False))

    curl_result, _ = curl_H(
        config=_make_config(),
        H_pad=H_pad,
        psi_E={},
        objects=_mock_objects(),
        simulate_boundaries=True,
    )

    assert curl_result.shape == (3, n, n, n)
    assert jnp.all(jnp.isfinite(curl_result))


def test_curl_H_nonuniform_metric_no_boundaries():
    """Backward-difference H curls recover linear physical derivatives at cell centers."""
    config = _make_nonuniform_config()
    nx, ny, nz = config.grid.shape
    x_centers = config.grid.centers(0)
    z_centers = config.grid.centers(2)
    X, _Y, Z = jnp.meshgrid(x_centers, config.grid.y_edges[:-1], z_centers, indexing="ij")

    H = jnp.stack([Z, jnp.zeros((nx, ny, nz), dtype=jnp.float32), -X], axis=0)
    H_pad = pad_fields(H, (False, False, False))

    curl_result, _ = curl_H(
        config=config,
        H_pad=H_pad,
        psi_E={},
        objects=_mock_objects(),
        simulate_boundaries=False,
    )

    assert curl_result.shape == (3, nx, ny, nz)
    assert jnp.allclose(curl_result[1][1:, :, 1:], 2.0, atol=1e-6)


def test_curl_H_nonuniform_quadratic_field_matches_local_physical_derivative():
    """Backward Yee derivatives use the inter-center distance (dy[j]+dy[j-1])/2."""
    config = _make_nonuniform_config()
    nx, ny, nz = config.grid.shape
    y_centers = config.grid.centers(1)
    z_centers = config.grid.centers(2)
    _X, Y, Z = jnp.meshgrid(config.grid.x_edges[:-1], y_centers, z_centers, indexing="ij")

    H = jnp.stack(
        [
            jnp.zeros((nx, ny, nz), dtype=jnp.float32),
            Z**2,
            Y**2,
        ],
        axis=0,
    )
    H_pad = pad_fields(H, (False, False, False))

    curl_result, _ = curl_H(
        config=config,
        H_pad=H_pad,
        psi_E={},
        objects=_mock_objects(),
        simulate_boundaries=False,
    )

    expected_y = y_centers[1:] + y_centers[:-1]
    expected_z = z_centers[1:] + z_centers[:-1]
    expected = expected_y[None, :, None] - expected_z[None, None, :]
    assert jnp.allclose(curl_result[0][:, 1:, 1:], expected, atol=1e-6)
