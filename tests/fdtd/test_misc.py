# tests/fdtd/test_misc.py

import jax.numpy as jnp
import pytest

import fdtdx.fdtd.misc as misc


class MockPML:
    """Minimal mock of PerfectlyMatchedLayer matching the API used in misc.py."""

    def __init__(self, name="pml1"):
        self.name = name

    def boundary_interface_slice(self):
        # Matches usage arr[:, *pml.boundary_interface_slice()]
        # Select the first column (so arr[:, 0:1])
        return (slice(0, 1),)

    def interface_slice(self):
        return (slice(0, 1),)


class MockArrayContainer:
    """Mock replacement for ArrayContainer.

    - Fields (E, H) are plain jnp arrays.
    - Supports arrays.at["E"].set(...) to set the whole field (used in add_boundary_interfaces).
    - Slicing and arr.at[..., ...].set(...) for individual jnp arrays is delegated to jax.numpy arrays.
    """

    def __init__(self, E, H):
        self.E = E
        self.H = H

    class _AtHelper:
        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, key):
            # Expect key to be the field name string, e.g. "E" or "H"
            if not isinstance(key, str):
                raise ValueError("MockArrayContainer.at expects a field-name string key")

            parent = self.parent
            field = key

            class Setter:
                def __init__(self, parent, field):
                    self.parent = parent
                    self.field = field

                def set(self, v):
                    # replace the whole field array
                    setattr(self.parent, self.field, v)
                    return self.parent

            return Setter(parent, field)

    @property
    def at(self):
        return MockArrayContainer._AtHelper(self)


@pytest.fixture
def mock_arraycontainer():
    return MockArrayContainer


@pytest.fixture
def mock_pml():
    # return a list because misc functions expect a sequence of PML objects
    return [MockPML("pml1")]


def test_collect_boundary_interfaces(mock_arraycontainer, mock_pml):
    arrays = mock_arraycontainer(E=jnp.array([[1, 2], [3, 4]]), H=jnp.array([[5, 6], [7, 8]]))

    result = misc.collect_boundary_interfaces(arrays, mock_pml)

    assert "pml1_E" in result
    assert "pml1_H" in result

    # pml.boundary_interface_slice() → (slice(0,1),)
    expected_E = arrays.E[:, 0:1]
    expected_H = arrays.H[:, 0:1]

    assert jnp.array_equal(result["pml1_E"], expected_E)
    assert jnp.array_equal(result["pml1_H"], expected_H)


def test_add_boundary_interfaces(mock_arraycontainer, mock_pml):
    arrays = mock_arraycontainer(E=jnp.array([[1, 2], [3, 4]]), H=jnp.array([[5, 6], [7, 8]]))

    values = {"pml1_E": jnp.array([[9], [10]]), "pml1_H": jnp.array([[11], [12]])}

    updated = misc.add_boundary_interfaces(arrays, values, mock_pml)

    # boundary slices should be updated
    assert jnp.array_equal(updated.E[:, 0:1], values["pml1_E"])
    assert jnp.array_equal(updated.H[:, 0:1], values["pml1_H"])

    # non-boundary entries should remain unchanged
    assert jnp.array_equal(updated.E[:, 1:], jnp.array([[2], [4]]))
    assert jnp.array_equal(updated.H[:, 1:], jnp.array([[6], [8]]))


def test_round_trip_collect_add(mock_arraycontainer, mock_pml):
    """Ensure collect → add restores the original arrays."""
    original = mock_arraycontainer(E=jnp.array([[1, 2], [3, 4]]), H=jnp.array([[5, 6], [7, 8]]))

    # Collect boundary values
    values = misc.collect_boundary_interfaces(original, mock_pml)

    # Tamper: overwrite boundary in a copy
    tampered = mock_arraycontainer(
        E=original.E.at[:, 0:1].set(jnp.array([[99], [99]])),
        H=original.H.at[:, 0:1].set(jnp.array([[88], [88]])),
    )

    # Add back original values
    restored = misc.add_boundary_interfaces(tampered, values, mock_pml)

    assert jnp.array_equal(restored.E, original.E)
    assert jnp.array_equal(restored.H, original.H)


def test_compute_anisotropic_update_matrices_without_conductivity():
    """Test compute_anisotropic_update_matrices with zero conductivity."""
    spatial_shape = (5, 5, 5)
    # Material property tensor with unique values (3, 3, Nx, Ny, Nz)
    inv_material_prop = jnp.arange(3 * 3 * 5 * 5 * 5).reshape(3, 3, 5, 5, 5)
    # Sigma is None for nonconductive materials
    sigma = None
    c = 0.5
    eta_factor = 1.0

    A, B = misc.compute_anisotropic_update_matrices(inv_material_prop, sigma, c, eta_factor)

    # For materials without conductivity, A should be identity
    expected_A = jnp.eye(3)[:, :, None, None, None] * jnp.ones((1, 1, *spatial_shape))
    assert jnp.allclose(A, expected_A, atol=1e-6)

    # For materials without conductivity, B should be c * inv_material_prop
    expected_B = c * inv_material_prop
    assert jnp.allclose(B, expected_B, atol=1e-6)


def test_compute_anisotropic_update_matrices_with_conductivity():
    """Test compute_anisotropic_update_matrices with non-zero conductivity."""
    spatial_shape = (5, 5, 5)
    # Material property tensor with unique values (3, 3, Nx, Ny, Nz)
    inv_material_prop = jnp.arange(3 * 3 * 5 * 5 * 5).reshape(3, 3, 5, 5, 5)
    # Sigma is a tensor (3, 3, Nx, Ny, Nz) for conductive materials
    sigma = jnp.eye(3)[:, :, None, None, None] * 0.5 * jnp.ones((1, 1, *spatial_shape))
    c = 0.5
    eta_factor = 1.0

    A, B = misc.compute_anisotropic_update_matrices(inv_material_prop, sigma, c, eta_factor)

    # With conductivity, A should not be identity
    expected_A = jnp.eye(3)[:, :, None, None, None] * jnp.ones((1, 1, *spatial_shape))
    assert not jnp.allclose(A, expected_A)

    # Check that A and B have the correct shape and are finite
    assert A.shape == (3, 3, *spatial_shape)
    assert B.shape == (3, 3, *spatial_shape)
    assert jnp.all(jnp.isfinite(A))
    assert jnp.all(jnp.isfinite(B))


def test_compute_anisotropic_update_matrices_reverse_without_conductivity():
    """Test compute_anisotropic_update_matrices_reverse with zero conductivity."""
    spatial_shape = (5, 5, 5)
    # Material property tensor with unique values (3, 3, Nx, Ny, Nz)
    inv_material_prop = jnp.arange(3 * 3 * 5 * 5 * 5).reshape(3, 3, 5, 5, 5)
    # Sigma is None for nonconductive materials
    sigma = None
    c = 0.5
    eta_factor = 1.0

    A_rev, B_rev = misc.compute_anisotropic_update_matrices_reverse(inv_material_prop, sigma, c, eta_factor)

    # For isotropic materials without conductivity, A_rev should be identity
    expected_A_rev = jnp.eye(3)[:, :, None, None, None] * jnp.ones((1, 1, *spatial_shape))
    assert jnp.allclose(A_rev, expected_A_rev, atol=1e-6)

    # B_rev should be c * inv_material_prop
    expected_B_rev = c * inv_material_prop
    assert jnp.allclose(B_rev, expected_B_rev, atol=1e-6)


def test_compute_anisotropic_update_matrices_reverse_with_conductivity():
    """Test compute_anisotropic_update_matrices_reverse with non-zero conductivity."""
    spatial_shape = (5, 5, 5)
    # Material property tensor with unique values (3, 3, Nx, Ny, Nz)
    inv_material_prop = jnp.arange(3 * 3 * 5 * 5 * 5).reshape(3, 3, 5, 5, 5)
    # Sigma is a tensor (3, 3, Nx, Ny, Nz) for conductive materials
    sigma = jnp.eye(3)[:, :, None, None, None] * 0.5 * jnp.ones((1, 1, *spatial_shape))
    c = 0.5
    eta_factor = 1.0

    A_rev, B_rev = misc.compute_anisotropic_update_matrices_reverse(inv_material_prop, sigma, c, eta_factor)

    # With conductivity, A_rev should not be identity
    expected_A_rev = jnp.eye(3)[:, :, None, None, None] * jnp.ones((1, 1, *spatial_shape))
    assert not jnp.allclose(A_rev, expected_A_rev)

    # Check that A_rev and B_rev have the correct shape and are finite
    assert A_rev.shape == (3, 3, *spatial_shape)
    assert B_rev.shape == (3, 3, *spatial_shape)
    assert jnp.all(jnp.isfinite(A_rev))
    assert jnp.all(jnp.isfinite(B_rev))


def test_avg_anisotropic_E_component():
    """Test avg_anisotropic_E_component."""
    # Create a padded E field (3, Nx=3, Ny=3, Nz=3)
    field = jnp.array(
        [
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 2, 0], [3, 4, 5], [0, 6, 0]], [[0, 0, 0], [0, 7, 0], [0, 0, 0]]],
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 2, 0], [3, 4, 5], [0, 6, 0]], [[0, 0, 0], [0, 7, 0], [0, 0, 0]]],
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 2, 0], [3, 4, 5], [0, 6, 0]], [[0, 0, 0], [0, 7, 0], [0, 0, 0]]],
        ]
    )

    Ex_avg_y = misc.avg_anisotropic_E_component(field, component=0, location=1)  # calc Ex at location of Ey
    Ex_avg_z = misc.avg_anisotropic_E_component(field, component=0, location=2)  # calc Ex at location of Ez
    Ey_avg_x = misc.avg_anisotropic_E_component(field, component=1, location=0)  # calc Ey at location of Ex
    Ey_avg_z = misc.avg_anisotropic_E_component(field, component=1, location=2)  # calc Ey at location of Ez
    Ez_avg_x = misc.avg_anisotropic_E_component(field, component=2, location=0)  # calc Ez at location of Ex
    Ez_avg_y = misc.avg_anisotropic_E_component(field, component=2, location=1)  # calc Ez at location of Ey

    expected_Ex_avg_y = jnp.array([[[(4 + 6 + 1 + 0) / 4]]])
    expected_Ex_avg_z = jnp.array([[[(4 + 5 + 1 + 0) / 4]]])
    expected_Ey_avg_x = jnp.array([[[(4 + 7 + 2 + 0) / 4]]])
    expected_Ey_avg_z = jnp.array([[[(4 + 5 + 2 + 0) / 4]]])
    expected_Ez_avg_x = jnp.array([[[(4 + 7 + 3 + 0) / 4]]])
    expected_Ez_avg_y = jnp.array([[[(4 + 6 + 3 + 0) / 4]]])

    assert jnp.allclose(Ex_avg_y, expected_Ex_avg_y)
    assert jnp.allclose(Ex_avg_z, expected_Ex_avg_z)
    assert jnp.allclose(Ey_avg_x, expected_Ey_avg_x)
    assert jnp.allclose(Ey_avg_z, expected_Ey_avg_z)
    assert jnp.allclose(Ez_avg_x, expected_Ez_avg_x)
    assert jnp.allclose(Ez_avg_y, expected_Ez_avg_y)


def test_avg_anisotropic_H_component():
    """Test avg_anisotropic_H_component."""
    # Create a padded H field (3, Nx=3, Ny=3, Nz=3)
    field = jnp.array(
        [
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 2, 0], [3, 4, 5], [0, 6, 0]], [[0, 0, 0], [0, 7, 0], [0, 0, 0]]],
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 2, 0], [3, 4, 5], [0, 6, 0]], [[0, 0, 0], [0, 7, 0], [0, 0, 0]]],
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 2, 0], [3, 4, 5], [0, 6, 0]], [[0, 0, 0], [0, 7, 0], [0, 0, 0]]],
        ]
    )

    Hx_avg_y = misc.avg_anisotropic_H_component(field, component=0, location=1)  # calc Hx at location of Hy
    Hx_avg_z = misc.avg_anisotropic_H_component(field, component=0, location=2)  # calc Hx at location of Hz
    Hy_avg_x = misc.avg_anisotropic_H_component(field, component=1, location=0)  # calc Hy at location of Hx
    Hy_avg_z = misc.avg_anisotropic_H_component(field, component=1, location=2)  # calc Hy at location of Hz
    Hz_avg_x = misc.avg_anisotropic_H_component(field, component=2, location=0)  # calc Hz at location of Hx
    Hz_avg_y = misc.avg_anisotropic_H_component(field, component=2, location=1)  # calc Hz at location of Hy

    expected_Hx_avg_y = jnp.array([[[(4 + 2 + 7 + 0) / 4]]])
    expected_Hx_avg_z = jnp.array([[[(4 + 3 + 7 + 0) / 4]]])
    expected_Hy_avg_x = jnp.array([[[(4 + 1 + 6 + 0) / 4]]])
    expected_Hy_avg_z = jnp.array([[[(4 + 3 + 6 + 0) / 4]]])
    expected_Hz_avg_x = jnp.array([[[(4 + 1 + 5 + 0) / 4]]])
    expected_Hz_avg_y = jnp.array([[[(4 + 2 + 5 + 0) / 4]]])

    assert jnp.allclose(Hx_avg_y, expected_Hx_avg_y)
    assert jnp.allclose(Hx_avg_z, expected_Hx_avg_z)
    assert jnp.allclose(Hy_avg_x, expected_Hy_avg_x)
    assert jnp.allclose(Hy_avg_z, expected_Hy_avg_z)
    assert jnp.allclose(Hz_avg_x, expected_Hz_avg_x)
    assert jnp.allclose(Hz_avg_y, expected_Hz_avg_y)
