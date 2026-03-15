# tests/unit/fdtd/test_misc.py

import jax.numpy as jnp
import pytest

import fdtdx.fdtd.misc as misc


class MockPML:
    """Minimal mock of PerfectlyMatchedLayer matching the API used in misc.py."""

    def __init__(self, name="pml1", start=0, end=1):
        self.name = name
        self._start = start
        self._end = end

    def interface_slice(self):
        return (slice(self._start, self._end),)


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

    expected_E = arrays.E[:, 0:1]
    expected_H = arrays.H[:, 0:1]

    assert jnp.array_equal(result["pml1_E"], expected_E)
    assert jnp.array_equal(result["pml1_H"], expected_H)


def test_collect_boundary_interfaces_single_field(mock_arraycontainer, mock_pml):
    """custom fields_to_collect=('E',) only collects E keys."""
    arrays = mock_arraycontainer(E=jnp.array([[1, 2], [3, 4]]), H=jnp.array([[5, 6], [7, 8]]))

    result = misc.collect_boundary_interfaces(arrays, mock_pml, fields_to_collect=("E",))

    assert "pml1_E" in result
    assert "pml1_H" not in result
    assert jnp.array_equal(result["pml1_E"], arrays.E[:, 0:1])


def test_collect_boundary_interfaces_multiple_pmls(mock_arraycontainer):
    """Multiple PML objects each produce their own keys."""
    pmls = [MockPML("pml1", start=0, end=1), MockPML("pml2", start=1, end=2)]
    arrays = mock_arraycontainer(E=jnp.array([[1, 2, 3], [4, 5, 6]]), H=jnp.array([[7, 8, 9], [10, 11, 12]]))

    result = misc.collect_boundary_interfaces(arrays, pmls)

    assert "pml1_E" in result
    assert "pml1_H" in result
    assert "pml2_E" in result
    assert "pml2_H" in result
    assert jnp.array_equal(result["pml1_E"], arrays.E[:, 0:1])
    assert jnp.array_equal(result["pml2_E"], arrays.E[:, 1:2])


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


def test_add_boundary_interfaces_single_field(mock_arraycontainer, mock_pml):
    """custom fields_to_add=('H',) only updates H, leaves E untouched."""
    arrays = mock_arraycontainer(E=jnp.array([[1, 2], [3, 4]]), H=jnp.array([[5, 6], [7, 8]]))

    values = {"pml1_H": jnp.array([[11], [12]])}
    updated = misc.add_boundary_interfaces(arrays, values, mock_pml, fields_to_add=("H",))

    assert jnp.array_equal(updated.H[:, 0:1], values["pml1_H"])
    # E untouched
    assert jnp.array_equal(updated.E, jnp.array([[1, 2], [3, 4]]))


def test_add_boundary_interfaces_multiple_pmls(mock_arraycontainer):
    """Multiple PML objects each restore their own boundary slice."""
    pmls = [MockPML("pml1", start=0, end=1), MockPML("pml2", start=1, end=2)]
    arrays = mock_arraycontainer(
        E=jnp.array([[1, 2, 3], [4, 5, 6]]),
        H=jnp.array([[7, 8, 9], [10, 11, 12]]),
    )
    values = {
        "pml1_E": jnp.array([[99], [99]]),
        "pml2_E": jnp.array([[88], [88]]),
        "pml1_H": jnp.array([[77], [77]]),
        "pml2_H": jnp.array([[66], [66]]),
    }
    updated = misc.add_boundary_interfaces(arrays, values, pmls)

    assert jnp.array_equal(updated.E[:, 0:1], values["pml1_E"])
    assert jnp.array_equal(updated.E[:, 1:2], values["pml2_E"])
    assert jnp.array_equal(updated.H[:, 0:1], values["pml1_H"])
    assert jnp.array_equal(updated.H[:, 1:2], values["pml2_H"])


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
    inv_material_prop = jnp.arange(3 * 3 * 5 * 5 * 5).reshape(3, 3, 5, 5, 5)
    sigma = None
    c = 0.5
    eta_factor = 1.0

    A, B = misc.compute_anisotropic_update_matrices(inv_material_prop, sigma, c, eta_factor)

    # For materials without conductivity, A should be identity
    expected_A = jnp.eye(3)[:, :, None, None, None] * jnp.ones((1, 1, *spatial_shape))
    assert jnp.allclose(A, expected_A, atol=1e-6)

    # B should be c * inv_material_prop
    expected_B = c * inv_material_prop
    assert jnp.allclose(B, expected_B, atol=1e-6)


def test_compute_anisotropic_update_matrices_with_conductivity():
    """Test compute_anisotropic_update_matrices with non-zero conductivity."""
    spatial_shape = (5, 5, 5)
    inv_material_prop = jnp.arange(3 * 3 * 5 * 5 * 5).reshape(3, 3, 5, 5, 5)
    sigma = jnp.eye(3)[:, :, None, None, None] * 0.5 * jnp.ones((1, 1, *spatial_shape))
    c = 0.5
    eta_factor = 1.0

    A, B = misc.compute_anisotropic_update_matrices(inv_material_prop, sigma, c, eta_factor)

    # With conductivity, A should not be identity
    expected_A = jnp.eye(3)[:, :, None, None, None] * jnp.ones((1, 1, *spatial_shape))
    assert not jnp.allclose(A, expected_A)

    assert A.shape == (3, 3, *spatial_shape)
    assert B.shape == (3, 3, *spatial_shape)
    assert jnp.all(jnp.isfinite(A))
    assert jnp.all(jnp.isfinite(B))


def test_compute_anisotropic_update_matrices_reverse_without_conductivity():
    """Test compute_anisotropic_update_matrices_reverse with zero conductivity."""
    spatial_shape = (5, 5, 5)
    inv_material_prop = jnp.arange(3 * 3 * 5 * 5 * 5).reshape(3, 3, 5, 5, 5)
    sigma = None
    c = 0.5
    eta_factor = 1.0

    A_rev, B_rev = misc.compute_anisotropic_update_matrices_reverse(inv_material_prop, sigma, c, eta_factor)

    expected_A_rev = jnp.eye(3)[:, :, None, None, None] * jnp.ones((1, 1, *spatial_shape))
    assert jnp.allclose(A_rev, expected_A_rev, atol=1e-6)

    expected_B_rev = c * inv_material_prop
    assert jnp.allclose(B_rev, expected_B_rev, atol=1e-6)


def test_compute_anisotropic_update_matrices_reverse_with_conductivity():
    """Test compute_anisotropic_update_matrices_reverse with non-zero conductivity."""
    spatial_shape = (5, 5, 5)
    inv_material_prop = jnp.arange(3 * 3 * 5 * 5 * 5).reshape(3, 3, 5, 5, 5)
    sigma = jnp.eye(3)[:, :, None, None, None] * 0.5 * jnp.ones((1, 1, *spatial_shape))
    c = 0.5
    eta_factor = 1.0

    A_rev, B_rev = misc.compute_anisotropic_update_matrices_reverse(inv_material_prop, sigma, c, eta_factor)

    expected_A_rev = jnp.eye(3)[:, :, None, None, None] * jnp.ones((1, 1, *spatial_shape))
    assert not jnp.allclose(A_rev, expected_A_rev)

    assert A_rev.shape == (3, 3, *spatial_shape)
    assert B_rev.shape == (3, 3, *spatial_shape)
    assert jnp.all(jnp.isfinite(A_rev))
    assert jnp.all(jnp.isfinite(B_rev))


def test_compute_anisotropic_matrices_forward_reverse_agree_no_sigma():
    """Without conductivity, forward and reverse matrices are equal (both identity A, same B)."""
    spatial_shape = (3, 3, 3)
    inv_material_prop = jnp.eye(3)[:, :, None, None, None] * jnp.ones((1, 1, *spatial_shape))

    A_fwd, B_fwd = misc.compute_anisotropic_update_matrices(inv_material_prop, None, 0.5, 1.0)
    A_rev, B_rev = misc.compute_anisotropic_update_matrices_reverse(inv_material_prop, None, 0.5, 1.0)

    assert jnp.allclose(A_fwd, A_rev, atol=1e-6)
    assert jnp.allclose(B_fwd, B_rev, atol=1e-6)


def test_compute_anisotropic_matrices_forward_reverse_differ_with_sigma():
    """With conductivity, forward A and reverse A differ (M1 != M2 implies M1^-1 M2 != M2^-1 M1)."""
    spatial_shape = (3, 3, 3)
    inv_material_prop = jnp.eye(3)[:, :, None, None, None] * jnp.ones((1, 1, *spatial_shape))
    sigma = jnp.eye(3)[:, :, None, None, None] * 0.5 * jnp.ones((1, 1, *spatial_shape))

    A_fwd, _ = misc.compute_anisotropic_update_matrices(inv_material_prop, sigma, 0.5, 1.0)
    A_rev, _ = misc.compute_anisotropic_update_matrices_reverse(inv_material_prop, sigma, 0.5, 1.0)

    assert not jnp.allclose(A_fwd, A_rev)


def test_avg_anisotropic_E_component():
    """Test avg_anisotropic_E_component for all off-diagonal (component != location) cases."""
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


def test_avg_anisotropic_E_component_diagonal():
    """Test avg_anisotropic_E_component when component == location (diagonal cases).

    roll(field[c], (-1, 1), axis=(c, c)) is a net-zero roll (shifts cancel), so the formula becomes:
    (2*field[c] + roll(field[c], -1, axis=c) + roll(field[c], 1, axis=c)) / 4
    """
    field = jnp.array(
        [
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 2, 0], [3, 4, 5], [0, 6, 0]], [[0, 0, 0], [0, 7, 0], [0, 0, 0]]],
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 2, 0], [3, 4, 5], [0, 6, 0]], [[0, 0, 0], [0, 7, 0], [0, 0, 0]]],
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 2, 0], [3, 4, 5], [0, 6, 0]], [[0, 0, 0], [0, 7, 0], [0, 0, 0]]],
        ]
    )

    Ex_avg_x = misc.avg_anisotropic_E_component(field, component=0, location=0)
    Ey_avg_y = misc.avg_anisotropic_E_component(field, component=1, location=1)
    Ez_avg_z = misc.avg_anisotropic_E_component(field, component=2, location=2)

    # component=0, location=0 at interior [1,1,1]:
    # (field[0][1,1,1] + field[0][2,1,1] + field[0][0,1,1] + field[0][1,1,1]) / 4 = (4+7+1+4)/4
    expected_Ex_avg_x = jnp.array([[[(4 + 7 + 1 + 4) / 4]]])
    # component=1, location=1 at interior [1,1,1]:
    # (field[1][1,1,1] + field[1][1,2,1] + field[1][1,0,1] + field[1][1,1,1]) / 4 = (4+6+2+4)/4
    expected_Ey_avg_y = jnp.array([[[(4 + 6 + 2 + 4) / 4]]])
    # component=2, location=2 at interior [1,1,1]:
    # (field[2][1,1,1] + field[2][1,1,2] + field[2][1,1,0] + field[2][1,1,1]) / 4 = (4+5+3+4)/4
    expected_Ez_avg_z = jnp.array([[[(4 + 5 + 3 + 4) / 4]]])

    assert jnp.allclose(Ex_avg_x, expected_Ex_avg_x)
    assert jnp.allclose(Ey_avg_y, expected_Ey_avg_y)
    assert jnp.allclose(Ez_avg_z, expected_Ez_avg_z)


def test_avg_anisotropic_H_component():
    """Test avg_anisotropic_H_component for all off-diagonal (component != location) cases."""
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


def test_avg_anisotropic_H_component_diagonal():
    """Test avg_anisotropic_H_component when component == location (diagonal cases).

    roll(field[c], (1, -1), axis=(c, c)) is a net-zero roll (shifts cancel), so the formula becomes:
    (2*field[c] + roll(field[c], 1, axis=c) + roll(field[c], -1, axis=c)) / 4
    """
    field = jnp.array(
        [
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 2, 0], [3, 4, 5], [0, 6, 0]], [[0, 0, 0], [0, 7, 0], [0, 0, 0]]],
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 2, 0], [3, 4, 5], [0, 6, 0]], [[0, 0, 0], [0, 7, 0], [0, 0, 0]]],
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 2, 0], [3, 4, 5], [0, 6, 0]], [[0, 0, 0], [0, 7, 0], [0, 0, 0]]],
        ]
    )

    Hx_avg_x = misc.avg_anisotropic_H_component(field, component=0, location=0)
    Hy_avg_y = misc.avg_anisotropic_H_component(field, component=1, location=1)
    Hz_avg_z = misc.avg_anisotropic_H_component(field, component=2, location=2)

    # component=0, location=0 at interior [1,1,1]:
    # (field[0][1,1,1] + field[0][0,1,1] + field[0][2,1,1] + field[0][1,1,1]) / 4 = (4+1+7+4)/4
    expected_Hx_avg_x = jnp.array([[[(4 + 1 + 7 + 4) / 4]]])
    # component=1, location=1 at interior [1,1,1]:
    # (field[1][1,1,1] + field[1][1,0,1] + field[1][1,2,1] + field[1][1,1,1]) / 4 = (4+2+6+4)/4
    expected_Hy_avg_y = jnp.array([[[(4 + 2 + 6 + 4) / 4]]])
    # component=2, location=2 at interior [1,1,1]:
    # (field[2][1,1,1] + field[2][1,1,0] + field[2][1,1,2] + field[2][1,1,1]) / 4 = (4+3+5+4)/4
    expected_Hz_avg_z = jnp.array([[[(4 + 3 + 5 + 4) / 4]]])

    assert jnp.allclose(Hx_avg_x, expected_Hx_avg_x)
    assert jnp.allclose(Hy_avg_y, expected_Hy_avg_y)
    assert jnp.allclose(Hz_avg_z, expected_Hz_avg_z)
