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
