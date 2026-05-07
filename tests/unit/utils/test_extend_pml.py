"""Unit tests for utils/extend_pml.py."""

import warnings
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.utils.extend_pml import extend_material_to_pml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleContainer:
    """Minimal ArrayContainer stand-in that supports aset() without the real TreeClass."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def aset(self, name: str, val) -> "SimpleContainer":
        kw = {k: getattr(self, k) for k in vars(self)}
        kw[name] = val
        return SimpleContainer(**kw)


def _make_pml(axis: int, direction: str, thickness: int, grid_shape: tuple) -> MagicMock:
    """Build a MagicMock that mimics PerfectlyMatchedLayer."""
    pml = MagicMock()
    pml.axis = axis
    pml.direction = direction
    pml.thickness = thickness
    N = grid_shape[axis]
    T = thickness
    s = [slice(0, grid_shape[i]) for i in range(3)]
    s[axis] = slice(0, T) if direction == "-" else slice(N - T, N)
    pml.grid_slice = tuple(s)
    return pml


def _make_objects(pml_list, grid_shape: tuple, background_permittivity: float = 1.0) -> MagicMock:
    """Build a MagicMock that mimics ObjectContainer."""
    from fdtdx.materials import Material

    objects = MagicMock()
    objects.pml_objects = pml_list
    objects.volume.grid_shape = grid_shape
    objects.volume.material = Material(permittivity=background_permittivity)
    return objects


def _make_arrays(inv_perm_fill: float, shape: tuple = (1, 10, 5, 5)) -> SimpleContainer:
    """Build a SimpleContainer with all-scalar fill values."""
    return SimpleContainer(
        inv_permittivities=jnp.full(shape, inv_perm_fill),
        inv_permeabilities=1.0,  # scalar float — non-magnetic default
        electric_conductivity=None,
        magnetic_conductivity=None,
    )


# ---------------------------------------------------------------------------
# Tests: minus-direction PML
# ---------------------------------------------------------------------------


class TestExtendMaterialToPmlMinus:
    """PML on the negative (min) side of an axis."""

    def _make_setup(self):
        """axis=0 "-", T=2, grid=(10,5,5). Interior edge at index 2 set to 0.5."""
        grid_shape = (10, 5, 5)
        pml = _make_pml(axis=0, direction="-", thickness=2, grid_shape=grid_shape)
        objects = _make_objects([pml], grid_shape)

        # Fill with default 1.0, then set interior edge (idx=2) to 0.5
        inv_perm = jnp.ones((1, 10, 5, 5))
        inv_perm = inv_perm.at[:, 2, :, :].set(0.5)
        arrays = SimpleContainer(
            inv_permittivities=inv_perm,
            inv_permeabilities=1.0,
            electric_conductivity=None,
            magnetic_conductivity=None,
        )
        return objects, arrays

    def test_minus_direction_basic(self):
        """PML cells (indices 0-1) should be filled with the interior-edge value 0.5."""
        objects, arrays = self._make_setup()
        result = extend_material_to_pml(objects, arrays)
        pml_region = np.asarray(result.inv_permittivities[:, 0:2, :, :])
        np.testing.assert_allclose(pml_region, 0.5)

    def test_interior_not_modified(self):
        """Cells at index ≥ 2 must remain unchanged after the call."""
        objects, arrays = self._make_setup()
        original_interior = np.asarray(arrays.inv_permittivities[:, 2:, :, :])
        result = extend_material_to_pml(objects, arrays)
        np.testing.assert_allclose(np.asarray(result.inv_permittivities[:, 2:, :, :]), original_interior)


# ---------------------------------------------------------------------------
# Tests: plus-direction PML
# ---------------------------------------------------------------------------


class TestExtendMaterialToPmlPlus:
    """PML on the positive (max) side of an axis."""

    def test_plus_direction_basic(self):
        """PML cells (indices 8-9) should be filled with the interior-edge value 0.5."""
        grid_shape = (10, 5, 5)
        pml = _make_pml(axis=0, direction="+", thickness=2, grid_shape=grid_shape)
        objects = _make_objects([pml], grid_shape)

        # Interior edge for "+" PML with T=2, N=10 is at index N-T-1 = 7
        inv_perm = jnp.ones((1, 10, 5, 5))
        inv_perm = inv_perm.at[:, 7, :, :].set(0.5)
        arrays = SimpleContainer(
            inv_permittivities=inv_perm,
            inv_permeabilities=1.0,
            electric_conductivity=None,
            magnetic_conductivity=None,
        )

        result = extend_material_to_pml(objects, arrays)
        pml_region = np.asarray(result.inv_permittivities[:, 8:, :, :])
        np.testing.assert_allclose(pml_region, 0.5)


# ---------------------------------------------------------------------------
# Tests: warning behaviour
# ---------------------------------------------------------------------------


class TestExtendMaterialWarning:
    """Warning logic when PML has non-default values."""

    def test_warning_when_pml_has_non_default_values(self):
        """A UserWarning must be raised when the PML region holds a non-default value."""
        grid_shape = (10, 5, 5)
        pml = _make_pml(axis=0, direction="-", thickness=2, grid_shape=grid_shape)
        objects = _make_objects([pml], grid_shape, background_permittivity=1.0)

        # PML region (indices 0-1) filled with 0.3 — clearly not the default 1/1.0 = 1.0
        inv_perm = jnp.ones((1, 10, 5, 5))
        inv_perm = inv_perm.at[:, 0:2, :, :].set(0.3)
        arrays = SimpleContainer(
            inv_permittivities=inv_perm,
            inv_permeabilities=1.0,
            electric_conductivity=None,
            magnetic_conductivity=None,
        )

        with pytest.warns(UserWarning, match="non-default"):
            extend_material_to_pml(objects, arrays)

    def test_no_warning_when_pml_has_default_values(self):
        """No warning when the PML region already holds the background default value."""
        grid_shape = (10, 5, 5)
        pml = _make_pml(axis=0, direction="-", thickness=2, grid_shape=grid_shape)
        objects = _make_objects([pml], grid_shape, background_permittivity=1.0)

        # Everything is 1.0 = 1 / background_permittivity — the expected default
        arrays = _make_arrays(inv_perm_fill=1.0)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should not raise
            extend_material_to_pml(objects, arrays)


# ---------------------------------------------------------------------------
# Tests: optional-field skipping
# ---------------------------------------------------------------------------


class TestExtendMaterialSkips:
    """Fields that should be skipped rather than processed."""

    def test_skip_inv_permeabilities_when_float(self):
        """inv_permeabilities stays a float when it is a scalar (non-magnetic simulation)."""
        grid_shape = (10, 5, 5)
        pml = _make_pml(axis=0, direction="-", thickness=2, grid_shape=grid_shape)
        objects = _make_objects([pml], grid_shape)
        arrays = _make_arrays(inv_perm_fill=1.0)
        assert isinstance(arrays.inv_permeabilities, float)

        result = extend_material_to_pml(objects, arrays)
        assert isinstance(result.inv_permeabilities, float)
        assert result.inv_permeabilities == 1.0

    def test_skip_electric_conductivity_when_none(self):
        """electric_conductivity remains None when not set."""
        grid_shape = (10, 5, 5)
        pml = _make_pml(axis=0, direction="-", thickness=2, grid_shape=grid_shape)
        objects = _make_objects([pml], grid_shape)
        arrays = _make_arrays(inv_perm_fill=1.0)
        assert arrays.electric_conductivity is None

        result = extend_material_to_pml(objects, arrays)
        assert result.electric_conductivity is None

    def test_skip_magnetic_conductivity_when_none(self):
        """magnetic_conductivity remains None when not set."""
        grid_shape = (10, 5, 5)
        pml = _make_pml(axis=0, direction="-", thickness=2, grid_shape=grid_shape)
        objects = _make_objects([pml], grid_shape)
        arrays = _make_arrays(inv_perm_fill=1.0)
        assert arrays.magnetic_conductivity is None

        result = extend_material_to_pml(objects, arrays)
        assert result.magnetic_conductivity is None


# ---------------------------------------------------------------------------
# Tests: multiple PML boundaries
# ---------------------------------------------------------------------------


class TestExtendMaterialMultiplePml:
    """Two PML objects on different axes should both be extended correctly."""

    def test_multiple_pml_boundaries(self):
        """Both PML regions are filled from their respective interior edges.

        We check only the non-corner, unambiguously-owned slices to avoid
        sensitivity to processing order in the overlap region.
        """
        grid_shape = (10, 8, 12)
        pml_x = _make_pml(axis=0, direction="-", thickness=2, grid_shape=grid_shape)  # edge at x=2
        pml_z = _make_pml(axis=2, direction="+", thickness=3, grid_shape=grid_shape)  # edge at z=8

        objects = _make_objects([pml_x, pml_z], grid_shape)

        # Set x=2 interior edge AFTER z=8 so x=2,z=8 ends up as 0.4 (no conflict in x-edge).
        inv_perm = jnp.ones((1, 10, 8, 12))
        inv_perm = inv_perm.at[:, :, :, 8].set(0.6)  # z=8 interior edge for pml_z
        inv_perm = inv_perm.at[:, 2, :, :].set(0.4)  # x=2 interior edge for pml_x (overrides z=8 at x=2)

        arrays = SimpleContainer(
            inv_permittivities=inv_perm,
            inv_permeabilities=1.0,
            electric_conductivity=None,
            magnetic_conductivity=None,
        )

        result = extend_material_to_pml(objects, arrays)

        # x-PML (x=0:2) away from z-PML corner: all values come from x=2 edge (0.4)
        np.testing.assert_allclose(np.asarray(result.inv_permittivities[:, 0:2, :, 0:9]), 0.4)
        # z-PML (z=9:12) away from x-PML corner: values come from z=8 edge (0.6)
        np.testing.assert_allclose(np.asarray(result.inv_permittivities[:, 3:, :, 9:12]), 0.6)
