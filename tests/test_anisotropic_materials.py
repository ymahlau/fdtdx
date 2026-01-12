"""Tests for anisotropic material support."""

import jax.numpy as jnp
import pytest

from fdtdx.materials import (
    Material,
    compute_allowed_electric_conductivities,
    compute_allowed_magnetic_conductivities,
    compute_allowed_permeabilities,
    compute_allowed_permittivities,
)


class TestMaterialClass:
    """Tests for the Material class with isotropic and anisotropic inputs."""

    def test_isotropic_material_scalar_input(self):
        """Test that scalar inputs are converted to tuples."""
        mat = Material(permittivity=2.5, permeability=1.2)
        assert mat.permittivity == (2.5, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 2.5)
        assert mat.permeability == (1.2, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.2)
        assert mat.electric_conductivity == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert mat.magnetic_conductivity == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_isotropic_material_default_values(self):
        """Test default material values."""
        mat = Material()
        assert mat.permittivity == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        assert mat.permeability == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        assert mat.electric_conductivity == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert mat.magnetic_conductivity == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_diagonally_anisotropic_material_tuple_input(self):
        """Test that tuple inputs are stored correctly."""
        mat = Material(
            permittivity=(2.0, 2.5, 3.0),
            permeability=(1.0, 1.2, 1.5),
            electric_conductivity=(0.1, 0.2, 0.3),
            magnetic_conductivity=(0.01, 0.02, 0.03),
        )
        assert mat.permittivity == (2.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 3.0)
        assert mat.permeability == (1.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.5)
        assert mat.electric_conductivity == (0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3)
        assert mat.magnetic_conductivity == (0.01, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.03)

    def test_fully_anisotropic_material_tuple_input(self):
        """Test that tuple inputs are stored correctly."""
        mat = Material(
            permittivity=(2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0),
            permeability=(1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9),
            electric_conductivity=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            magnetic_conductivity=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09),
        )
        assert mat.permittivity == (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
        assert mat.permeability == (1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9)
        assert mat.electric_conductivity == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        assert mat.magnetic_conductivity == (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09)

    def test_mixed_isotropic_and_diagonally_anisotropic(self):
        """Test material with some isotropic and some anisotropic properties."""
        mat = Material(
            permittivity=(2.0, 2.5, 3.0),
            permeability=1.0,
        )
        assert mat.permittivity == (2.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 3.0)
        assert mat.permeability == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def test_mixed_isotropic_and_fully_anisotropic(self):
        """Test material with some isotropic and some fully anisotropic properties."""
        mat = Material(
            permittivity=(2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0),
            permeability=1.0,
        )
        assert mat.permittivity == (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
        assert mat.permeability == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def test_mixed_diagonally_anisotropic_and_fully_anisotropic(self):
        """Test material with some diagonally anisotropic and some fully anisotropic properties."""
        mat = Material(
            permittivity=(2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0),
            permeability=(1.0, 1.5, 2.0),
        )
        assert mat.permittivity == (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
        assert mat.permeability == (1.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.0)

    def test_invalid_tuple_length(self):
        """Test that tuples with wrong length raise an error."""
        with pytest.raises(ValueError, match="must have exactly 3 or 9 elements"):
            Material(permittivity=(2.0, 2.5))

        with pytest.raises(ValueError, match="must have exactly 3 or 9 elements"):
            Material(permeability=(1.0, 1.2, 1.5, 2.0))

    def test_is_all_isotropic_property(self):
        """Test the is_all_isotropic property."""
        mat = Material(permittivity=2.0, permeability=1.5)
        assert mat.is_all_isotropic is True

        mat = Material(permittivity=(2.0, 2.0, 2.0), permeability=(1.0, 1.0, 1.0))
        assert mat.is_all_isotropic is True

        mat = Material(permittivity=(2.0, 2.5, 3.0), permeability=(1.0, 1.0, 1.0))
        assert mat.is_all_isotropic is False

        mat = Material(permittivity=(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0), permeability=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        assert mat.is_all_isotropic is True

        mat = Material(permittivity=(2.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 3.0), permeability=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        assert mat.is_all_isotropic is False

    def test_is_all_diagonally_anisotropic_property(self):
        """Test the is_all_diagonally_anisotropic property."""
        mat = Material(permittivity=2.0, permeability=1.5)
        assert mat.is_all_diagonally_anisotropic is True

        mat = Material(permittivity=(2.0, 2.0, 2.0), permeability=(1.0, 1.0, 1.0))
        assert mat.is_all_diagonally_anisotropic is True

        mat = Material(permittivity=(2.0, 2.5, 3.0), permeability=(1.0, 1.0, 1.0))
        assert mat.is_all_diagonally_anisotropic is True

        mat = Material(permittivity=(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0), permeability=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        assert mat.is_all_diagonally_anisotropic is True

        mat = Material(permittivity=(2.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 3.0), permeability=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        assert mat.is_all_diagonally_anisotropic is True

        mat = Material(permittivity=(2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0), permeability=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        assert mat.is_all_diagonally_anisotropic is False

    def test_is_isotropic_property(self):
        """Test the is_magnetic property."""
        mat = Material(permittivity=1.0)
        assert mat.is_isotropic_permittivity is True

        mat = Material(permittivity=(1.0, 1.0, 1.0))
        assert mat.is_isotropic_permittivity is True

        mat = Material(permittivity=(1.0, 1.5, 2.0))
        assert mat.is_isotropic_permittivity is False

        mat = Material(permittivity=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        assert mat.is_isotropic_permittivity is True

        mat = Material(permittivity=(1.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.0))
        assert mat.is_isotropic_permittivity is False

    def test_is_diagonally_anisotropic_property(self):
        """Test the is_magnetic property."""
        mat = Material(permittivity=1.0)
        assert mat.is_diagonally_anisotropic_permittivity is True

        mat = Material(permittivity=(1.0, 1.0, 1.0))
        assert mat.is_diagonally_anisotropic_permittivity is True

        mat = Material(permittivity=(1.0, 1.5, 2.0))
        assert mat.is_diagonally_anisotropic_permittivity is True

        mat = Material(permittivity=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        assert mat.is_diagonally_anisotropic_permittivity is True

        mat = Material(permittivity=(1.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.0))
        assert mat.is_diagonally_anisotropic_permittivity is True

        mat = Material(permittivity=(1.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0))
        assert mat.is_diagonally_anisotropic_permittivity is False

    def test_is_magnetic_property(self):
        """Test the is_magnetic property."""
        mat = Material(permeability=1.0)
        assert mat.is_magnetic is False

        mat = Material(permeability=2.0)
        assert mat.is_magnetic is True

        mat = Material(permeability=(1.0, 1.0, 1.0))
        assert mat.is_magnetic is False

        mat = Material(permeability=(1.0, 1.5, 2.0))
        assert mat.is_magnetic is True

        mat = Material(permeability=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        assert mat.is_magnetic is False

        mat = Material(permeability=(1.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.0))
        assert mat.is_magnetic is True

        mat = Material(permeability=(1.0, 1.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0))
        assert mat.is_magnetic is True

    def test_is_electrically_conductive_property(self):
        """Test the is_electrically_conductive property."""
        mat = Material(electric_conductivity=0.0)
        assert mat.is_electrically_conductive is False

        mat = Material(electric_conductivity=1.5)
        assert mat.is_electrically_conductive is True

        mat = Material(electric_conductivity=(0.0, 0.0, 0.0))
        assert mat.is_electrically_conductive is False

        mat = Material(electric_conductivity=(0.1, 0.0, 0.0))
        assert mat.is_electrically_conductive is True

        mat = Material(electric_conductivity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert mat.is_electrically_conductive is False

        mat = Material(electric_conductivity=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert mat.is_electrically_conductive is True

    def test_is_magnetically_conductive_property(self):
        """Test the is_magnetically_conductive property."""
        mat = Material(magnetic_conductivity=0.0)
        assert mat.is_magnetically_conductive is False

        mat = Material(magnetic_conductivity=1.5)
        assert mat.is_magnetically_conductive is True

        mat = Material(magnetic_conductivity=(0.0, 0.0, 0.0))
        assert mat.is_magnetically_conductive is False

        mat = Material(magnetic_conductivity=(0.1, 0.0, 0.0))
        assert mat.is_magnetically_conductive is True

        mat = Material(magnetic_conductivity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert mat.is_magnetically_conductive is False

        mat = Material(magnetic_conductivity=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert mat.is_magnetically_conductive is True


class TestMaterialHelperFunctions:
    """Tests for material helper functions."""

    def test_compute_allowed_permittivities_isotropic(self):
        """Test compute_allowed_permittivities_isotropic returns list of tuples."""
        materials = {
            "air": Material(permittivity=1.0),
            "iso": Material(permittivity=2.0),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials.values()])
        result = compute_allowed_permittivities(materials, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic)

        assert len(result) == 2
        assert all(isinstance(p, tuple) for p in result)
        assert all(len(p) == 1 for p in result)

        # Should be sorted by first component
        assert result[0] == (1.0,)
        assert result[1] == (2.0,)

    def test_compute_allowed_permittivities_diagonally_anisotropic(self):
        """Test compute_allowed_permittivities_diagonally_anisotropic returns list of tuples."""
        materials = {
            "air": Material(permittivity=1.0),
            "aniso": Material(permittivity=(2.0, 2.5, 3.0)),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials.values()])
        result = compute_allowed_permittivities(materials, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic)

        assert len(result) == 2
        assert all(isinstance(p, tuple) for p in result)
        assert all(len(p) == 3 for p in result)

        # Should be sorted by first component
        assert result[0] == (1.0, 1.0, 1.0)
        assert result[1] == (2.0, 2.5, 3.0)

    def test_compute_allowed_permittivities_fully_anisotropic(self):
        """Test compute_allowed_permittivities_fully_anisotropic returns list of tuples."""
        materials = {
            "air": Material(permittivity=1.0),
            "aniso": Material(permittivity=(2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials.values()])
        result = compute_allowed_permittivities(materials, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic)

        assert len(result) == 2
        assert all(isinstance(p, tuple) for p in result)
        assert all(len(p) == 9 for p in result)

        # Should be sorted by first component
        assert result[0] == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        assert result[1] == (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)

    def test_compute_allowed_permeabilities(self):
        """Test compute_allowed_permeabilities returns list of tuples."""
        materials = {
            "non_mag": Material(permeability=1.0),
            "magnetic": Material(permeability=(1.0, 1.5, 2.0)),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials.values()])
        result = compute_allowed_permeabilities(materials, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic)

        assert len(result) == 2
        assert result[0] == (1.0, 1.0, 1.0)
        assert result[1] == (1.0, 1.5, 2.0)

    def test_compute_allowed_electric_conductivities(self):
        """Test compute_allowed_electric_conductivities returns list of tuples."""
        materials = {
            "insulator": Material(electric_conductivity=0.0),
            "conductor": Material(electric_conductivity=(0.5, 1.0, 1.5)),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials.values()])
        result = compute_allowed_electric_conductivities(materials, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic)

        assert len(result) == 2
        assert result[0] == (0.0, 0.0, 0.0)
        assert result[1] == (0.5, 1.0, 1.5)

    def test_compute_allowed_magnetic_conductivities(self):
        """Test compute_allowed_magnetic_conductivities returns list of tuples."""
        materials = {
            "low_loss": Material(magnetic_conductivity=0.0),
            "lossy": Material(magnetic_conductivity=(0.1, 0.2, 0.3)),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials.values()])
        result = compute_allowed_magnetic_conductivities(materials, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic)

        assert len(result) == 2
        assert result[0] == (0.0, 0.0, 0.0)
        assert result[1] == (0.1, 0.2, 0.3)

    def test_material_ordering_by_first_component(self):
        """Test that materials are ordered by first component of properties."""
        materials = {
            "mat_high": Material(permittivity=(3.0, 1.0, 1.0)),
            "mat_low": Material(permittivity=(1.0, 5.0, 5.0)),
            "mat_mid": Material(permittivity=(2.0, 2.0, 2.0)),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials.values()])
        result = compute_allowed_permittivities(materials, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic)

        # Should be sorted by first component: 1.0, 2.0, 3.0
        assert result[0][0] == 1.0
        assert result[1][0] == 2.0
        assert result[2][0] == 3.0


class TestArrayShapes:
    """Tests for array shapes with anisotropic materials."""

    def test_permittivity_array_conversion_isotropic(self):
        """Test that isotropic permittivity tuples convert correctly to JAX arrays."""
        materials_dict = {
            "air": Material(permittivity=1.0),
            "iso": Material(permittivity=2.0),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials_dict.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials_dict.values()])
        perms = compute_allowed_permittivities(materials_dict, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic)
        perm_array = jnp.asarray(perms)

        # Should have shape (num_materials, 1)
        assert perm_array.shape == (2, 1)
        assert jnp.allclose(perm_array[0], jnp.array([1.0]))
        assert jnp.allclose(perm_array[1], jnp.array([2.0]))

    def test_permittivity_array_conversion_diagonally_anisotropic(self):
        """Test that diagonally anisotropic permittivity tuples convert correctly to JAX arrays."""
        materials_dict = {
            "air": Material(permittivity=1.0),
            "aniso": Material(permittivity=(2.0, 2.5, 3.0)),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials_dict.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials_dict.values()])
        perms = compute_allowed_permittivities(materials_dict, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic)
        perm_array = jnp.asarray(perms)

        # Should have shape (num_materials, 3)
        assert perm_array.shape == (2, 3)
        assert jnp.allclose(perm_array[0], jnp.array([1.0, 1.0, 1.0]))
        assert jnp.allclose(perm_array[1], jnp.array([2.0, 2.5, 3.0]))

    def test_permittivity_array_conversion_fully_anisotropic(self):
        """Test that fully anisotropic permittivity tuples convert correctly to JAX arrays."""
        materials_dict = {
            "air": Material(permittivity=1.0),
            "aniso": Material(permittivity=(2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials_dict.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials_dict.values()])
        perms = compute_allowed_permittivities(materials_dict, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic)
        perm_array = jnp.asarray(perms)

        # Should have shape (num_materials, 9)
        assert perm_array.shape == (2, 9)
        assert jnp.allclose(perm_array[0], jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
        assert jnp.allclose(perm_array[1], jnp.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]))

    def test_inverse_permittivity_calculation_isotropic(self):
        """Test that inverse permittivity is calculated correctly for isotropic materials."""
        materials_dict = {
            "mat": Material(permittivity=2.0),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials_dict.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials_dict.values()])
        perms = compute_allowed_permittivities(materials_dict, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic)
        perm_array = jnp.asarray(perms)
        inv_perms = 1 / perm_array

        expected = jnp.array([[0.5]])
        assert jnp.allclose(inv_perms, expected)

    def test_inverse_permittivity_calculation_diagonally_anisotropic(self):
        """Test that inverse permittivity is calculated correctly for diagonally anisotropic materials."""
        materials_dict = {
            "mat": Material(permittivity=(2.0, 4.0, 5.0)),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials_dict.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials_dict.values()])
        perms = compute_allowed_permittivities(materials_dict, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic)
        perm_array = jnp.asarray(perms)
        inv_perms = 1 / perm_array

        expected = jnp.array([[0.5, 0.25, 0.2]])
        assert jnp.allclose(inv_perms, expected)

    def test_inverse_permittivity_calculation_fully_anisotropic(self):
        """Test that inverse permittivity is calculated correctly for fully anisotropic materials."""
        materials_dict = {
            "mat": Material(permittivity=(2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0)),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials_dict.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials_dict.values()])
        perms = compute_allowed_permittivities(materials_dict, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic)
        perm_matrix = jnp.array(perms).reshape(3, 3)
        inv_perms = jnp.linalg.inv(perm_matrix).flatten()

        expected = jnp.array([0.5, 0.0, 0.0, -0.5, 0.5, 0.0, 0.0, -0.5, 0.5])
        assert jnp.allclose(inv_perms, expected)

    def test_component_indexing(self):
        """Test indexing into material component arrays."""
        materials_dict = {
            "mat1": Material(permittivity=(1.0, 2.0, 3.0)),
            "mat2": Material(permittivity=(4.0, 5.0, 6.0)),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials_dict.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials_dict.values()])
        perms = compute_allowed_permittivities(materials_dict, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic)
        perm_array = jnp.asarray(perms)  # shape: (2, 3)

        # Access x-component (index 0) of all materials
        x_components = perm_array[:, 0]
        assert jnp.allclose(x_components, jnp.array([1.0, 4.0]))

        # Access y-component (index 1) of all materials
        y_components = perm_array[:, 1]
        assert jnp.allclose(y_components, jnp.array([2.0, 5.0]))

        # Access z-component (index 2) of all materials
        z_components = perm_array[:, 2]
        assert jnp.allclose(z_components, jnp.array([3.0, 6.0]))


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with isotropic materials."""

    def test_old_style_material_creation(self):
        """Test that old code creating materials with scalars still works."""
        # Old style: Material(permittivity=2.0)
        mat = Material(permittivity=2.0, permeability=1.0, electric_conductivity=0.5)

        # Should be converted to tuples internally
        assert isinstance(mat.permittivity, tuple)
        assert isinstance(mat.permeability, tuple)
        assert isinstance(mat.electric_conductivity, tuple)

        # All components should be equal
        assert mat.permittivity == (2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0)
        assert mat.permeability == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        assert mat.electric_conductivity == (0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
        assert mat.is_all_isotropic is True
        assert mat.is_all_diagonally_anisotropic is True

    def test_material_helper_functions_with_isotropic(self):
        """Test that helper functions work with purely isotropic materials."""
        materials = {
            "air": Material(permittivity=1.0),
            "glass": Material(permittivity=2.5),
            "metal": Material(permittivity=10.0),
        }

        perms = compute_allowed_permittivities(materials, isotropic=True)
        assert len(perms) == 3
        # All should be tuples even though input was scalar
        assert all(isinstance(p, tuple) and len(p) == 1 for p in perms)
        assert perms[0] == (1.0,)
        assert perms[1] == (2.5,)
        assert perms[2] == (10.0,)
