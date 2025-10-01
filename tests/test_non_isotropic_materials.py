"""Tests for non-isotropic (anisotropic) material support."""

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
    """Tests for the Material class with isotropic and non-isotropic inputs."""

    def test_isotropic_material_scalar_input(self):
        """Test that scalar inputs are converted to tuples."""
        mat = Material(permittivity=2.5, permeability=1.2)
        assert mat.permittivity == (2.5, 2.5, 2.5)
        assert mat.permeability == (1.2, 1.2, 1.2)
        assert mat.electric_conductivity == (0.0, 0.0, 0.0)
        assert mat.magnetic_conductivity == (0.0, 0.0, 0.0)

    def test_isotropic_material_default_values(self):
        """Test default material values."""
        mat = Material()
        assert mat.permittivity == (1.0, 1.0, 1.0)
        assert mat.permeability == (1.0, 1.0, 1.0)
        assert mat.electric_conductivity == (0.0, 0.0, 0.0)
        assert mat.magnetic_conductivity == (0.0, 0.0, 0.0)

    def test_non_isotropic_material_tuple_input(self):
        """Test that tuple inputs are stored correctly."""
        mat = Material(
            permittivity=(2.0, 2.5, 3.0),
            permeability=(1.0, 1.2, 1.5),
            electric_conductivity=(0.1, 0.2, 0.3),
            magnetic_conductivity=(0.01, 0.02, 0.03),
        )
        assert mat.permittivity == (2.0, 2.5, 3.0)
        assert mat.permeability == (1.0, 1.2, 1.5)
        assert mat.electric_conductivity == (0.1, 0.2, 0.3)
        assert mat.magnetic_conductivity == (0.01, 0.02, 0.03)

    def test_mixed_isotropic_and_non_isotropic(self):
        """Test material with some isotropic and some non-isotropic properties."""
        mat = Material(
            permittivity=(2.0, 2.5, 3.0),  # non-isotropic
            permeability=1.0,  # isotropic
        )
        assert mat.permittivity == (2.0, 2.5, 3.0)
        assert mat.permeability == (1.0, 1.0, 1.0)

    def test_invalid_tuple_length(self):
        """Test that tuples with wrong length raise an error."""
        with pytest.raises(ValueError, match="must have exactly 3 elements"):
            Material(permittivity=(2.0, 2.5))

        with pytest.raises(ValueError, match="must have exactly 3 elements"):
            Material(permeability=(1.0, 1.2, 1.5, 2.0))

    def test_is_isotropic_property(self):
        """Test the is_isotropic property."""
        iso = Material(permittivity=2.0, permeability=1.5)
        assert iso.is_isotropic is True

        aniso = Material(permittivity=(2.0, 2.5, 3.0))
        assert aniso.is_isotropic is False

        # Material with all same values should be isotropic
        mat = Material(permittivity=(2.0, 2.0, 2.0), permeability=(1.0, 1.0, 1.0))
        assert mat.is_isotropic is True

    def test_is_magnetic_property(self):
        """Test the is_magnetic property."""
        non_mag = Material(permeability=1.0)
        assert non_mag.is_magnetic is False

        mag_iso = Material(permeability=2.0)
        assert mag_iso.is_magnetic is True

        mag_aniso = Material(permeability=(1.0, 1.5, 1.0))
        assert mag_aniso.is_magnetic is True

        # All components = 1.0 should be non-magnetic
        mat = Material(permeability=(1.0, 1.0, 1.0))
        assert mat.is_magnetic is False

    def test_is_electrically_conductive_property(self):
        """Test the is_electrically_conductive property."""
        non_cond = Material(electric_conductivity=0.0)
        assert non_cond.is_electrically_conductive is False

        cond_iso = Material(electric_conductivity=1.5)
        assert cond_iso.is_electrically_conductive is True

        cond_aniso = Material(electric_conductivity=(0.1, 0.0, 0.0))
        assert cond_aniso.is_electrically_conductive is True

    def test_is_magnetically_conductive_property(self):
        """Test the is_magnetically_conductive property."""
        non_cond = Material(magnetic_conductivity=0.0)
        assert non_cond.is_magnetically_conductive is False

        cond_iso = Material(magnetic_conductivity=0.5)
        assert cond_iso.is_magnetically_conductive is True

        cond_aniso = Material(magnetic_conductivity=(0.01, 0.02, 0.0))
        assert cond_aniso.is_magnetically_conductive is True


class TestMaterialHelperFunctions:
    """Tests for material helper functions."""

    def test_compute_allowed_permittivities(self):
        """Test compute_allowed_permittivities returns list of tuples."""
        materials = {
            "air": Material(permittivity=1.0),
            "aniso": Material(permittivity=(2.0, 2.5, 3.0)),
        }
        result = compute_allowed_permittivities(materials)

        assert len(result) == 2
        assert all(isinstance(p, tuple) for p in result)
        assert all(len(p) == 3 for p in result)

        # Should be sorted by first component
        assert result[0] == (1.0, 1.0, 1.0)
        assert result[1] == (2.0, 2.5, 3.0)

    def test_compute_allowed_permeabilities(self):
        """Test compute_allowed_permeabilities returns list of tuples."""
        materials = {
            "non_mag": Material(permeability=1.0),
            "magnetic": Material(permeability=(1.0, 1.5, 2.0)),
        }
        result = compute_allowed_permeabilities(materials)

        assert len(result) == 2
        assert result[0] == (1.0, 1.0, 1.0)
        assert result[1] == (1.0, 1.5, 2.0)

    def test_compute_allowed_electric_conductivities(self):
        """Test compute_allowed_electric_conductivities returns list of tuples."""
        materials = {
            "insulator": Material(electric_conductivity=0.0),
            "conductor": Material(electric_conductivity=(0.5, 1.0, 1.5)),
        }
        result = compute_allowed_electric_conductivities(materials)

        assert len(result) == 2
        assert result[0] == (0.0, 0.0, 0.0)
        assert result[1] == (0.5, 1.0, 1.5)

    def test_compute_allowed_magnetic_conductivities(self):
        """Test compute_allowed_magnetic_conductivities returns list of tuples."""
        materials = {
            "low_loss": Material(magnetic_conductivity=0.0),
            "lossy": Material(magnetic_conductivity=(0.1, 0.2, 0.3)),
        }
        result = compute_allowed_magnetic_conductivities(materials)

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
        result = compute_allowed_permittivities(materials)

        # Should be sorted by first component: 1.0, 2.0, 3.0
        assert result[0][0] == 1.0
        assert result[1][0] == 2.0
        assert result[2][0] == 3.0


class TestArrayShapes:
    """Tests for array shapes with non-isotropic materials."""

    def test_permittivity_array_conversion(self):
        """Test that tuples convert correctly to JAX arrays."""
        materials_dict = {
            "air": Material(permittivity=1.0),
            "aniso": Material(permittivity=(2.0, 2.5, 3.0)),
        }
        perms = compute_allowed_permittivities(materials_dict)
        perm_array = jnp.asarray(perms)

        # Should have shape (num_materials, 3)
        assert perm_array.shape == (2, 3)
        assert jnp.allclose(perm_array[0], jnp.array([1.0, 1.0, 1.0]))
        assert jnp.allclose(perm_array[1], jnp.array([2.0, 2.5, 3.0]))

    def test_inverse_permittivity_calculation(self):
        """Test that inverse permittivity is calculated correctly."""
        materials_dict = {
            "mat": Material(permittivity=(2.0, 4.0, 5.0)),
        }
        perms = compute_allowed_permittivities(materials_dict)
        perm_array = jnp.asarray(perms)
        inv_perms = 1 / perm_array

        expected = jnp.array([[0.5, 0.25, 0.2]])
        assert jnp.allclose(inv_perms, expected)

    def test_component_indexing(self):
        """Test indexing into material component arrays."""
        materials_dict = {
            "mat1": Material(permittivity=(1.0, 2.0, 3.0)),
            "mat2": Material(permittivity=(4.0, 5.0, 6.0)),
        }
        perms = compute_allowed_permittivities(materials_dict)
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
        assert mat.permittivity == (2.0, 2.0, 2.0)
        assert mat.is_isotropic is True

    def test_material_helper_functions_with_isotropic(self):
        """Test that helper functions work with purely isotropic materials."""
        materials = {
            "air": Material(permittivity=1.0),
            "glass": Material(permittivity=2.5),
            "metal": Material(permittivity=10.0),
        }

        perms = compute_allowed_permittivities(materials)
        assert len(perms) == 3
        # All should be tuples even though input was scalar
        assert all(isinstance(p, tuple) and len(p) == 3 for p in perms)
