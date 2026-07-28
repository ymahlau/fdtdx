"""Tests for anisotropic material support."""

import jax.numpy as jnp
import pytest

from fdtdx.materials import (
    Material,
    compute_allowed_electric_conductivities,
    compute_allowed_magnetic_conductivities,
    compute_allowed_permeabilities,
    compute_allowed_permittivities,
    isotropic_property_value,
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

    def test_fully_anisotropic_material_nested_tuple_input(self):
        """Test that nested tuple inputs are stored correctly."""
        mat = Material(
            permittivity=((2.0, 2.5, 3.0), (3.5, 4.0, 4.5), (5.0, 5.5, 6.0)),
            permeability=((1.0, 1.2, 1.3), (1.4, 1.5, 1.6), (1.7, 1.8, 1.9)),
            electric_conductivity=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9)),
            magnetic_conductivity=((0.01, 0.02, 0.03), (0.04, 0.05, 0.06), (0.07, 0.08, 0.09)),
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

    def test_mixed_isotropic_and_nested_fully_anisotropic(self):
        """Test material with some isotropic and some nested fully anisotropic properties."""
        mat = Material(
            permittivity=((2.0, 2.5, 3.0), (3.5, 4.0, 4.5), (5.0, 5.5, 6.0)),
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

    def test_mixed_diagonally_anisotropic_and_nested_fully_anisotropic(self):
        """Test material with some diagonally anisotropic and some nested fully anisotropic properties."""
        mat = Material(
            permittivity=((2.0, 2.5, 3.0), (3.5, 4.0, 4.5), (5.0, 5.5, 6.0)),
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

        with pytest.raises(ValueError, match="Nested tuple must have 3 elements in each row"):
            Material(permittivity=((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)))

        with pytest.raises(ValueError, match="Invalid material property tuple"):
            Material(permeability=((1.0, 2.0), 3.0, 4.0))

    def test_is_all_isotropic_property(self):
        """Test the is_all_isotropic property."""
        mat = Material(permittivity=2.0, permeability=1.5)
        assert mat.is_all_isotropic is True

        mat = Material(permittivity=(2.0, 2.0, 2.0), permeability=(1.0, 1.0, 1.0))
        assert mat.is_all_isotropic is True

        mat = Material(permittivity=(2.0, 2.5, 3.0), permeability=(1.0, 1.0, 1.0))
        assert mat.is_all_isotropic is False

        mat = Material(
            permittivity=(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0),
            permeability=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        assert mat.is_all_isotropic is True

        mat = Material(
            permittivity=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)),
            permeability=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        assert mat.is_all_isotropic is True

        mat = Material(
            permittivity=(2.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 3.0),
            permeability=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        assert mat.is_all_isotropic is False

        mat = Material(
            permittivity=((2.0, 0.0, 0.0), (0.0, 2.5, 0.0), (0.0, 0.0, 3.0)),
            permeability=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        assert mat.is_all_isotropic is False

    def test_isotropic_property_value(self):
        """Test scalar extraction from normalized isotropic material properties."""
        mat = Material(permittivity=2.5)
        assert isotropic_property_value(mat.permittivity, "permittivity") == 2.5

        invalid_cases = [
            (Material(permittivity=(1.0, 2.0, 1.0)).permittivity, "isotropic"),
            (Material(permittivity=True).permittivity, "finite real isotropic"),
            (Material(permittivity=1.0 + 0.1j).permittivity, "real"),
            (Material(permittivity=float("inf")).permittivity, "finite"),
        ]
        for prop, match in invalid_cases:
            with pytest.raises(ValueError, match=match):
                isotropic_property_value(prop, "permittivity")

    def test_is_all_diagonally_anisotropic_property(self):
        """Test the is_all_diagonally_anisotropic property."""
        mat = Material(permittivity=2.0, permeability=1.5)
        assert mat.is_all_diagonally_anisotropic is True

        mat = Material(permittivity=(2.0, 2.0, 2.0), permeability=(1.0, 1.0, 1.0))
        assert mat.is_all_diagonally_anisotropic is True

        mat = Material(permittivity=(2.0, 2.5, 3.0), permeability=(1.0, 1.0, 1.0))
        assert mat.is_all_diagonally_anisotropic is True

        mat = Material(
            permittivity=(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0),
            permeability=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        assert mat.is_all_diagonally_anisotropic is True

        mat = Material(
            permittivity=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)),
            permeability=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        assert mat.is_all_diagonally_anisotropic is True

        mat = Material(
            permittivity=(2.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 3.0),
            permeability=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        assert mat.is_all_diagonally_anisotropic is True

        mat = Material(
            permittivity=((2.0, 0.0, 0.0), (0.0, 2.5, 0.0), (0.0, 0.0, 3.0)),
            permeability=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        assert mat.is_all_diagonally_anisotropic is True

        mat = Material(
            permittivity=(2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0),
            permeability=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        assert mat.is_all_diagonally_anisotropic is False

        mat = Material(
            permittivity=((2.0, 2.5, 3.0), (3.5, 4.0, 4.5), (5.0, 5.5, 6.0)),
            permeability=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
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

        mat = Material(permittivity=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        assert mat.is_isotropic_permittivity is True

        mat = Material(permittivity=(1.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.0))
        assert mat.is_isotropic_permittivity is False

        mat = Material(permittivity=((1.0, 0.0, 0.0), (0.0, 1.5, 0.0), (0.0, 0.0, 2.0)))
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

        mat = Material(permittivity=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        assert mat.is_diagonally_anisotropic_permittivity is True

        mat = Material(permittivity=(1.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.0))
        assert mat.is_diagonally_anisotropic_permittivity is True

        mat = Material(permittivity=((1.0, 0.0, 0.0), (0.0, 1.5, 0.0), (0.0, 0.0, 2.0)))
        assert mat.is_diagonally_anisotropic_permittivity is True

        mat = Material(permittivity=(1.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0))
        assert mat.is_diagonally_anisotropic_permittivity is False

        mat = Material(permittivity=((1.0, 2.5, 3.0), (3.5, 4.0, 4.5), (5.0, 5.5, 6.0)))
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

        mat = Material(permeability=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        assert mat.is_magnetic is False

        mat = Material(permeability=(1.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.0))
        assert mat.is_magnetic is True

        mat = Material(permeability=((1.0, 0.0, 0.0), (0.0, 1.5, 0.0), (0.0, 0.0, 2.0)))
        assert mat.is_magnetic is True

        mat = Material(permeability=(1.0, 1.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        assert mat.is_magnetic is True

        mat = Material(permeability=((1.0, 1.5, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
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

        mat = Material(electric_conductivity=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        assert mat.is_electrically_conductive is False

        mat = Material(electric_conductivity=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert mat.is_electrically_conductive is True

        mat = Material(electric_conductivity=((0.1, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
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

        mat = Material(magnetic_conductivity=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        assert mat.is_magnetically_conductive is False

        mat = Material(magnetic_conductivity=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert mat.is_magnetically_conductive is True

        mat = Material(magnetic_conductivity=((0.1, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
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
        result = compute_allowed_permittivities(
            materials, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )

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
        result = compute_allowed_permittivities(
            materials, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )

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
        result = compute_allowed_permittivities(
            materials, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )

        assert len(result) == 2
        assert all(isinstance(p, tuple) for p in result)
        assert all(len(p) == 9 for p in result)

        # Should be sorted by first component
        assert result[0] == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        assert result[1] == (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)

    def test_compute_allowed_permeabilities_fully_anisotropic_nested_tuple(self):
        """Test compute_allowed_permeabilities_fully_anisotropic_nested_tuple returns list of tuples."""
        materials = {
            "air": Material(permeability=1.0),
            "aniso": Material(permeability=((2.0, 2.5, 3.0), (3.5, 4.0, 4.5), (5.0, 5.5, 6.0))),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials.values()])
        result = compute_allowed_permeabilities(
            materials, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )

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
        result = compute_allowed_permeabilities(
            materials, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )

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
        result = compute_allowed_electric_conductivities(
            materials, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )

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
        result = compute_allowed_magnetic_conductivities(
            materials, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )

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
        result = compute_allowed_permittivities(
            materials, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )

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
        perms = compute_allowed_permittivities(
            materials_dict, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )
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
        perms = compute_allowed_permittivities(
            materials_dict, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )
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
        perms = compute_allowed_permittivities(
            materials_dict, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )
        perm_array = jnp.asarray(perms)

        # Should have shape (num_materials, 9)
        assert perm_array.shape == (2, 9)
        assert jnp.allclose(perm_array[0], jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
        assert jnp.allclose(perm_array[1], jnp.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]))

    def test_permittivity_array_conversion_fully_anisotropic_nested_tuple(self):
        """Test that fully anisotropic permittivity nested tuples convert correctly to JAX arrays."""
        materials_dict = {
            "air": Material(permittivity=1.0),
            "aniso": Material(permittivity=((2.0, 2.5, 3.0), (3.5, 4.0, 4.5), (5.0, 5.5, 6.0))),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials_dict.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials_dict.values()])
        perms = compute_allowed_permittivities(
            materials_dict, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )
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
        perms = compute_allowed_permittivities(
            materials_dict, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )
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
        perms = compute_allowed_permittivities(
            materials_dict, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )
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
        perms = compute_allowed_permittivities(
            materials_dict, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )
        perm_matrix = jnp.array(perms).reshape(3, 3)
        inv_perms = jnp.linalg.inv(perm_matrix).flatten()

        expected = jnp.array([0.5, 0.0, 0.0, -0.5, 0.5, 0.0, 0.0, -0.5, 0.5])
        assert jnp.allclose(inv_perms, expected)

    def test_inverse_permittivity_calculation_fully_anisotropic_nested_tuple(self):
        """Test that inverse permittivity is calculated correctly for fully anisotropic nested tuple materials."""
        materials_dict = {
            "mat": Material(permittivity=((2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (2.0, 2.0, 2.0))),
        }
        is_isotropic = all([mat.is_all_isotropic for mat in materials_dict.values()])
        is_diagonally_anisotropic = all([mat.is_all_diagonally_anisotropic for mat in materials_dict.values()])
        perms = compute_allowed_permittivities(
            materials_dict, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )
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
        perms = compute_allowed_permittivities(
            materials_dict, isotropic=is_isotropic, diagonally_anisotropic=is_diagonally_anisotropic
        )
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


class TestLossyMaterialConstructors:
    """Tests for the complex-permittivity / refractive-index / loss-tangent constructors.

    These convert a frequency-domain loss specification into the equivalent real
    permittivity + electric conductivity (and the magnetic dual) at a reference
    frequency: sigma = omega0 * eps0 * eps'' (sign convention exp(-i omega t),
    so positive imaginary part = loss).
    """

    _WL = 1.55e-6  # reference wavelength (m)

    def _omega(self):
        import math

        from fdtdx import constants

        return 2.0 * math.pi * constants.c / self._WL

    def test_from_complex_permittivity_scalar(self):
        import math

        from fdtdx import constants

        mat = Material.from_complex_permittivity(4.0 + 0.5j, wavelength=self._WL)
        omega = self._omega()
        assert math.isclose(mat.permittivity[0], 4.0)
        assert math.isclose(mat.permittivity[4], 4.0)
        assert math.isclose(mat.permittivity[8], 4.0)
        assert math.isclose(mat.electric_conductivity[0], omega * constants.eps0 * 0.5, rel_tol=1e-12)
        assert mat.is_electrically_conductive
        # purely electric loss leaves the material non-magnetic
        assert not mat.is_magnetic
        assert not mat.is_magnetically_conductive

    def test_from_refractive_index_scalar(self):
        import math

        from fdtdx import constants

        n, k = 1.5, 0.01
        mat = Material.from_refractive_index(n + 1j * k, wavelength=self._WL)
        omega = self._omega()
        assert math.isclose(mat.permittivity[0], n**2 - k**2, rel_tol=1e-12)
        assert math.isclose(mat.electric_conductivity[0], 2 * omega * constants.eps0 * n * k, rel_tol=1e-9)

    def test_from_refractive_index_lossless_has_no_conductivity(self):
        import math

        mat = Material.from_refractive_index(1.5, wavelength=self._WL)
        assert math.isclose(mat.permittivity[0], 2.25, rel_tol=1e-12)
        assert not mat.is_electrically_conductive
        assert mat.electric_conductivity == (0.0,) * 9

    def test_from_loss_tangent_scalar(self):
        import math

        from fdtdx import constants

        eps_real, tand = 4.0, 0.01
        mat = Material.from_loss_tangent(eps_real, tand, wavelength=self._WL)
        omega = self._omega()
        assert math.isclose(mat.permittivity[0], eps_real)
        assert math.isclose(mat.electric_conductivity[0], omega * constants.eps0 * eps_real * tand, rel_tol=1e-12)

    def test_diagonal_anisotropy(self):
        import math

        from fdtdx import constants

        mat = Material.from_complex_permittivity((2.0 + 0.1j, 3.0 + 0.2j, 4.0 + 0.0j), wavelength=self._WL)
        omega = self._omega()
        assert mat.permittivity == (2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0)
        assert math.isclose(mat.electric_conductivity[0], omega * constants.eps0 * 0.1, rel_tol=1e-12)
        assert math.isclose(mat.electric_conductivity[4], omega * constants.eps0 * 0.2, rel_tol=1e-12)
        assert math.isclose(mat.electric_conductivity[8], 0.0, abs_tol=1e-30)

    def test_complex_permeability_dual(self):
        import math

        from fdtdx import constants

        mat = Material.from_complex_permittivity(2.0 + 0.0j, wavelength=self._WL, permeability=1.0 + 0.05j)
        omega = self._omega()
        assert math.isclose(mat.permeability[0], 1.0)
        assert math.isclose(mat.magnetic_conductivity[0], omega * constants.mu0 * 0.05, rel_tol=1e-12)
        assert mat.is_magnetically_conductive

    def test_reference_via_wave_character(self):
        import math

        from fdtdx.core.wavelength import WaveCharacter

        mat_wl = Material.from_complex_permittivity(4.0 + 0.5j, wavelength=self._WL)
        mat_wc = Material.from_complex_permittivity(4.0 + 0.5j, reference=WaveCharacter(wavelength=self._WL))
        assert math.isclose(mat_wl.electric_conductivity[0], mat_wc.electric_conductivity[0], rel_tol=1e-12)

    def test_reference_via_frequency_matches_wavelength(self):
        import math

        from fdtdx import constants

        freq = constants.c / self._WL
        mat_f = Material.from_complex_permittivity(4.0 + 0.5j, frequency=freq)
        mat_wl = Material.from_complex_permittivity(4.0 + 0.5j, wavelength=self._WL)
        assert math.isclose(mat_f.electric_conductivity[0], mat_wl.electric_conductivity[0], rel_tol=1e-12)

    def test_gain_is_allowed(self):
        # No gain guard: a negative imaginary part yields a negative conductivity.
        mat = Material.from_complex_permittivity(4.0 - 0.5j, wavelength=self._WL)
        assert mat.electric_conductivity[0] < 0.0

    def test_requires_exactly_one_reference(self):
        import pytest

        with pytest.raises(ValueError):
            Material.from_complex_permittivity(4.0 + 0.5j)
        with pytest.raises(ValueError):
            Material.from_complex_permittivity(4.0 + 0.5j, wavelength=self._WL, frequency=1e14)

    def test_loss_tangent_length_mismatch_raises(self):
        import pytest

        with pytest.raises(ValueError):
            Material.from_loss_tangent((4.0, 3.0, 2.0), (0.01, 0.02), wavelength=self._WL)

    def test_nested_tuple_rejected(self):
        import pytest

        with pytest.raises(ValueError):
            Material.from_complex_permittivity(((2.0 + 0.1j, 0.0, 0.0),), wavelength=self._WL)


class TestStaticNegativePermittivityWarning:
    def test_negative_diagonal_entry_warns(self):
        with pytest.warns(UserWarning, match="unconditionally unstable"):
            Material(permittivity=(-2.0, 1.0, 1.0))

    def test_zero_diagonal_entry_warns(self):
        with pytest.warns(UserWarning, match="unconditionally unstable"):
            Material(permittivity=0.0)

    def test_negative_eps_inf_warns_even_with_dispersion(self):
        from fdtdx.dispersion import DispersionModel, DrudePole

        disp = DispersionModel(poles=(DrudePole(plasma_frequency=2e15, damping=1e13),))
        with pytest.warns(UserWarning, match="unconditionally unstable"):
            Material(permittivity=-2.0, dispersion=disp)

    def test_positive_permittivity_does_not_warn(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Material(permittivity=2.25)
            Material(permittivity=(2.0, 3.0, 4.0))
            # negative off-diagonals of a positive-definite tensor are fine
            Material(permittivity=(2.0, -0.5, 0.0, -0.5, 2.0, 0.0, 0.0, 0.0, 2.0))


class TestHasIsotropicDispersion:
    def test_non_dispersive_material(self):
        assert Material(permittivity=2.25).has_isotropic_dispersion

    def test_isotropic_dispersion(self):
        from fdtdx.dispersion import DispersionModel, LorentzPole

        disp = DispersionModel(poles=(LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0),))
        assert Material(permittivity=2.25, dispersion=disp).has_isotropic_dispersion

    def test_per_axis_dispersion(self):
        from fdtdx.dispersion import DispersionModel, DrudePole

        disp = DispersionModel(poles=(DrudePole(plasma_frequency=(2e15, 0.0, 0.0), damping=1e13),))
        assert not Material(permittivity=2.25, dispersion=disp).has_isotropic_dispersion


class TestDispersiveDivisorStability:
    """Per-material implicit-update divisor validation for CCPR poles.

    The CCPR polarization couples to E^{n+1} through c4, so update_E divides by
    ``divisor = 1 + inv_eps * sum(c4) [+ conductivity term]``. A large negative
    Re(residue) makes c4 negative and can drive the divisor to zero (huge ring-up)
    or below zero (NaN). dt ~ 1.9e-16 corresponds to a ~1e-7 m grid at
    courant_factor 0.99.
    """

    DT = 1.9e-16

    @staticmethod
    def _ccpr_material(re_residue, eps_inf=2.0, electric_conductivity=0.0):
        from fdtdx.dispersion import CCPRPole, DispersionModel

        pole = CCPRPole(pole=complex(-1e13, -2e15), residue=complex(re_residue, 1e15))
        return Material(
            permittivity=eps_inf,
            electric_conductivity=electric_conductivity,
            dispersion=DispersionModel(poles=(pole,)),
        )

    def test_non_positive_divisor_raises_with_name_and_courant_hint(self):
        from fdtdx.materials import validate_dispersive_divisor_stability

        mat = self._ccpr_material(-6e15)  # drives the divisor below zero
        with pytest.raises(ValueError, match="gold") as exc:
            validate_dispersive_divisor_stability({"gold": mat}, dt=self.DT, courant_factor=0.99)
        msg = str(exc.value)
        assert "non-positive" in msg
        assert "courant_factor" in msg

    def test_near_zero_divisor_warns(self):
        from fdtdx.materials import validate_dispersive_divisor_stability

        # A mildly-negative residue keeps the divisor positive but small; a wide
        # explicit threshold makes the warn band robust (default 0.01 is narrow).
        mat = self._ccpr_material(-4e15)
        with pytest.warns(UserWarning, match="courant_factor"):
            validate_dispersive_divisor_stability(
                {"gold": mat}, dt=self.DT, courant_factor=0.99, near_zero_threshold=0.5
            )

    def test_recommended_courant_factor_is_below_current_and_stabilizes(self):
        from fdtdx.materials import _min_dispersive_divisor, validate_dispersive_divisor_stability

        mat = self._ccpr_material(-6e15)
        with pytest.raises(ValueError) as exc:
            validate_dispersive_divisor_stability({"gold": mat}, dt=self.DT, courant_factor=0.99)
        # Parse the recommended courant_factor out of the message.
        msg = str(exc.value)
        cf_max = float(msg.split("lower courant_factor to <= ")[1].split(" ")[0])
        assert 0.0 < cf_max < 0.99
        # dt scales with courant_factor: at the recommended cf the divisor recovers.
        dt_safe = self.DT * (cf_max / 0.99)
        assert _min_dispersive_divisor(mat, dt_safe)[0] >= 0.01 - 1e-9

    def test_recommended_courant_factor_is_conservative_for_mixed_sign_poles(self):
        # The divisor is a sum of individually-monotonic c4 terms with mixed
        # residue signs, so it need not be monotonic in the time step. The
        # recommendation must be the FIRST crossing from zero: EVERY scale below
        # it must be stable, not just the returned point.
        from fdtdx.dispersion import CCPRPole, DispersionModel
        from fdtdx.materials import _min_dispersive_divisor, validate_dispersive_divisor_stability

        poles = (
            CCPRPole(pole=complex(-2e14, -2e15), residue=complex(-1.2e16, 1e15)),  # strong b < 0
            CCPRPole(pole=complex(-4e15, -8e15), residue=complex(3e15, 5e14)),  # b > 0
        )
        mat = Material(permittivity=2.0, dispersion=DispersionModel(poles=poles))
        with pytest.raises(ValueError) as exc:
            validate_dispersive_divisor_stability({"m": mat}, dt=self.DT, courant_factor=0.99)
        cf_max = float(str(exc.value).split("lower courant_factor to <= ")[1].split(" ")[0])
        s_max = cf_max / 0.99
        # Conservativeness invariant: divisor >= threshold at every scale up to s_max.
        scales = [s_max * k / 400 for k in range(1, 401)]
        assert min(_min_dispersive_divisor(mat, s * self.DT)[0] for s in scales) >= 0.01 - 1e-9

    def test_stable_ccpr_material_does_not_raise_or_warn(self):
        import warnings

        from fdtdx.materials import validate_dispersive_divisor_stability

        mat = self._ccpr_material(-2e15)  # divisor ~ 0.6, comfortably positive
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_dispersive_divisor_stability({"safe": mat}, dt=self.DT, courant_factor=0.99)

    def test_lorentz_and_drude_are_skipped(self):
        import warnings

        from fdtdx.dispersion import DispersionModel, DrudePole, LorentzPole
        from fdtdx.materials import validate_dispersive_divisor_stability

        lorentz = Material(
            permittivity=2.0,
            dispersion=DispersionModel(poles=(LorentzPole(resonance_frequency=2e15, damping=1e13, delta_epsilon=1.5),)),
        )
        drude = Material(
            permittivity=1.0,
            dispersion=DispersionModel(poles=(DrudePole(plasma_frequency=1.37e16, damping=1e14),)),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_dispersive_divisor_stability({"lorentz": lorentz, "drude": drude}, dt=self.DT, courant_factor=0.99)

    def test_per_axis_instability_names_axis(self):
        from fdtdx.materials import validate_dispersive_divisor_stability

        # Anisotropic eps_inf: only the low-permittivity x axis destabilizes.
        mat = self._ccpr_material(-6e15, eps_inf=(2.0, 8.0, 8.0))
        with pytest.raises(ValueError, match="axis x"):
            validate_dispersive_divisor_stability({"aniso": mat}, dt=self.DT, courant_factor=0.99)

    def test_conductivity_term_relaxes_the_bound(self):
        from fdtdx.materials import _min_dispersive_divisor

        # The conductivity term is >= 0, so adding loss increases the divisor.
        lossless = self._ccpr_material(-6e15, electric_conductivity=0.0)
        lossy = self._ccpr_material(-6e15, electric_conductivity=5e5)
        assert _min_dispersive_divisor(lossy, self.DT)[0] > _min_dispersive_divisor(lossless, self.DT)[0]

    def test_oriented_pole_alongside_ccpr_is_handled(self):
        import warnings

        from fdtdx.dispersion import CCPRPole, DispersionModel, LorentzPole
        from fdtdx.materials import validate_dispersive_divisor_stability

        # An oriented pole has no dE/dt coupling and must not trip the
        # per-axis-only coefficient path when a CCPR pole sits next to it.
        poles = (
            LorentzPole(resonance_frequency=2e15, damping=1e13, delta_epsilon=1.5, orientation=(1.0, 1.0, 0.0)),
            CCPRPole(pole=complex(-1e13, -2e15), residue=complex(-2e15, 1e15)),
        )
        mat = Material(permittivity=2.0, dispersion=DispersionModel(poles=poles))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_dispersive_divisor_stability({"mixed": mat}, dt=self.DT, courant_factor=0.99)


class TestComplexTensorConstructors:
    """Complex full-tensor (9-component / nested 3x3) permittivity constructors."""

    _WL = 1.55e-6  # reference wavelength (m)

    def _omega(self):
        import math

        from fdtdx import constants

        return 2.0 * math.pi * constants.c / self._WL

    def test_from_complex_permittivity_flat_9_tuple(self):
        import math

        from fdtdx import constants

        eps = (4.0 + 0.5j, 0.5 + 0.1j, 0.0, 0.5 + 0.1j, 3.0 + 0.2j, 0.0, 0.0, 0.0, 2.0 + 0.0j)
        mat = Material.from_complex_permittivity(eps, wavelength=self._WL)
        omega = self._omega()
        for i, e in enumerate(eps):
            assert math.isclose(mat.permittivity[i], complex(e).real, abs_tol=1e-15)
            assert math.isclose(
                mat.electric_conductivity[i], omega * constants.eps0 * complex(e).imag, rel_tol=1e-12, abs_tol=1e-15
            )
        assert not mat.is_diagonally_anisotropic_permittivity

    def test_from_complex_permittivity_nested_3x3(self):
        nested = (
            (4.0 + 0.5j, 0.5 + 0.1j, 0.0),
            (0.5 + 0.1j, 3.0 + 0.2j, 0.0),
            (0.0, 0.0, 2.0 + 0.0j),
        )
        flat = tuple(entry for row in nested for entry in row)
        mat_nested = Material.from_complex_permittivity(nested, wavelength=self._WL)
        mat_flat = Material.from_complex_permittivity(flat, wavelength=self._WL)
        assert mat_nested.permittivity == mat_flat.permittivity
        assert mat_nested.electric_conductivity == mat_flat.electric_conductivity

    def test_from_complex_permittivity_hermitian_gives_antisymmetric_sigma(self):
        # A Hermitian eps (gyrotropic: imaginary off-diagonals) maps to an
        # antisymmetric real conductivity tensor.
        g = 0.3
        eps = (2.0, 1j * g, 0.0, -1j * g, 2.0, 0.0, 0.0, 0.0, 2.0)
        mat = Material.from_complex_permittivity(eps, wavelength=self._WL)
        assert mat.permittivity == (2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0)
        assert mat.electric_conductivity[1] == pytest.approx(-mat.electric_conductivity[3])
        assert mat.electric_conductivity[1] != 0.0

    def test_from_complex_permittivity_tensor_permeability(self):
        import math

        from fdtdx import constants

        mu = (1.0 + 0.1j, 0.0, 0.0, 0.0, 2.0 + 0.0j, 0.0, 0.0, 0.0, 1.5 + 0.05j)
        mat = Material.from_complex_permittivity(4.0, permeability=mu, wavelength=self._WL)
        omega = self._omega()
        assert mat.permeability == (1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.5)
        assert math.isclose(mat.magnetic_conductivity[0], omega * constants.mu0 * 0.1, rel_tol=1e-12)
        assert math.isclose(mat.magnetic_conductivity[8], omega * constants.mu0 * 0.05, rel_tol=1e-12)

    def test_from_complex_permittivity_singular_real_part_raises(self):
        # Real part with a zero eigenvalue (zz entry zero) cannot be inverted.
        eps = (2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0 + 0.5j)
        with pytest.raises(ValueError, match="singular"):
            Material.from_complex_permittivity(eps, wavelength=self._WL)

    def test_from_complex_permittivity_singular_permeability_raises(self):
        mu = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 + 0.5j)
        with pytest.raises(ValueError, match="permeability tensor is singular"):
            Material.from_complex_permittivity(4.0, permeability=mu, wavelength=self._WL)

    def test_from_complex_permittivity_malformed_nested_raises(self):
        with pytest.raises(ValueError, match="3x3"):
            Material.from_complex_permittivity(((1.0, 0.0), (0.0, 1.0)), wavelength=self._WL)

    def test_from_refractive_index_rejects_full_tensor(self):
        with pytest.raises(ValueError, match="from_complex_permittivity"):
            Material.from_refractive_index((1.5, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.5), wavelength=self._WL)
        with pytest.raises(ValueError, match="from_complex_permittivity"):
            Material.from_refractive_index(((1.5, 0.0, 0.0), (0.0, 1.5, 0.0), (0.0, 0.0, 1.5)), wavelength=self._WL)

    def test_from_loss_tangent_flat_9_tuple(self):
        import math

        from fdtdx import constants

        eps = (4.0, 0.5, 0.0, 0.5, 3.0, 0.0, 0.0, 0.0, 2.0)
        tand = 0.01
        mat = Material.from_loss_tangent(eps, tand, wavelength=self._WL)
        omega = self._omega()
        assert mat.permittivity == eps
        for i, e in enumerate(eps):
            assert math.isclose(
                mat.electric_conductivity[i], omega * constants.eps0 * e * tand, rel_tol=1e-12, abs_tol=1e-18
            )

    def test_from_loss_tangent_rejects_nested(self):
        with pytest.raises(ValueError, match="nested"):
            Material.from_loss_tangent(((4.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 2.0)), 0.01, wavelength=self._WL)
