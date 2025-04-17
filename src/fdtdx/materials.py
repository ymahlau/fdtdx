import math

from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit


@extended_autoinit
class Material(ExtendedTreeClass):
    """
    Represents an electromagnetic material with specific electrical and magnetic properties.

    This class stores the fundamental electromagnetic properties of a material for use
    in electromagnetic simulations.

    Args:
        permittivity (float): The relative permittivity (dielectric constant) of the material,
            which describes how the electric field is affected by the material. Higher values
            indicate greater electric polarization in response to an applied electric field.

        permeability (float, optional): The relative permeability of the material, which
            describes how the magnetic field is affected by the material. Higher values
            indicate greater magnetic response to an applied magnetic field.
            Defaults to 1.0 (non-magnetic material).

        electric_conductivity (float, optional): The electrical conductivity of the material in siemens
            per meter (S/m), which describes how easily electric current can flow through it.
            Higher values indicate materials that conduct electricity more easily.
            Defaults to 0.0 (perfect insulator).

        magnetic_conductivity (float, optional): The magnetic conductivity, or magnetic loss of the material.
            This is an artificial parameter for numerical applications and does not represent an actual physical unit,
            even though often described in Ohm/m. The naming can be misleading, because it does not actually describe
            a conductivity, but rather an "equivalent magnetic loss parameter".
            Defaults to 0.0.
    """

    permittivity: float = 1.0
    permeability: float = 1.0
    electric_conductivity: float = 0.0
    magnetic_conductivity: float = 0.0

    @property
    def is_magnetic(self) -> bool:
        if isinstance(self.permeability, complex):
            return True
        return not math.isclose(self.permeability, 1.0)

    @property
    def is_electrically_conductive(self) -> bool:
        return not math.isclose(self.electric_conductivity, 0.0)

    @property
    def is_magnetically_conductive(self) -> bool:
        return not math.isclose(self.magnetic_conductivity, 0.0)


def compute_ordered_material_name_tuples(
    materials: dict[str, Material],
) -> list[tuple[str, Material]]:
    """
    Returns a list of materials ordered by their properties.

    The ordering priority is:
    1. Permittivity (descending)
    2. Permeability (descending)
    3. Electric conductivity (descending)
    4. Magnetic conductivity (descending)

    Args:
        materials: Dictionary mapping material names to Material objects

    Returns:
        List of Material objects ordered by their properties
    """
    return sorted(
        materials.items(),
        key=lambda m: (m[1].permittivity, m[1].permeability, m[1].electric_conductivity, m[1].magnetic_conductivity),
        reverse=True,
    )


def compute_allowed_permittivities(
    materials: dict[str, Material],
) -> list[float]:
    ordered_materials = compute_ordered_material_name_tuples(materials)
    return [o[1].permittivity for o in ordered_materials]


def compute_allowed_permeabilities(
    materials: dict[str, Material],
) -> list[float]:
    ordered_materials = compute_ordered_material_name_tuples(materials)
    return [o[1].permeability for o in ordered_materials]


def compute_allowed_electric_conductivities(
    materials: dict[str, Material],
) -> list[float]:
    ordered_materials = compute_ordered_material_name_tuples(materials)
    return [o[1].electric_conductivity for o in ordered_materials]


def compute_allowed_magnetic_conductivities(
    materials: dict[str, Material],
) -> list[float]:
    ordered_materials = compute_ordered_material_name_tuples(materials)
    return [o[1].magnetic_conductivity for o in ordered_materials]


def compute_ordered_names(
    materials: dict[str, Material],
) -> list[str]:
    ordered_materials = compute_ordered_material_name_tuples(materials)
    return [o[0] for o in ordered_materials]


def compute_ordered_materials(
    materials: dict[str, Material],
) -> list[Material]:
    ordered_materials = compute_ordered_material_name_tuples(materials)
    return [o[1] for o in ordered_materials]


@extended_autoinit
class ContinuousMaterialRange(ExtendedTreeClass):
    start_material: Material
    end_material: Material
