import math

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field


@autoinit
class Material(TreeClass):
    """
    Represents an electromagnetic material with specific electrical and magnetic properties.

    This class stores the fundamental electromagnetic properties of a material for use
    in electromagnetic simulations.

    Attributes:
        permittivity (float, optional): The relative permittivity (dielectric constant) of the material,
            which describes how the electric field is affected by the material. Higher values
            indicate greater electric polarization in response to an applied electric field.
            Defaults to 1.0.
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

    permittivity: float = frozen_field(default=1.0)
    permeability: float = frozen_field(default=1.0)
    electric_conductivity: float = frozen_field(default=0.0)
    magnetic_conductivity: float = frozen_field(default=0.0)

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
    1. Permittivity (ascending)
    2. Permeability (ascending)
    3. Electric conductivity (ascending)
    4. Magnetic conductivity (ascending)

    Args:
        materials (dict[str, Material]): Dictionary mapping material names to Material objects.

    Returns:
        list[tuple[str, Material]]: List of Material objects ordered by their properties.
    """
    return sorted(
        materials.items(),
        key=lambda m: (m[1].permittivity, m[1].permeability, m[1].electric_conductivity, m[1].magnetic_conductivity),
        reverse=False,
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
