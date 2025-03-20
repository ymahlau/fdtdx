from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit


@extended_autoinit
class Material(ExtendedTreeClass):
    permittivity: float
    permeability: float = 1.0
    conductivity: float = 0.0


def compute_ordered_material_name_tuples(
    materials: dict[str, Material],
) -> list[tuple[str, Material]]:
    """
    Returns a list of materials ordered by their properties.

    The ordering priority is:
    1. Permittivity (descending)
    2. Permeability (descending)
    3. Conductivity (descending)

    Args:
        materials: Dictionary mapping material names to Material objects

    Returns:
        List of Material objects ordered by their properties
    """
    return sorted(
        materials.items(),
        key=lambda m: (m[1].permittivity, m[1].permeability, m[1].conductivity),
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
