from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit


@extended_autoinit
class Material(ExtendedTreeClass):
    permittivity: float
    permeability: float = 1.0
    conductivity: float = 0.0




def ordered_material_list(
    materials: dict[str, Material],
) -> list[Material]:
    """
    Returns a list of materials ordered by their properties.
    
    The ordering priority is:
    1. Permittivity (ascending)
    2. Permeability (ascending)
    3. Conductivity (ascending)
    
    Args:
        materials: Dictionary mapping material names to Material objects
        
    Returns:
        List of Material objects ordered by their properties
    """
    return sorted(
        materials.values(),
        key=lambda m: (m.permittivity, m.permeability, m.conductivity)
    )
    

def allowed_permittivities(
    materials: dict[str, Material],
) -> list[float]:
    ordered_materials = ordered_material_list(materials)
    return [o.permittivity for o in ordered_materials]


@extended_autoinit
class ContinuousMaterialRange(ExtendedTreeClass):
    start_material: Material
    end_material: Material
