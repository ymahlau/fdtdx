import math
from typing import overload

from fdtdx.core.jax.pytrees import TreeClass, frozen_field


def _normalize_material_property(value: float | tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalize material property to a 3-tuple for (x, y, z) components.

    Args:
        value: Either a scalar (isotropic) or 3-tuple (anisotropic) material property

    Returns:
        tuple[float, float, float]: Material property as (x, y, z) components
    """
    if isinstance(value, tuple):
        if len(value) != 3:
            raise ValueError(f"Material property tuple must have exactly 3 elements, got {len(value)}")
        return value
    else:
        # Isotropic: broadcast scalar to all three components
        return (value, value, value)


class Material(TreeClass):
    """
    Represents an electromagnetic material with specific electrical and magnetic properties.

    This class stores the fundamental electromagnetic properties of a material for use
    in electromagnetic simulations. Supports both isotropic and anisotropic materials.

    Note:
        All material properties are stored internally as 3-tuples (x, y, z components).
        Scalar inputs are automatically broadcast to all three components.

    """

    #: The relative permittivity (dielectric constant) of the material, which describes how the electric field is affected by the material.
    #: Higher values indicate greater electric polarization in response to an applied electric field.
    #: For isotropic materials, provide a scalar float. For anisotropic materials, provide a tuple of 3 floats (εx, εy, εz).
    #: Stored internally as a 3-tuple. Defaults to (1.0, 1.0, 1.0).
    permittivity: tuple[float, float, float] = frozen_field(
        default=(1.0, 1.0, 1.0),
        on_setattr=[_normalize_material_property],
    )

    #: The relative permeability of the material, which describes how the magnetic field is affected by the material.
    #: Higher values indicate greater magnetic response to an applied magnetic field.
    #: For isotropic materials, provide a scalar float. For anisotropic materials, provide a tuple of 3 floats (μx, μy, μz).
    #: Stored internally as a 3-tuple. Defaults to (1.0, 1.0, 1.0).
    permeability: tuple[float, float, float] = frozen_field(
        default=(1.0, 1.0, 1.0),
        on_setattr=[_normalize_material_property],
    )

    #: The electrical conductivity of the material in siemens per meter (S/m), which describes how easily electric current can flow through it.
    #: Higher values indicate materials that conduct electricity more easily.
    #: For isotropic materials, provide a scalar float. For anisotropic materials, provide a tuple of 3 floats (σx, σy, σz).
    #: Stored internally as a 3-tuple. Defaults to (0.0, 0.0, 0.0).
    electric_conductivity: tuple[float, float, float] = frozen_field(
        default=(0.0, 0.0, 0.0),
        on_setattr=[_normalize_material_property],
    )

    #: The magnetic conductivity, or magnetic loss of the material.
    #: This is an artificial parameter for numerical applications and does not represent an actual physical unit,
    #: even though often described in Ohm/m. The naming can be misleading, because it does not actually describe
    #:  a conductivity, but rather an "equivalent magnetic loss parameter".
    #: For isotropic materials, provide a scalar float. For anisotropic materials, provide a tuple of 3 floats.
    #: Stored internally as a 3-tuple. Defaults to (0.0, 0.0, 0.0).
    magnetic_conductivity: tuple[float, float, float] = frozen_field(
        default=(0.0, 0.0, 0.0),
        on_setattr=[_normalize_material_property],
    )

    @overload
    def __init__(
        self,
        *,
        permittivity: float = 1.0,
        permeability: float = 1.0,
        electric_conductivity: float = 0.0,
        magnetic_conductivity: float = 0.0,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        permittivity: tuple[float, float, float] = (1.0, 1.0, 1.0),
        permeability: tuple[float, float, float] = (1.0, 1.0, 1.0),
        electric_conductivity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        magnetic_conductivity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None: ...

    def __init__(
        self,
        *,
        permittivity: float | tuple[float, float, float] = (1.0, 1.0, 1.0),
        permeability: float | tuple[float, float, float] = (1.0, 1.0, 1.0),
        electric_conductivity: float | tuple[float, float, float] = (0.0, 0.0, 0.0),
        magnetic_conductivity: float | tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        object.__setattr__(self, "permittivity", _normalize_material_property(permittivity))
        object.__setattr__(self, "permeability", _normalize_material_property(permeability))
        object.__setattr__(self, "electric_conductivity", _normalize_material_property(electric_conductivity))
        object.__setattr__(self, "magnetic_conductivity", _normalize_material_property(magnetic_conductivity))

    @property
    def is_all_isotropic(self) -> bool:
        """Check if all material properties are isotropic (all components equal).

        Returns:
            bool: True if material is isotropic, False if anisotropic
        """

        return (
            _is_property_isotropic(self.permittivity)
            and _is_property_isotropic(self.permeability)
            and _is_property_isotropic(self.electric_conductivity)
            and _is_property_isotropic(self.magnetic_conductivity)
        )

    @property
    def is_isotropic_permittivity(self) -> bool:
        """Check if material has isotropic permittivity (all components equal).

        Returns:
            bool: True if material has isotropic permittivity
        """
        return _is_property_isotropic(self.permittivity)

    @property
    def is_isotropic_permeability(self) -> bool:
        """Check if material has isotropic permeability (all components equal).

        Returns:
            bool: True if material has isotropic permeability
        """
        return _is_property_isotropic(self.permeability)

    @property
    def is_isotropic_electric_conductivity(self) -> bool:
        """Check if material has isotropic electric conductivity (all components equal).

        Returns:
            bool: True if material has isotropic electric conductivity
        """
        return _is_property_isotropic(self.electric_conductivity)

    @property
    def is_isotropic_magnetic_conductivity(self) -> bool:
        """Check if material has isotropic magnetic conductivity (all components equal).

        Returns:
            bool: True if material has isotropic magnetic conductivity
        """
        return _is_property_isotropic(self.magnetic_conductivity)

    @property
    def is_magnetic(self) -> bool:
        """Check if material has magnetic properties (permeability != 1.0 for any component).

        Returns:
            bool: True if material is magnetic
        """
        perm = self.permeability
        if isinstance(perm[0], complex) or isinstance(perm[1], complex) or isinstance(perm[2], complex):
            return True
        return not (math.isclose(perm[0], 1.0) and math.isclose(perm[1], 1.0) and math.isclose(perm[2], 1.0))

    @property
    def is_electrically_conductive(self) -> bool:
        """Check if material is electrically conductive (conductivity != 0.0 for any component).

        Returns:
            bool: True if material is electrically conductive
        """
        cond = self.electric_conductivity
        return not (math.isclose(cond[0], 0.0) and math.isclose(cond[1], 0.0) and math.isclose(cond[2], 0.0))

    @property
    def is_magnetically_conductive(self) -> bool:
        """Check if material has magnetic conductivity (magnetic loss != 0.0 for any component).

        Returns:
            bool: True if material has magnetic conductivity
        """
        cond = self.magnetic_conductivity
        return not (math.isclose(cond[0], 0.0) and math.isclose(cond[1], 0.0) and math.isclose(cond[2], 0.0))


def _is_property_isotropic(prop: tuple[float, float, float]) -> bool:
    return math.isclose(prop[0], prop[1]) and math.isclose(prop[1], prop[2])


def compute_ordered_material_name_tuples(
    materials: dict[str, Material],
) -> list[tuple[str, Material]]:
    """
    Returns a list of materials ordered by their properties.

    The ordering priority is:
    1. Permittivity (ascending, using first component for ordering)
    2. Permeability (ascending, using first component for ordering)
    3. Electric conductivity (ascending, using first component for ordering)
    4. Magnetic conductivity (ascending, using first component for ordering)

    Args:
        materials (dict[str, Material]): Dictionary mapping material names to Material objects.

    Returns:
        list[tuple[str, Material]]: List of Material objects ordered by their properties.
    """
    return sorted(
        materials.items(),
        key=lambda m: (
            m[1].permittivity[0],
            m[1].permeability[0],
            m[1].electric_conductivity[0],
            m[1].magnetic_conductivity[0],
        ),
        reverse=False,
    )


def compute_allowed_permittivities(
    materials: dict[str, Material],
    isotropic: bool = False,
) -> list[tuple[float, ...]]:
    """Get list of permittivity tuples for all materials in sorted order.

    Args:
        materials: Dictionary mapping material names to Material objects
        isotropic: If True, return single-element tuples (εx,); if False, return 3-tuples (εx, εy, εz)

    Returns:
        list[tuple[float, ...]]: List of permittivity tuples
    """
    ordered_materials = compute_ordered_material_name_tuples(materials)
    if isotropic:
        return [(o[1].permittivity[0],) for o in ordered_materials]
    return [o[1].permittivity for o in ordered_materials]


def compute_allowed_permeabilities(
    materials: dict[str, Material],
    isotropic: bool = False,
) -> list[tuple[float, ...]]:
    """Get list of permeability tuples for all materials in sorted order.

    Args:
        materials: Dictionary mapping material names to Material objects
        isotropic: If True, return single-element tuples (μx,); if False, return 3-tuples (μx, μy, μz)

    Returns:
        list[tuple[float, ...]]: List of permeability tuples
    """
    ordered_materials = compute_ordered_material_name_tuples(materials)
    if isotropic:
        return [(o[1].permeability[0],) for o in ordered_materials]
    return [o[1].permeability for o in ordered_materials]


def compute_allowed_electric_conductivities(
    materials: dict[str, Material],
    isotropic: bool = False,
) -> list[tuple[float, ...]]:
    """Get list of electric conductivity tuples for all materials in sorted order.

    Args:
        materials: Dictionary mapping material names to Material objects
        isotropic: If True, return single-element tuples (σx,); if False, return 3-tuples (σx, σy, σz)

    Returns:
        list[tuple[float, ...]]: List of electric conductivity tuples
    """
    ordered_materials = compute_ordered_material_name_tuples(materials)
    if isotropic:
        return [(o[1].electric_conductivity[0],) for o in ordered_materials]
    return [o[1].electric_conductivity for o in ordered_materials]


def compute_allowed_magnetic_conductivities(
    materials: dict[str, Material],
    isotropic: bool = False,
) -> list[tuple[float, ...]]:
    """Get list of magnetic conductivity tuples for all materials in sorted order.

    Args:
        materials: Dictionary mapping material names to Material objects
        isotropic: If True, return single-element tuples; if False, return 3-tuples

    Returns:
        list[tuple[float, ...]]: List of magnetic conductivity tuples
    """
    ordered_materials = compute_ordered_material_name_tuples(materials)
    if isotropic:
        return [(o[1].magnetic_conductivity[0],) for o in ordered_materials]
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
