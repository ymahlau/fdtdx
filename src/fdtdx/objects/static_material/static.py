from fdtdx.core.jax.pytrees import extended_autoinit, field
from fdtdx.materials import ContinuousMaterialRange, Material
from fdtdx.objects.object import OrderableObject


@extended_autoinit
class StaticMaterialObject(OrderableObject):
    material: Material | dict[str, Material] | ContinuousMaterialRange = field(kind="KW_ONLY")  # type: ignore
