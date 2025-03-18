from fdtdx.core.jax.pytrees import extended_autoinit, field, frozen_field
from fdtdx.core.plotting.colors import LIGHT_BLUE, LIGHT_BROWN, LIGHT_GREY
from fdtdx.materials import Material
from fdtdx.objects.static_material.static import StaticMaterialObject


@extended_autoinit
class UniformMaterialObject(StaticMaterialObject):
    material: Material = field(kind="KW_ONLY")  # type: ignore
    color: tuple[float, float, float] = LIGHT_GREY


@extended_autoinit
class SimulationVolume(UniformMaterialObject):
    """Background material for the entire simulation volume.

    Defines the default material properties for the simulation background.
    Usually represents air/vacuum with εᵣ=1.0 and μᵣ=1.0.
    """
    placement_order = -1000
    material: Material = field(
        default = Material(
            permittivity=1.0,
            permeability=1.0,
            conductivity=0.0,
        ),
        kind="KW_ONLY",
    )


@extended_autoinit
class Substrate(UniformMaterialObject):
    """Material representing a substrate layer.

    Used to model substrate materials like silicon dioxide.
    Visualized in light brown color by default.
    """

    color: tuple[float, float, float] = LIGHT_BROWN


@extended_autoinit
class WaveGuide(UniformMaterialObject):
    """Material for optical waveguides.

    Used to model waveguide structures that can guide electromagnetic waves.
    Visualized in light blue color by default.

    Attributes:
        permittivity: Required relative permittivity of the waveguide material
        color: RGB tuple for visualization, defaults to light blue
    """
    
    color: tuple[float, float, float] = LIGHT_BLUE
