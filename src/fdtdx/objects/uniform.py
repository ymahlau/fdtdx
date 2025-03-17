from fdtdx.core.jax.pytrees import extended_autoinit, field, frozen_field
from fdtdx.core.plotting.colors import LIGHT_BLUE, LIGHT_BROWN, LIGHT_GREY
from fdtdx.materials import Material
from fdtdx.objects.object import SimulationObject


@extended_autoinit
class UniformMaterialObject(SimulationObject):
    """Object with uniform permittivity and permeability throughout its volume.

    A material object that applies constant permittivity and permeability values
    to its entire volume in the simulation grid. Used as base class for specific
    material implementations.

    Attributes:
        permittivity: Relative permittivity (εᵣ) of the material
        permeability: Relative permeability (μᵣ) of the material, defaults to 1.0
        color: RGB tuple for visualization, defaults to light grey
    """

    material: Material = field(kind="KW_ONLY")
    placement_order: int = frozen_field(default=0)
    color: tuple[float, float, float] = LIGHT_GREY


@extended_autoinit
class SimulationVolume(UniformMaterialObject):
    """Background material for the entire simulation volume.

    Defines the default material properties for the simulation background.
    Usually represents air/vacuum with εᵣ=1.0 and μᵣ=1.0.
    """
    placement_order = -1000
    material = Material(
        permittivity=1.0,
        permeability=1.0,
        conductivity=0.0,
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
