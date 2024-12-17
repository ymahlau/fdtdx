import jax
import jax.numpy as jnp
import pytreeclass as tc

from fdtdx.core.jax.pytrees import extended_autoinit, frozen_field
from fdtdx.core.plotting.colors import LIGHT_BLUE, LIGHT_BROWN, LIGHT_GREY
from fdtdx.objects.object import SimulationObject


@extended_autoinit
class UniformMaterial(SimulationObject):
    """Object with uniform permittivity and permeability throughout its volume.

    A material object that applies constant permittivity and permeability values
    to its entire volume in the simulation grid. Used as base class for specific
    material implementations.

    Attributes:
        permittivity: Relative permittivity (εᵣ) of the material
        permeability: Relative permeability (μᵣ) of the material, defaults to 1.0
        color: RGB tuple for visualization, defaults to light grey
    """

    permittivity: float = frozen_field(init=True, kind="KW_ONLY")  # type: ignore
    permeability: float = tc.field(default=1.0, init=True, kind="KW_ONLY")  # type: ignore
    color: tuple[float, float, float] = LIGHT_GREY

    def get_inv_permittivity(
        self,
        prev_inv_permittivity: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:  # permittivity and info dict
        """Calculate inverse permittivity for this material's volume.

        Args:
            prev_inv_permittivity: Previous inverse permittivity values
            params: Optional parameters for permittivity calculation

        Returns:
            tuple containing:
                - Array of inverse permittivity values (1/εᵣ)
                - Dictionary with additional information
        """
        del params
        res = jnp.ones_like(prev_inv_permittivity) / self.permittivity
        return res, {}

    def get_inv_permeability(
        self,
        prev_inv_permeability: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:  # permeability and info dict
        """Calculate inverse permeability for this material's volume.

        Args:
            prev_inv_permeability: Previous inverse permeability values
            params: Optional parameters for permeability calculation

        Returns:
            tuple containing:
                - Array of inverse permeability values (1/μᵣ)
                - Dictionary with additional information
        """
        del params
        res = jnp.ones_like(prev_inv_permeability) / self.permeability
        return res, {}


@tc.autoinit
class NoMaterial(SimulationObject):
    """
    Object that does not modify the permittivity nor permeability
    """

    placement_order: int = -1000

    def get_inv_permittivity(
        self,
        prev_inv_permittivity: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:  # permittivity and info dict
        del params
        return prev_inv_permittivity, {}

    def get_inv_permeability(
        self,
        prev_inv_permeability: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:  # permeability and info dict
        del params
        return prev_inv_permeability, {}


@tc.autoinit
class SimulationVolume(UniformMaterial):
    """Background material for the entire simulation volume.

    Defines the default material properties for the simulation background.
    Usually represents air/vacuum with εᵣ=1.0 and μᵣ=1.0.
    """

    permittivity: float = tc.field(default=1.0, init=True, kind="KW_ONLY")  # type: ignore
    permeability: float = tc.field(default=1.0, init=True, kind="KW_ONLY")  # type: ignore


@extended_autoinit
class Substrate(UniformMaterial):
    """Material representing a substrate layer.

    Used to model substrate materials like silicon dioxide.
    Visualized in light brown color by default.
    """

    color: tuple[float, float, float] = LIGHT_BROWN


@tc.autoinit
class WaveGuide(UniformMaterial):
    """Material for optical waveguides.

    Used to model waveguide structures that can guide electromagnetic waves.
    Visualized in light blue color by default.

    Attributes:
        permittivity: Required relative permittivity of the waveguide material
        color: RGB tuple for visualization, defaults to light blue
    """

    permittivity: float = tc.field(init=True, kind="KW_ONLY")  # type: ignore
    color: tuple[float, float, float] = LIGHT_BLUE
