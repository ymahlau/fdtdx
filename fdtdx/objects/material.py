import jax
import jax.numpy as jnp
import pytreeclass as tc

from fdtdx.core.plotting.colors import LIGHT_BLUE, LIGHT_GREY, LIGHT_BROWN
from fdtdx.objects.object import SimulationObject
from fdtdx.core.physics.constants import (
    relative_permittivity_substrate,
)


@tc.autoinit
class UniformMaterial(SimulationObject):
    """
    Object with a single float of permittivity and permeabiliy
    """
    permittivity: float = 1.0
    permeability: float = 1.0
    color: tuple[float, float, float] = LIGHT_GREY
    
    def get_inv_permittivity(
        self,
        prev_inv_permittivity: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:  # permittivity and info dict
        del params
        res = jnp.ones_like(prev_inv_permittivity) / self.permittivity
        return res, {}
    
    def get_inv_permeability(
        self,
        prev_inv_permeability: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:  # permeability and info dict
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
    pass

@tc.autoinit
class Substrate(UniformMaterial):
    permittivity: float = relative_permittivity_substrate
    color: tuple[float, float, float] = LIGHT_BROWN

@tc.autoinit
class WaveGuide(UniformMaterial):
    permittivity: float =  tc.field(init=True, kind="KW_ONLY")  # type: ignore
    color: tuple[float, float, float] = LIGHT_BLUE
