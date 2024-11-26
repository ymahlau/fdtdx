from abc import ABC
import pytreeclass as tc

import jax
from fdtdx.objects.object import SimulationObject
from fdtdx.core.physics import constants


@tc.autoinit
class WaveLengthDependentObject(SimulationObject, ABC):
    _period: float | None = tc.field(default=None, alias="period")  # type: ignore
    _wavelength: float | None = tc.field(default=None, alias="wavelength")  # type: ignore
    
    @property
    def period(self) -> float:
        if self._period is not None and self._wavelength is not None:
            raise Exception("Need to set either wavelength or period")
        if self._period is not None:
            return self._period
        if self._wavelength is not None:
            return self._wavelength / constants.c
        raise Exception("Need to set either wavelength or period")

    @property
    def wavelength(self) -> float:
        if self._period is not None and self._wavelength is not None:
            raise Exception("Need to set either wavelength or period")
        if self._wavelength is not None:
            return self._wavelength
        if self._period is not None:
            return self._period * constants.c
        raise Exception("Need to set either wavelength or period")

    @property
    def frequency(self):
        return 1 / self.period

@tc.autoinit
class WaveLengthDependentNoMaterial(WaveLengthDependentObject):
        
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
        
