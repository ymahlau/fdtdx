from abc import ABC

import jax
import pytreeclass as tc

from fdtdx.core.physics import constants
from fdtdx.objects.object import SimulationObject


@tc.autoinit
class WaveLengthDependentObject(SimulationObject, ABC):
    """Base class for objects whose properties depend on wavelength/period.

    An abstract base class for simulation objects that have wavelength-dependent
    behavior. Provides properties to handle either wavelength or period specification,
    ensuring only one is set at a time.

    Attributes:
        _period: Optional period in seconds. Mutually exclusive with _wavelength.
        _wavelength: Optional wavelength in meters. Mutually exclusive with _period.
    """

    _period: float | None = tc.field(default=None, alias="period")  # type: ignore
    _wavelength: float | None = tc.field(default=None, alias="wavelength")  # type: ignore

    @property
    def period(self) -> float:
        """Gets the period in seconds.

        Returns:
            float: The period in seconds, either directly set or computed from wavelength.

        Raises:
            Exception: If neither period nor wavelength is set, or if both are set.
        """
        if self._period is not None and self._wavelength is not None:
            raise Exception("Need to set either wavelength or period")
        if self._period is not None:
            return self._period
        if self._wavelength is not None:
            return self._wavelength / constants.c
        raise Exception("Need to set either wavelength or period")

    @property
    def wavelength(self) -> float:
        """Gets the wavelength in meters.

        Returns:
            float: The wavelength in meters, either directly set or computed from period.

        Raises:
            Exception: If neither period nor wavelength is set, or if both are set.
        """
        if self._period is not None and self._wavelength is not None:
            raise Exception("Need to set either wavelength or period")
        if self._wavelength is not None:
            return self._wavelength
        if self._period is not None:
            return self._period * constants.c
        raise Exception("Need to set either wavelength or period")

    @property
    def frequency(self) -> float:
        """Gets the frequency in Hz.

        Returns:
            float: The frequency in Hz, computed as 1/period.
        """
        return 1 / self.period


@tc.autoinit
class WaveLengthDependentNoMaterial(WaveLengthDependentObject):
    """A wavelength-dependent object that doesn't modify material properties.

    Implements WaveLengthDependentObject for cases where the object doesn't affect
    the permittivity or permeability of the simulation volume.
    """

    def get_inv_permittivity(
        self,
        prev_inv_permittivity: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:  # permittivity and info dict
        """Returns unchanged inverse permittivity.

        Args:
            prev_inv_permittivity: The existing inverse permittivity array
            params: Optional parameter dictionary

        Returns:
            tuple: (Unchanged inverse permittivity array, Empty info dict)
        """
        del params
        return prev_inv_permittivity, {}

    def get_inv_permeability(
        self,
        prev_inv_permeability: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:  # permeability and info dict
        """Returns unchanged inverse permeability.

        Args:
            prev_inv_permeability: The existing inverse permeability array
            params: Optional parameter dictionary

        Returns:
            tuple: (Unchanged inverse permeability array, Empty info dict)
        """
        del params
        return prev_inv_permeability, {}
