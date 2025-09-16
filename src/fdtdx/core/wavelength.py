from fdtdx import constants
from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field


@autoinit
class WaveCharacter(TreeClass):
    """Class describing a wavelength/period/frequency in free space. Importantly, the wave characteristic conversion is
    based on a free space wave when using the wavelength (For conversion, a refractive index of 1 is used).
    """

    #: Phase shift in radians. Defaults to 0.
    phase_shift: float = frozen_field(default=0.0)

    #: Optional period in seconds. Mutually exclusive with wavelength and frequency.
    #: Defaults to None.
    period: float | None = frozen_field(default=None)

    #: Optional wavelength in meters for free space propagation.
    #: Mutually exclusive with period and frequency. Defaults to None.
    wavelength: float | None = frozen_field(default=None)

    #: Optional frequency in Hz. Mutually exclusive with period and wavelength.
    frequency: float | None = frozen_field(default=None)

    def __post_init__(
        self,
    ):
        self._check_input()

    def _check_input(self):
        if sum([self.period is not None, self.frequency is not None, self.wavelength is not None]) != 1:
            raise Exception("Need to set exactly one of Period, Frequency or Wavelength in WaveCharacter")
        
    def get_period(self) -> float:
        if self.period is None:
            if self.wavelength is not None:
               return self.wavelength / constants.c
            elif self.frequency is not None:
                return 1.0 / self.frequency
            else:
                raise Exception("This should never happen")
        assert self.period is not None, "This should never happen"
        return self.period

    def get_wavelength(self) -> float:
        if self.wavelength is None:
            if self.period is not None:
                return self.period * constants.c
            elif self.frequency is not None:
               return constants.c / self.frequency
            else:
                raise Exception("This should never happen")
        assert self.wavelength is not None, "This should never happen"
        return self.wavelength

    def get_frequency(self) -> float:
        if self.frequency is None:
            if self.period is not None:
                return 1.0 / self.period
            elif self.wavelength is not None:
                return constants.c / self.wavelength
            else:
                raise Exception("This should never happen")
        assert self.frequency is not None, "This should never happen"
        return self.frequency
