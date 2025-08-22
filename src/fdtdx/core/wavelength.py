from fdtdx import constants
from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field
from fdtdx.units.typing import SI
from fdtdx.units.unitful import Unitful


@autoinit
class WaveCharacter(TreeClass):
    """Class describing a wavelength/period/frequency in free space. Importantly, the wave characteristic conversion is
    based on a free space wave when using the wavelength (For conversion, a refractive index of 1 is used).

    Attributes:
        phase_shift (float, optional): Phase shift in radians. Defaults to 0.
        period (Unitful | None, optional): Optional period in seconds. Mutually exclusive with wavelength and frequency.
            Defaults to None.
        wavelength (Unitful | None, optional): Optional wavelength in meters for free space propagation.
            Mutually exclusive with period and frequency. Defaults to None.
        frequency (Unitful | None, optional): Optional frequency in Hz. Mutually exclusive with period and wavelength.
    """

    phase_shift: float = frozen_field(default=0.0)
    period: Unitful | None = frozen_field(default=None)
    wavelength: Unitful | None = frozen_field(default=None)
    frequency: Unitful | None = frozen_field(default=None)

    def __post_init__(
        self,
    ):
        self._check_input()

    def _check_input(self):
        if sum([self.period is not None, self.frequency is not None, self.wavelength is not None]) != 1:
            raise Exception("Need to set exactly one of Period, Frequency or Wavelength in WaveCharacter")
        if self.period is None:
            if self.wavelength is not None:
                assert self.wavelength.unit.dim == {SI.m: 1}, f"Please specify wavelength in meter, not {self.wavelength}"
                self.period = self.wavelength / constants.c
            elif self.frequency is not None:
                assert self.frequency.unit.dim == {SI.s: -1}, f"Please specify frequency in Hz, not {self.period}"
                self.period = 1.0 / self.frequency
            else:
                raise Exception("This should never happen")

        if self.wavelength is None:
            if self.period is not None:
                assert self.period.unit.dim == {SI.s: 1}, f"Please specify period in seconds, not {self.period}"
                self.wavelength = self.period * constants.c
            elif self.frequency is not None:
                self.wavelength = constants.c / self.frequency
            else:
                raise Exception("This should never happen")

        if self.frequency is None:
            if self.period is not None:
                assert self.period.unit.dim == {SI.s: 1}, f"Please specify period in seconds, not {self.period}"
                self.frequency = 1.0 / self.period
                assert self.frequency.unit.dim == {SI.s: -1}, f"Please specify frequency in Hz, not {self.period}"
            elif self.wavelength is not None:
                assert self.wavelength.unit.dim == {SI.m: 1}, f"Please specify wavelength in meter, not {self.wavelength}"
                self.frequency = constants.c / self.wavelength
            else:
                raise Exception("This should never happen")

    def get_period(self) -> Unitful:
        assert self.period is not None, "This should never happen"
        return self.period

    def get_wavelength(self) -> Unitful:
        assert self.wavelength is not None, "This should never happen"
        return self.wavelength

    def get_frequency(self) -> Unitful:
        assert self.frequency is not None, "This should never happen"
        return self.frequency
