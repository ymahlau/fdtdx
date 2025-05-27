"""Constraint modules for enforcing physical and fabrication requirements.

This package provides constraint modules that can be applied during optimization
to ensure designs meet physical realizability and fabrication requirements.
Key constraints include:

- Minimum feature size constraints
- Binary material constraints (air/dielectric only)
- Connectivity constraints (no floating material)
- Fabrication constraints (no trapped air, etc.)

The constraints are implemented as modules that can be chained together
and applied during the optimization process.
"""

from .linear_polarization import (
    GaussianPlaneSource,
    LinearlyPolarizedPlaneSource,
    UniformPlaneSource,
)
from .mode import ModePlaneSource
from .profile import GaussianPulseProfile, SingleFrequencyProfile
from .source import HardConstantAmplitudePlanceSource

__all__ = [
    "LinearlyPolarizedPlaneSource",
    "GaussianPlaneSource",
    "UniformPlaneSource",
    "HardConstantAmplitudePlanceSource",
    "SingleFrequencyProfile",
    "GaussianPulseProfile",
    "ModePlaneSource",
]
