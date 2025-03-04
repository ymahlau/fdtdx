"""Objects module for the FDTDX electromagnetic simulation framework.

This module provides classes and utilities for defining and manipulating objects
within FDTD simulations. It includes support for:

- Material objects with wavelength-dependent properties
- Geometric constraints and positioning
- Boundary conditions and PML layers
- Sources and detectors
- Object containers for managing multiple simulation elements

The objects module enables intuitive positioning and sizing of elements in absolute
or relative coordinates within the simulation scene, with automatic constraint resolution.
"""

from .container import (
    ArrayContainer,
    ObjectContainer,
    ParameterContainer,
    SimulationState,
)
from .initialization import place_objects
from .material import (
    SimulationVolume,
    Substrate,
    UniformMaterial,
    WaveGuide,
)

__all__ = [
    "SimulationVolume",
    "UniformMaterial",
    "Substrate",
    "WaveGuide",
    "place_objects",
    "ParameterContainer",
    "SimulationState",
    "ArrayContainer",
    "ObjectContainer",
]
