# API Documentation

The FDTDX API is organized into several main components:

## Objects
The [Objects](objects/index.md) package defines simulation components:
- Sources and detectors
- Boundary conditions
- Object positioning
- Devices with a variable shape that can be optimized

## FDTD
The [FDTD](fdtd.md) package provides simulation algorithms:
- FDTD implementations
- Memory efficient simulations for automatic differentiation

## Config
The [Config](config.md) package provides configuration files for specifying simulation parameters

## Constants
The [Constants](constants.md) package provides a small list of commonly used material constants

## Interfaces
The [Interfaces](interfaces.md) package contains functions for interface compression, enabling memory efficient autodiff.

## Materials
The [Materials](materials.md) package contains the basic structure for defining a material in FDTDX.

## Typing
The [Typing](materials.md) package contains a number of type hints used throughout the FDTDX package

## Utils
The [Utility](utils.md) package contains utility functions for logging and plotting simulations

## Core Package
The [Core](core.md) package provides fundamental data structures and utilities used mostly internally by FDTDX