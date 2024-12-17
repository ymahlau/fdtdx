# Base Classes

## Core Classes
::: fdtdx.objects.object.SimulationObject
Base class for all simulation objects with positioning and sizing capabilities.

::: fdtdx.objects.object.UniqueName
Utility for generating unique object identifiers.

## Positioning Constraints
::: fdtdx.objects.object.PositionConstraint
Defines relative positioning between simulation objects.

::: fdtdx.objects.object.SizeConstraint  
Controls size relationships between objects.

::: fdtdx.objects.object.SizeExtensionConstraint
Extends objects to reach other objects or boundaries.

::: fdtdx.objects.object.GridCoordinateConstraint
Aligns objects to specific grid coordinates.

::: fdtdx.objects.object.RealCoordinateConstraint
Positions objects at specific physical coordinates.

## Wavelength-Dependent Objects
::: fdtdx.objects.wavelength.WaveLengthDependentObject
Base class for objects with wavelength/period-dependent properties.

::: fdtdx.objects.wavelength.WaveLengthDependentNoMaterial
Non-material modifying wavelength-dependent object.
