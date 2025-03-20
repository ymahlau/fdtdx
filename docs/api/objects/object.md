# Base Classes

## Core Classes
::: fdtdx.objects.object.SimulationObject
Base class for all simulation objects with positioning and sizing capabilities. All Objects in FDTDX are cuboid shaped by default, but with multi-material objects more complicated shapes can be realized as well.

::: fdtdx.objects.SimulationVolume
An object of cuboid shape describing the size and default background material of the simulation volume.

::: fdtdx.objects.UniformMaterialObject
An object that has a uniform material throughout its entire volume.

## Convenience Wrapper
::: fdtdx.objects.Substrate
A substrate of uniform material. This is just a wrapper for an object with uniform material.

::: fdtdx.objects.Waveguide
A waveguide of uniform material. This is just a wrapper for an object with uniform material.

## Positioning/Sizing Constraints
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



