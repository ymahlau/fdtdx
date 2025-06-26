##
# Core Classes
::: fdtdx.SimulationObject
Base class for all simulation objects with positioning and sizing capabilities. All Objects in FDTDX are cuboid shaped by default, but with multi-material objects more complicated shapes can be realized as well.

::: fdtdx.SimulationVolume
An object of cuboid shape describing the size and default background material of the simulation volume.

::: fdtdx.UniformMaterialObject
An object that has a uniform material throughout its entire volume.

# Positioning/Sizing Constraints
::: fdtdx.PositionConstraint
Defines relative positioning between simulation objects.

::: fdtdx.SizeConstraint  
Controls size relationships between objects.

::: fdtdx.SizeExtensionConstraint
Extends objects to reach other objects or boundaries.

::: fdtdx.GridCoordinateConstraint
Aligns objects to specific grid coordinates.

::: fdtdx.RealCoordinateConstraint
Positions objects at specific physical coordinates.



