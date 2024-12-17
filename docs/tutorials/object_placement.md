# Object Placement Guide

This guide explains how to position objects in a simulation scene in FDTDX.


## Basic Positioning
In FDTDX, objects are positioned either directly or relation to other objects through constriants. 
The user should specify these constraints and collect them in a list.

The first step should always be to define the size of the simulation volume. 
FDTDX always uses metrical units, i.e. meters or grid positions referring to the Yee-grid,
 which depends on the resolution used.
```python
# Create a simulation volume
volume = SimulationVolume(
    partial_real_shape=(4e-6, 4e-6, 1.5e-6),
)
# Create a list of placement constriants
placement_constraints = []
```

Now, we can start to position some objects in the simulation scene. 
We start with a substrate at the bottom of simulation. 
To this end, we specify a constraint that aligns the objects in the z-axis (axis 2). 

Positional constraints define an anchor point for both objects, which are constrainted to be at the same position. 
The position of the anchor point can be specified in a relative coordinate system of each object. 
A relative coordinate system means that a position of -1 would place the anchor at the left boundary of the object, 
a position of 0 at the middle and a position of 1 at the right boundary.

In case of the substrate, we want the lower boundary of the substrate to be aligned with the lower boundary of the simulation volume. 
This ensures that the substrate is placed exactly at the bottom of the simulation.


```python
# create substrate
substrate = UniformMaterial(
    partial_real_shape=(None, None, 0.6e-6),
    permittivity=constants.relative_permittivity_silica,
)
# place at the bottom of simulation volume
constraint = substrate.place_relative_to(
    volume,
    axes=2,
    own_positions=-1,
    other_positions=-1,
    margins=0,
    grid_margins=0,
)
placement_constraints.append(constraint)
```

The margins and grid_margins arguments are optional and would allow to speficy a fixed distance between 
the anchor points. The margins argument is in units of meters, the grid margins in units of yee-grid cells.

There exist a number of useful shorthands for rapid placements. Some of them are listed below that place 
some cubes in the scene:

```python
# place an object on top (z-axis / 2) of another object
cube1 = UniformMaterial(
    partial_real_shape=(0.1e-6, 0.1e-6, 0.1e-6),
    permittivity=constants.relative_permittivity_silicon,
)
placement_constraints.append(
    cube1.place_above(substrate)
)

# place an object at the center of another object
placement_constraints.append(
    cube1.place_at_center(
        substrate,
        axes=(0, 1),
    )
)
```

## Size Configuration

Object sizes can be specified in a number of ways. 
Firstly, one can directly set the size of an object in the init method.
This can either be a specified in Yee-grid cells or metrical units (meter).

```python
# size in meters
UniformMaterial(
    partial_real_shape=(0.1e-6, 0.1e-6, 0.1e-6),
    ...
)

# size in grid units
UniformMaterial(
    partial_grid_shape=(5, 3, 2),
    ...
)

# partial combination
UniformMaterial(
    partial_real_shape=(None, 0.5e-6, None),
    partial_grid_shape=(3, None, None),
    ...
)
```
If the size of an object is only partially defined and does not have any constraints,
the size is set to the size of the simulation volume in the respective axis.

The size of an object can also be set in relation to another object:
```python
size_constraint = object1.size_relative_to(
    object2,
    axes=(0, 2),
    other_axes=(0, 1),
    proportions=(1, 0.5),
    offsets=(0, 0),
    grid_offsets=(0, 0),
)
placement_constraints.append(size_constraint)
```
This would set the size of object1 in x-direction to the same size as object2 in x-direction.
It is also possible to combine different axes of objects. In the example above,
object1 is constrained in axis 2 to half the size of object2 in axis 1.

A useful convenience wrapper is the following:
```python
object1.same_size(object2, axes=(0,1))
```

The last method to set the size of an object is to constrain the size, such that it extends up to another object in the simulation scene.
```python
constraint = object1.extend_to(
    object2,
    axis=0,
    direction="+,
)
placement_constraints.append(constraint)
```
This would constrain the size of an object such that its upper boundary ("+") extends directly up to object2 in axis 0.


## Be careful when combining Positional Constraints and default size
A common mistake is to combine a positional constraint that sets the position of the object, but not setting the size of the object.
For example, the following code would likely lead to an error:
```python
object1 = UniformMaterial(
    partial_real_shape=(None, 0.1e-6, 0.1e-6),
    ...
)
pos_constraint = object1.place_relative_to(
    object2,
    axes=0,
    own_positions=0,
    other_positions=0,
)
placement_constraints.append(pos_constraint)
```
This code creates an object1, which does not have a size specified for axis 0. 
Therefore, the size of object1 in axis 0 is set to the size of the simulation volume.
However, the positional constraint requires the center of object1 to be at the same position as the center of object2.
Unless object2 is positioned at the very center of the simulation volume, this is not possible and results in an error.


See the [Objects API Reference](../api/objects/index.md) for complete details on all positioning and sizing options.





