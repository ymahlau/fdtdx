# Incorporation of Fabrication Constraints

When using inverse design with FDTDX, fabrication constraints have to be specified. The basic building block for an object optimizable by inverse design is a Device:

```python
from fdtdx import (
    Device,
    constants,
    Material
)

material_config = {
    "Air": Material(permittivity=constants.relative_permittivity_air),
    "Polymer": Material(permittivity=constants.relative_permittivity_ma_N_1400_series),
}
device_scatter = DiscreteDevice(
    name="Device",
    partial_real_shape=(1e-6, 1e-6, 1e-6),
    material=material_config,
    param_transforms=...,  # <- This needs to be filled
    partial_voxel_real_shape=(0.2e-6, 0.2e-6, 0.2e-6),
)
```

The parameter mapping, which is left empty above, specifies the mapping from continuous latent parameters to materials used in the simulation.

# Simple example
At the beginning of optimization, the latent parameters of a device are always initialized randomly in the interval [0, 1]. Depending on the constraint mapping, these parameteters are mapped to inverse permittivities. Let's look at an example of a simple constraint mapping:

```python
from fdtdx import ClosestIndex

param_transforms = [ClosestIndex()]
```
The constraint mapping consists of a chain of modules, or in other words a chain of transformations followed by a discretization. Let's look at the module in detail:
- ClosestIndex(): This module quantizes the latent variables to the closest integer. Since latent parameters are initialized randomly in the interval [0, 1], this module maps the continuous parameters to either the index 0 or 1. Since this operation is not differentiable, we employ a straight-through-estimator (STE), which simply copies the gradient from the quantized values to the original values in the backward pass.

This mapping constraints each voxel independently of the other voxels to the inverse permittivity of either air or polymer. However, often more elaborate fabrication constraints are needed in practice, which we introduce in the following sections.



# Silicon Device with minimum feature constraint
Now let's develop a constraint mapping for silicon photonics, which restricts the minimum feature size of a device.

```python
from fdtdx import (
    StandardToPlusOneMinusOneRange,
    BrushConstraint2D,
    circular_brush,
)

brush_diameter_in_voxels = round(100e-9 / config.resolution)
param_transforms = [
    StandardToPlusOneMinusOneRange()
    BrushConstraint2D(
        brush=circular_brush(diameter=brush_diameter_in_voxels),
        axis=2,
    ),
]
```

This mapping does not just quantize latent paramters to material indices, but also makes sure that the device adheres to a minimum feature size with regard to a specific brush. In this example, we used a circular brush of 100nm. In other words, one could "paint" the design with a brush of this size.
In more detail:
- StandardToPlusOneMinusOneRange(): maps the standard [0, 1] range to [-1, 1]. This is necessary, because the BrushConstraint2D expects the input to be in this range.
- BrushConstraint2D(): maps the output of the previous module to permittivity indices similar to ClosestIndex() described above. However, it also makes sure that the design adheres to a minimum feature size regarding a specific brush shape. The axis argument defines the axis perpendicular to the 2D-design plane used. In our example, the perpendicular axis is 2 (in other words z/upwards). Therefore, the minimum feature constraint is enforced in the XY-plane.

# 3D Fabrication Constraints for Two-Photon-Polymerization
Lastly, let's look at a more involved constraint mapping used to Two-Photon Polymerization (2PP). In 2PP, a laser is focused on liquid monomer to harden the material. This allows the creation of fully three-dimensional designs. 

Resulting from this fabrication technique multiple constraints arise. Firstly, basic physical knowledge tells us that no material can float in the air without a connection to the ground. In 3D-design, we have to explicitly incorporate this constraint, which was not necessary in 2D (in 2D, all voxels are always connected to the ground). Secondly, there cannot be enclosed air cavities in a design for 2PP. An enclosed cavity would trap unpolmerized monomer and destroy the structural integrity of the design. However, in practice it is often not necessary to explicitly encode this constraint in the simulation. Enclosed cavities seldomly increase a figure of merit and therefore only rarely appear in an optimized design. But, it is important to check after the simulation is finished if any enclosed cavitities exist in the design.


```python
from fdtdx import (
    BOTTOM_Z_PADDING_CONFIG_REPEAT,
    BinaryMedianFilterModule,
    RemoveFloatingMaterial,
    ClosestIndex,
)

param_transforms = [
    ClosestIndex(),
    BinaryMedianFilterModule(
        kernel_sizes=(5, 5, 5),
        padding_cfg=BOTTOM_Z_PADDING_CONFIG_REPEAT,
        num_repeats=2,
    ),
    RemoveFloatingMaterial(),
]

```
This constraint mapping is one possibility to implement constraints for 2PP. The two new modules are:
- BinaryMedianFilterModule: This module does a soft enforcement of a minimum feature size by smoothing the incoming indices (produced by the previous module) with a median filter. The kernel size describes the size of the smoothing kernel in Yee grid cells. The padding config describes how the boundaries of the design are padded for smoothing. The BOTTOM_Z_PADDING_CONFIG_REPEAT uses a repeat of the boundary values except at the bottom of the design, where the design is padded with non-air-material. Heuristically, this gives the design better ground contact. The num_repeats argument specifies how often the smoothing filter is applied. However, in contrast to the BrushConstraint2D, this is only an approximation and does not always enforce the minimum feature size.
- RemoveFloatingMaterial: As the name suggests, this module goes through the indices generated by the previous module (BinaryMedianFilter) and removes any floating material without ground connection. Ground connection is computed using a simple flood fill algorithm and all voxels with floating material are converted to the background material (usually air).

