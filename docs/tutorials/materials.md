# Materials Guide

In FDTDX, objects can have different permittivities and permeabilities. Currently, the conductivity of all materials is assumed to be zero, but we are planning to implement conductive materials in the very near future.

Also, currently neither dispersion nor non-linear materials are implemented. The implementation of dispersion is scheduled in the near-mid future and afterwards an implementation of non-linear materials will follow.

This guide is currently very short and will be expanded with them implementations mentioned above.

## UniformMaterial
The most basic and also probably most useful object is the UniformMaterialObject. As the name suggests, it has a single material.
```python
from fdtdx import (
    constants,
    UniformMaterialObject,
    Material,
    colors,
)

uniform_obj = UniformMaterialObject(
    partial_real_shape=(0.6e-6, 0.6e-6, 0.6e-6),
    material=Material(permittivity=constants.relative_permittivity_silica),
    # permeability is one by default
    permeability=1.0,
    color=colors.CYAN,
    name="uniform_obj",
)
```
The name and color attribute are only used for plotting and do not have any effect on the simulation.

## Device
For inverse design, it is necessary to model objects that can either be one or the other materials. In some applications, it might even be necessary to model objects consisting of more than two materials.

In this example, we create a device consisting of voxels that are either air or polymer.
```python
material_config = {
    "Air": Material(permittivity=constants.relative_permittivity_air),
    "Polymer": Material(permittivity=constants.relative_permittivity_ma_N_1400_series),
}
device = Device(
    name="Device",
    partial_real_shape=(1e-6, 1e-6, 1e-6),
    material=material_config,
    parameter_mapping=...,
    partial_voxel_real_shape=(0.2e-6, 0.2e-6, 0.2e-6),
)
```
The device has a permittivity config, which defines the different permittivity options. This is currently only implemented for permittivity, but we will expand it in the future to metallic materials as well. 

The partial_voxel_real_shape argument specifies the size of the uniform material voxels within the device. In this case, voxels, of 200nm^3 have a single permittivity. Since the device has a shape of 1µm^3, there are 5x5x5=125 of these voxels within the device. 

Importantly, the size of the device needs to be divisible by the voxel size. Additionally, the voxel size needs to be suffiently larger than the resolution of the Yee-grid in the simulation. For example, if the resolution of the Yee-grid is also 200nm, then this simulation will not produce accurate results. As a rule of thumb, the resolution of the Yee-grid should be at least three times smaller than the size of the voxels.

The device has one latent parameter for every voxel. Initially, these latent parameters are uniformly random in the interval [0, 1]. The constraint mapping defines how these latent parameters are mapped to actual inverse permittivity choices. A detailed guide on this topic can be found [here](./parameter_mapping.md).
