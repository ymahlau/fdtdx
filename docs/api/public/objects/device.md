##
# Devices
In FDTDX, devices are objects whose shape can be optimized. A device has a corresponding set of latent parameters, which are mapped to produce the current shape of the device.
::: fdtdx.Device

# Parameter Mapping
::: fdtdx.ParameterTransformation

# Projections
::: fdtdx.TanhProjection
::: fdtdx.SubpixelSmoothedProjection

# Tranformation of latent parameters
::: fdtdx.StandardToPlusOneMinusOneRange
::: fdtdx.StandardToCustomRange
::: fdtdx.GaussianSmoothing2D

# Discretizations
::: fdtdx.ClosestIndex
::: fdtdx.PillarDiscretization
::: fdtdx.BrushConstraint2D
::: fdtdx.circular_brush

# Discrete PostProcessing
::: fdtdx.BinaryMedianFilterModule
::: fdtdx.ConnectHolesAndStructures
::: fdtdx.RemoveFloatingMaterial
::: fdtdx.BinaryMedianFilterModule

# Symmetries
::: fdtdx.DiagonalSymmetry2D