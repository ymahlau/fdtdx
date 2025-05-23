##
# Devices
In FDTDX, devices are objects whose shape can be optimized. A device has a corresponding set of latent parameters, which are mapped to produce the current shape of the device.
::: fdtdx.objects.device.Device

# Parameter Mapping
::: fdtdx.objects.device.ParameterTransformation

# Projections
::: fdtdx.objects.device.TanhProjection
::: fdtdx.objects.device.SubpixelSmoothedProjection

# Tranformation of latent parameters
::: fdtdx.objects.device.StandardToPlusOneMinusOneRange
::: fdtdx.objects.device.StandardToCustomRange
::: fdtdx.objects.device.GaussianSmoothing2D

# Discretizations
::: fdtdx.objects.device.ClosestIndex
::: fdtdx.objects.device.PillarDiscretization
::: fdtdx.objects.device.BrushConstraint2D
::: fdtdx.objects.device.circular_brush

# Discrete PostProcessing
::: fdtdx.objects.device.BinaryMedianFilterModule
::: fdtdx.objects.device.ConnectHolesAndStructures
::: fdtdx.objects.device.RemoveFloatingMaterial
::: fdtdx.objects.device.BinaryMedianFilterModule

# Symmetries
::: fdtdx.objects.device.DiagonalSymmetry2D