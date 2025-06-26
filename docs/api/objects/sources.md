##
# Sources
A collection of source objects to induce light into a simulation. The spatial profile of the sources can be either a plane, a gaussian or a mode. The temporal profile is set through the correponding attribute. All sources are implemented using the Total-Field/Scattered-Field formulation (also known as soft sources), except for the sources explicitly marked as "hard". These directly set the electric/magnetic field to a fixed value.

::: fdtdx.GaussianPlaneSource
A source with a spatial profile of a gaussian.

::: fdtdx.UniformPlaneSource
A source with a spatial profile of a uniform plane.

:::fdtdx.ModePlaneSource
A source with the spatial profile of a mode. The mode is computed automatically and by default a first order mode is used. In the future, we will develop a better interface to support other modes as well.

# Temporal Profiles
::: fdtdx.SingleFrequencyProfile
A temporal profile which exhibits just a single wave throughout the whole simulation time.

::: fdtdx.GaussianPulseProfile
A temporal pulse of a gaussian envelope.
