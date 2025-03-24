##
# Initialization of Boundaries
Currently, the simulation boundary can either be a Perfectly Matched Layer for absorbing all incoming light, or a periodic boundary which wraps around to the other side of the simulation.

::: fdtdx.objects.boundaries.BoundaryConfig
Configuration object for specifying at which side of the simulation wich type of boundary should be used. Also allows specification of the PML thickness and other parameters if used.

::: fdtdx.objects.boundaries.boundary_objects_from_config
Initializes the corresponding boundary objects based on the config object.

# Boundary Objects
::: fdtdx.objects.boundaries.PerfectlyMatchedLayer

::: fdtdx.objects.boundaries.PeriodicBoundary