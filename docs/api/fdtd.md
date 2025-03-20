# Object Placement and Parameters 
::: fdtdx.fdtd.place_objects
Main entry point for placing and initializing simulation objects.

::: fdtdx.fdtd.apply_params
Applies parameters to devices and updates source states to be ready for simulation.

# Core FDTD Algorithms

## Reversible FDTD
::: fdtdx.fdtd.reversible_fdtd
Time-reversal symmetric FDTD implementation with memory-efficient autodiff.

### Checkpointed FDTD
::: fdtdx.fdtd.checkpointed_fdtd
Gradient checkpointing FDTD implementation for memory-performance tradeoff when using autodiff. In most use-cases this performs worse than the reversible FDTD.

## Backward FDTD in time
::: fdtdx.fdtd.full_backward
Complete backward FDTD propagation from current state to start time. This can be used to check if the compression of boundary interfaces still lead to a physically accurate backward pass.

## Custom Time Evolution
::: fdtdx.fdtd.custom_fdtd_forward
Customizable FDTD implementation for partial time evolution and analysis. If used smartly, this can make simulation a bit faster, but in most use-cases this is not necessary.

# Python Objects used for FDTD simulation
::: fdtdx.fdtd.ArrayContainer
Container holding the electric/magnetic fields as well as permittivity/permeability arrays for simulation

::: fdtdx.fdtd.ObjectContainer
Container holding all the objects in a simulation scene

::: fdtdx.fdtd.ParameterContainer
Dictionary holding the parameters for every device in the simulation

::: fdtdx.fdtd.SimulationState
Simulation state returned by the FDTD simulations. This is a tuple of the simulation time step and an array container.
