##
# Object Placement and Parameters 

::: fdtdx.place_objects
Main entry point for placing and initializing simulation objects.

::: fdtdx.apply_params
Applies parameters to devices and updates source states to be ready for simulation.

# Core FDTD Algorithm

::: fdtdx.run_fdtd
Time-reversal symmetric FDTD implementation with memory-efficient autodiff.

# Python Objects used for FDTD simulation
::: fdtdx.ArrayContainer
Container holding the electric/magnetic fields as well as permittivity/permeability arrays for simulation

::: fdtdx.ObjectContainer
Container holding all the objects in a simulation scene

::: fdtdx.ParameterContainer
Dictionary holding the parameters for every device in the simulation

::: fdtdx.SimulationState
Simulation state returned by the FDTD simulations. This is a tuple of the simulation time step and an array container.
