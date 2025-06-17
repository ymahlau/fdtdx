## Conventions and normalizations used in FDTDX

### Where is the origin?
The origin is at the bottom left front corner of the simulation and corresponds to indices (0, 0, 0) in all arrays.

### Where is up?
In FDTDX, we assume that the following correspond to forward, sideways and up:
- For propagation in x-axis, sideways is the y-axis, and up is the z-axis
- For propagation in y-axis, sideways is the x-axis, and up is the z-axis
- For propagation in z-axis, sideways is the x-axis, and up is the y-axis
This convention is independent of the direction ("+", "-") of propagation.

### Which direction to go?
The forward direction ("+") corresponds to an increase of the index in an array. For example, from the index 5, the index 10 can be reached by going in the forward direction. Vice versa, "-" corresponds to backwards and index 5 can be reached from index 10 in this direction.

### Physical constants
For FDTDX, we assume that the free space permittivity ($\varepsilon_{0}$) and permeability ($\mu_{0}$) are 1.0. This convention can be used because FDTDX is scale-invariant (at least for linear materials). If more complicated material models are introduced enventually, this convention might have to be removed.

For everything else, we use metrical units. For example, length is measured in meter, time in seconds, frequency in 1/second. Light speed is 299792458 m/s.
