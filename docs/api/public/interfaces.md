##
# Interface Compression
This API can be used for automatic differentiation (autodiff) with time-reversibility, which is more memory efficient than other approaches. 
Additionally, some basic compression modules are implemented to reduce the memory footprint even further. However, they should be used with care since too much compression can reduce the gradient accuracy.

::: fdtdx.Recorder
A recorder object for recording the interfaces between simulation volume and PML boundary during the forward simulation

# Compression Modules
::: fdtdx.LinearReconstructEveryK
Compression module which only records every k time steps during the forward simulation. For reconstruction a linear interpolation between the recorded time steps is performed.

::: fdtdx.DtypeConversion
Compression module to save the interfaces at a lower datatype resolution. From experience, in most applications saving the interfaces in jnp.float16 or jnp.float8_e4m3fnuz is sufficient.
