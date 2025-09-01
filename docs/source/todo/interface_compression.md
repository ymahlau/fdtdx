# Gradient computation by time reversibility

FDTDX implements automatic differentiation by exploiting the time-reversibility of Maxwell's equations. You can find more details about the time-reversible gradient computation in our [paper](https://arxiv.org/abs/2407.10273). 

For this tutorial, the important point to note is that during the forward simulation, the interface region between PML and actual simulation volume needs to be saved at every time step. Even though this is better than the standard implementation of AutoDiff (which would save the whole 3D volume at every time step), this can still lead to large memory requirements if the simulation is large or the simulation time long. 

As a remedy, we implement a compression mechanism for these saved fields. The compression settings can be adjusted in the simulation config:
```python
from fdtdx import (
    GradientConfig,
    SimulationConfig,
    DtypeConversion,
    Recorder,
    LinearReconstructEveryK,
)
import jax.numpy as jnp

gradient_config = GradientConfig(
    recorder=Recorder(
        modules=[
            LinearReconstructEveryK(2),
            DtypeConversion(dtype=jnp.float16),
        ]
    )
)
config = SimulationConfig(
    time=300e-15,
    resolution=100e-9,
    dtype=jnp.float32,
    courant_factor=0.99,
    gradient_config=gradient_config,  # <- This needs to be set for gradient computation
)
```
Similarly to the [constraint mappings](./parameter_mapping.md), the recorder of the gradient config is defined by a list of modules, which are applied consecutively. In this example, the following two modules are used:
- LinearReconstructEveryK: Firstly, this module only saves the boundary fields at every second time step. During reconstruction, the missing values are recomputed by linearly interpolating between the saved time steps. The attribute k=2 defines the step size.
- DtypeConversion: The output of the previous module is converted to a different data type. In our example, the simulation runs with 32 bit floating point precision and the module converts these values to 16 bit precision, again saving 50% of the required memory.

At the moment, these are the only two important compression modules implemented. Experience has shown that in almost all cases 8bit precision is also sufficient, namely the data type "jnp.float8_e4m3fnuz". 

Regarding the number of time steps, a rule of thumb is that 10 time steps per period should be saved for accurate results. Often lower saving intervals also suffice, but one needs to make sure that this is actually the case. So for example, if the simulation performs 30 time steps per period (this depends on the Courant-Friedrichs-Levy Condition), then a compression of LinearReconstructEveryK(3) should be used to save 10 time steps. The number of time steps per period can be computed by:
```python
from fdtdx import constants
wavelength = 1.55e-6
period = constants.wavelength_to_period(wavelength)
steps_per_period = period / config.time_step_duration
```

## Gradient Computation

The actual gradient computation can be invoked using the standard jax.grad method on the fdtd_reversible function call. In pseudocode this might look something like this:
```python
def loss_function(params, ...)
    arrays, new_objects, info = apply_params(arrays, objects, params, key)

    _, arrays = reversible_fdtd(
        arrays=arrays,
        objects=new_objects,
        config=config,
        key=key,
    )
    loss = - figure_of_merit(arrays.detector_states)
    return loss

grad_function = jax.grad(loss_fn)
grad_loss_wrt_params = grad_function(params)
```
Of course figure_of_merit can be any objective function that should be optimized. The apply_params function internally calls the [Parameter mapping](./parameter_mapping.md) of the device and sets the proper inverse permittivities for the simulation.