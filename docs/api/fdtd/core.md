# Core FDTD Algorithms

## Memory-Efficient Implementations

### Reversible FDTD
::: fdtdx.fdtd.reversible_fdtd
Time-reversal symmetric FDTD implementation with O(1) memory usage.

### Checkpointed FDTD
::: fdtdx.fdtd.checkpointed_fdtd
Gradient checkpointing FDTD implementation for memory-performance tradeoff.

## Custom Time Evolution
::: fdtdx.fdtd.custom_fdtd_forward
Customizable FDTD implementation for partial time evolution and analysis.

## Forward Propagation
::: fdtdx.fdtd.forward.forward
Standard forward FDTD time stepping implementation.

::: fdtdx.fdtd.forward.forward_single_args_wrapper
JAX-compatible wrapper for forward propagation.

## Backward Propagation
::: fdtdx.fdtd.backward.full_backward
Complete backward FDTD propagation from current state to start time.

::: fdtdx.fdtd.backward.backward
Single step backward FDTD propagation.
