import jax
import jax.core
import jax.numpy as jnp


def check_not_tracing(fn_name: str) -> None:
    """Raise a clear error if called inside a JAX JIT trace.

    Setup functions allocate concrete device memory and resolve static grid
    geometry. Both require eager execution and are incompatible with JIT
    tracing. This guard surfaces that constraint with an actionable message
    instead of a cryptic internal JAX error.

    Works by checking whether a freshly created jnp array is a Tracer —
    which it is under JIT, and is not in eager mode.
    """
    if isinstance(jnp.empty(()), jax.core.Tracer):
        raise TypeError(
            f"`{fn_name}` is a setup function and cannot be called inside "
            f"`jax.jit()`. It allocates device arrays and resolves concrete "
            f"grid geometry, both of which require eager execution.\n\n"
            f"Move all fdtdx setup calls outside any JIT boundary and only "
            f"JIT-compile the simulation time-stepping loop:\n\n"
            f"  # Correct\n"
            f"  objects, arrays, config, _ = fdtdx.place_objects(...)\n"
            f"  result = jax.jit(run_simulation)(arrays, ...)\n\n"
            f"  # Wrong\n"
            f"  @jax.jit\n"
            f"  def run():\n"
            f"      objects, arrays, config, _ = fdtdx.place_objects(...)  # ERROR\n"
            f"      return run_simulation(arrays, ...)\n"
        )
