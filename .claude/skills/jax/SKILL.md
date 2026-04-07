---
name: jax
description: Composable transformations of Python+NumPy programs. Differentiate, vectorize, JIT-compile to GPU/TPU. Built for high-performance machine learning research and complex scientific simulations. Use for automatic differentiation, GPU/TPU acceleration, higher-order derivatives, physics-informed machine learning, differentiable simulations, and automatic vectorization.
version: 0.4
license: Apache-2.0
user-invocable: false
---

# JAX - Autograd and XLA (Accelerated Linear Algebra)

JAX is a framework that combines a NumPy-like API with a powerful system of composable function transformations: Grad (differentiation), Jit (compilation), Vmap (vectorization), and Pmap (parallelization).

## When to Use

- High-performance scientific simulations requiring GPU/TPU acceleration.
- Custom machine learning research where PyTorch/TF abstractions are too restrictive.
- Calculating higher-order derivatives (Hessians, Jacobians) for optimization.
- Physics-informed machine learning and differentiable simulations.
- Automatic vectorization of functions (no more manual batching).
- Running the same code on CPU, GPU, and TPU without changes.

## Reference Documentation

**Official docs**: https://jax.readthedocs.io/  
**GitHub**: https://github.com/google/jax  
**Search patterns**: `jax.numpy`, `jax.jit`, `jax.grad`, `jax.vmap`, `jax.random`

## Core Principles

### Pure Functions (Immutability)
JAX is built on functional programming. All functions must be pure: they should not have side effects (like modifying a global variable) and must return the same output for the same input. JAX arrays are immutable.

### XLA (Just-In-Time Compilation)
JAX uses XLA to compile and optimize Python/NumPy code into efficient machine code for specific hardware.

### Manual PRNG Handling
Unlike NumPy, JAX requires explicit management of random state (keys) to ensure reproducibility in parallel environments.

## Quick Reference

### Installation

```bash
# CPU
pip install jax jaxlib

# GPU (Check documentation for specific CUDA versions)
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Standard Imports

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap, random
```

### Basic Pattern - Differentiate and JIT

```python
import jax.numpy as jnp
from jax import grad, jit

# 1. Define a pure function
def f(x):
    return jnp.sin(x) + x**2

# 2. Transform: Create a gradient function
df_dx = grad(f)

# 3. Transform: Compile for speed
f_fast = jit(f)

# 4. Use
val = f_fast(2.0)
slope = df_dx(2.0)
```

## Critical Rules

### ✅ DO

- **Use jax.numpy (jnp)** - It mirrors NumPy but supports JAX transformations.
- **Write Pure Functions** - Ensure functions only depend on inputs and don't modify external state.
- **Handle PRNG Keys Manually** - Use `key, subkey = random.split(key)` for every random operation.
- **Use vmap for Batching** - Write code for a single sample and let JAX handle the batch dimension.
- **Set static_argnums in JIT** - If a JIT'ed function takes a non-array argument (like a string or integer used in a loop), mark it as static.
- **In-place updates via .at** - Since arrays are immutable, use `x = x.at[idx].set(val)`.

### ❌ DON'T

- **Use in-place updates** - `x[idx] = val` will raise an error.
- **Use standard numpy (np)** - Standard NumPy arrays don't support JAX transformations.
- **Use Side Effects** - Don't use `print()` or modify global variables inside JIT-compiled functions.
- **Forget to block_until_ready()** - JAX uses asynchronous execution. For accurate timing, use `result.block_until_ready()`.

## Anti-Patterns (NEVER)

```python
import jax.numpy as jnp
from jax import jit, random

# ❌ BAD: Modifying a global variable inside a function
counter = 0
@jit
def bad_func(x):
    global counter
    counter += 1 # ❌ Side effect! Will only run once during compilation
    return x * 2

# ❌ BAD: Standard NumPy random (not reproducible/parallel-safe)
# val = np.random.randn(5) 

# ✅ GOOD: JAX PRNG
key = random.key(42)
val = random.normal(key, (5,))

# ❌ BAD: In-place assignment
# x[0] = 1.0 

# ✅ GOOD: Functional update
x = jnp.zeros(5)
x = x.at[0].set(1.0)
```

## Function Transformations

### Grad (Differentiation)

```python
def loss(params, x, y):
    pred = jnp.dot(x, params)
    return jnp.mean((pred - y)**2)

# Gradient of loss with respect to the 1st argument (params)
grads = grad(loss)(params, x, y)

# Higher-order: Hessian
hessian = jax.hessian(loss)(params, x, y)
```

### Jit (Just-In-Time Compilation)

```python
@jit
def complex_math(x):
    # This whole block is compiled into one XLA kernel
    y = jnp.exp(x)
    return jnp.sin(y) / jnp.sqrt(x)

# First call: Compiles (slow)
# Subsequent calls: Super fast
```

### Vmap (Automatic Vectorization)

```python
def model(params, x):
    return jnp.dot(params, x)

# model works on 1D x. How to apply to a 2D batch of X?
# in_axes=(None, 0): don't map params, map the 0th axis of x
batch_model = vmap(model, in_axes=(None, 0))

batch_preds = batch_model(params, X_batch)
```

## Random Numbers (jax.random)

### The State Management

```python
key = random.key(0)

# Never reuse the same key!
key, subkey = random.split(key)
data = random.normal(subkey, (10,))

key, subkey = random.split(key)
noise = random.uniform(subkey, (10,))
```

## Working with PyTrees

### Handling complex data structures (Dicts, Lists, Tuples)

JAX transformations work on "PyTrees" — nested containers of arrays.

```python
params = {'weights': jnp.ones((5,)), 'bias': 0.0}

def predict(p, x):
    return jnp.dot(x, p['weights']) + p['bias']

# grad and jit handle the dictionary automatically
grads = grad(predict)(params, x)
```

## Practical Workflows

### 1. Differentiable Physics: Solving a Simple ODE

```python
def system_dynamics(state, t):
    # Simple harmonic oscillator
    x, v = state
    dxdt = v
    dvdt = -0.5 * x
    return jnp.array([dxdt, dvdt])

def loss_fn(initial_state, target_x):
    # Simulate for 10 steps using simple Euler
    state = initial_state
    dt = 0.1
    for i in range(10):
        state += system_dynamics(state, i*dt) * dt
    return (state[0] - target_x)**2

# We can take the gradient of the simulation with respect to initial state!
optimize_initial_state = grad(loss_fn)
```

### 2. Parameter Sweep with vmap

```python
def simulation(param):
    # Some complex computation
    return jnp.sum(jnp.linspace(0, param, 100))

# Parallelize simulation over a range of parameters
params = jnp.linspace(1, 10, 100)
results = vmap(simulation)(params)
```

### 3. Distributed Training with pmap

```python
# pmap replicates the function across multiple GPUs
# (assuming 8 GPUs are available)
# x = jnp.zeros((8, 1024))
# results = pmap(jnp.sin)(x)
```

## Performance Optimization

### Static Arguments in JIT

If your function uses a loop based on an input value, that value must be static.

```python
from functools import partial

@partial(jit, static_argnums=(1,))
def power_loop(x, n):
    for i in range(n):
        x = x * x
    return x
```

### Avoid Python Control Flow

Prefer JAX control flow (`cond`, `while_loop`, `fori_loop`) for better XLA optimization.

```python
from jax.lax import cond

def safe_divide(x, y):
    return cond(y == 0, lambda _: 0.0, lambda _: x / y, operand=None)
```

## Common Pitfalls and Solutions

### The "Tracer" Error

Inside JIT, JAX doesn't see actual numbers, it sees "Tracers".

```python
# ❌ Problem:
# @jit
# def func(x):
#     if x > 0: return x # Error! JAX doesn't know x's value during compile

# ✅ Solution:
# Use jax.lax.cond or mark x as static_argnum
```

### NaN Gradients

If your function has singularities (like `sqrt(0)`), gradients will be NaN.

```python
# ✅ Solution: Add a small epsilon
def safe_sqrt(x):
    return jnp.sqrt(x + 1e-8)
```

### Memory Leaks on GPU

JAX pre-allocates 90% of GPU memory by default.

```python
# ✅ Solution: Set environment variable
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
```

## Best Practices

1. **Always use pure functions** - No side effects, deterministic outputs
2. **Manage PRNG keys explicitly** - Split keys for every random operation
3. **Use JIT for hot loops** - Compile functions that are called repeatedly
4. **Leverage vmap for batching** - Write single-sample code, let JAX handle batches
5. **Mark static arguments** - Use `static_argnums` for non-array parameters in JIT
6. **Use functional updates** - Always use `.at` methods for array modifications
7. **Profile before optimizing** - Use `jax.profiler` to find bottlenecks
8. **Handle device placement** - Use `jax.device_put()` to control where arrays live
9. **Test on CPU first** - Debug on CPU, then scale to GPU/TPU
10. **Understand compilation costs** - First JIT call is slow, subsequent calls are fast

JAX is the ultimate playground for differentiable science. By treating math as a series of functional transformations, it unlocks speeds and complexities that were previously impossible in Python.