---
name: jax-pde
description: Advanced sub-skill for JAX focused on solving Partial Differential Equations (PDEs) and Differentiable Physics. Covers Finite Difference Methods (FDM), Neural Operators, and Physics-Informed Neural Networks (PINNs).
version: 0.4
license: Apache-2.0
user-invocable: false
---

# JAX - Differentiable Physics & PDEs

JAX is uniquely suited for physics because it can differentiate through numerical solvers. This guide covers how to implement traditional PDE solvers that are "optimization-friendly" and how to build neural-hybrid physical models.

## When to Use

- Solving Navier-Stokes, Wave, or Heat equations on GPU.
- Implementing Physics-Informed Neural Networks (PINNs).
- Performing Inverse Design (finding material properties from observations).
- Creating differentiable simulations for robotics or climate modeling.
- Sensitivity analysis of physical systems.

## Core Principles

### 1. Differentiation through the Solver

In JAX, if you write an Euler or Runge-Kutta integrator using `jax.numpy`, you can automatically calculate ∂Result/∂InitialCondition or ∂Result/∂Viscosity.

### 2. Staggered Grids & Vmap

Physical fields (velocity, pressure) are often stored on grids. JAX's `vmap` allows you to parallelize solvers across different boundary conditions or parameter sets instantly.

### 3. The Adjoint Method

For very large systems, JAX's reverse-mode autodiff effectively implements the "Adjoint State Method" used in traditional CFD/Geophysics for gradient calculation.

## Implementation Patterns

### 1. PINNs (Physics-Informed Neural Networks)

```python
import jax.numpy as jnp
from jax import grad, vmap

# A simple MLP representing the solution u(x, t)
def model(params, x, t):
    # standard neural net logic...
    return result

# Residual of the PDE: u_t + u*u_x - nu*u_xx = 0 (Burgers Equation)
def pde_loss(params, x, t, nu):
    u = lambda x, t: model(params, x, t)
    
    # Automatic derivatives of the MODEL
    u_t = grad(u, argnums=1)(x, t)
    u_x = grad(u, argnums=0)(x, t)
    u_xx = grad(grad(u, argnums=0), argnums=0)(x, t)
    
    return jnp.mean((u_t + u * u_x - nu * u_xx)**2)
```

### 2. Differentiable Finite Difference Solver

```python
@jit
def update_step(u, dt, dx, nu):
    """One step of a diffusion solver."""
    # Vectorized Laplacian using shifts (Zero-copy views)
    u_left = jnp.roll(u, -1)
    u_right = jnp.roll(u, 1)
    laplacian = (u_left + u_right - 2*u) / (dx**2)
    return u + dt * nu * laplacian

# We can now differentiate this solver!
def loss(initial_u, target_u):
    final_u = integrate_pde(initial_u) # Loop of update_step
    return jnp.sum((final_u - target_u)**2)

grad_initial_condition = grad(loss)(initial_u, target_u)
```

## Critical Rules

### ✅ DO

- **Use jax.lax.scan for time loops** - Standard Python for loops create massive XLA graphs. `scan` compiles the loop into a single efficient kernel.
- **Normalize your Grids** - Like ML, PINNs converge faster if x, t are scaled to [0,1] or [-1,1].
- **Combine Data and Physics** - Use PINNs where you have some sensor data + the physical law to "fill the gaps".
- **Use Double Precision for Physics** - Use `jax.config.update("jax_enable_x64", True)` for sensitive numerical solvers.

### ❌ DON'T

- **Don't use PINNs for everything** - Traditional solvers (FDM/FEM) are much faster for "forward" problems. PINNs excel at "inverse" problems.
- **Don't ignore Boundary Conditions (BCs)** - In PINNs, BCs must be added to the loss function: Loss = PDE_loss + BC_loss.
- **Don't forget the 'Ghost Cells'** - When implementing FDM, handle boundaries carefully to avoid artifacts.

## Practical Workflows: Inverse Problem

### Finding Viscosity from a Video of Fluid

```python
def objective(nu_guess):
    # 1. Run simulation with nu_guess
    final_state = run_simulation(initial_state, nu_guess)
    # 2. Compare with experimental data
    return jnp.mean((final_state - experimental_frame)**2)

# Gradient descent to find the real physical property
optimal_nu = optimize(grad(objective))
```

JAX PDE transforms physics from a static simulation into a dynamic, optimizable landscape. It allows researchers to ask "What physical parameters produced this result?" and find the answer through the power of gradients.