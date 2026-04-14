---
name: fdtdx
description: FDTDX framework knowledge — JAX-based FDTD simulation patterns, Yee grid conventions, pytree immutability, constraint system, gradient strategies, and inverse design workflows. Use when writing or modifying fdtdx code.
user-invocable: false
---

# FDTDX Framework Knowledge

## Immutability & PyTree Pattern

Every class inherits from `TreeClass` (wraps `pytreeclass.TreeClass`). Objects are **frozen JAX pytrees** — never mutate in place.

**Always use `.aset()` for updates:**
```python
# Single field
config = config.aset("gradient_config", grad_cfg)

# Nested path
obj = obj.aset("nested->field", value)
```

**Field types** (from `core/jax/pytrees.py`):
- `field()` — standard mutable pytree leaf (KW_ONLY by default)
- `frozen_field()` — excluded from pytree traversal (metadata, not differentiated)
- `private_field()` — not in `__init__`, set after construction
- `frozen_private_field()` — frozen + private combined
- `@autoinit` — decorator that auto-generates `__init__` from type hints

**Key implication:** Since objects are pytrees, they can flow through `jax.jit`, `jax.grad`, `jax.vmap`, etc. Use `frozen_field` for anything that should NOT be traced/differentiated (names, config flags, etc.).

## Simulation Pipeline

The canonical execution order is always:

```python
# 1. Define objects and constraints
volume = fdtdx.SimulationVolume(partial_real_shape=(Lx, Ly, Lz))
source = fdtdx.GaussianPlaneSource(...)
detector = fdtdx.PoyntingFluxDetector(...)
# ... add constraints ...

# 2. Resolve constraints and initialize arrays
objects, arrays, params, config, key = fdtdx.place_objects(
    object_list=object_list,
    config=config,
    constraints=constraints,
    key=key,
)

# 3. Apply device parameters to permittivity arrays
arrays, objects, info = fdtdx.apply_params(arrays, objects, params, key)

# 4. Run simulation
state = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key)
time_step, arrays = state

# 5. (Optional) Backward pass for gradient recording
_, arrays = fdtdx.full_backward(state=state, objects=objects, config=config, key=key)
```

**`place_objects()`** resolves all constraints iteratively (up to 1000 iterations), places objects on the grid, initializes E/H/PML field arrays, material arrays, detector states, and recording state.

**`apply_params()`** runs the device parameter transformation pipeline and writes resulting permittivities into the array container. For CONTINUOUS output, uses linear interpolation between materials. For DISCRETE output, uses straight-through estimator (STE).

**`run_fdtd()`** dispatches to either `reversible_fdtd()` or `checkpointed_fdtd()` based on `config.gradient_config`.

## Yee Grid Conventions

Axis mapping: 0=x, 1=y, 2=z. Field arrays have shape `(3, Nx, Ny, Nz)` where index 0 is the component index.

**Staggered field positions (Taflove convention):**
```python
E_x: (i+1/2, j,     k    )     H_x: (i,     j+1/2, k+1/2)
E_y: (i,     j+1/2, k    )     H_y: (i+1/2, j,     k+1/2)
E_z: (i,     j,     k+1/2)     H_z: (i+1/2, j+1/2, k    )
```

**Leapfrog time stepping:** E at integer steps, H at half-steps. Single time step order:
1. Update E fields (curl of H)
2. Update H fields (curl of E)
3. Inject sources
4. Record detectors

**Detector interpolation:** E and H are co-located at the E_z grid point `(i, j, k+1/2)` via multi-point averaging before recording.

## Field Normalization

FDTDX uses **eta0-normalized H fields** — the impedance of free space (eta0 ~ 376.73 Ohm) is absorbed into the field update equations rather than appearing explicitly.

**Update equations (isotropic, lossless):**
```
E^(n+1) = E^n + c * curl(H) * inv_permittivity
H^(n+1/2) = H^(n-1/2) - c * curl(E) * inv_permeability
```
where `c = courant_number = courant_factor / sqrt(3)` (default: 0.99/sqrt(3) ~ 0.571).

**With conductivity (lossy):**
```
factor_E = 1 - c * sigma_E * eta0 * inv_eps / 2
E = factor_E * E + c * curl(H) * inv_eps
E = E / (1 + c * sigma_E * eta0 * inv_eps / 2)

factor_H = 1 - c * sigma_H / eta0 * inv_mu / 2
H = factor_H * H - c * curl(E) * inv_mu
H = H / (1 + c * sigma_H / eta0 * inv_mu / 2)
```

Note the asymmetry: sigma_E multiplied by eta0, sigma_H divided by eta0.

## Material System

Materials store **inverse** permittivity/permeability (`inv_permittivities`, `inv_permeabilities`) to avoid division in the hot loop.

**Internal representation:** Always 9-tuple `(xx, xy, xz, yx, yy, yz, zx, zy, zz)` for the full 3x3 tensor, but array sizing adapts:
- **Isotropic** (all objects scalar): 1-component arrays
- **Diagonally anisotropic** (any object has 3 components): 3-component arrays
- **Fully anisotropic** (any object has 3x3 tensor): 9-component arrays

This is determined globally — if ANY object is anisotropic, ALL material arrays expand. The detection happens in `ObjectContainer` properties like `all_objects_isotropic_permittivity`.

**Conductivity is scaled by resolution** during initialization (line ~459 in `initialization.py`).

## Constraint System

Objects are positioned relative to each other via constraint objects. Key constraint builders on `SimulationObject`:

```python
# Position relative to another object
obj.place_relative_to(other, axes=(2,), own_positions=("x1",), other_positions=("x2",))
obj.place_at_center(volume)
obj.place_above(other, margin=0.5e-6)
obj.face_to_face_same_side(other, axis=2, position="x2", margin=0.1e-6)

# Size constraints
obj.same_size(volume, axes=(0, 1))
obj.size_relative_to(volume, axes=(2,), proportions=(0.5,))
obj.extend_to(boundary, axis=2, side="x2")

# Grid-level positioning
obj.set_grid_coordinates(axes=(2,), sides=("x1",), coordinates=(10,))
```

**Constraint types:** `PositionConstraint`, `SizeConstraint`, `SizeExtensionConstraint`, `GridCoordinateConstraint`, `RealCoordinateConstraint`.

Objects specify `partial_real_shape` (meters) or `partial_grid_shape` (voxels) with `None` for unconstrained dimensions that will be resolved by constraints.

## Boundary Conditions

**PML (Perfectly Matched Layer):** CPML formulation with polynomial-graded sigma, kappa, alpha profiles. Uses 6 auxiliary psi fields `(psi_Ex, psi_Ey, psi_Ez, psi_Hx, psi_Hy, psi_Hz)` each with shape `(Nx, Ny, Nz)`. PML breaks time-reversal symmetry, so interface fields must be recorded for reversible gradients.

**PEC / PMC:** Zero tangential E (PEC) or H (PMC) at boundary. Applied via `apply_field_reset()`.

**Bloch Boundary:** Phase-shifted periodic conditions with `bloch_vector=(kx, ky, kz)`. When any k component is nonzero, complex fields are required. `PeriodicBoundary` is an alias for `BlochBoundary` with `bloch_vector=(0,0,0)`.

**BoundaryConfig helper:**
```python
bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="pml", thickness=10)
bound_dict, constraint_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
```

## Gradient Strategies

**Reversible FDTD** (`method="reversible"`):
- Exploits time-reversibility of Maxwell's equations
- O(1) field memory, O(T) boundary memory (PML interfaces only)
- Uses `@jax.custom_vjp` — forward pass runs simulation recording boundaries, backward pass reconstructs fields in reverse
- Requires a `Recorder` with optional compression modules (e.g., `DtypeConversion(dtype=jnp.bfloat16)`)
- Only computes gradients w.r.t. `inv_permittivities` and `inv_permeabilities`

**Checkpointed FDTD** (`method="checkpointed"`):
- Standard gradient checkpointing via `eqxi.while_loop(kind="checkpointed")`
- Configurable memory/compute tradeoff via `num_checkpoints`

**Setup pattern:**
```python
recorder = fdtdx.Recorder(modules=[fdtdx.DtypeConversion(dtype=jnp.bfloat16)])
gradient_config = fdtdx.GradientConfig(method="reversible", recorder=recorder)
config = config.aset("gradient_config", gradient_config)
```

## Device & Parameter Transformations

Devices are optimizable regions with a parameter transformation pipeline:

```python
device = fdtdx.Device(
    materials={"air": air, "si": silicon},
    param_transforms=[
        fdtdx.StandardToInversePermittivityRange(),
        fdtdx.GaussianSmoothing2D(sigma=1.0),
        fdtdx.TanhProjection(beta=4.0),
        fdtdx.ClosestIndex(),
    ],
    partial_real_shape=(...),
    partial_voxel_real_shape=(...),  # voxel grid can differ from sim grid
)
```

**Pipeline order:** projection -> smoothing -> discretization -> discrete post-processing -> symmetry.

**Parameter types flow:** `CONTINUOUS` (float values interpolating between materials) -> `DISCRETE` (integer material indices). The STE (straight-through estimator) bridges discrete forward with continuous gradients.

**Voxel indirection:** Devices have their own voxel grid independent of the simulation grid, allowing coarse optimization on a fine simulation mesh.

## Sources

**TFSF (Total-Field/Scattered-Field):** Plane sources inject fields at a boundary offset +0.25 on the Yee grid along the propagation axis.

**Source types:**
- `UniformPlaneSource` — uniform amplitude across plane
- `GaussianPlaneSource` — Gaussian beam profile with configurable `radius`
- `ModePlaneSource` — injects a computed waveguide mode profile
- `PointDipoleSource` — point dipole with configurable polarization axis and orientation angles

**Temporal profiles:** `SingleFrequencyProfile` (CW) or `GaussianPulseProfile` (pulsed).

**On/Off control:** `OnOffSwitch` pre-computes boolean arrays for the entire simulation duration during `place_on_grid()`.

## Detectors

All detectors use `OnOffSwitch` for temporal gating. State is stored as `DetectorState = Dict[str, Array]`.

- `FieldDetector` — records raw E/H field components
- `EnergyDetector` — records electromagnetic energy density
- `PoyntingFluxDetector` — records directional power flow (key for transmission/reflection)
- `PhasorDetector` — records complex phasor amplitudes at specific frequencies
- `DiffractiveDetector` — records complex diffraction efficiencies per order
- `ModeOverlapDetector` — computes overlap integral with a guided mode (inherits from PhasorDetector)

**Accessing results:** `arrays.detector_states["name"]["key"]`

All state arrays have a leading time dimension: `(num_time_steps_on, ...)`. Use index `-1` for the final accumulated value.

**FieldDetector** — key: `"fields"`
- `reduce_volume=False`: `(T, num_components, nx, ny, nz)`
- `reduce_volume=True`: `(T, num_components)`

**EnergyDetector** — key: `"energy"` or slice keys
- `as_slices=False, reduce_volume=False`: `(T, nx, ny, nz)`
- `as_slices=False, reduce_volume=True`: `(T, 1)` (scalar)
- `as_slices=True`: three keys `"XY Plane"` `(T, nx, ny)`, `"XZ Plane"` `(T, nx, nz)`, `"YZ Plane"` `(T, ny, nz)` — cannot combine with `reduce_volume=True`

**PoyntingFluxDetector** — key: `"poynting_flux"`
- Default (`reduce_volume=True`, scalar): `(T, 1)` — total flux through surface
- `keep_all_components=True, reduce_volume=True`: `(T, 3)`
- `reduce_volume=False`: `(T, nx, ny, nz)` or `(T, 3, nx, ny, nz)` with `keep_all_components`

**PhasorDetector** — key: `"phasor"`, dtype: complex
- Time dim is always 1 (frequency-domain accumulation)
- `reduce_volume=False`: `(1, num_wavelengths, num_components, nx, ny, nz)`
- `reduce_volume=True`: `(1, num_wavelengths, num_components)`
- Component index matches order of the `components` tuple

**DiffractiveDetector** — key: `"diffractive"`, dtype: complex
- Time dim is always 1
- Shape: `(1, num_frequencies, num_orders)`

**ModeOverlapDetector** — inherits PhasorDetector, always uses all 6 field components. Use `compute_overlap_to_mode()` to get the scalar overlap.

## Testing Patterns

**Three test tiers** (auto-marked via conftest.py):
- `unit` — individual components, no simulation runs
- `integration` — object placement, initialization, multi-component interaction
- `simulation` — full FDTD runs validating physics

**Physics validation pattern (two-run normalization):**
```python
# Reference run (e.g., all PML) and test run (e.g., with PEC) share a helper:
def _run(objects, constraints, config):
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key,
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=config, key=key)
    return arrays

def _mean_flux(arrays, name):
    flux = np.array(arrays.detector_states[name]["poynting_flux"][:, 0])
    return float(np.mean(flux[-N_AVG_STEPS:]))

ref_flux = _mean_flux(_run(ref_objects, ref_constraints, config), "detector")
test_flux = _mean_flux(_run(test_objects, test_constraints, config), "detector")
transmission = test_flux / ref_flux
```

**Steady-state extraction:** Average over last N optical periods:
```python
steps_per_period = int(round(wavelength / (c0 * dt)))
n_avg = 10 * steps_per_period
steady_state = float(np.mean(flux[-n_avg:]))
```

**Gradient validation:**
```python
loss, grads = jax.value_and_grad(loss_fn)(params, arrays, objects, config, key)
assert jnp.isfinite(loss)
assert jnp.all(jnp.isfinite(grads))
```

**When simulation tests fail marginally:** Increase resolution (more cells per wavelength) rather than relaxing tolerances — the physics should converge, not the assertions weaken.

## Common Pitfalls

- **Forgetting `.aset()`**: Direct attribute assignment on TreeClass objects silently fails or raises. Always use `.aset()`.
- **Material array sizing is global**: Adding one anisotropic object forces ALL material arrays to expand. Check `ObjectContainer` isotropy properties.
- **PML + reversible gradients**: PML breaks time-reversal. Must set up `Recorder` and `recording_state` for boundary interfaces.
- **Complex fields**: Bloch boundaries with nonzero k-vector automatically require complex fields. Check `config.use_complex_fields`.
- **Conductivity scaling**: Conductivity values are multiplied by `config.resolution` during `_init_arrays()`. Don't pre-scale.
- **Inverse storage**: Material arrays store `1/epsilon` and `1/mu`, not epsilon and mu directly.
- **Detector timing**: Detectors only record at timesteps where their `OnOffSwitch` is active. Check `switch` configuration if data appears missing.
- **donate_argnames**: When JIT-compiling simulation functions, use `donate_argnames=["arrays"]` to allow JAX to reuse array memory.
