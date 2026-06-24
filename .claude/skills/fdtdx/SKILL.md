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

**`place_objects()`** first resolves the grid policy into a concrete `RectilinearGrid` from the volume shape (so constraints solve against real edge coordinates — see Grid Policies), then resolves all constraints iteratively (up to 1000 iterations), places objects on the grid, initializes E/H/PML field arrays, material arrays, detector states, and recording state.

**`apply_params()`** runs the device parameter transformation pipeline and writes resulting permittivities into the array container. For CONTINUOUS output, uses linear interpolation between materials. For DISCRETE output, uses straight-through estimator (STE).

**`run_fdtd()`** dispatches to either `reversible_fdtd()` or `checkpointed_fdtd()` based on `config.gradient_config`.

## Grid Policies & Coordinate System

**The coordinate origin is at the CENTER of the simulation domain** (changed in #363, Jun 2026 — previously the lower-left corner). Real-space coordinates span symmetrically: an axis of physical length `L` runs from `-L/2` to `+L/2`. All `partial_real_position` / real-coordinate-constraint values are measured from the domain center (0 = center, negative = lower half, positive = upper half). Any pre-#363 code that placed objects in corner-relative `[0, L]` coordinates must be shifted by `-L/2`.

**Three grid representations** (`src/fdtdx/core/grid.py`):
- `UniformGrid(spacing, center=(0,0,0))` — *policy*. Cubic cells of one scalar `spacing`; user intent before the volume shape is known.
- `QuasiUniformGrid(dx, dy, dz, center=(0,0,0))` — *policy* (new in #363, exported as `fdtdx.QuasiUniformGrid`). Independent per-axis spacing (rectangular-parallelepiped cells); each axis internally uniform. `is_uniform` is True only when `dx==dy==dz`.
- `RectilinearGrid(x_edges, y_edges, z_edges)` — *realized* solver grid: explicit strictly-increasing edge arrays (`nx+1` per axis). The single canonical metric source the FDTD loop reads. Build directly, via `.uniform(shape, spacing, center=...)`, or `.custom(...)`.

Policies `.resolve(shape)` → `RectilinearGrid`, centered on `center`. **`QuasiUniformGrid.resolve` requires an even cell count on every axis** (raises `ValueError` otherwise — the center must land on a cell edge); `UniformGrid` does not enforce this.

**The grid is resolved FIRST, before constraint solving** (#363 reordered `place_objects`). The volume's `partial_grid_shape` (or `partial_real_shape ÷ policy spacing`) pins a concrete `RectilinearGrid` into `config.grid` up front, so every object sees resolved physical edge coordinates during placement (no temporary-coordinate handling). Under `config.symmetry` the grid is re-resolved/sliced onto the reduced upper half.

**Config grid helpers:**
- `config.resolved_grid` → `RectilinearGrid | None` (`None` while still an unresolved policy).
- `config.has_nonuniform_grid` → True only once resolved to genuinely non-uniform edges.
- `config.resolve_grid(shape)` → concrete grid; `config.uniform_spacing()` → scalar (raises for non-uniform / unequal `QuasiUniformGrid`).
- `config.time_step_duration` uses the grid's CFL bound: uniform → `courant_factor/√3 · spacing/c`; non-uniform `RectilinearGrid` → `courant_factor / (c·√(1/dx_min²+1/dy_min²+1/dz_min²))`.

There is no longer a global `config.resolution` scalar — ask the grid for physical distances (`cell_widths(axis)`, `centers(axis)`, `edges(axis)`, `face_area(...)`, `cell_volume(...)`, `min_spacing`).

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

**With dispersion (ADE correction):** After the lossless/lossy E update but before the final divide by `(1 + c*sigma_E*eta0*inv_eps/2)`, add the per-pole polarization increment. For each pole `p`:
```text
P_p^(n+1) = c1_p * P_p^n + c2_p * P_p^(n-1) + c3_p * E^n
E        += inv_eps * sum_p (P_p^n - P_p^(n+1))
```
`P` is stored normalized as `P/eps_0`, so it has the same units as `E` and no eta0 factor enters. The reverse-time update in `update_E_reverse` inverts this recurrence (`c2 ~ -1` in the physical regime keeps the inversion numerically stable).

## Material System

Materials store **inverse** permittivity/permeability (`inv_permittivities`, `inv_permeabilities`) to avoid division in the hot loop.

**Internal representation:** Always 9-tuple `(xx, xy, xz, yx, yy, yz, zx, zy, zz)` for the full 3x3 tensor, but array sizing adapts:
- **Isotropic** (all objects scalar): 1-component arrays
- **Diagonally anisotropic** (any object has 3 components): 3-component arrays
- **Fully anisotropic** (any object has 3x3 tensor): 9-component arrays

This is determined globally — if ANY object is anisotropic, ALL material arrays expand. The detection happens in `ObjectContainer` properties like `all_objects_isotropic_permittivity`.

**Conductivity is scaled by the grid cell spacing** during initialization in `_init_arrays` (`src/fdtdx/fdtd/initialization.py`) — the scale factor is `constants.c * config.time_step_duration / config.courant_number` (equal to the cell width for a uniform grid). There is no `config.resolution` to pre-scale against.

**Material fields:**
- `permittivity`, `permeability`, `electric_conductivity`, `magnetic_conductivity` — 9-tuples (scalar/3-tuple/nested-3x3 inputs auto-normalized).
- `dispersion: DispersionModel | None` — attaches an ADE dispersion model. When set, `permittivity` is the high-frequency permittivity ε∞ and the full ε(ω) = ε∞ + χ(ω).
- `is_dispersive` property → True iff `dispersion` has at least one pole.

## Dispersive Materials (ADE)

Linear dispersion is implemented via the Auxiliary Differential Equation (ADE) method in `src/fdtdx/dispersion.py`. A `DispersionModel` is a sum of 2nd-order poles, each solving `p̈ + γ ṗ + ω₀² p = K E` for a normalized polarization `p = P/ε₀`.

**Pole classes** (all inherit from `Pole`, stored as `frozen_field` inside `DispersionModel`):
- `LorentzPole(resonance_frequency, damping, delta_epsilon)` — `χ(ω) = Δε·ω₀² / (ω₀² − ω² − iγω)`
- `DrudePole(plasma_frequency, damping)` — `χ(ω) = −ωₚ² / (ω² + iγω)` (special case ω₀ = 0)
- New pole types: subclass `Pole` and expose `omega_0`, `gamma`, `coupling_sq` (K = Δε·ω₀² for Lorentz, ωₚ² for Drude).

**Discrete-time recurrence** (central differences, evaluated once at setup via `compute_pole_coefficients(poles, dt)`):
```text
p^(n+1) = c1·p^n + c2·p^(n-1) + c3·E^n
c1 = (2 − ω₀²·dt²) / (1 + γ·dt/2)
c2 = −(1 − γ·dt/2) / (1 + γ·dt/2)
c3 =  (K·dt²)      / (1 + γ·dt/2)
```
Stability needs `γ·dt < 2`; physically `γ·dt ≪ 1`, so `c2 ≈ −1` and the reverse-time inversion in `update_E_reverse` is well-conditioned.

**ArrayContainer fields** (all `None` unless any object is dispersive):
- `dispersive_P_curr`, `dispersive_P_prev` — shape `(num_poles, 3, Nx, Ny, Nz)`, field-dtype (complex if `use_complex_fields`). Not differentiable (state-only; `None` cotangent in both gradient paths).
- `dispersive_c1`, `dispersive_c2`, `dispersive_c3` — shape `(num_poles, 1, Nx, Ny, Nz)` (middle axis broadcasts over field components). Config dtype. Differentiable: cotangents flow through them in both the reversible and checkpointed paths.
- `dispersive_inv_c2` — cached `1/c2`, closure-captured and `stop_gradient`'d so gradients flow through `c2` only and don't double-count.

**Leading pole axis size:** `objects.max_num_dispersive_poles` — the max pole count across all `UniformMaterialObject`, `Device`, `StaticMultiMaterialObject`. Materials with fewer poles get zero-padded slots, so non-dispersive cells automatically contribute zero. `UniformMaterialObject` always writes the full zero-padded coefficient stack into its `grid_slice`, so a non-dispersive object placed over a dispersive one cleanly clears stale coefficients.

**Restriction:** Dispersive materials cannot currently be combined with fully anisotropic (off-diagonal) permittivity tensors — `_init_arrays` raises `NotImplementedError`. Isotropic and diagonally anisotropic ε are both fine.

**Devices with dispersive materials:** `apply_params` interpolates ADE coefficients the same way it interpolates `inv_permittivities` — linearly between the two bracketing materials for `CONTINUOUS` output, straight-through-estimator for `DISCRETE`. This is not equivalent to interpolating the pole *parameters*, but it keeps gradients smooth for inverse design.

**Evaluating χ(ω) / ε(ω) from stored coefficients** (useful in sources, detectors, setup-time analysis):
- `susceptibility_from_coefficients(c1, c2, c3, omega, dt)` → JAX complex array of per-cell χ(ω), summed over poles.
- `effective_inv_permittivity(inv_eps, c1, c2, c3, omega, dt)` → real 1/Re(ε∞ + χ(ω)); used by sources to sample the true medium at the carrier frequency (imaginary part is already handled by the ADE loop — injecting it would double-count).
- `compute_eps_spectrum_from_coefficients(c1, c2, c3, inv_eps_inf, omegas, dt, weights=None)` → host-side numpy; volume-averaged complex ε(ω) spectrum for a block of cells.
- `compute_impedance_corrected_temporal_profile(raw_samples, dt, eps_spectrum, eps_center)` → applies the FIR filter `G(ω) = √(ε(ω)/ε(ω_c))` to an E-side temporal profile, producing the H-side profile for broadband TFSF injection.

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

## Simulation Symmetry

Mirror-symmetry exploitation: build the **full** model, set `config.symmetry`, and fdtdx runs the reduced half/quarter/octant internally (up to 8× less memory/compute), then you unfold results back to the full domain. Implemented in `src/fdtdx/fdtd/symmetry.py`; the FDTD time loop is untouched (the symmetry plane is just an added PEC/PMC wall object).

**Encoding** — `symmetry: tuple[int, int, int]` on `SimulationConfig`, order `(x, y, z)`:
- `0` = no symmetry on this axis
- `-1` = **PEC** mirror (electric wall) on the axis center plane — tangential E odd, normal E even
- `+1` = **PMC** mirror (magnetic wall) — tangential H odd, normal H even

Distinct from manually placing PEC/PMC via `BoundaryConfig` (that still works unchanged); `config.symmetry` is the additive auto-reduce path.

**Requirements / behavior:**
- Each symmetric axis **must resolve to an even cell count** (else `place_objects` raises `ValueError`) — guarantees an exact split and cell-for-cell unfold.
- The **upper half is kept** so the plane lands at the reduced domain's min edge (matching the mode solver's "wall at min edge" convention). Objects are clipped to that half during `place_objects`; centered objects keep their upper half, objects entirely in the discarded half are dropped (with a warning).
- The min-side boundary on each symmetric axis is replaced by the PEC/PMC wall; the far (max) side keeps whatever the user set (use PML there, not periodic).
- `ModePlaneSource` / `ModeOverlapDetector` get their mode-solver `symmetry` 2-tuple **auto-derived** (PMC→1, PEC/none→0 on the two transverse axes) unless explicitly set.
- The user must place objects symmetrically about the center plane — asymmetric models are warned about but not corrected (true of every FDTD symmetry feature).

**Usage:**
```python
config = fdtdx.SimulationConfig(
    grid=fdtdx.UniformGrid(spacing=...),
    time=...,
    symmetry=(0, -1, 1),
)  # PEC y-plane, PMC z-plane
# ... build the FULL volume/sources/detectors/boundaries as usual ...
objects, arrays, params, config, _ = fdtdx.place_objects(...)   # reduced internally
arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
_, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key)  # runs on reduced domain

# Unfold to full domain (explicit, post-processing — NOT auto-run by run_fdtd):
full = fdtdx.unfold_detector_states(arrays, objects, config)     # full-domain detector_states
E_full = fdtdx.unfold_fields(arrays.fields.E, config.symmetry, "E")  # (3, Nx, Ny, Nz)
```

**Unfold helpers** (`fdtdx.unfold_fields`, `fdtdx.unfold_detector_states`, `fdtdx.unfold_source_mode`, `fdtdx.unfold_array`):
- `unfold_fields(field, symmetry, field_type)` — reconstruct a full `(3, Nx, Ny, Nz)` E/H array via per-component parity mirror. The general escape hatch — derive any quantity from the full fields.
- `unfold_detector_states(arrays, objects, config)` — pure post-processing that rebuilds each detector's full-domain output from its stored reduced output + parity (no in-loop cost, no flags). Spatial outputs are mirrored per component; `reduce_volume` sums/means are rescaled per component (even doubles/keeps, **odd vanishes**); `as_slices` energy planes are mirrored in-plane.
- `unfold_source_mode(source, config)` → `(E_full, H_full)` — reconstruct the full-domain mode profile a `ModePlaneSource` *injects* (its solved-on-the-reduced-cross-section `_E`/`_H`). Unfolds only the transverse axes (the propagation axis is never a symmetry plane). Run `apply_params` first. For the fields *recorded during the run*, prefer a detector on the source plane + `unfold_detector_states`.
- **Guardrails:** unfolding a non-symmetric model (`symmetry=(0,0,0)`) raises `ValueError`; `place_objects` warns that results are on the reduced domain until unfolded.
- **Not unfoldable:** `DiffractiveDetector` raises `NotImplementedError` (its diffraction-order basis depends on domain size — unfold the fields and recompute instead).
- **Mode-overlap S-params** are already correct on the reduced domain (source + detector share the reduced plane), so they need no unfolding.

**Mode sources are fully wired:** under symmetry, a `ModePlaneSource`'s cross-section is clipped to the reduced grid, its mode-solver `symmetry` 2-tuple is auto-derived from `config.symmetry`, and `compute_mode` solves/injects the half/quarter mode with the matching PEC/PMC wall. Use `unfold_source_mode` to inspect the reconstructed full profile.

**Gradient note:** the differentiable simulation runs on the reduced domain (correct and cheaper); unfolding is a post-hoc step on the output arrays.

## Gradient Strategies

**Reversible FDTD** (`method="reversible"`):
- Exploits time-reversibility of Maxwell's equations
- O(1) field memory, O(T) boundary memory (PML interfaces only)
- Uses `@jax.custom_vjp` — forward pass runs simulation recording boundaries, backward pass reconstructs fields in reverse
- Requires a `Recorder` with optional compression modules (e.g., `DtypeConversion(dtype=jnp.bfloat16)`)
- Differentiable primals: `inv_permittivities`, `inv_permeabilities`, and (when present) `dispersive_c1/c2/c3`. Conductivity arrays and `dispersive_inv_c2` are closure-captured non-primals; `dispersive_P_curr/prev` thread through as state-only primals with `None` cotangent.
- Dispersive reverse update: the ADE recurrence `P^(n+1) = c1·P^n + c2·P^(n-1) + c3·E^n` is algebraically inverted to recover `P^(n-1)` (see `update_E_reverse`). For lossy + dispersive + conductive cells the reverse E update subtracts `inv_eps * sum(P^n − P^(n+1))` before dividing by the loss factor.

**Checkpointed FDTD** (`method="checkpointed"`):
- Standard gradient checkpointing via `eqxi.while_loop(kind="checkpointed")`
- Configurable memory/compute tradeoff via `num_checkpoints`
- Dispersive coefficients flow gradient naturally through the tape.

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

## GDS Import

`fdtdx.gds_layer_stack(gds_source, cell_name, layers, materials, simulation_volume, gds_center, ...)` builds `GDSLayerObject`s (one per `GDSLayerSpec`) from a GDS layout (`src/fdtdx/objects/static_material/gds_layer_stack.py`).

- `gds_center: tuple[float, float]` is the GDS coordinate (metres) that maps to the **center** of the simulation volume (center-origin convention, #363). `gds_center=(0, 0)` puts the GDS origin at the domain center; `(500e-9, 0)` shifts the layout 500 nm.
- `GDSLayerSpec` fields: `gds_layer`, `material_name`, `thickness`, `gds_datatype=0`, `z_base`, `name`, `etch_by` (layer/datatype pairs subtracted via boolean NOT before voxelization), and `color: Color | None = XKCD_LIGHT_GREY` — display color for `plot_set_up`; `None` leaves the object uncolored (#357).
- An empty `layers` list raises `ValueError` (#357/#363).
- `sources_from_gds_ports` / `detectors_from_gds_ports` place mode sources/detectors from GDS port markers (port `propagation_axis` must be 0 or 1).

## Sources

**TFSF (Total-Field/Scattered-Field):** Plane sources inject fields at a boundary offset +0.25 on the Yee grid along the propagation axis.

**Source types:**
- `UniformPlaneSource` — uniform amplitude across plane
- `GaussianPlaneSource` — Gaussian beam profile with configurable `radius`
- `ModePlaneSource` — injects a computed waveguide mode profile
- `PointDipoleSource` — point dipole with configurable `polarization` axis (0/1/2) plus optional `azimuth_angle`/`elevation_angle` (degrees) to tilt off-axis; also `source_type` ∈ {"electric","magnetic"}.

**Temporal profiles** (`src/fdtdx/objects/sources/profile.py`, all subclass `TemporalProfile`): `SingleFrequencyProfile` (CW, with a linear startup ramp), `GaussianPulseProfile` (pulsed), `CustomTimeSignalProfile` (arbitrary sampled waveform), and the carrier-free *window* profiles `GaussianWindowProfile`/`TukeyWindowProfile` (used as detector apodization — see Temporal Apodization). The window/envelope shapes are shared functions in `src/fdtdx/core/physics/temporal_windows.py` (`gaussian_envelope`, `linear_rampup`, `tukey_envelope`, `windowed_dft`) — `GaussianPulseProfile`/`SingleFrequencyProfile` build their envelope/ramp from them, so there is one implementation. `TemporalProfile.sample_time_signal` / `frequency_spectrum` sample the signal at the FDTD cadence.

**Polarization & off-normal incidence (shared helper):** `LinearlyPolarizedPlaneSource` (→ `UniformPlaneSource`/`GaussianPlaneSource`) takes `fixed_E_polarization_vector`/`fixed_H_polarization_vector` plus `azimuth_angle`/`elevation_angle` (degrees) to tilt the wave vector off the plane normal. The resolve-polarization → wave-vector → tilt sequence lives in **`tilted_polarization_vectors`** (`src/fdtdx/core/misc.py`), reused by both the plane sources and `GaussianModeOverlapDetector`/`gaussian_mode_fields` (so an analytic Gaussian *reference mode* and an injected Gaussian *source* derive polarization/tilt identically). It returns `(e_pol, h_pol, wave_vector)` via `normalize_polarization_for_source` + `get_wave_vector_raw` + `rotate_vector` with `axes_tpl = (horizontal_axis, vertical_axis, propagation_axis)`.

**Injected power spectrum** (`Source.injected_power_spectrum(frequencies, *, apodization=None)`): analytic per-frequency source power, no run. For `TFSFPlaneSource` (plane/mode) it uses the exact separable injection `E(r,n)=_E(r)·s_E(n+δ_E)`, `H(r,n)=_H(r)·s_H(n+δ_H)` → `P(f)=½ Re[Ŝ_E·conj(Ŝ_H)·G]` (`Ŝ` = windowed DFT of the temporal signal, `G` = spatial Poynting factor of `_E/_H` with per-cell Yee-offset phases). `PointDipoleSource.injected_power_spectrum` **raises** — a dipole's radiated power is environment-dependent (Purcell), so use the *measured* `radiated_power_spectrum` (homogeneous run → nominal, real run → actual). See Spectral Power & Transmission.

**On/Off control:** `OnOffSwitch` pre-computes boolean arrays for the entire simulation duration during `place_on_grid()`.

**`SimulationObject.apply()` signature** — `apply_params` passes dispersive coefficients through to every object:
```python
def apply(self, *, key, inv_permittivities, inv_permeabilities,
          dispersive_c1=None, dispersive_c2=None, dispersive_c3=None): ...
```
Coefficient arrays are passed with `stop_gradient` (matching how `inv_permittivities` is passed to source apply) — the FDTD VJP itself still differentiates through them, this only avoids gradient noise from the source amplitude path. Objects that don't use them (detectors, boundaries, uniform material objects) just `del` the kwargs; sources use them to sample the real medium at the carrier frequency.

**Carrier-frequency impedance in dispersive media:** Sources inside a dispersive background call `effective_inv_permittivity(...)` to get `1/Re(ε∞ + χ(ω_c))` before computing impedance and energy normalization — otherwise they would use only ε∞ and inject with the wrong amplitude ratio. This happens in `LinearlyPolarizedPlaneSource.apply`, `ModePlaneSource.apply`, and `PointDipoleSource.apply`. `PointDipoleSource` additionally uses `_contract_orientation` (einsum over the flattened 9-tensor) so off-diagonal ε coupling is picked up correctly for tilted dipoles. `ModeOverlapDetector.apply` uses the same correction so the reference mode profile is solved against ε(ω_c).

**Broadband TFSF correction** (`_build_dispersive_H_filter` in `src/fdtdx/objects/sources/tfsf.py`): When a source sits in a dispersive medium and its `temporal_profile` is wideband (e.g. `GaussianPulseProfile`), the η(ω_c) rescale alone leaks unphysical reflections at off-carrier frequencies. `TFSFPlaneSource` precomputes a filtered H-side temporal profile `s_H(t)` with spectrum `S(ω)·√(ε(ω)/ε(ω_c))` (stored in `_temporal_H_filter`, shape `(time_steps_total,)`) and looks it up per step via `jnp.interp` at the Yee half-step offset. Non-dispersive case leaves `_temporal_H_filter = None` and the inner loop falls back to the raw `temporal_profile.get_amplitude` call — so non-dispersive behavior is bit-identical.

Bulk ε(ω) is averaged uniformly over the source cells — correct for `LinearlyPolarizedPlaneSource`, a first-order approximation for `ModePlaneSource` (captures bulk dispersion of the guiding medium, not geometric modal dispersion).

## Detectors

All detectors use `OnOffSwitch` for temporal gating. State is stored as `DetectorState = Dict[str, Array]`.

- `FieldDetector` — records raw E/H field components
- `EnergyDetector` — records electromagnetic energy density
- `PoyntingFluxDetector` — records directional power flow (key for transmission/reflection)
- `PhasorDetector` — records complex phasor amplitudes at specific frequencies; supports smooth temporal `apodization` (see Temporal Apodization)
- `DiffractiveDetector` — records complex diffraction efficiencies per order
- `BaseModeOverlapDetector` (abstract) — owns the overlap-integral machinery; `_compute_mode_fields` is the only abstract hook. Subclasses: `ModeOverlapDetector` (solver-based reference mode), `CustomModeOverlapDetector` (user `mode_function` callable), `GaussianModeOverlapDetector` (analytic Gaussian). All frequency-indexed (one reference mode per `wave_characters` entry).

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
- `scaling_mode="continuous"` (default, scale `2/Σw`) reconstructs CW amplitude; `"pulse"` (scale 1) is the raw windowed DFT — use `"pulse"` for `flux_spectrum`/`transmission`.

**DiffractiveDetector** — key: `"diffractive"`, dtype: complex
- Time dim is always 1
- Shape: `(1, num_frequencies, num_orders)`

**Mode-overlap detector family** (`src/fdtdx/objects/detectors/mode.py`) — all inherit `BaseModeOverlapDetector(PhasorDetector, ABC)`, always use all 6 components. The base owns the overlap integral, the stored reference modes (`_mode_E`/`_mode_H` stacked as `(num_freqs, 3, *spatial)`), the face-area weights, and the `apply()` skeleton (slice ε/μ → per-freq dispersive `effective_inv_permittivity` correction → call the abstract `_compute_mode_fields` hook → stack). **Multi-frequency (#362):** one reference mode per `wave_characters` entry.
- `compute_overlap(state)` → complex `(num_freqs,)`. Bare scalar pre-#362, so single-frequency → shape `(1,)` (index `[0]`).
- `compute_overlap_to_mode(state, mode_E, mode_H, wave_character_index=0)` → scalar per frequency.
- Subclasses implement `_compute_mode_fields`: **`ModeOverlapDetector`** calls the tidy3d mode solver (`direction`/`mode_index`/`filter_pol`/`bend_*`/`symmetry`). **`CustomModeOverlapDetector`** evaluates a user `mode_function(coordinates, frequency, propagation_axis, inv_permittivity) -> (E, H)` callable on the detector plane (`normalize=True` re-Poynting-normalizes). **`GaussianModeOverlapDetector`** builds an analytic Gaussian via `gaussian_mode_fields` with the *same* polarization/angle knobs as the source (`fixed_E/H_polarization_vector`, `azimuth_angle`/`elevation_angle`, `polarization_axis`, `mode_radius`, `center`); tilt adds an `exp(ik·r)` transverse phase ramp (normal incidence stays real). `gaussian_mode_function(...)` returns a `mode_function` for the custom detector.
- The slice handed to `_compute_mode_fields` is already the dispersion-corrected `effective_inv_permittivity` at each carrier frequency, so Custom/Gaussian detectors are dispersion-aware automatically (their `n`/wavenumber use ε(ω_c), not ε∞).

**S-parameters** (`fdtdx.calculate_sparam` / `calculate_sparams`, `src/fdtdx/utils/sparams.py`) are **frequency-indexed (#362):** the returned `{(detector_name, input_port_name): amplitude}` maps to a complex array indexed by frequency — shape `(1,)` for the single-frequency detectors `setup_sparams_simulation` builds. The isinstance gate is `BaseModeOverlapDetector`, so custom/Gaussian detectors work in S-param calculations too.

## Temporal Apodization

`PhasorDetector` (and the whole mode-overlap family) accept `apodization: TemporalProfile | None` — a smooth window applied per-step before the running DFT, replacing the leaky hard-rectangular `OnOffSwitch` gate alone. Reuse the temporal-profile abstraction: pass a carrier-free `TukeyWindowProfile(start_time, end_time, alpha)` (Hann at `alpha=1`), `GaussianWindowProfile(center_time, sigma_time)`, or any `CustomTimeSignalProfile`/`TemporalProfile`.

- Precomputed in `PhasorDetector.place_on_grid`: `_window_at_time_step_arr = on_mask · w[n]`, `_window_sum = Σ w`. `update` multiplies the per-step contribution by `w[time_step]` and the continuous-mode scale becomes `2/_window_sum` (coherent-gain correction → CW amplitude stays ~1).
- **`apodization=None` is bit-identical to before** (`w≡1` on on-steps, `_window_sum=num_time_steps_on=N`).
- For `transmission` to cancel correctly the source signal must be windowed with the **same** apodization — `transmission` forwards `out_detector.apodization` to `injected_power_spectrum`.

## Spectral Power & Transmission

`src/fdtdx/utils/spectra.py` — Per-frequency results under distinct names. All share the raw windowed-DFT convention (`flux_spectrum` un-scales the detector's `static_scale`), so ratios are unit-consistent.
- `flux_spectrum(detector, arrays)` → real `(num_freqs,)`: net Poynting flux `½ Re ∮ (E×H*)·n̂ dA` from a **plane** `PhasorDetector` recording all 6 components (`reduce_volume=False`).
- `injected_power_spectrum(source, frequencies, *, apodization=None)` → wraps `source.injected_power_spectrum` (analytic; validated to match a co-located source-plane `flux_spectrum` within ~3% in a homogeneous medium).
- `transmission(out_detector, arrays, source, *, frequencies=None)` = `flux_spectrum / injected_power_spectrum` — the transmitted power fraction. The pulse spectrum and matching apodization window cancel, so it is pulse-shape-independent.
- `radiated_power_spectrum(face_detectors, arrays)` → sums signed `flux_spectrum` over `(detector, outward_sign)` box faces: the **measured** actual/total radiated power (any source/count, Purcell-aware) — the dipole power tool and the validation oracle for the analytic path.

**Use a pulse, not CW, for transmission:** a ramped CW under-integrates the steady state at far planes over finite time, so `transmission` drifts below 1. A `GaussianPulseProfile` fully transits every plane, making the recorded spectrum (hence the transmitted fraction) position-independent.

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
- **Complex fields**: Bloch boundaries with nonzero k-vector automatically require complex fields. Check `config.use_complex_fields`. When complex fields are in effect, ADE polarization arrays (`dispersive_P_curr/prev`) are also allocated as complex.
- **Conductivity scaling**: Conductivity values are multiplied by the grid cell spacing (`constants.c * config.time_step_duration / config.courant_number`) during `_init_arrays()`. Don't pre-scale. There is no `config.resolution` scalar anymore — derive distances from the grid.
- **Inverse storage**: Material arrays store `1/epsilon` and `1/mu`, not epsilon and mu directly. For dispersive materials, `Material.permittivity` represents ε∞ only — the full ε(ω) must be reconstructed via the dispersion model.
- **Detector timing**: Detectors only record at timesteps where their `OnOffSwitch` is active. Check `switch` configuration if data appears missing.
- **donate_argnames**: When JIT-compiling simulation functions, use `donate_argnames=["arrays"]` to allow JAX to reuse array memory.
- **Dispersive + full anisotropic**: Not supported — `_init_arrays` raises `NotImplementedError`. Use diagonal anisotropy if you need directional ε alongside dispersion.
- **Dispersive pole count is max'd globally**: The `num_poles` leading axis size = `objects.max_num_dispersive_poles`. Adding one 3-pole material allocates 3 pole slots for every dispersive cell in the sim; non-dispersive cells still have their `c1/c2/c3` set to zero (ADE term vanishes) but consume array memory.
- **Dispersive source impedance**: Inside a dispersive medium, never use ε∞ as the source's effective permittivity — call `effective_inv_permittivity` at ω_c. Broadband pulses additionally need the `_temporal_H_filter` path to avoid TFSF leakage at off-carrier frequencies.
- **Stacking objects with mixed dispersion**: `UniformMaterialObject` always writes a full zero-padded pole-coefficient stack into its `grid_slice`, so placing a non-dispersive object over a dispersive one cleanly overwrites stale coefficients. Rely on this rather than assuming "no dispersion = leave coefficients alone".
- **Symmetry results look wrong / are half-size**: with `config.symmetry` set, `run_fdtd` returns *reduced-domain* arrays — you must call `fdtdx.unfold_detector_states` / `fdtdx.unfold_fields` to get full-domain results (see Simulation Symmetry). `place_objects` warns about this. Unfolding a non-symmetric model raises.
- **Symmetry needs even cells + symmetric model**: each symmetric axis must resolve to an even cell count (`place_objects` raises otherwise), and the user's full model must actually be mirror-symmetric about the center plane — asymmetric objects are only warned about. Use PML (not periodic) on the far side of a symmetric axis.
- **Coordinate origin is the domain center** (#363): real positions / real-coordinate constraints run from `-L/2` to `+L/2` with 0 at the center, not the lower-left corner. Shift any corner-relative coordinates by `-L/2`. GDS `gds_center` maps a GDS coordinate to the domain center, not the corner.
- **QuasiUniformGrid needs even cell counts**: every axis must resolve to an even number of cells or `QuasiUniformGrid.resolve` raises (center lands on a cell edge). `UniformGrid` does *not* enforce this, so migrating an odd-shaped uniform sim to `QuasiUniformGrid` can surprise you with a `ValueError`.
- **Mode-overlap / S-param results are frequency-indexed** (#362): `ModeOverlapDetector.compute_overlap` and `calculate_sparam` return arrays indexed by frequency (shape `(1,)` for single-frequency setups), not bare scalars. Index `[0]` if you want the scalar.
- **`transmission` needs a pulse + a matching window**: with a CW source `transmission` drifts below 1 at far planes (finite-time steady-state under-integration) — use a `GaussianPulseProfile`. The output detector and the source must use the **same** `apodization` (forwarded automatically by `transmission`) or the window won't cancel. `flux_spectrum` needs a plane `PhasorDetector` with all 6 components and `reduce_volume=False`.
- **Dipole power is measured, not analytic**: `PointDipoleSource.injected_power_spectrum` raises — radiated power is Purcell/environment-dependent. Enclose the dipole in a phasor-detector box and use `radiated_power_spectrum` (homogeneous background → nominal, real structure → actual power).
- **Apodization reuses `TemporalProfile`, isn't a new class**: a detector's `apodization` is a (carrier-free) `TemporalProfile` — `TukeyWindowProfile`/`GaussianWindowProfile`. Window shapes live once in `core/physics/temporal_windows.py`; don't re-implement Hann/Gaussian. `apodization=None` stays bit-identical to the old hard gate.
