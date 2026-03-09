# Test Development

## Setup
- Use `.venv` directly, not `uv` (uv reverts JAX fixes)
- Run unit tests: `.venv/bin/python -m pytest tests/unit -m unit`
- Run integration tests: `.venv/bin/python -m pytest tests/integration -m integration`
- Run simulation tests: `.venv/bin/python -m pytest tests/simulation -m simulation`
- Coverage report: `.venv/bin/python -m pytest tests/unit --cov=fdtdx --cov-report=html`
- Coverage must run on the full unit suite (not individual files) to avoid JAX crash

## Test Locations
- `tests/unit/` — unit tests, `@pytest.mark.unit`, mirrors `src/fdtdx/` structure
- `tests/integration/` — tests requiring `place_objects`/`apply_params`, `@pytest.mark.integration`
- `tests/simulation/` — full FDTD runs, `@pytest.mark.simulation` (auto-applied via conftest.py)

## Simulation Tests Status

| Test File | Tests | Status |
|-----------|-------|--------|
| physics/test_plane_wave.py | phase velocity + wave impedance in vacuum + dielectric | ✓ passing (4 tests) |
| physics/test_fresnel.py | Fresnel T and R+T=1 (ε_r=4, normal incidence) | ✓ passing |
| physics/test_skin_depth.py | attenuation α in lossy conductor (σ=1e4 S/m) | ✓ passing (1 test) |
| physics/test_birefringence.py | k and Z for ordinary + extraordinary axes | ✓ passing (4 tests) |

Unit and integration test phases are complete.

## Guidelines
- Avoid redundant tests - maintain coverage without duplication
- Use mocks when needed (e.g., JAX backend detection in config.py, ArrayContainer in vti.py)
- Reuse existing test code and real classes (e.g., WaveCharacter) when possible
- Use `git mv` when moving test files to preserve history
- When existing tests in `tests/` contain both unit and integration tests, split them:
  move unit tests to `tests/unit/`, integration tests to `tests/integration/`

---

# Physics Simulation Tests

## Goal
Small, fast FDTD simulations that compare results to analytic solutions. Catch regressions
when adding dispersion, complex fields, conductivity, anisotropy, etc.

## Common 1D-like Setup
All tests use this base configuration:
- **Propagation**: +z direction
- **Domain**: `4e-6 × 150e-9 × 150e-9` m (z × x × y) — 3 cells transverse
- **Resolution**: `100e-9` m (10 cells/λ), giving ~2–5% numerical dispersion
- **PML**: uniform 10 cells all sides (`BoundaryConfig.from_uniform_bound(thickness=10)`)
- **Source**: `UniformPlaneSource`, direction=`"+"`, `SingleFrequencyProfile`, λ=1 μm
- **Time**: ~100 fs (≈30 periods; 4-period startup ramp built into `SingleFrequencyProfile`)
- **Tolerances**: 5% for lossless, 10% for lossy (consistent with 10 cells/λ dispersion)

Phasor detector data access:
```python
# state["phasor"] shape: (1, num_freqs, num_components, *grid_or_scalar)
# components: Ex=0, Ey=1, Ez=2, Hx=3, Hy=4, Hz=5
phasor = arrays.detector_states["d1"]["phasor"][0, 0]  # (num_components,)
amplitude = float(jnp.abs(phasor[0]))   # |Ex|
phase     = float(jnp.angle(phasor[0])) # angle(Ex)
```

## File Structure
```
tests/simulation/physics/
├── __init__.py
├── conftest.py          # shared build_1d_sim_base() helper
├── test_plane_wave.py   # phase velocity, impedance
├── test_fresnel.py      # Fresnel reflection/transmission
├── test_skin_depth.py   # skin depth in conductor
└── test_birefringence.py # birefringence in anisotropic material
```

The `conftest.py` helper `build_1d_sim_base()` returns `(objects, constraints, config)` with
the volume, PML, and source already added so each test only appends detectors and materials.

## Planned Tests

### test_plane_wave.py

**Test 1 — Phase velocity in vacuum**
- Two `PhasorDetector`s (reduce_volume=True) separated by `d = 0.5 μm`
- Analytic: `Δφ = 2π·d/λ = π`; tolerance ±0.16 rad (5%)

**Test 2 — Phase velocity in dielectric (ε_r=4, n=2)**
- `UniformMaterialObject` filling domain, same two detectors, `d = 0.25 μm`
- Analytic: `Δφ = 2π·n·d/λ = π`; or compare `k_measured = Δφ/d` to `k = 2πn/λ`

**Test 3 — Wave impedance (E/H ratio)**
- Single `PhasorDetector` measuring Ex (idx 0) and Hy (idx 4)
- Vacuum: `|Ex|/|Hy| = Z₀ ≈ 377 Ω`; dielectric (ε_r=4): `Z₀/n = 188.5 Ω`

### test_fresnel.py

**Test 4 — Normal incidence transmission (two-run normalization)**
- Run A (reference): uniform vacuum → `PoyntingFluxDetector` at z=3.0 μm → `S₀`
- Run B (interface): `UniformMaterialObject(ε_r=4)` filling right half → same detector → `S_T`
- `T_measured = S_T / S₀`; analytic: `T = 4n₁n₂/(n₁+n₂)² = 8/9 ≈ 0.889`

**Test 5 — Power conservation (R + T = 1)**
- From runs A/B above, also place a `PoyntingFluxDetector` on the vacuum side
- Analytic: `R = ((n₁-n₂)/(n₁+n₂))² = 1/9 ≈ 0.111`, verify `T + R ≈ 1`

### test_skin_depth.py

**Test 6 — Exponential field decay in a conductor**

Choose `σ = 1e4 S/m`, `λ = 1 μm`:
```
γ = σ/(ωε₀) = 1e4 / (1.885e15 × 8.85e-12) ≈ 0.60
α = (2π/λ) · Im(√(1 + i·γ)) = (2π/1μm) · 0.288 = 1.81 μm⁻¹
δ_analytic = 1/α ≈ 0.552 μm
```
- Left half: vacuum; right half: `Material(electric_conductivity=1e4)`
- Three `PhasorDetector`s inside conductor at z = 2.2, 2.75, 3.3 μm (≈1δ apart)
- Fit `A₂/A₁ = exp(-Δz/δ)` → `δ_measured`; compare to `δ_analytic = 0.552 μm`

Note: use exact general formula `δ = 1/[k₀·Im(√(ε_r + iσ/(ωε₀)))]`, not good-conductor
approximation `√(2/(ωμ₀σ))` (only valid when σ >> ωε₀).

### test_birefringence.py

Anisotropic slab: `ε = (2.0, 4.0, 1.0)` → `n_x = √2 ≈ 1.414`, `n_y = 2.0`

**Test 7a — x-polarized wave** (`fixed_E_polarization_vector=(1,0,0)`) through L=1.5 μm slab
- PhasorDetectors before and after; measure Ex phase difference
- Analytic: `Δφ_x = 2π·n_x·L/λ = 2π·√2·1.5 ≈ 13.33 rad`

**Test 7b — y-polarized wave** (`fixed_E_polarization_vector=(0,1,0)`) same slab
- Analytic: `Δφ_y = 2π·n_y·L/λ = 2π·2·1.5 = 6π ≈ 18.85 rad`

**Test 8 — Birefringent retardation**
- `Δφ_y - Δφ_x = 2π·(n_y - n_x)·L/λ = 2π·(2 - √2)·1.5 ≈ 5.51 rad`

## Future Tests (after new features land)
| Feature | Test |
|---------|------|
| Drude dispersion | Phase velocity vs. frequency matches Drude model |
| Lorentz oscillator | Resonance frequency in permittivity spectrum |
| Complex-valued simulation (complex64) | Fresnel result matches float32 within ~1e-5 |
| Magnetic materials (μ≠1) | Impedance `Z = Z₀√(μ/ε)` and phase velocity `v = c/√(με)` |
