"""Physics simulation test: PMC total reflection.

A plane wave (z-propagating, x-polarized) hits a PMC wall at z-max.
PML at z-min absorbs the source, PMC at z-max reflects all energy.
Periodic boundaries in x and y.

For PMC, H_tangential = 0 at the wall.  For z-propagation, H_x and H_y
are tangential to the z-face.  The dominant H component (H_y for
x-polarized E) is zeroed at the wall.

Two independent checks:

  1. Net-flux magnitude |R| ≈ 1:
     Like PEC, PMC produces total reflection and a standing wave with
     zero time-averaged Poynting flux.  This alone CANNOT distinguish PMC
     from PEC (both reflect totally).

  2. Standing-wave phase discriminator (PMC ≠ PEC):
     The standing-wave phase differs:
       PEC → E-field NODE at wall      (E_tangential = 0, Ex ∝ sin(k·d))
       PMC → E-field ANTINODE at wall  (H_tangential = 0, Ex ∝ cos(k·d))
     Two PhasorDetectors (Ex) — "near" 1 cell from the wall and "far" a
     quarter-wavelength further — must satisfy, for PMC (whose E-antinode
     sits on the integer Ex grid plane, a full cell from the near sample),
       |Ex|_near / |Ex|_far = |cos(kΔ)| / |cos(kΔ+π/2)| = 1/|tan(kΔ)| ≫ 1
     (E antinode at wall), the inverse of the PEC node pattern.  A PEC↔PMC
     swap fails.  The resolution is raised to 40 cells/λ so the analytic
     ratio (≈ 6.3) is a sharp antinode signature.

Tolerance: 5 % relative error on R; standing-wave ratio bounded within
±25 % of the analytic 1/|tan(kΔ)|.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6
# 40 cells/λ (raised from 20): keeps kΔ small so the near-wall standing-wave
# antinode (1/|tan(kΔ)|) is a sharp PEC↔PMC discriminator, not just |R|≈1.
_RESOLUTION = 25e-9
_PML_CELLS = 10
_DOMAIN_XY = 3 * _RESOLUTION
_DOMAIN_Z = 4e-6

_Z_CELLS = round(_DOMAIN_Z / _RESOLUTION)  # = 160 (z-max wall index = _Z_CELLS - 1)
_SOURCE_Z = _PML_CELLS + 2
_DET_Z = 80

# Standing-wave phase discriminator detectors (relative to the z-max wall):
#   near: 1 cell from wall (d = Δ);  far: a quarter-wave further (d = Δ + λ/4)
_QWAVE_CELLS = round(_WAVELENGTH / 4 / _RESOLUTION)  # = 10 at 40 cells/λ
_DET_NEAR_Z = _Z_CELLS - 2  # 1 cell from z-max wall
_DET_FAR_Z = _Z_CELLS - 2 - _QWAVE_CELLS  # quarter-wave further from the wall

_K0 = 2 * np.pi / _WAVELENGTH
_KD = _K0 * _RESOLUTION  # per-cell phase, kΔ
# PMC: Ex ∝ cos(k·d), d = distance from the wall (E antinode at the wall).
# Unlike PEC (whose E-node sits on the half-integer Yee plane, half a cell from
# the Ex sample), the PMC E-antinode falls on the integer Ex grid plane, so the
# "near" Ex sample sits a FULL cell from the antinode: d = Δ, far = Δ + λ/4, and
#   near/far = |cos(kΔ)| / |cos(kΔ + π/2)| = 1/|tan(kΔ)| ≈ 6.3.
# (This full-cell offset is confirmed empirically to ~1 %.)
_PMC_RATIO_ANALYTIC = 1.0 / abs(np.tan(_KD))

_SIM_TIME = 120e-15
_TOLERANCE = 0.05

_DT_APPROX = 0.99 * _RESOLUTION / (3e8 * np.sqrt(3))
_STEPS_PER_PERIOD = round(_WAVELENGTH / (3e8 * _DT_APPROX))
_N_AVG_STEPS = 10 * _STEPS_PER_PERIOD


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_base(z_max_type="pml"):
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        time=_SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(_DOMAIN_XY, _DOMAIN_XY, _DOMAIN_Z),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            "min_x": "periodic",
            "max_x": "periodic",
            "min_y": "periodic",
            "max_y": "periodic",
            "max_z": z_max_type,
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_SOURCE_Z,)),
        ]
    )
    objects.append(source)

    return objects, constraints, config, volume, wave


def _add_flux_det(name, z_idx, volume, objects, constraints):
    det = fdtdx.PoyntingFluxDetector(
        name=name,
        partial_grid_shape=(None, None, 1),
        direction="+",
        reduce_volume=True,
        plot=False,
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(0, 1)),
            det.place_at_center(volume, axes=(0, 1)),
            det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(z_idx,)),
        ]
    )
    objects.append(det)


def _add_ex_phasor(name, z_idx, wave, volume, objects, constraints):
    """PhasorDetector recording Ex at one z-plane (standing-wave probe)."""
    det = fdtdx.PhasorDetector(
        name=name,
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave,),
        reduce_volume=True,
        components=("Ex",),
        plot=False,
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(0, 1)),
            det.place_at_center(volume, axes=(0, 1)),
            det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(z_idx,)),
        ]
    )
    objects.append(det)


def _run(objects, constraints, config):
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=config, key=key)
    return arrays


def _mean_flux(arrays, name):
    flux = np.array(arrays.detector_states[name]["poynting_flux"][:, 0])
    return float(np.mean(flux[-_N_AVG_STEPS:]))


def _ex_amplitude(arrays, name) -> float:
    return float(np.abs(arrays.detector_states[name]["phasor"][0, 0, 0]))


# ── Tests ────────────────────────────────────────────────────────────────────


def test_pmc_total_reflection():
    """PMC wall: |R| ≈ 1 (within 5 %) AND E-field ANTINODE at the wall.

    Two checks:
      1. Two-run net-flux normalization (R = 1 - S_pmc/S_ref ≈ 1.0).
      2. Standing-wave phase discriminator: near/far |Ex| ratio ≈ 1/|tan(kΔ)|
         (E antinode at wall), so near ≫ far — the OPPOSITE of PEC. A
         PEC↔PMC swap therefore fails this test.
    """
    # Reference: PML on all z-faces
    obj_ref, con_ref, cfg_ref, vol_ref, _ = _build_base(z_max_type="pml")
    _add_flux_det("flux", _DET_Z, vol_ref, obj_ref, con_ref)
    S_ref = _mean_flux(_run(obj_ref, con_ref, cfg_ref), "flux")

    # PMC run (with standing-wave probes near and λ/4 from the wall)
    obj_pmc, con_pmc, cfg_pmc, vol_pmc, wave_pmc = _build_base(z_max_type="pmc")
    _add_flux_det("flux", _DET_Z, vol_pmc, obj_pmc, con_pmc)
    _add_ex_phasor("near", _DET_NEAR_Z, wave_pmc, vol_pmc, obj_pmc, con_pmc)
    _add_ex_phasor("far", _DET_FAR_Z, wave_pmc, vol_pmc, obj_pmc, con_pmc)
    arrays_pmc = _run(obj_pmc, con_pmc, cfg_pmc)
    S_pmc = _mean_flux(arrays_pmc, "flux")

    assert S_ref > 0, f"Reference flux is zero/negative: {S_ref}"

    # --- Check 1: net-flux magnitude |R| ≈ 1 ---
    T_measured = S_pmc / S_ref
    R_measured = 1.0 - T_measured
    R_analytic = 1.0

    rel_err = abs(R_measured - R_analytic) / R_analytic
    assert rel_err < _TOLERANCE, (
        f"PMC reflection: R_measured={R_measured:.4f}, R_analytic={R_analytic:.4f}, "
        f"T_measured={T_measured:.4f}, relative error={rel_err:.3f}"
    )

    # --- Check 2: standing-wave phase — E ANTINODE at the PMC wall ---
    amp_near = _ex_amplitude(arrays_pmc, "near")
    amp_far = _ex_amplitude(arrays_pmc, "far")
    assert amp_far > 0, "PMC far detector measured zero Ex"
    ratio = amp_near / amp_far
    # Analytic antinode signature 1/|tan(kΔ)| ≈ 6.3 (matches measurement to
    # ~1 %); bounded tightly to ±25 %. PEC gives a node ratio ≈ 0.079, so a
    # PEC↔PMC swap fails this lower bound by ~60x.
    assert 0.75 * _PMC_RATIO_ANALYTIC < ratio < 1.25 * _PMC_RATIO_ANALYTIC, (
        f"PMC standing wave: near/far |Ex| ratio={ratio:.4f}, expected ≈ "
        f"1/|tan(kΔ)|={_PMC_RATIO_ANALYTIC:.4f} (E antinode at wall). "
        f"|Ex|_near={amp_near:.4e}, |Ex|_far={amp_far:.4e}"
    )
