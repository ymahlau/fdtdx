"""Physics simulation test: PEC total reflection.

A plane wave (z-propagating, x-polarized) hits a PEC wall at z-max.
PML at z-min absorbs the source, PEC at z-max reflects all energy.
Periodic boundaries in x and y.

Domain layout (25 nm resolution = 40 cells/λ, z is propagation axis):
  160 cells total in z (4 µm):
    cells   0-  9 : PML (0.25 µm)
    cells  10-159 : active region, PEC terminates at z-max

  Source      : z-index 12 (2 cells into active region)
  Flux detector: z-index 80 (midway in active region)

Two independent checks:

  1. Net-flux magnitude  |R| ≈ 1:
     In steady state, a PEC wall produces a perfect standing wave, whose
     time-averaged Poynting flux is zero.  Two-run normalization
     (reference = PML everywhere) gives T = S_standing / S_reference ≈ 0,
     hence R = 1 - T ≈ 1.0.  This alone CANNOT distinguish PEC from PMC
     (both reflect totally).

  2. Standing-wave phase discriminator (PEC ≠ PMC):
     A PEC wall forces E_tangential = 0 at the wall, so Ex ∝ sin(k·d)
     with d = distance from wall — an E-field NODE at the wall, antinode
     at λ/4.  Two PhasorDetectors (Ex) — "near" 1 cell from the wall and
     "far" a quarter-wavelength further — must therefore satisfy (with the
     Yee half-cell offset that puts the near Ex sample at d = Δ/2)
       |Ex|_near / |Ex|_far = |sin(kΔ/2)| / |sin(kΔ/2 + π/2)| = |tan(kΔ/2)| ≪ 1.
     (PMC instead gives the reciprocal ≫ 1; a PEC↔PMC swap fails.)
     The resolution is raised to 40 cells/λ so kΔ is small and the
     analytic ratio (≈ 0.079) is a sharp node signature.

Tolerance: 5 % relative error on R; standing-wave ratio bounded within
±25 % of the analytic |tan(kΔ/2)|.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6
# 40 cells/λ (raised from 20): keeps kΔ small so the near-wall standing-wave
# node (|tan(kΔ)|) is a sharp PEC↔PMC discriminator, not just |R|≈1.
_RESOLUTION = 25e-9
_PML_CELLS = 10
_DOMAIN_XY = 3 * _RESOLUTION  # 3 cells, periodic
_DOMAIN_Z = 4e-6  # 160 cells

_Z_CELLS = round(_DOMAIN_Z / _RESOLUTION)  # = 160 (z-max wall index = _Z_CELLS - 1)
_SOURCE_Z = _PML_CELLS + 2  # = 12
_DET_Z = 80  # flux detector in middle of active region

# Standing-wave phase discriminator detectors (relative to the z-max wall):
#   near: 1 cell from wall (d = Δ);  far: a quarter-wave further (d = Δ + λ/4)
_QWAVE_CELLS = round(_WAVELENGTH / 4 / _RESOLUTION)  # = 10 at 40 cells/λ
_DET_NEAR_Z = _Z_CELLS - 2  # 1 cell from z-max wall
_DET_FAR_Z = _Z_CELLS - 2 - _QWAVE_CELLS  # quarter-wave further from the wall

_K0 = 2 * np.pi / _WAVELENGTH
_KD = _K0 * _RESOLUTION  # per-cell phase, kΔ
# PEC: Ex ∝ sin(k·d), d = distance from the E-tangential (wall) plane.
# On the Yee grid the tangential-E plane (where Ex is pinned to 0) sits half a
# cell from the Ex sample point, so the "near" detector (1 cell from the wall)
# samples at d = Δ/2 and the "far" detector at d = Δ/2 + λ/4. Hence
#   near/far = |sin(kΔ/2)| / |sin(kΔ/2 + π/2)| = |tan(kΔ/2)| ≈ 0.079.
# (This half-cell offset is confirmed empirically to 4 sig-figs.)
_PEC_RATIO_ANALYTIC = abs(np.tan(_KD / 2))

_SIM_TIME = 120e-15  # 120 fs ≈ 36 optical periods
_TOLERANCE = 0.05

# Time-averaging
_DT_APPROX = 0.99 * _RESOLUTION / (3e8 * np.sqrt(3))
_STEPS_PER_PERIOD = round(_WAVELENGTH / (3e8 * _DT_APPROX))
_N_AVG_STEPS = 10 * _STEPS_PER_PERIOD


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_base(z_max_type="pml"):
    """Build domain with periodic xy, PML z-min, and configurable z-max."""
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


def test_pec_total_reflection():
    """PEC wall: |R| ≈ 1 (within 5 %) AND E-field NODE at the wall.

    Two checks:
      1. Two-run net-flux normalization (R = 1 - S_pec/S_ref ≈ 1.0).
      2. Standing-wave phase discriminator: near/far |Ex| ratio ≈ |tan(kΔ)|
         (E node at wall), so near ≪ far — the OPPOSITE of PMC. A
         PEC↔PMC swap therefore fails this test.
    """
    # Reference: PML on all z-faces
    obj_ref, con_ref, cfg_ref, vol_ref, _ = _build_base(z_max_type="pml")
    _add_flux_det("flux", _DET_Z, vol_ref, obj_ref, con_ref)
    S_ref = _mean_flux(_run(obj_ref, con_ref, cfg_ref), "flux")

    # PEC run (with standing-wave probes near and λ/4 from the wall)
    obj_pec, con_pec, cfg_pec, vol_pec, wave_pec = _build_base(z_max_type="pec")
    _add_flux_det("flux", _DET_Z, vol_pec, obj_pec, con_pec)
    _add_ex_phasor("near", _DET_NEAR_Z, wave_pec, vol_pec, obj_pec, con_pec)
    _add_ex_phasor("far", _DET_FAR_Z, wave_pec, vol_pec, obj_pec, con_pec)
    arrays_pec = _run(obj_pec, con_pec, cfg_pec)
    S_pec = _mean_flux(arrays_pec, "flux")

    assert S_ref > 0, f"Reference flux is zero/negative: {S_ref}"

    # --- Check 1: net-flux magnitude |R| ≈ 1 ---
    T_measured = S_pec / S_ref
    R_measured = 1.0 - T_measured
    R_analytic = 1.0

    rel_err = abs(R_measured - R_analytic) / R_analytic
    assert rel_err < _TOLERANCE, (
        f"PEC reflection: R_measured={R_measured:.4f}, R_analytic={R_analytic:.4f}, "
        f"T_measured={T_measured:.4f}, relative error={rel_err:.3f}"
    )

    # --- Check 2: standing-wave phase — E NODE at the PEC wall ---
    amp_near = _ex_amplitude(arrays_pec, "near")
    amp_far = _ex_amplitude(arrays_pec, "far")
    assert amp_far > 0, "PEC far detector measured zero Ex"
    ratio = amp_near / amp_far
    # Analytic node signature |tan(kΔ/2)| ≈ 0.079 (matches the measured value
    # to ~4 sig-figs); bounded tightly to ±25 %. PMC gives the reciprocal
    # (≈ 12.7), so a PEC↔PMC swap fails this bound by ~160x.
    assert 0.75 * _PEC_RATIO_ANALYTIC < ratio < 1.25 * _PEC_RATIO_ANALYTIC, (
        f"PEC standing wave: near/far |Ex| ratio={ratio:.4f}, expected ≈ "
        f"|tan(kΔ/2)|={_PEC_RATIO_ANALYTIC:.4f} (E node at wall). "
        f"|Ex|_near={amp_near:.4e}, |Ex|_far={amp_far:.4e}"
    )
