"""Physics simulation test: quarter-domain with PEC + PMC symmetry boundaries.

Validates that PEC and PMC boundaries correctly enforce field conditions
when used together as symmetry planes for a quarter-domain simulation.

Physics:
  An Ey-polarized plane wave propagating in +x is compatible with both:
  - PEC at min_y (zeros tangential Ex, Ez — both zero for this polarization)
  - PMC at min_z (zeros tangential Hx, Hy — both zero for this polarization)

  The normal components (Ey at PEC y-face, Hz at PMC z-face) are free and
  carry the wave energy.  The wave should propagate undisturbed through the
  quarter domain with the same phase velocity as in free space.

  This test catches the specific bug where PEC/PMC boundaries fail to
  explicitly enforce tangential field zeroing after the E/H update,
  causing field corruption and mode degradation.

Test strategy:
  1. test_quarter_domain_propagation:
     Two phasor detectors measure phase difference → k_measured ≈ k₀ within 1%.
     This validates propagation / numerical dispersion only.  Because the wave
     is transversely uniform, the measured k is essentially insensitive to the
     PEC/PMC transverse walls, so this test does NOT by itself detect boundary
     corruption — that is the job of tests 2 and 3.

  2. test_quarter_domain_vs_full_domain:
     Compare the Ey phasor field of a quarter domain (PEC min_y, PMC min_z),
     unfolded by mirroring, to a full domain (periodic y, periodic z).  The two
     use the same discretization, so the reconstruction is near-exact: the MAX
     relative error over the slice must stay ≤ 2e-3, confirming the boundaries
     act as faithful mirrors.

  3. test_quarter_domain_field_enforcement:
     Verify tangential E = 0 at PEC and tangential H = 0 at PMC by measuring
     field components ON each boundary plane.  These tangential components are
     ~zero by source polarization, so the realistic residual is roundoff.

Domain: PEC min_y, PMC min_z, PML elsewhere.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6
_RESOLUTION = 25e-9  # 40 cells/λ for accurate boundary measurement
_PML_CELLS = 10
_DOMAIN_YZ = 1e-6  # transverse extent (quarter domain)
_FULL_DOMAIN_YZ = 2 * _DOMAIN_YZ  # transverse extent (full domain, doubled in Y and Z)
_DOMAIN_X = 5e-6  # propagation direction

_SOURCE_X = _PML_CELLS + 2
_DET1_X = _SOURCE_X + 20  # well past source
_DET2_X = _DET1_X + 10  # 10 cells further

_SIM_TIME = 150e-15

# Time-averaging constants
_DT_APPROX = 0.99 * _RESOLUTION / (3e8 * np.sqrt(3))
_STEPS_PER_PERIOD = round(_WAVELENGTH / (3e8 * _DT_APPROX))


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_quarter_domain():
    """Build quarter-domain sim with PEC at min_y and PMC at min_z."""
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        time=_SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(_DOMAIN_X, _DOMAIN_YZ, _DOMAIN_YZ),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            "min_y": "pec",
            "max_y": "periodic",
            "min_z": "pmc",
            "max_z": "periodic",
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    # Ey-polarized plane wave propagating in +x
    # Compatible with PEC at min_y (Ey is normal to y-face) and
    # PMC at min_z (Hz is normal to z-face).
    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(1, None, None),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(0, 1, 0),
        normalize_by_energy=False,
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(1, 2)),
            source.place_at_center(volume, axes=(1, 2)),
            source.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_SOURCE_X,)),
        ]
    )
    objects.append(source)

    return objects, constraints, config, volume, wave


def _build_full_domain():
    """Build full-domain reference sim with periodic y and z.

    Domain is twice as large in Y and Z as the quarter domain, representing
    the true full domain that the quarter domain (with PEC/PMC mirrors) models.
    """
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        time=_SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(_DOMAIN_X, _FULL_DOMAIN_YZ, _FULL_DOMAIN_YZ),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            "min_y": "periodic",
            "max_y": "periodic",
            "min_z": "periodic",
            "max_z": "periodic",
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(1, None, None),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(0, 1, 0),
        normalize_by_energy=False,
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(1, 2)),
            source.place_at_center(volume, axes=(1, 2)),
            source.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_SOURCE_X,)),
        ]
    )
    objects.append(source)

    return objects, constraints, config, volume, wave


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


# ── Tests ────────────────────────────────────────────────────────────────────


def test_quarter_domain_propagation():
    """Ey plane wave propagates through the quarter domain with correct phase velocity.

    Measures k from the phase difference between two on-axis phasor detectors and
    checks it against k₀ = 2π/λ within 1% (the residual is numerical dispersion at
    40 cells/λ, ≈ 0.08%).  Because the plane wave is transversely uniform, this k is
    essentially insensitive to the PEC/PMC transverse walls, so this test mainly
    validates propagation — it does NOT by itself detect boundary corruption
    (see test_quarter_domain_vs_full_domain and test_quarter_domain_field_enforcement).
    """
    objects, constraints, config, volume, wave = _build_quarter_domain()

    # Two phasor detectors for phase measurement
    for name, x_coord in [("ph_near", _DET1_X), ("ph_far", _DET2_X)]:
        det = fdtdx.PhasorDetector(
            name=name,
            partial_grid_shape=(1, None, None),
            wave_characters=(wave,),
            components=("Ey",),
            reduce_volume=True,
            plot=False,
        )
        constraints.extend(
            [
                det.same_size(volume, axes=(1, 2)),
                det.place_at_center(volume, axes=(1, 2)),
                det.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(x_coord,)),
            ]
        )
        objects.append(det)

    arrays = _run(objects, constraints, config)

    p_near = complex(arrays.detector_states["ph_near"]["phasor"][0, 0, 0])
    p_far = complex(arrays.detector_states["ph_far"]["phasor"][0, 0, 0])

    assert abs(p_near) > 1e-30, "Near phasor is zero — wave not launched"
    assert abs(p_far) > 1e-30, "Far phasor is zero — wave not propagating"

    # Measure phase velocity
    delta_phi = (np.angle(p_far) - np.angle(p_near)) % (2 * np.pi)
    if delta_phi > np.pi:
        delta_phi -= 2 * np.pi
    separation = (_DET2_X - _DET1_X) * _RESOLUTION
    k_measured = abs(delta_phi) / separation
    k_expected = 2 * np.pi / _WAVELENGTH

    rel_err = abs(k_measured - k_expected) / k_expected
    assert rel_err < 0.01, (
        f"Phase velocity error: k_measured/k₀ = {k_measured / k_expected:.4f} (rel_err={rel_err:.3f}, expected < 0.01)"
    )


def test_quarter_domain_vs_full_domain():
    """Quarter-domain Ey phasor field, when unfolded, matches the full-domain reference.

    The full domain is twice as large in Y and Z as the quarter domain.
    PEC at min_y and PMC at min_z act as perfect mirrors for this polarization:
      - Ey has even symmetry about y=0 (PEC mirror: normal E unchanged)
      - Ey has even symmetry about z=0 (PMC mirror: tangential E unchanged)

    The quarter-domain spatial phasor is mirrored in Y then Z to reconstruct
    the full 2D slice, which is then compared element-wise to the full-domain field.
    Since both runs share the same discretization, the mirror reconstruction is
    near-exact: the MAX relative error over the slice must stay ≤ 2e-3.
    """
    # --- Full domain (periodic y, z) — doubled in Y and Z ---
    objects_full, con_full, cfg_full, vol_full, wave = _build_full_domain()
    det_full = fdtdx.PhasorDetector(
        name="det_full",
        partial_grid_shape=(1, None, None),
        wave_characters=(wave,),
        components=("Ey",),
        reduce_volume=False,
        plot=False,
    )
    con_full.extend(
        [
            det_full.same_size(vol_full, axes=(1, 2)),
            det_full.place_at_center(vol_full, axes=(1, 2)),
            det_full.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_DET1_X,)),
        ]
    )
    objects_full.append(det_full)
    arrays_full = _run(objects_full, con_full, cfg_full)
    # phasor shape: (1, n_freqs, n_components, nx=1, ny_full, nz_full)
    full_ey = arrays_full.detector_states["det_full"]["phasor"][0, 0, 0, 0, :, :]

    # --- Quarter domain (PEC min_y, PMC min_z) ---
    objects_qtr, con_qtr, cfg_qtr, vol_qtr, wave = _build_quarter_domain()
    det_qtr = fdtdx.PhasorDetector(
        name="det_qtr",
        partial_grid_shape=(1, None, None),
        wave_characters=(wave,),
        components=("Ey",),
        reduce_volume=False,
        plot=False,
    )
    con_qtr.extend(
        [
            det_qtr.same_size(vol_qtr, axes=(1, 2)),
            det_qtr.place_at_center(vol_qtr, axes=(1, 2)),
            det_qtr.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_DET1_X,)),
        ]
    )
    objects_qtr.append(det_qtr)
    arrays_qtr = _run(objects_qtr, con_qtr, cfg_qtr)
    # phasor shape: (1, n_freqs, n_components, nx=1, ny_qtr, nz_qtr)
    qtr_ey = arrays_qtr.detector_states["det_qtr"]["phasor"][0, 0, 0, 0, :, :]

    assert jnp.max(jnp.abs(full_ey)) > 1e-30, "Full-domain Ey phasor is zero — wave not launched"
    assert jnp.max(jnp.abs(qtr_ey)) > 1e-30, "Quarter-domain Ey phasor is zero — wave not launched"

    # Unfold the quarter-domain field by mirroring:
    #   PEC at min_y → Ey even about y=0: prepend reversed copy along Y
    #   PMC at min_z → Ey even about z=0: prepend reversed copy along Z
    unfolded_y = jnp.concatenate([qtr_ey[::-1, :], qtr_ey], axis=0)  # (2*ny_qtr, nz_qtr)
    unfolded_ey = jnp.concatenate([unfolded_y[:, ::-1], unfolded_y], axis=1)  # (2*ny_qtr, 2*nz_qtr)

    assert unfolded_ey.shape == full_ey.shape, (
        f"Unfolded quarter shape {unfolded_ey.shape} != full domain shape {full_ey.shape}. "
        f"Check that _FULL_DOMAIN_YZ == 2 * _DOMAIN_YZ."
    )

    # Compare complex fields element-wise: |unfolded - full| captures both amplitude and phase.
    # The PEC/PMC walls and the periodic full domain share the same discretization, so the
    # mirror reconstruction is near-exact — bound the worst-case (MAX) relative error, not the
    # mean (which would hide a localized boundary defect).
    ref_max = float(jnp.max(jnp.abs(full_ey)))
    max_complex_err = float(jnp.max(jnp.abs(unfolded_ey - full_ey)))
    rel_err = max_complex_err / ref_max

    assert rel_err <= 2e-3, (
        f"Unfolded quarter vs full domain complex field mismatch: "
        f"max |Δ| / max|full| = {rel_err:.2e} (expected <= 2e-3). "
        f"max|full|={ref_max:.4e}, max|unfolded|={float(jnp.max(jnp.abs(unfolded_ey))):.4e}"
    )


def test_quarter_domain_field_enforcement():
    """Tangential E = 0 at PEC, tangential H = 0 at PMC.

    Places phasor detectors ON the PEC (min_y) and PMC (min_z) boundary planes
    (grid index 0) to verify the correct field components are enforced to zero.

    At PEC (y=0): Ex and Ez should be zero, Ey should be nonzero.
    At PMC (z=0): Hx and Hy should be zero, Hz should be nonzero.

    For this Ey-polarized plane wave the only nonzero fields are Ey and Hz, so the
    tangential components are zero by polarization and the boundary forces them to
    exactly zero on the wall — the realistic residual ratio is roundoff (< 1e-3).
    """
    objects, constraints, config, volume, wave = _build_quarter_domain()

    # Detector ON PEC boundary plane (min_y, grid index 0)
    det_pec = fdtdx.PhasorDetector(
        name="pec_boundary",
        partial_grid_shape=(1, 1, None),
        wave_characters=(wave,),
        components=("Ex", "Ey", "Ez"),
        reduce_volume=True,
        plot=False,
    )
    constraints.extend(
        [
            det_pec.same_size(volume, axes=(2,)),
            det_pec.place_at_center(volume, axes=(2,)),
            det_pec.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_DET1_X,)),
            # Place ON min_y boundary plane (grid index 0)
            det_pec.set_grid_coordinates(axes=(1,), sides=("-",), coordinates=(0,)),
        ]
    )
    objects.append(det_pec)

    # Detector ON PMC boundary plane (min_z, grid index 0)
    det_pmc = fdtdx.PhasorDetector(
        name="pmc_boundary",
        partial_grid_shape=(1, None, 1),
        wave_characters=(wave,),
        components=("Hx", "Hy", "Hz"),
        reduce_volume=True,
        plot=False,
    )
    constraints.extend(
        [
            det_pmc.same_size(volume, axes=(1,)),
            det_pmc.place_at_center(volume, axes=(1,)),
            det_pmc.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_DET1_X,)),
            # Place ON min_z boundary plane (grid index 0)
            det_pmc.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(0,)),
        ]
    )
    objects.append(det_pmc)

    # Reference detector in the interior
    det_ref = fdtdx.PhasorDetector(
        name="interior",
        partial_grid_shape=(1, None, None),
        wave_characters=(wave,),
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        reduce_volume=True,
        plot=False,
    )
    constraints.extend(
        [
            det_ref.same_size(volume, axes=(1, 2)),
            det_ref.place_at_center(volume, axes=(1, 2)),
            det_ref.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_DET1_X,)),
        ]
    )
    objects.append(det_ref)

    arrays = _run(objects, constraints, config)

    # --- PEC boundary check: tangential E should be ~0, normal E should be nonzero ---
    pec_phasor = arrays.detector_states["pec_boundary"]["phasor"]
    amp_ex_pec = float(jnp.abs(pec_phasor[0, 0, 0]))  # Ex (tangential to y-face)
    amp_ey_pec = float(jnp.abs(pec_phasor[0, 0, 1]))  # Ey (normal to y-face)
    amp_ez_pec = float(jnp.abs(pec_phasor[0, 0, 2]))  # Ez (tangential to y-face)

    assert amp_ey_pec > 1e-30, "Ey at PEC boundary is zero — wave not present"

    # Tangential components should be much smaller than normal
    if amp_ey_pec > 0:
        ratio_ex = amp_ex_pec / amp_ey_pec
        ratio_ez = amp_ez_pec / amp_ey_pec
        assert ratio_ex < 1e-3, (
            f"PEC: |Ex|/|Ey| = {ratio_ex:.2e} on boundary (expected < 1e-3). Tangential Ex not properly enforced."
        )
        assert ratio_ez < 1e-3, (
            f"PEC: |Ez|/|Ey| = {ratio_ez:.2e} on boundary (expected < 1e-3). Tangential Ez not properly enforced."
        )

    # --- PMC boundary check: tangential H should be ~0, normal H should be nonzero ---
    pmc_phasor = arrays.detector_states["pmc_boundary"]["phasor"]
    amp_hx_pmc = float(jnp.abs(pmc_phasor[0, 0, 0]))  # Hx (tangential to z-face)
    amp_hy_pmc = float(jnp.abs(pmc_phasor[0, 0, 1]))  # Hy (tangential to z-face)
    amp_hz_pmc = float(jnp.abs(pmc_phasor[0, 0, 2]))  # Hz (normal to z-face)

    assert amp_hz_pmc > 1e-30, "Hz at PMC boundary is zero — wave not present"

    if amp_hz_pmc > 0:
        ratio_hx = amp_hx_pmc / amp_hz_pmc
        ratio_hy = amp_hy_pmc / amp_hz_pmc
        assert ratio_hx < 1e-3, (
            f"PMC: |Hx|/|Hz| = {ratio_hx:.2e} on boundary (expected < 1e-3). Tangential Hx not properly enforced."
        )
        assert ratio_hy < 1e-3, (
            f"PMC: |Hy|/|Hz| = {ratio_hy:.2e} on boundary (expected < 1e-3). Tangential Hy not properly enforced."
        )
