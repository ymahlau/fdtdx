"""Physics simulation test: Bloch with k=0 degenerates to periodic.

Runs the same plane wave simulation with periodic BCs (real fields) and
with Bloch BCs at k_bloch = (0, 0, 0) forced onto the COMPLEX field path.

At k=0 ``BlochBoundary.needs_complex_fields`` is False, so auto-detection
keeps the Bloch run on real fields — meaning the test would never actually
exercise the complex code path it claims to validate.  We therefore set
``use_complex_fields=True`` on the Bloch run's SimulationConfig to force
complex64 field arrays and the complex-valued PML.  Since exp(i*0*L) = 1,
the Bloch boundary must still behave identically to periodic.  This now
genuinely validates:
  1. Complex field arrays reproduce the real-field physics exactly
  2. Bloch phase correction with unit phase factor is correct
  3. PML still works correctly with complex-valued fields

Comparison metric: PhasorDetector Ex/Hy amplitude and phase (complex Bloch
run vs real periodic reference) must match within 1e-4 relative difference.

Domain: 3-cell periodic/Bloch in x,y, PML on z, plane wave in +z.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6
_RESOLUTION = 50e-9
_PML_CELLS = 10
_DOMAIN_XY = 3 * _RESOLUTION
_DOMAIN_Z = 4e-6

_SOURCE_Z = _PML_CELLS + 2
_DET_Z = _SOURCE_Z + 5

_SIM_TIME = 120e-15
_REL_TOL = 1e-4


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build(bc_type, bloch_vector=(0.0, 0.0, 0.0), force_complex=False):
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        time=_SIM_TIME,
        dtype=jnp.float32,
        # Force complex64 fields + complex PML even though k=0 Bloch does not
        # require them (needs_complex_fields is False at k=0). Without this the
        # Bloch run silently stays on the real path and the test is vacuous.
        use_complex_fields=True if force_complex else None,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(_DOMAIN_XY, _DOMAIN_XY, _DOMAIN_Z),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            "min_x": bc_type,
            "max_x": bc_type,
            "min_y": bc_type,
            "max_y": bc_type,
        },
        bloch_vector=bloch_vector,
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

    det = fdtdx.PhasorDetector(
        name="det",
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave,),
        reduce_volume=True,
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        plot=False,
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(0, 1)),
            det.place_at_center(volume, axes=(0, 1)),
            det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_DET_Z,)),
        ]
    )
    objects.append(det)

    return objects, constraints, config


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


def _ex_phasor(arrays) -> complex:
    return complex(arrays.detector_states["det"]["phasor"][0, 0, 0])


def _hy_phasor(arrays) -> complex:
    return complex(arrays.detector_states["det"]["phasor"][0, 0, 4])


# ── Tests ────────────────────────────────────────────────────────────────────


def test_bloch_zero_matches_periodic_ex():
    """Bloch(k=0, complex fields) Ex phasor matches real periodic Ex within 1e-4."""
    # Periodic reference (real fields)
    obj_p, con_p, cfg_p = _build("periodic")
    arrays_p = _run(obj_p, con_p, cfg_p)
    ex_periodic = _ex_phasor(arrays_p)

    # Bloch with k=0, forced onto the complex field + complex PML path
    obj_b, con_b, cfg_b = _build("bloch", bloch_vector=(0.0, 0.0, 0.0), force_complex=True)
    assert cfg_b.use_complex_fields is True, "Bloch run must force complex fields"
    arrays_b = _run(obj_b, con_b, cfg_b)
    ex_bloch = _ex_phasor(arrays_b)

    assert abs(ex_periodic) > 0, "Periodic Ex phasor is zero"
    assert abs(ex_bloch) > 0, "Bloch Ex phasor is zero"

    # Amplitude comparison
    amp_diff = abs(abs(ex_bloch) - abs(ex_periodic)) / abs(ex_periodic)
    assert amp_diff < _REL_TOL, (
        f"|Ex| mismatch: periodic={abs(ex_periodic):.6e}, bloch={abs(ex_bloch):.6e}, relative diff={amp_diff:.2e}"
    )

    # Phase comparison
    phase_diff = abs(np.angle(ex_bloch) - np.angle(ex_periodic))
    # Wrap to [0, π]
    phase_diff = min(phase_diff, 2 * np.pi - phase_diff)
    assert phase_diff < _REL_TOL, (
        f"Ex phase mismatch: periodic={np.angle(ex_periodic):.6f}, "
        f"bloch={np.angle(ex_bloch):.6f}, diff={phase_diff:.2e}"
    )


def test_bloch_zero_matches_periodic_hy():
    """Bloch(k=0, complex fields) Hy phasor matches real periodic Hy within 1e-4."""
    obj_p, con_p, cfg_p = _build("periodic")
    arrays_p = _run(obj_p, con_p, cfg_p)
    hy_periodic = _hy_phasor(arrays_p)

    obj_b, con_b, cfg_b = _build("bloch", bloch_vector=(0.0, 0.0, 0.0), force_complex=True)
    assert cfg_b.use_complex_fields is True, "Bloch run must force complex fields"
    arrays_b = _run(obj_b, con_b, cfg_b)
    hy_bloch = _hy_phasor(arrays_b)

    assert abs(hy_periodic) > 0, "Periodic Hy phasor is zero"
    assert abs(hy_bloch) > 0, "Bloch Hy phasor is zero"

    amp_diff = abs(abs(hy_bloch) - abs(hy_periodic)) / abs(hy_periodic)
    assert amp_diff < _REL_TOL, (
        f"|Hy| mismatch: periodic={abs(hy_periodic):.6e}, bloch={abs(hy_bloch):.6e}, relative diff={amp_diff:.2e}"
    )

    phase_diff = abs(np.angle(hy_bloch) - np.angle(hy_periodic))
    phase_diff = min(phase_diff, 2 * np.pi - phase_diff)
    assert phase_diff < _REL_TOL, (
        f"Hy phase mismatch: periodic={np.angle(hy_periodic):.6f}, "
        f"bloch={np.angle(hy_bloch):.6f}, diff={phase_diff:.2e}"
    )
