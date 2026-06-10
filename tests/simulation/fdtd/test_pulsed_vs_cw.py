"""Simulation test: pulsed source agrees with continuous-wave source.

Validates that the frequency-domain transmission spectrum obtained from a
single broadband pulsed simulation matches the transmission obtained from
individual continuous-wave (CW) simulations at each wavelength.

Domain layout (25 nm resolution, z is propagation axis):
  200 cells total in z (5 µm):
    cells   0-  9 : PML (0.25 µm)
    cells  10-189 : active region
    cells 190-199 : PML (0.25 µm)

  Source      : z-index 12 (vacuum side, 2 cells into active region)
  Layer 1     : z-index 80-100 (20 cells, ε=2.25, n=1.5)
  Layer 2     : z-index 100-115 (15 cells, ε=6.25, n=2.5)
  Layer 3     : z-index 115-135 (20 cells, ε=4.0, n=2.0)
  Detector    : z-index 160 (past all layers)

Transverse (x, y): 3 cells each with periodic boundaries.

Method:
  Pulsed:  GaussianPulseProfile (center=1 µm, spectral_width=WaveCharacter(3 µm))
           + PhasorDetector at all test wavelengths simultaneously.
           Two runs (with and without stack) give the normalized pulsed spectrum.

  CW:      SingleFrequencyProfile at each test wavelength.
           Two runs (with and without stack) give the normalized CW transmission.

  Both are normalized as |phasor_stack| / |phasor_reference| and compared.

Test wavelengths: [0.85 µm, 1.0 µm, 1.15 µm]
Tolerance: 5 % relative error.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ──────────────────────────────────────────────────────────
_RESOLUTION = 25e-9
_PML_CELLS = 10
_DOMAIN_XY = 3 * _RESOLUTION  # 3 cells, periodic
_DOMAIN_Z = 5e-6  # 200 cells total

_SOURCE_Z = _PML_CELLS + 2  # = 12
_DET_Z = 160  # past all layers

# 3-layer stack positions (grid indices)
_LAYER1_START = 80
_LAYER1_CELLS = 20
_LAYER2_START = 100
_LAYER2_CELLS = 15
_LAYER3_START = 115
_LAYER3_CELLS = 20

_SIM_TIME = 200e-15  # 200 fs — enough for pulse transit and CW steady state

# Pulsed source parameters
_CENTER_WL = 1e-6
# spectral_width as a WaveCharacter: get_frequency() gives the sigma in Hz.
# WaveCharacter(wavelength=3e-6) → c/3e-6 ≈ 100 THz sigma → covers ~0.7-1.3 µm
_SPECTRAL_WIDTH_WL = 3e-6

# Test wavelengths (must lie well within the pulse bandwidth)
_TEST_WAVELENGTHS = [0.85e-6, 1.0e-6, 1.15e-6]

_TOLERANCE = 0.05


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_base(temporal_profile, wave_character):
    """Build base geometry: volume, PML+periodic boundaries, source."""
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
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=wave_character,
        temporal_profile=temporal_profile,
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

    return objects, constraints, config, volume


def _add_layers(volume, objects, constraints):
    """Add the 3-layer material stack."""
    layer_specs = [
        (_LAYER1_START, _LAYER1_CELLS, 2.25),
        (_LAYER2_START, _LAYER2_CELLS, 6.25),
        (_LAYER3_START, _LAYER3_CELLS, 4.0),
    ]
    for start_z, n_cells, eps in layer_specs:
        layer = fdtdx.UniformMaterialObject(
            partial_grid_shape=(None, None, n_cells),
            material=fdtdx.Material(permittivity=eps),
        )
        constraints.extend(
            [
                layer.same_size(volume, axes=(0, 1)),
                layer.place_at_center(volume, axes=(0, 1)),
                layer.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(start_z,)),
            ]
        )
        objects.append(layer)


def _add_phasor_det(name, z_idx, wave_characters, volume, objects, constraints):
    """Add a PhasorDetector measuring the given wave_characters at z_idx."""
    det = fdtdx.PhasorDetector(
        name=name,
        partial_grid_shape=(None, None, 1),
        wave_characters=wave_characters,
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


def _phasor_amplitudes(arrays, det_name):
    """Extract per-frequency phasor amplitude from a PhasorDetector state.

    PhasorDetector state["phasor"] shape: (1, num_freq, num_components, ...)
    Returns shape (num_freq,): mean amplitude over Ex component (index 0).
    """
    phasor = np.array(arrays.detector_states[det_name]["phasor"])  # (1, F, C)
    # Take Ex component (component index 0) and collapse time axis
    return np.abs(phasor[0, :, 0])  # (F,)


# ── Test ──────────────────────────────────────────────────────────────────────


def test_pulsed_vs_cw_transmission():
    """Normalized transmission from pulsed and CW sources agree within 5 %.

    For each test wavelength:
      T_pulsed = |phasor_stack_pulsed[wl]| / |phasor_ref_pulsed[wl]|
      T_CW     = |phasor_stack_CW[wl]|     / |phasor_ref_CW[wl]|
      assert |T_pulsed - T_CW| / T_pulsed < 5 %
    """
    test_wave_chars = [fdtdx.WaveCharacter(wavelength=wl) for wl in _TEST_WAVELENGTHS]

    # ── Pulsed simulations ────────────────────────────────────────────────────
    pulsed_profile = fdtdx.GaussianPulseProfile(
        center_wave=fdtdx.WaveCharacter(wavelength=_CENTER_WL),
        spectral_width=fdtdx.WaveCharacter(wavelength=_SPECTRAL_WIDTH_WL),
    )
    center_wave = fdtdx.WaveCharacter(wavelength=_CENTER_WL)

    # Reference pulsed run (no stack)
    obj_ref, con_ref, cfg_ref, vol_ref = _build_base(pulsed_profile, center_wave)
    _add_phasor_det("det", _DET_Z, test_wave_chars, vol_ref, obj_ref, con_ref)
    arrays_pulsed_ref = _run(obj_ref, con_ref, cfg_ref)

    # Stack pulsed run
    obj_stack, con_stack, cfg_stack, vol_stack = _build_base(pulsed_profile, center_wave)
    _add_layers(vol_stack, obj_stack, con_stack)
    _add_phasor_det("det", _DET_Z, test_wave_chars, vol_stack, obj_stack, con_stack)
    arrays_pulsed_stack = _run(obj_stack, con_stack, cfg_stack)

    amp_pulsed_ref = _phasor_amplitudes(arrays_pulsed_ref, "det")
    amp_pulsed_stack = _phasor_amplitudes(arrays_pulsed_stack, "det")
    T_pulsed = amp_pulsed_stack / amp_pulsed_ref  # (num_wavelengths,)

    # ── CW simulations ────────────────────────────────────────────────────────
    T_cw = np.zeros(len(_TEST_WAVELENGTHS))

    for i, wl in enumerate(_TEST_WAVELENGTHS):
        wave = fdtdx.WaveCharacter(wavelength=wl)
        cw_profile = fdtdx.SingleFrequencyProfile()

        # Reference CW run (no stack)
        obj_r, con_r, cfg_r, vol_r = _build_base(cw_profile, wave)
        _add_phasor_det("det", _DET_Z, [wave], vol_r, obj_r, con_r)
        arr_r = _run(obj_r, con_r, cfg_r)

        # Stack CW run
        obj_s, con_s, cfg_s, vol_s = _build_base(cw_profile, wave)
        _add_layers(vol_s, obj_s, con_s)
        _add_phasor_det("det", _DET_Z, [wave], vol_s, obj_s, con_s)
        arr_s = _run(obj_s, con_s, cfg_s)

        amp_r = _phasor_amplitudes(arr_r, "det")  # (1,)
        amp_s = _phasor_amplitudes(arr_s, "det")  # (1,)
        T_cw[i] = float(amp_s[0] / amp_r[0])

    # ── Comparison ────────────────────────────────────────────────────────────
    for i, wl in enumerate(_TEST_WAVELENGTHS):
        rel_err = abs(T_pulsed[i] - T_cw[i]) / T_pulsed[i]
        assert rel_err < _TOLERANCE, (
            f"wl={wl * 1e6:.2f} µm: T_pulsed={T_pulsed[i]:.4f}, T_CW={T_cw[i]:.4f}, "
            f"relative error={rel_err:.3f} > {_TOLERANCE}"
        )


# ── Mode overlap pulsed vs CW ─────────────────────────────────────────────────
#
# Si/SiO2 slab waveguide (propagation along x, periodic in y).
# One broadband pulsed run with ModeOverlapDetector carrying all 3 wave_characters
# is compared against 3 independent CW runs each with a single wave_character.
# Agreement validates that multi-frequency phasor accumulation produces the same
# S-parameter at each frequency as running the frequencies independently.
#
# Domain (50 nm resolution, x is propagation axis):
#   x: 3 µm total (60 cells); PML 10 cells each side; active 40 cells
#   y: 150 nm (3 cells), periodic — slab approximation
#   z: 2 µm total (40 cells); PML 10 cells each side
#   Si core: 250 nm (5 cells) centred in z
#   Source: x-index 12; output detector: x-index 25

_WG_RESOLUTION = 50e-9
_WG_PML = 10
_WG_DOMAIN_X = 3e-6
_WG_DOMAIN_Y = 3 * _WG_RESOLUTION
_WG_DOMAIN_Z = 2e-6
_WG_CORE_HEIGHT = 250e-9
_WG_SOURCE_X = _WG_PML + 2  # = 12
_WG_DET_X = 25
_WG_SIM_TIME = 300e-15  # 300 fs: ~69 periods at 1300 nm, enough for CW startup ramp to be negligible
_WG_CENTER_WL = 1.45e-6

_WG_EPS_SI = fdtdx.constants.relative_permittivity_silicon
_WG_EPS_SIO2 = fdtdx.constants.relative_permittivity_silica

_WG_TEST_WCS = [
    fdtdx.WaveCharacter(wavelength=1.30e-6),
    fdtdx.WaveCharacter(wavelength=1.45e-6),
    fdtdx.WaveCharacter(wavelength=1.55e-6),
]
_WG_TOLERANCE = 0.05


def _build_waveguide_scene(temporal_profile, source_wc, det_wave_characters):
    """Return (objects, constraints, config) for the Si/SiO2 slab waveguide."""
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_WG_RESOLUTION),
        time=_WG_SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(_WG_DOMAIN_X, _WG_DOMAIN_Y, _WG_DOMAIN_Z),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_WG_PML,
        override_types={"min_y": "periodic", "max_y": "periodic"},
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    cladding = fdtdx.UniformMaterialObject(
        name="cladding",
        partial_real_shape=(None, None, None),
        material=fdtdx.Material(permittivity=_WG_EPS_SIO2),
    )
    constraints.extend(cladding.same_position_and_size(volume))
    objects.append(cladding)

    core = fdtdx.UniformMaterialObject(
        name="core",
        partial_real_shape=(None, None, _WG_CORE_HEIGHT),
        material=fdtdx.Material(permittivity=_WG_EPS_SI),
    )
    constraints.extend([core.same_size(volume, axes=(0, 1)), core.place_at_center(volume, axes=(0, 1, 2))])
    objects.append(core)

    source = fdtdx.ModePlaneSource(
        name="source",
        partial_grid_shape=(1, None, None),
        wave_character=source_wc,
        temporal_profile=temporal_profile,
        direction="+",
        mode_index=0,
        filter_pol="te",
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(1, 2)),
            source.place_at_center(volume, axes=(1, 2)),
            source.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_WG_SOURCE_X,)),
        ]
    )
    objects.append(source)

    input_det = fdtdx.ModeOverlapDetector(
        name="det_source",
        wave_characters=tuple(det_wave_characters),
        direction="+",
        mode_index=0,
        filter_pol="te",
        scaling_mode="pulse",
    )
    constraints.extend([input_det.same_size(source), input_det.same_position(source, grid_margins=(1, 0, 0))])
    objects.append(input_det)

    det_out = fdtdx.ModeOverlapDetector(
        name="det_out",
        partial_grid_shape=(1, None, None),
        wave_characters=tuple(det_wave_characters),
        direction="+",
        mode_index=0,
        filter_pol="te",
        scaling_mode="pulse",
    )
    constraints.extend(
        [
            det_out.same_size(volume, axes=(1, 2)),
            det_out.place_at_center(volume, axes=(1, 2)),
            det_out.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_WG_DET_X,)),
        ]
    )
    objects.append(det_out)

    return objects, constraints, config


def test_pulsed_vs_cw_mode_overlap():
    """Multi-freq pulsed ModeOverlapDetector agrees with single-freq CW runs within 5 %.

    One broadband pulsed run carries all 3 wave_characters on a single
    ModeOverlapDetector.  Three independent CW runs each carry one wave_character.
    |S_pulsed(wl)| must agree with |S_CW(wl)| at every wavelength, proving that
    multi-frequency phasor accumulation does not cross-contaminate frequency channels
    and that per-frequency normalization is correct.
    """
    center_wc = fdtdx.WaveCharacter(wavelength=_WG_CENTER_WL)
    pulsed_profile = fdtdx.GaussianPulseProfile(
        center_wave=center_wc,
        spectral_width=fdtdx.WaveCharacter(wavelength=_WG_CENTER_WL * 10),
    )
    key = jax.random.PRNGKey(0)

    # ── Pulsed run (all 3 frequencies at once) ────────────────────────────────
    objs, cons, cfg = _build_waveguide_scene(pulsed_profile, center_wc, _WG_TEST_WCS)
    obj_container, arrays, _, cfg, _ = fdtdx.place_objects(object_list=objs, config=cfg, constraints=cons, key=key)
    arrays = fdtdx.extend_material_to_pml(objects=obj_container, arrays=arrays)
    result_pulsed, _ = fdtdx.calculate_sparam(
        objects=obj_container,
        arrays=arrays,
        config=cfg,
        input_port_name="source",
        show_progress=False,
    )
    s_pulsed = np.array(result_pulsed[("det_out", "source")])  # shape (3,)

    # ── CW runs (one per frequency) ───────────────────────────────────────────
    s_cw = []
    for wc in _WG_TEST_WCS:
        objs, cons, cfg = _build_waveguide_scene(fdtdx.SingleFrequencyProfile(), wc, [wc])
        obj_container, arrays, _, cfg, _ = fdtdx.place_objects(object_list=objs, config=cfg, constraints=cons, key=key)
        arrays = fdtdx.extend_material_to_pml(objects=obj_container, arrays=arrays)
        result, _ = fdtdx.calculate_sparam(
            objects=obj_container,
            arrays=arrays,
            config=cfg,
            input_port_name="source",
            show_progress=False,
        )
        s_cw.append(complex(np.array(result[("det_out", "source")])[0]))

    # ── Comparison ────────────────────────────────────────────────────────────
    for i, wc in enumerate(_WG_TEST_WCS):
        wl_nm = int(wc.wavelength * 1e9)
        rel_err = abs(abs(s_pulsed[i]) - abs(s_cw[i])) / abs(s_cw[i])
        assert rel_err < _WG_TOLERANCE, (
            f"wl={wl_nm} nm: |S_pulsed|={abs(s_pulsed[i]):.4f}, "
            f"|S_CW|={abs(s_cw[i]):.4f}, rel_err={rel_err:.3f} > {_WG_TOLERANCE}"
        )
