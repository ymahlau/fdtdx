"""Simulation test: pulsed source agrees with continuous-wave source.

Validates that the frequency-domain transmission spectrum obtained from a
single broadband pulsed simulation matches the transmission obtained from
individual continuous-wave (CW) simulations at each wavelength.

Domain layout (25 nm resolution, z is propagation axis):
  200 cells total in z (5 µm):
    cells   0–  9 : PML (0.25 µm)
    cells  10–189 : active region
    cells 190–199 : PML (0.25 µm)

  Source      : z-index 12 (vacuum side, 2 cells into active region)
  Layer 1     : z-index 80–100 (20 cells, ε=2.25, n=1.5)
  Layer 2     : z-index 100–115 (15 cells, ε=6.25, n=2.5)
  Layer 3     : z-index 115–135 (20 cells, ε=4.0, n=2.0)
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
# WaveCharacter(wavelength=3e-6) → c/3e-6 ≈ 100 THz sigma → covers ~0.7–1.3 µm
_SPECTRAL_WIDTH_WL = 3e-6

# Test wavelengths (must lie well within the pulse bandwidth)
_TEST_WAVELENGTHS = [0.85e-6, 1.0e-6, 1.15e-6]

_TOLERANCE = 0.05


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_base(temporal_profile, wave_character):
    """Build base geometry: volume, PML+periodic boundaries, source."""
    config = fdtdx.SimulationConfig(
        resolution=_RESOLUTION,
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
