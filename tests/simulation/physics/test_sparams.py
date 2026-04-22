"""Physics simulation tests: calculate_sparam for a simple waveguide.

Tests that a lossless Si/SiO2 slab waveguide transmits ~100% of
fundamental-TE-mode light, measured at multiple ModeOverlapDetectors
at increasing distances from the source.

Geometry (propagation along x, periodic in y = infinite slab approximation):
  Domain:  3 µm x 150 nm x 2 µm (including PML)
  PML: 10 cells in x and z; periodic BC in y (no PML needed in y)
  Core: Si (ε=12.25), 250 nm slab centred in z (5 cells, divides evenly)
  Cladding: SiO₂ (ε=2.25), fills the volume
  Source: ModePlaneSource at x-index 12, TE mode 0
  Detectors: ModeOverlapDetectors at x-index 20, 25, 30
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_CENTER_WAVELENGTH = 1.55e-6
_RESOLUTION = 50e-9
_PML_CELLS = 10  # 500 nm per face

# y is periodic (slab approximation) — just 3 cells, no PML needed
_DOMAIN_Y = 3 * _RESOLUTION  # 150 nm

# z is the confinement direction: PML(500nm) + cladding + wg + cladding + PML(500nm)
_WG_HEIGHT = 250e-9  # 5 cells in z — divides evenly on the 50 nm grid
_DOMAIN_Z = 2e-6  # 40 cells — ample cladding above/below the 250 nm core

# x is the propagation axis
_DOMAIN_X = 3e-6  # 60 cells

# Grid indices (left face along x)
_SOURCE_X = _PML_CELLS + 2  # = 12
_DET1_X = 20  # 8 cells from source
_DET2_X = 25  # 5 cells further
_DET3_X = 30  # 5 cells further still

_SIM_TIME = 120e-15  # 120 fs ≈ 23 optical periods at λ = 1550 nm

_EPS_SI = fdtdx.constants.relative_permittivity_silicon  # 12.25
_EPS_SIO2 = fdtdx.constants.relative_permittivity_silica  # 2.25

_TOLERANCE = 0.01  # 1% relative tolerance on |S|²


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_waveguide_sparams():
    """Return (objects, constraints, config) for the Si/SiO2 slab waveguide.

    Periodic BC in y gives an infinite-slab approximation.  The mode solver
    sees only the z cross-section (Si core + SiO2 cladding), producing a
    purely real TE mode that the ModePlaneSource can inject exactly.
    """
    config = fdtdx.SimulationConfig(
        resolution=_RESOLUTION,
        time=_SIM_TIME,
        dtype=jnp.float32,
    )
    print(f"{config.time_steps_total=}")
    print(f"{config.time_step_duration=}")
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(_DOMAIN_X, _DOMAIN_Y, _DOMAIN_Z),
    )
    objects.append(volume)

    # PML in x and z; periodic in y (slab waveguide — infinite in y)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            "min_y": "periodic",
            "max_y": "periodic",
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    # SiO2 cladding fills the entire volume
    cladding = fdtdx.UniformMaterialObject(
        name="cladding",
        partial_real_shape=(None, None, None),
        material=fdtdx.Material(permittivity=_EPS_SIO2),
    )
    constraints.extend(cladding.same_position_and_size(volume))
    objects.append(cladding)

    # Si slab core, centred in z (periodic/infinite in y via BC)
    core = fdtdx.UniformMaterialObject(
        name="core",
        partial_real_shape=(None, None, _WG_HEIGHT),
        material=fdtdx.Material(permittivity=_EPS_SI),
    )
    constraints.extend(
        [
            core.same_size(volume, axes=(0, 1)),
            core.place_at_center(volume, axes=(0, 1, 2)),
        ]
    )
    objects.append(core)

    center_wave = fdtdx.WaveCharacter(wavelength=_CENTER_WAVELENGTH)
    wave_range = fdtdx.WaveCharacter(wavelength=_CENTER_WAVELENGTH * 10)
    profile = fdtdx.GaussianPulseProfile(center_wave=center_wave, spectral_width=wave_range)

    # Input: ModePlaneSource exciting TE fundamental slab mode
    source = fdtdx.ModePlaneSource(
        name="source",
        partial_grid_shape=(1, None, None),
        wave_character=center_wave,
        temporal_profile=profile,
        direction="+",
        mode_index=0,
        filter_pol="te",
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(1, 2)),
            source.place_at_center(volume, axes=(1, 2)),
            source.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_SOURCE_X,)),
        ]
    )
    objects.append(source)

    input_det = fdtdx.ModeOverlapDetector(
        name="det_source",
        wave_characters=(center_wave,),
        direction="+",
        mode_index=0,
        filter_pol="te",
        scaling_mode="pulse",
    )
    constraints.extend([input_det.same_size(source), input_det.same_position(source, grid_margins=(1, 0, 0))])
    objects.append(input_det)

    # Three output detectors at closely spaced x-positions along the waveguide
    for name, x_idx in [("det_near", _DET1_X), ("det_mid", _DET2_X), ("det_far", _DET3_X)]:
        det = fdtdx.ModeOverlapDetector(
            name=name,
            partial_grid_shape=(1, None, None),
            wave_characters=(center_wave,),
            direction="+",
            mode_index=0,
            filter_pol="te",
            scaling_mode="pulse",
        )
        constraints.extend(
            [
                det.same_size(volume, axes=(1, 2)),
                det.place_at_center(volume, axes=(1, 2)),
                det.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(x_idx,)),
            ]
        )
        objects.append(det)

    return objects, constraints, config


# ── Test ─────────────────────────────────────────────────────────────────────


def test_waveguide_sparam_transmission():
    """S-parameters at multiple detector positions are all close to 1.0.

    A ModePlaneSource injects the fundamental TE mode into a Si/SiO2 slab
    waveguide (periodic in y).  Three ModeOverlapDetectors at x-cells 20,
    25, and 30 measure the mode overlap.  Since the waveguide is lossless
    at 1550 nm, all |S|² values should exceed 0.90.
    """
    objects, constraints, config = _build_waveguide_sparams()

    key = jax.random.PRNGKey(0)
    obj_container, arrays, _params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays = fdtdx.extend_material_to_pml(objects=obj_container, arrays=arrays)

    result, _detector_states = fdtdx.calculate_sparam(
        objects=obj_container,
        arrays=arrays,
        config=config,
        input_port_name="source",
        show_progress=False,
    )

    for (det_name, src_name), s_param in result.items():
        assert src_name == "source", f"Unexpected source name in key: {src_name!r}"
        power = float(abs(np.array(s_param)) ** 2)
        print(f"{power=}")
        assert power > (1.0 - _TOLERANCE) and power <= 1.0001, (
            f"|S({det_name!r}, 'source')|² = {power}, "
            f"expected > {1.0 - _TOLERANCE:.2f} "
            f"(lossless Si/SiO2 slab waveguide should transmit large % of TE mode power)"
        )
