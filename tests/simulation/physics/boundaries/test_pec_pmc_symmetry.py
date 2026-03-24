"""Physics simulation test: PEC vs PMC standing wave field pattern.

Validates that PEC and PMC walls produce the correct standing wave
field conditions by measuring E_x amplitude near the reflecting wall.

Physics:
  A plane wave reflecting from a perfect conductor creates a standing
  wave.  The E-field pattern depends on the conductor type:

  PEC (E_tangential = 0 at wall):
    E_x ∝ sin(k·d) where d = distance from wall
    → E_x = 0 at wall (node)
    → E_x maximal at d = λ/4 (antinode)

  PMC (H_tangential = 0 at wall):
    E_x ∝ cos(k·d) where d = distance from wall
    → E_x maximal at wall (antinode)
    → E_x = 0 at d = λ/4 (node)

Test strategy:
  Place two PhasorDetectors in each run:
    "near" — 1 cell from the wall (d ≈ 0)
    "far"  — λ/4 from the wall (d = λ/4)

  PEC: |Ex_near| << |Ex_far|   (node at wall, antinode at λ/4)
  PMC: |Ex_near| >> |Ex_far|   (antinode at wall, node at λ/4)

  Quantitative check (using two-run normalization):
    near/far ratio for PEC should be < 0.2
    near/far ratio for PMC should be > 5.0

Domain: periodic xy, PML z-min, PEC/PMC z-max.
Resolution: 25 nm = 40 cells/λ for accurate near-wall measurement.
"""

import jax
import jax.numpy as jnp

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6
_RESOLUTION = 12.5e-9  # 80 cells/λ for accurate near-wall measurement
_PML_CELLS = 10
_DOMAIN_XY = 3 * _RESOLUTION
_DOMAIN_Z = 5e-6  # 400 cells

_Z_CELLS = int(round(_DOMAIN_Z / _RESOLUTION))  # = 400
_SOURCE_Z = _PML_CELLS + 2  # = 12

# Detector positions relative to z-max wall
# "near": 1 cell from wall = 12.5nm, sin(k*d) ≈ 0.08
# "far": λ/4 from wall = 20 cells at 80 cells/λ, sin(k*d) = 1.0
_DET_NEAR_Z = _Z_CELLS - 2  # 1 cell from z-max
_DET_FAR_Z = _Z_CELLS - 20 - 1  # λ/4 from z-max

_SIM_TIME = 150e-15  # 150 fs — extra time for standing wave to develop


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build(z_max_type):
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

    # Near-wall detector
    det_near = fdtdx.PhasorDetector(
        name="near",
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave,),
        reduce_volume=True,
        components=("Ex",),
        plot=False,
    )
    constraints.extend(
        [
            det_near.same_size(volume, axes=(0, 1)),
            det_near.place_at_center(volume, axes=(0, 1)),
            det_near.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_DET_NEAR_Z,)),
        ]
    )
    objects.append(det_near)

    # Far detector (λ/4 from wall)
    det_far = fdtdx.PhasorDetector(
        name="far",
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave,),
        reduce_volume=True,
        components=("Ex",),
        plot=False,
    )
    constraints.extend(
        [
            det_far.same_size(volume, axes=(0, 1)),
            det_far.place_at_center(volume, axes=(0, 1)),
            det_far.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_DET_FAR_Z,)),
        ]
    )
    objects.append(det_far)

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


def _ex_amplitude(arrays, name) -> float:
    # phasor shape with reduce_volume=True and 1 component: (1, num_freqs, num_components)
    # first dim is latent time step
    phasor = arrays.detector_states[name]["phasor"]
    return float(jnp.abs(phasor[0, 0, 0]))


# ── Tests ────────────────────────────────────────────────────────────────────


def test_pec_standing_wave():
    """PEC wall: E_x node at wall (near << far).

    Near-wall Ex should be close to zero (sin-like envelope with
    node at wall).  Far detector at λ/4 is at the antinode.
    Ratio near/far should be < 0.2.
    """
    obj, con, cfg = _build("pec")
    arrays = _run(obj, con, cfg)

    amp_near = _ex_amplitude(arrays, "near")
    amp_far = _ex_amplitude(arrays, "far")

    assert amp_far > 0, "Far detector measured zero Ex"

    ratio = amp_near / amp_far
    assert ratio < 0.2, (
        f"PEC standing wave: near/far ratio={ratio:.3f} (expected < 0.2). "
        f"|Ex_near|={amp_near:.4e}, |Ex_far|={amp_far:.4e}"
    )


def test_pmc_standing_wave():
    """PMC wall: E_x antinode at wall (near >> far).

    Near-wall Ex should be at maximum (cos-like envelope with antinode
    at wall).  Far detector at λ/4 is at the node.
    Ratio near/far should be > 5.0.
    """
    obj, con, cfg = _build("pmc")
    arrays = _run(obj, con, cfg)

    amp_near = _ex_amplitude(arrays, "near")
    amp_far = _ex_amplitude(arrays, "far")

    assert amp_near > 0, "Near detector measured zero Ex"

    ratio = amp_near / amp_far if amp_far > 0 else float("inf")
    assert ratio > 5.0, (
        f"PMC standing wave: near/far ratio={ratio:.3f} (expected > 5.0). "
        f"|Ex_near|={amp_near:.4e}, |Ex_far|={amp_far:.4e}"
    )
