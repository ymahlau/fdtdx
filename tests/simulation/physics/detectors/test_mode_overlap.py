"""Physics simulation tests: GaussianModeOverlapDetector against a real Gaussian beam.

A ``GaussianPlaneSource`` launches a beam into vacuum and a bank of
``GaussianModeOverlapDetector`` s on one downstream plane measure it. Every assertion is a
*ratio* between detectors sharing that plane, so the mode normalization and the overlap
integral's own scaling cancel and what is left is physics: the measured coupling is
compared against the closed-form overlap of the two Gaussians involved.

The beam is deliberately wide (``w = 1.18 um``, Rayleigh range ``4.4 um``) so the detector
plane sits at ``0.05`` Rayleigh ranges — near-collimated, so a flat-phase reference is the
matched one — and so the beam's own angular spread stays well below the tilt under test.
Mismatched references are all *narrower* than the beam, which keeps them fully inside the
detector plane and the analytic formula free of truncation bias.

``GaussianPlaneSource`` parameterizes its profile as ``exp(-r^2 / (2 (radius std)^2))`` with
a hard aperture at ``radius``, so the ``1/e`` amplitude radius the detector wants is
``radius * std * sqrt(2)`` — see ``_BEAM_RADIUS``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.constants import c as c0

_WAVELENGTH = 1e-6
_RESOLUTION = 50e-9
_PML_CELLS = 10
_DOMAIN_XY = 8e-6
_DOMAIN_Z = 4e-6
_SIM_TIME = 120e-15

_SOURCE_Z = _PML_CELLS + 2
_DET_Z = _SOURCE_Z + 4  # 200 nm downstream

# Source aperture radius, and the 1/e amplitude radius of the beam it actually launches.
_SOURCE_RADIUS = 2.5e-6
_BEAM_RADIUS = _SOURCE_RADIUS * (1 / 3) * np.sqrt(2.0)  # std defaults to 1/3

_WAVENUMBER = 2.0 * np.pi / _WAVELENGTH
_RAYLEIGH_RANGE = np.pi * _BEAM_RADIUS**2 / _WAVELENGTH
_PROPAGATION_DISTANCE = (_DET_Z - _SOURCE_Z) * _RESOLUTION

_NARROW_FACTORS = (0.6, 0.4)
_TILT_DEGREES = 30.0
_DIVERGENCE_DEGREES = 30.0

_DT_APPROX = 0.99 * _RESOLUTION / (c0 * np.sqrt(3))
_STEPS_PER_PERIOD = round(_WAVELENGTH / (c0 * _DT_APPROX))


def _radius_overlap(w_field: float, w_ref: float) -> float:
    """Analytic power coupling between two collimated circular Gaussians."""
    return (2.0 * w_field * w_ref / (w_field**2 + w_ref**2)) ** 2


def _curvature_overlap(w: float, divergence_degrees: float) -> float:
    """Analytic power coupling between a flat-phase Gaussian and a curved one of equal width.

    From ``eta = 4 / (w^4 [(2/w^2)^2 + c^2])`` with ``c = k tan(angle) / (2 w)`` the
    quadratic phase coefficient; the envelopes are identical, so only the phase mismatch
    survives.
    """
    c = _WAVENUMBER * np.tan(np.deg2rad(divergence_degrees)) / (2.0 * w)
    return 1.0 / (1.0 + c**2 * w**4 / 4.0)


def _detector(name: str, wave, **kwargs) -> fdtdx.GaussianModeOverlapDetector:
    """A GaussianModeOverlapDetector on the shared measurement plane."""
    params = {
        "name": name,
        "partial_grid_shape": (None, None, 1),
        "wave_characters": (wave,),
        "mode_radius": _BEAM_RADIUS,
        "direction": "+",
        "fixed_E_polarization_vector": (1.0, 0.0, 0.0),
    }
    params.update(kwargs)
    return fdtdx.GaussianModeOverlapDetector(**params)


def _build():
    """Assemble the free-space domain, the source and the detector bank."""
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        time=_SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(partial_real_shape=(_DOMAIN_XY, _DOMAIN_XY, _DOMAIN_Z))
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML_CELLS)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.GaussianPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
        radius=_SOURCE_RADIUS,
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_SOURCE_Z,)),
        ]
    )
    objects.append(source)

    detectors = [
        _detector("matched", wave),
        _detector("backward", wave, direction="-"),
        _detector("tilted", wave, azimuth_angle=_TILT_DEGREES),
        _detector("diverging", wave, divergence_angle=_DIVERGENCE_DEGREES),
    ]
    detectors += [_detector(f"narrow_{factor}", wave, mode_radius=factor * _BEAM_RADIUS) for factor in _NARROW_FACTORS]
    for det in detectors:
        constraints.extend(
            [
                det.same_size(volume, axes=(0, 1)),
                det.place_at_center(volume, axes=(0, 1)),
                det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_DET_Z,)),
            ]
        )
    objects.extend(detectors)

    return objects, constraints, config


@pytest.fixture(scope="module")
def overlap_powers():
    """Run the simulation once and return ``{detector name: |overlap|^2}``."""
    objects, constraints, config = _build()
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=config, key=key)

    powers = {}
    for det in obj_container.detectors:
        if isinstance(det, fdtdx.BaseModeOverlapDetector):
            overlap = det.compute_overlap(arrays.detector_states[det.name])
            powers[det.name] = float(jnp.abs(overlap[0]) ** 2)
    print({name: f"{value:.4e}" for name, value in powers.items()})
    return powers


def test_beam_is_collimated_at_the_detector_plane():
    """Guard the premise the other tests rest on: the launched beam is near its waist."""
    assert _PROPAGATION_DISTANCE < 0.1 * _RAYLEIGH_RANGE
    assert _STEPS_PER_PERIOD > 10  # enough temporal sampling for the phasor accumulation


def test_mismatched_radius_follows_analytic_overlap(overlap_powers):
    """Coupling into a wrong-radius reference matches the two-Gaussian formula.

    This is what shows the overlap integral computes a coupling *efficiency* rather than
    merely a large number wherever the field is: the predicted ratios follow from the beam
    radii alone and are independent of every normalization in play.
    """
    matched = overlap_powers["matched"]
    for factor in _NARROW_FACTORS:
        expected = _radius_overlap(_BEAM_RADIUS, factor * _BEAM_RADIUS)
        measured = overlap_powers[f"narrow_{factor}"] / matched
        assert measured == pytest.approx(expected, rel=0.03), (
            f"radius x{factor}: measured ratio {measured:.4f} vs analytic {expected:.4f}"
        )


def test_divergence_follows_analytic_overlap(overlap_powers):
    """A curved reference loses exactly the predicted amount against a collimated beam.

    Fails if the curvature term is dropped (ratio -> 1) or carries the wrong magnitude. The
    residual few percent is the launched beam's own ~0.7 deg of curvature at this plane,
    which the analytic estimate treats as exactly flat.
    """
    expected = _curvature_overlap(_BEAM_RADIUS, _DIVERGENCE_DEGREES)
    measured = overlap_powers["diverging"] / overlap_powers["matched"]
    assert measured == pytest.approx(expected, rel=0.08), (
        f"divergence {_DIVERGENCE_DEGREES} deg: measured ratio {measured:.4f} vs analytic {expected:.4f}"
    )


def test_tilt_suppresses_coupling(overlap_powers):
    """A tilted reference barely couples to a normal-incidence beam.

    The transverse phase ramp is what suppresses it — a 1D estimate gives
    ``exp(-(k sin(theta) w)^2 / 4) ~ 0.03`` here — so this fails if the ramp is dropped.
    """
    ratio = overlap_powers["tilted"] / overlap_powers["matched"]
    assert ratio < 0.05, f"tilted/matched = {ratio:.3e}; the phase ramp is not taking effect"


def test_backward_reference_reads_zero(overlap_powers):
    """A forward beam carries no backward modal power — direction selectivity.

    Would be ~1 if the overlap merely measured field magnitude on the plane.
    """
    ratio = overlap_powers["backward"] / overlap_powers["matched"]
    assert ratio < 1e-3, f"backward/forward = {ratio:.3e} is not negligible"


def test_matched_reference_is_the_best(overlap_powers):
    """No mismatched reference out-couples the matched one."""
    matched = overlap_powers["matched"]
    others = [name for name in overlap_powers if name != "matched"]
    for name in others:
        assert overlap_powers[name] < matched, f"{name} out-coupled the matched reference"
