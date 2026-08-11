"""Physics simulation test: PhasorDetector apodization suppresses spectral leakage.

A CW source is still ringing when the detector stops recording, so the rectangular gate ends
on a discontinuity and smears the tone's power across neighbouring frequencies. A Tukey
window tapers the ends and suppresses that, while the coherent-gain correction keeps the
on-frequency amplitude unchanged.

Both detectors are gated to the steady-state part of the run: the source's linear startup
ramp makes the field non-stationary for the first few periods, and a rectangular gate weights
that transient equally while a taper de-weights it, which would otherwise show up as an
amplitude difference unrelated to leakage.

Probe frequencies sit at *fractional* bin offsets. Integer offsets land exactly on the nulls
of the rectangular window's kernel, which hides the leakage entirely.
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
_DOMAIN_XY = 5 * _RESOLUTION
_DOMAIN_Z = 4e-6
_SIM_TIME = 120e-15

_SOURCE_Z = _PML_CELLS + 2
_DET_Z = _SOURCE_Z + 10

_RECORD_START = 40e-15  # well past the source's 13.3 fs startup ramp and the transit time

_BIN_OFFSETS = (2.5, 4.5)  # fractional DFT bins away from the tone


def _build():
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RESOLUTION), time=_SIM_TIME, dtype=jnp.float32)
    duration = (config.time_steps_total - 1) * config.time_step_duration
    # Bins are set by the *recorded* interval, not the whole run.
    bin_hz = 1.0 / (duration - _RECORD_START)
    f0 = c0 / _WAVELENGTH
    probe_freqs = [f0] + [f0 + off * bin_hz for off in _BIN_OFFSETS]
    waves = tuple(fdtdx.WaveCharacter(wavelength=float(c0 / f)) for f in probe_freqs)

    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_real_shape=(_DOMAIN_XY, _DOMAIN_XY, _DOMAIN_Z))
    objects.append(volume)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={"min_x": "periodic", "max_x": "periodic", "min_y": "periodic", "max_y": "periodic"},
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    source = fdtdx.UniformPlaneSource(
        name="source",
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
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

    # The window spans exactly the recorded interval, so it tapers both of its edges.
    window = fdtdx.TukeyWindow(start_time=_RECORD_START, end_time=duration, alpha=0.5)
    switch = fdtdx.OnOffSwitch(start_time=_RECORD_START)
    for name, apodization in (("rect", None), ("tukey", window)):
        det = fdtdx.PhasorDetector(
            name=name,
            partial_grid_shape=(None, None, 1),
            wave_characters=waves,
            reduce_volume=True,
            apodization=apodization,
            switch=switch,
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


@pytest.fixture(scope="module")
def spectra():
    """Run once; return ``{detector: |phasor| per probe frequency}`` for the Ex component."""
    objects, constraints, config = _build()
    key = jax.random.PRNGKey(0)
    oc, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, oc, _ = fdtdx.apply_params(arrays, oc, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key)
    out = {}
    for name in ("rect", "tukey"):
        phasor = np.asarray(arrays.detector_states[name]["phasor"])
        out[name] = np.abs(phasor[0, :, 0])  # (num_freqs,) for Ex
    return out


def test_apodization_preserves_on_frequency_amplitude(spectra):
    """The 2 / sum(w) coherent-gain correction leaves the tone's amplitude unchanged.

    Measured agreement is 8.5e-4 relative at these settings and holds to 5.3e-4 under a 4x
    refinement in space and time, so the deviation is a numerical floor (float32 DFT
    accumulation plus the 2w residual), not a discretization artifact. CPU and GPU agree to
    1.1e-7. A missing or wrong gain correction (the window's mean weight is 1 - alpha/2 = 0.75,
    a 25% error) misses by ~300x the tolerance.
    """
    rect, tukey = spectra["rect"][0], spectra["tukey"][0]
    assert rect > 0
    assert tukey == pytest.approx(rect, rel=2e-3), f"on-frequency amplitude changed: {rect:.4e} -> {tukey:.4e}"


@pytest.mark.parametrize("index,offset", list(enumerate(_BIN_OFFSETS, start=1)))
def test_apodization_suppresses_off_frequency_leakage(spectra, index, offset):
    """Off-tone response drops once the recording edges are tapered.

    Measured suppression at these settings is 3.01x at 2.5 bins and 3.85x at 4.5 bins;
    requiring 2.5x leaves ~17% headroom on the tighter of the two. Suppression is
    setting-dependent (2.85x at 2.5 bins under a 4x refinement), so these numbers apply to the
    constants above and must be re-measured if the domain or timing changes.
    """
    rect = spectra["rect"][index] / spectra["rect"][0]
    tukey = spectra["tukey"][index] / spectra["tukey"][0]
    # The leakage must genuinely be there, or "suppression" would be vacuous.
    assert rect > 0.03, f"{offset} bins: no rectangular leakage to suppress ({rect:.3e})"
    assert tukey < 0.4 * rect, f"{offset} bins: Tukey {tukey:.3e} vs rectangular {rect:.3e} (need <2.5x)"


def test_suppression_grows_with_bin_offset(spectra):
    """A tapered window's sidelobes fall off faster than a rectangular gate's, so suppression
    must increase with distance from the tone. Structural, no calibrated threshold: catches a
    window that is applied but wrong (bad alpha, wrong axis) where a flat ratio bound would not.
    """
    supp = [
        (spectra["rect"][i] / spectra["rect"][0]) / (spectra["tukey"][i] / spectra["tukey"][0])
        for i in range(1, len(_BIN_OFFSETS) + 1)
    ]
    assert supp == sorted(supp), f"suppression not monotonic in bin offset: {supp}"
