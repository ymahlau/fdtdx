"""Integration tests of the adjoint gradient method on tiny scenes.

The scenes are a few hundred steps on a 12^3 grid. The pulsed fields have not fully decayed in that time,
which the exact mode tolerates much better than the frequency mode, so the comparisons with the
checkpointed gradient use mode-specific tolerances; the precise comparisons are in
``tests/simulation/fdtd/test_adjoint_gradient.py``. Error paths are traced with ``jax.eval_shape`` and
do not run the simulation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from loguru import logger

import fdtdx
from fdtdx.constants import c as c0

_RESOLUTION = 100e-9
_WAVELENGTH = 1.0e-6
_N = 12
_PML = 3
_PULSE_TIME = 40e-15
_CW_TIME = 120e-15


def _pulse_profile():
    return fdtdx.GaussianPulseProfile(
        spectral_width=fdtdx.WaveCharacter(frequency=0.2 * c0 / _WAVELENGTH),
        center_wave=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
    )


def _at(obj, corner):
    return obj, obj.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=corner)


def _dipole(profile, corner=(5, 5, 3), name="dip"):
    source = fdtdx.PointDipoleSource(
        name=name,
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        polarization=0,
        temporal_profile=profile,
    )
    return _at(source, corner)


def _device(corner=(5, 5, 5)):
    device = fdtdx.Device(
        name="dev",
        partial_grid_shape=(2, 2, 2),
        partial_voxel_grid_shape=(1, 1, 1),
        materials={"air": fdtdx.Material(permittivity=1.0), "die": fdtdx.Material(permittivity=4.0)},
        param_transforms=[],
    )
    return _at(device, corner)


def _phasor(cw, wavelengths=(0.9e-6, 1.1e-6), corner=(4, 4, 8), name="ph", **kwargs):
    if cw:
        # a CW phasor is taken in the steady state
        kwargs.setdefault("switch", fdtdx.OnOffSwitch(start_time=_CW_TIME / 2))
    detector = fdtdx.PhasorDetector(
        name=name,
        partial_grid_shape=(4, 4, 1),
        wave_characters=[fdtdx.WaveCharacter(wavelength=w) for w in ((_WAVELENGTH,) if cw else wavelengths)],
        scaling_mode="continuous" if cw else "pulse",
        plot=False,
        components=("Ex", "Ey", "Hy"),
        **kwargs,
    )
    return _at(detector, corner)


def _block(name="block", corner=(4, 4, 7), shape=(4, 4, 1), **material):
    block = fdtdx.UniformMaterialObject(
        name=name, partial_grid_shape=shape, material=fdtdx.Material(**(material or {"permittivity": 2.0}))
    )
    return _at(block, corner)


def _gradient_config(method, **adjoint_options):
    if method == "checkpointed":
        return fdtdx.GradientConfig(method="checkpointed", num_checkpoints=20)
    if method == "default":
        return fdtdx.GradientConfig(method="adjoint", **adjoint_options)
    return fdtdx.GradientConfig(method="adjoint", adjoint_mode=method, **adjoint_options)


def _scene(
    method="default",
    *,
    cw=False,
    entries=None,
    sim_time=None,
    periodic=False,
    complex_fields=False,
    **adjoint_options,
):
    """Place a PML box with a dipole, a device and a phasor detector (or ``entries``)."""
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(_N, _N, _N))
    objects.append(volume)
    overrides = {f: "periodic" for f in ("min_x", "max_x", "min_y", "max_y")} if periodic else {}
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML, override_types=overrides)
    bound_dict, bound_constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(bound_constraints)
    profile = fdtdx.SingleFrequencyProfile() if cw else _pulse_profile()
    for obj, constraint in entries if entries is not None else (_dipole(profile), _device(), _phasor(cw)):
        objects.append(obj)
        constraints.append(constraint)
    config = fdtdx.SimulationConfig(
        time=sim_time or (_CW_TIME if cw else _PULSE_TIME),
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        backend="cpu",
        dtype=jnp.float32,
        gradient_config=_gradient_config(method, **adjoint_options),
        use_complex_fields=True if complex_fields else None,
    )
    key = jax.random.PRNGKey(0)
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    params = jax.tree.map(lambda p: jnp.full_like(p, 0.5), params)
    return obj, arrays, params, config


def _phasor_power(out, objects):
    del objects
    return jnp.sum(jnp.abs(out.detector_states["ph"]["phasor"]) ** 2)


def _loss_fn(obj, arrays, config, loss=_phasor_power, show_progress=False):
    """Loss of the inverse permittivity, the array whose gradient the adjoint method computes."""
    key = jax.random.PRNGKey(1)

    def loss_fn(inv_eps):
        out_state = fdtdx.run_fdtd(
            arrays.aset("inv_permittivities", inv_eps), obj, config, key, show_progress=show_progress
        )
        return loss(out_state[1], obj)

    return loss_fn


def _inv_eps(obj, arrays, params):
    arrays, _, _ = fdtdx.apply_params(arrays, obj, params, jax.random.PRNGKey(1))
    return arrays.inv_permittivities


def _gradient(obj, arrays, params, config, loss=_phasor_power):
    inv_eps = _inv_eps(obj, arrays, params)
    return np.asarray(jax.jit(jax.grad(_loss_fn(obj, arrays, config, loss)))(inv_eps))


def _trace_gradient(obj, arrays, params, config, loss=_phasor_power):
    """Trace the gradient (forward and backward pass) without running the simulation."""
    inv_eps = _inv_eps(obj, arrays, params)
    return jax.eval_shape(jax.grad(_loss_fn(obj, arrays, config, loss)), inv_eps)


def _relative_error_in(region, gradient, reference):
    return np.linalg.norm(gradient[region] - reference[region]) / np.linalg.norm(reference[region])


class _Warnings:
    """Collects loguru warnings."""

    def __enter__(self):
        self.messages = []
        self._sink = logger.add(lambda message: self.messages.append(str(message)), level="WARNING")
        return self

    def __exit__(self, *exc):
        logger.remove(self._sink)

    def contain(self, text):
        return any(text in m for m in self.messages)


@pytest.mark.parametrize("mode, rtol", [("exact", 1e-5), ("frequency", 2e-3)])
def test_pulse_gradient_inside_device(mode, rtol):
    """The adjoint gradient matches the checkpointed one inside the device and is zero elsewhere."""
    obj, arrays, params, config = _scene(mode)
    gradient = _gradient(obj, arrays, params, config)
    reference = _gradient(*_scene("checkpointed"))
    region = (slice(None), *obj["dev"].grid_slice)
    assert _relative_error_in(region, gradient, reference) < rtol
    outside = np.ones(gradient.shape, dtype=bool)
    outside[region] = False
    assert np.all(gradient[outside] == 0)


def test_cw_gradient_uses_the_frequency_mode():
    """CW sources run the frequency mode with its steady-state DFT window under the default config."""
    obj, arrays, params, config = _scene(cw=True)
    gradient = _gradient(obj, arrays, params, config)
    reference = _gradient(*_scene("checkpointed", cw=True))
    region = (slice(None), *obj["dev"].grid_slice)
    assert _relative_error_in(region, gradient, reference) < 2e-4


def test_cw_with_explicit_window_and_frequencies():
    switch = fdtdx.OnOffSwitch(start_time=60e-15)
    entries = (
        _dipole(fdtdx.SingleFrequencyProfile()),
        _device(),
        _phasor(True, switch=fdtdx.OnOffSwitch(start_time=60e-15)),
    )
    obj, arrays, params, config = _scene(
        "frequency",
        cw=True,
        entries=entries,
        adjoint_window=switch,
        adjoint_frequencies=(fdtdx.WaveCharacter(wavelength=_WAVELENGTH),),
    )
    assert np.all(np.isfinite(_gradient(obj, arrays, params, config)))


def test_cw_window_without_time_steps_raises():
    obj, arrays, params, config = _scene("frequency", cw=True, adjoint_window=fdtdx.OnOffSwitch(is_always_off=True))
    with pytest.raises(ValueError, match="contains no time step"):
        _trace_gradient(obj, arrays, params, config)


def test_frequency_mode_options_and_ignored_detector_frequencies():
    """Explicit adjoint frequencies split into several runs; detector frequencies outside them are ignored."""
    frequencies = (fdtdx.WaveCharacter(wavelength=0.9e-6), fdtdx.WaveCharacter(wavelength=0.91e-6))
    obj, arrays, params, config = _scene("frequency", adjoint_frequencies=frequencies, adjoint_time=30e-15)
    with _Warnings() as warnings:
        gradient = _gradient(obj, arrays, params, config)
    assert np.all(np.isfinite(gradient))
    assert warnings.contain("not in GradientConfig.adjoint_frequencies")


@pytest.mark.parametrize("mode", ["exact", "frequency"])
def test_named_gradient_region_and_magnetic_material(mode):
    """A material block as gradient region, without a device; inv_mu is an array (magnetic layer)."""
    entries = (
        _dipole(_pulse_profile()),
        _block(),
        _block("magnetic", corner=(3, 3, 6), shape=(6, 6, 1), permeability=1.5),
        _phasor(False),
    )
    obj, arrays, _, config = _scene(mode, entries=entries, adjoint_gradient_objects=("block",))
    assert isinstance(arrays.inv_permeabilities, jax.Array) and arrays.inv_permeabilities.ndim == 4
    key = jax.random.PRNGKey(1)

    def loss_fn(inv_eps, inv_mu):
        materials = arrays.aset("inv_permittivities", inv_eps).aset("inv_permeabilities", inv_mu)
        _, out = fdtdx.run_fdtd(materials, obj, config, key, show_progress=False)
        return _phasor_power(out, obj)

    grad_eps, grad_mu = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))(arrays.inv_permittivities, arrays.inv_permeabilities)
    region = (slice(None), *obj["block"].grid_slice)
    for grad in (np.asarray(grad_eps), np.asarray(grad_mu)):
        assert np.linalg.norm(grad[region]) > 0
        outside = np.ones(grad.shape, dtype=bool)
        outside[region] = False
        assert np.all(grad[outside] == 0)


@pytest.mark.parametrize("mode", ["exact", "frequency"])
def test_periodic_device_and_closed_surface_detector(mode):
    """A device and a per-face detector spanning a periodic axis; only one face is in the loss.

    The detector touches the domain edge, so its adjoint pattern uses the full-domain co-location.
    """
    detector = fdtdx.ClosedSurfacePhasorPoyntingFluxDetector(
        name="box",
        partial_grid_shape=(_N, 4, 3),
        wave_characters=[fdtdx.WaveCharacter(wavelength=_WAVELENGTH)],
        scaling_mode="pulse",
    )
    device = fdtdx.Device(
        name="dev",
        partial_grid_shape=(_N, 2, 2),
        partial_voxel_grid_shape=(1, 1, 1),
        materials={"air": fdtdx.Material(permittivity=1.0), "die": fdtdx.Material(permittivity=4.0)},
        param_transforms=[],
    )
    entries = (_dipole(_pulse_profile()), _at(device, (0, 5, 4)), _at(detector, (0, 4, 6)))
    obj, arrays, params, config = _scene(mode, entries=entries, periodic=True)

    def loss(out, objects):
        del objects
        return jnp.sum(jnp.abs(out.detector_states["box"]["phasor_axis2_max"]) ** 2)

    gradient = _gradient(obj, arrays, params, config, loss)
    assert np.linalg.norm(gradient[(slice(None), *obj["dev"].grid_slice)]) > 0


@pytest.mark.parametrize("mode", ["exact", "frequency"])
def test_loss_of_material_arrays_only(mode):
    """Without detector cotangents the adjoint run is skipped and the material cotangents pass through."""
    entries = (
        _dipole(_pulse_profile()),
        _device(),
        _block("magnetic", corner=(3, 3, 7), shape=(6, 6, 1), permeability=1.5),
        _phasor(False),
    )
    obj, arrays, _, config = _scene(mode, entries=entries)
    key = jax.random.PRNGKey(1)

    def loss_fn(inv_eps, inv_mu):
        materials = arrays.aset("inv_permittivities", inv_eps).aset("inv_permeabilities", inv_mu)
        _, out = fdtdx.run_fdtd(materials, obj, config, key, show_progress=False)
        return jnp.sum(out.inv_permittivities) + 2 * jnp.sum(out.inv_permeabilities)

    grad_eps, grad_mu = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))(arrays.inv_permittivities, arrays.inv_permeabilities)
    np.testing.assert_allclose(np.asarray(grad_eps), 1.0)
    np.testing.assert_allclose(np.asarray(grad_mu), 2.0)


@pytest.mark.parametrize("mode", ["exact", "frequency"])
def test_forward_run_without_gradient_matches_plain_simulation(mode):
    obj, arrays, params, config = _scene(mode)
    inv_eps = _inv_eps(obj, arrays, params)
    adjoint_loss = float(jax.jit(_loss_fn(obj, arrays, config))(inv_eps))
    plain_loss = float(jax.jit(_loss_fn(obj, arrays, config.aset("gradient_config", None)))(inv_eps))
    assert adjoint_loss == pytest.approx(plain_loss, rel=1e-6)


def test_inverse_detector_does_not_contribute():
    """Inverse detectors record the reverse-time reconstruction of other methods and are skipped."""
    entries = (
        _dipole(_pulse_profile()),
        _device(),
        _phasor(False),
        _phasor(False, corner=(4, 4, 7), name="inv", inverse=True),
    )
    obj, arrays, params, config = _scene("exact", entries=entries)

    def loss(out, objects):
        del objects
        return jnp.sum(jnp.abs(out.detector_states["inv"]["phasor"]) ** 2)

    _trace_gradient(obj, arrays, params, config, loss)


@pytest.mark.parametrize("mode", ["exact", "frequency"])
def test_progress_bars(mode, capsys):
    obj, arrays, params, config = _scene(mode)
    inv_eps = _inv_eps(obj, arrays, params)
    jax.block_until_ready(jax.grad(_loss_fn(obj, arrays, config, show_progress=True))(inv_eps))
    err = capsys.readouterr().err
    assert "FDTD (adjoint)" in err and "FDTD (adjoint, backward)" in err


def test_exact_mode_warns_about_a_broadband_source_and_band_leakage():
    """A noise-like custom source fills the spectrum; a coarse band tolerance leaves forward content outside."""
    rng = np.random.default_rng(0)
    dt = 0.99 * _RESOLUTION / (np.sqrt(3) * c0)
    signal = np.zeros(400)
    signal[5:60] = rng.normal(size=55)
    profile = fdtdx.CustomTimeSignalProfile(signal=jnp.asarray(signal), time_step_duration=dt)
    entries = (_dipole(profile), _device(), _phasor(False))
    obj, arrays, params, config = _scene("exact", entries=entries, adjoint_band_tolerance=0.3)
    with _Warnings() as warnings:
        gradient = _gradient(obj, arrays, params, config)
    assert np.all(np.isfinite(gradient))
    assert warnings.contain("lies outside the band of the exact adjoint mode")
    obj, arrays, params, config = _scene("exact", entries=entries, adjoint_band_tolerance=1e-9)
    with _Warnings() as warnings:
        _trace_gradient(obj, arrays, params, config)
    assert warnings.contain("not band-limited")


def test_frequency_mode_warns_about_a_detector_with_partial_recording():
    detector = _phasor(
        False,
        switch=fdtdx.OnOffSwitch(start_time=10e-15),
        apodization=fdtdx.TukeyWindow(start_time=10e-15, end_time=_PULSE_TIME),
        exact_interpolation=False,
    )
    entries = (_dipole(_pulse_profile()), _device(), detector)
    obj, arrays, params, config = _scene("frequency", entries=entries)
    with _Warnings() as warnings:
        _trace_gradient(obj, arrays, params, config)
    assert warnings.contain("does not record the full run")


def _error_scene(method="default", extra=(), **kwargs):
    entries = (_dipole(_pulse_profile()), _device(), _phasor(False), *extra)
    return _scene(method, entries=entries, **kwargs)


@pytest.mark.parametrize("mode", ["exact", "frequency"])
def test_loss_of_final_fields_raises(mode):
    def loss(out, objects):
        del objects
        return jnp.sum(out.fields.E**2)

    with pytest.raises(NotImplementedError, match="end of the simulation"):
        _trace_gradient(*_error_scene(mode), loss=loss)


@pytest.mark.parametrize("mode", ["exact", "frequency"])
def test_loss_of_time_domain_detector_raises(mode):
    flux = fdtdx.PoyntingFluxDetector(name="flux", partial_grid_shape=(4, 4, 1), direction="+")

    def loss(out, objects):
        del objects
        return jnp.sum(out.detector_states["flux"]["poynting_flux"])

    with pytest.raises(NotImplementedError, match="PhasorPoyntingFluxDetector"):
        _trace_gradient(*_error_scene(mode, extra=(_at(flux, (4, 4, 7)),)), loss=loss)


@pytest.mark.parametrize(
    "kwargs, extra, error, match",
    [
        ({"complex_fields": True}, (), NotImplementedError, "complex fields"),
        ({"adjoint_gradient_objects": ("blok",)}, (), ValueError, "not objects of the simulation"),
        ({"adjoint_gradient_objects": ("pml",)}, (_block("pml", corner=(0, 4, 4)),), ValueError, "overlaps the PML"),
        ({"adjoint_gradient_objects": ("cover",)}, (_block("cover", corner=(4, 4, 3)),), ValueError, "the source"),
        ({"adjoint_frequencies": (fdtdx.WaveCharacter(wavelength=1e-6),)}, (), ValueError, "frequency adjoint mode"),
        (
            {},
            (_dipole(fdtdx.SingleFrequencyProfile(), corner=(6, 6, 3), name="cw"),),
            ValueError,
            "mix",
        ),
        (
            {},
            (
                _block(
                    "lorentz",
                    corner=(4, 4, 7),
                    permittivity=2.0,
                    dispersion=fdtdx.DispersionModel(
                        poles=(fdtdx.LorentzPole(resonance_frequency=3e15, damping=1e14, delta_epsilon=0.5),)
                    ),
                ),
            ),
            NotImplementedError,
            "dispersive",
        ),
    ],
)
def test_invalid_setups_raise(kwargs, extra, error, match):
    with pytest.raises(error, match=match):
        _trace_gradient(*_error_scene(extra=extra, **kwargs))


def test_band_tolerance_with_cw_sources_raises():
    with pytest.raises(ValueError, match="only applies to the exact adjoint mode"):
        _trace_gradient(*_scene(cw=True, adjoint_band_tolerance=1e-4))


def test_simulation_without_device_needs_gradient_objects():
    entries = (_dipole(_pulse_profile()), _block(), _phasor(False))
    with pytest.raises(ValueError, match="adjoint_gradient_objects"):
        _trace_gradient(*_scene(entries=entries))


def test_frequency_mode_without_phasor_detector_raises():
    entries = (_dipole(_pulse_profile()), _device())
    with pytest.raises(ValueError, match="no PhasorDetector"):
        _trace_gradient(
            *_scene("frequency", entries=entries), loss=lambda out, objects: jnp.sum(out.inv_permittivities)
        )


@pytest.mark.parametrize("with_zero_source", [False, True])
def test_exact_mode_without_source_signal_raises(with_zero_source):
    entries = (_device(), _phasor(False))
    if with_zero_source:
        silent = fdtdx.CustomTimeSignalProfile(signal=jnp.zeros(16), time_step_duration=1e-16)
        entries = (_dipole(silent), *entries)
    with pytest.raises(ValueError, match="nonzero time signal"):
        _trace_gradient(*_scene("exact", entries=entries))


def test_too_short_adjoint_time_raises():
    with pytest.raises(ValueError, match="at least one time step"):
        _trace_gradient(*_scene("frequency", adjoint_time=1e-20))


def test_exact_mode_with_traced_source_signal_raises():
    """The band is chosen from the source signals at trace time; a traced custom signal cannot be sampled."""
    from types import SimpleNamespace

    from fdtdx.fdtd.adjoint import _source_spectrum

    obj, _, _, config = _scene("exact")

    def spectrum(signal):
        profile = fdtdx.CustomTimeSignalProfile(signal=signal, time_step_duration=1e-16)
        source = obj["dip"].aset("temporal_profile", profile)
        return _source_spectrum(SimpleNamespace(sources=[source]), config)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="depends on traced values"):
        jax.eval_shape(spectrum, jnp.sin(jnp.arange(64) * 0.3))
