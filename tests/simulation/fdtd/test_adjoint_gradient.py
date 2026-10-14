"""Simulation tests for the frequency-domain adjoint gradient method (``GradientConfig(method="adjoint")``).

Every test compares the adjoint gradient of a phasor-detector loss with the gradient of the exact
checkpointed path. The adjoint method is exact once the fields have decayed (pulsed sources) or are
periodic in the DFT window (CW sources); the scenes are small PML boxes in which both hold to float
precision, so the tolerances are close to machine precision. Scenes with pulsed sources run in both
adjoint modes ("adjoint" is the default frequency mode, "exact" the exact mode).
"""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.constants import c as c0

_RESOLUTION = 50e-9
_WAVELENGTH = 800e-9
_N = 16
_PML = 4
_PULSE_TIME = 100e-15
_CW_TIME = 200e-15

_MODES = pytest.mark.parametrize("mode", ["adjoint", "exact"])


@contextmanager
def _x64_enabled():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _pulse_profile():
    return fdtdx.GaussianPulseProfile(
        spectral_width=fdtdx.WaveCharacter(frequency=0.15 * c0 / _WAVELENGTH),
        center_wave=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
    )


def _box(objects, constraints, shape=(_N, _N, _N), override_types=None, thickness=_PML):
    volume = fdtdx.SimulationVolume(partial_grid_shape=shape)
    objects.append(volume)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=thickness, override_types=override_types or {})
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)
    return volume


def _device(name="dev", shape=(4, 4, 3), corner=(6, 6, 7), permittivity=4.0):
    device = fdtdx.Device(
        name=name,
        partial_grid_shape=shape,
        partial_voxel_grid_shape=(1, 1, 1),
        materials={"air": fdtdx.Material(permittivity=1.0), "die": fdtdx.Material(permittivity=permittivity)},
        param_transforms=[],
    )
    return device, device.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=corner)


def _dipole(profile, corner=(7, 7, 5), polarization=0, name="dip"):
    source = fdtdx.PointDipoleSource(
        name=name,
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        polarization=polarization,
        temporal_profile=profile,
    )
    return source, source.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=corner)


def _phasor(wavelengths, cw, corner=(5, 5, 11), shape=(6, 6, 1), sim_time=_PULSE_TIME, **kwargs):
    detector = fdtdx.PhasorDetector(
        name="ph",
        partial_grid_shape=shape,
        wave_characters=[fdtdx.WaveCharacter(wavelength=w) for w in wavelengths],
        scaling_mode="continuous" if cw else "pulse",
        plot=False,
        switch=fdtdx.OnOffSwitch(start_time=sim_time / 2) if cw else fdtdx.OnOffSwitch(),
        **kwargs,
    )
    return detector, detector.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=corner)


def _place(objects, constraints, config):
    key = jax.random.PRNGKey(0)
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    params = jax.tree.map(lambda p: jnp.full_like(p, 0.5), params)
    return obj, arrays, params, config


def _config(sim_time, dtype, method, symmetry=(0, 0, 0)):
    """``method`` is "checkpointed", "adjoint" (frequency mode) or "exact" (adjoint method, exact mode)."""
    if method == "checkpointed":
        grad_cfg = fdtdx.GradientConfig(method="checkpointed", num_checkpoints=40)
    elif method == "exact":
        grad_cfg = fdtdx.GradientConfig(method="adjoint", adjoint_mode="exact")
    else:
        grad_cfg = fdtdx.GradientConfig(method="adjoint")
    return fdtdx.SimulationConfig(
        time=sim_time,
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        backend="cpu",
        dtype=dtype,
        gradient_config=grad_cfg,
        symmetry=symmetry,
    )


def _phasor_power(arrays, objects):
    del objects
    return jnp.sum(jnp.abs(arrays.detector_states["ph"]["phasor"]) ** 2)


def _gradients(build, loss=_phasor_power, mode="adjoint"):
    """Loss and parameter gradient for the checkpointed method and the adjoint method in ``mode``."""
    results = {}
    for method in ("checkpointed", mode):
        obj, arrays, params, config = build(method)
        key = jax.random.PRNGKey(1)

        def loss_fn(params, arrays=arrays, obj=obj, config=config, key=key):
            arr, objs, _ = fdtdx.apply_params(arrays, obj, params, key)
            _, out = fdtdx.run_fdtd(arr, objs, config, key, show_progress=False)
            return loss(out, objs)

        value, grad = jax.jit(jax.value_and_grad(loss_fn))(params)
        name = "checkpointed" if method == "checkpointed" else "adjoint"
        results[name] = (float(value), np.concatenate([np.ravel(np.asarray(g)) for g in jax.tree.leaves(grad)]))
    return results


def _assert_match(results, rtol):
    (loss_c, grad_c), (loss_a, grad_a) = results["checkpointed"], results["adjoint"]
    assert loss_a == pytest.approx(loss_c, rel=1e-6)
    assert np.linalg.norm(grad_c) > 0
    rel = np.linalg.norm(grad_a - grad_c) / np.linalg.norm(grad_c)
    assert rel < rtol, f"adjoint vs checkpointed gradient: relative error {rel:.3e} >= {rtol:.1e}"


def _dipole_scene(cw, dtype, wavelengths=(760e-9, 800e-9, 850e-9), sim_time=None):
    sim_time = sim_time or (_CW_TIME if cw else _PULSE_TIME)

    def build(method):
        objects, constraints = [], []
        _box(objects, constraints)
        profile = fdtdx.SingleFrequencyProfile() if cw else _pulse_profile()
        for obj, constraint in (
            _dipole(profile),
            _device(),
            _phasor(wavelengths if not cw else (_WAVELENGTH,), cw, sim_time=sim_time, components=("Ex", "Ey", "Hy")),
        ):
            objects.append(obj)
            constraints.append(constraint)
        return _place(objects, constraints, _config(sim_time, dtype, method))

    return build


@pytest.mark.parametrize("cw", [False, True], ids=["pulse", "cw"])
def test_dipole_float32(cw):
    _assert_match(_gradients(_dipole_scene(cw, jnp.float32)), rtol=1e-4 if not cw else 5e-4)


@pytest.mark.parametrize("cw", [False, True], ids=["pulse", "cw"])
def test_dipole_float64(cw):
    with _x64_enabled():
        _assert_match(_gradients(_dipole_scene(cw, jnp.float64)), rtol=1e-8 if not cw else 2e-5)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_exact_dipole(dtype):
    if dtype == "float32":
        _assert_match(_gradients(_dipole_scene(False, jnp.float32), mode="exact"), rtol=1e-5)
    else:
        with _x64_enabled():
            _assert_match(_gradients(_dipole_scene(False, jnp.float64), mode="exact"), rtol=1e-8)


def test_exact_dense_frequencies_in_one_adjoint_run():
    """Twelve close frequencies, which the frequency mode splits into many adjoint runs, need one exact run."""
    from fdtdx.fdtd.adjoint import _build_spec

    wavelengths = tuple(np.linspace(760e-9, 850e-9, 12))
    build = _dipole_scene(False, jnp.float32, wavelengths=wavelengths)
    obj, _, _, config = build("adjoint")
    assert len(_build_spec(obj, config).groups) > 4
    _assert_match(_gradients(build, mode="exact"), rtol=1e-5)


def test_exact_detector_windows():
    """A gated, subsampled and apodized detector: the exact mode uses its true per-step weights."""

    def build(method):
        objects, constraints = [], []
        _box(objects, constraints)
        detector = fdtdx.PhasorDetector(
            name="ph",
            partial_grid_shape=(6, 6, 1),
            wave_characters=[fdtdx.WaveCharacter(wavelength=w) for w in (760e-9, 850e-9)],
            scaling_mode="pulse",
            plot=False,
            components=("Ex", "Ey", "Hy"),
            switch=fdtdx.OnOffSwitch(start_time=10e-15, end_time=90e-15),
            dft_subsample=3,
            apodization=fdtdx.TukeyWindow(start_time=15e-15, end_time=80e-15),
        )
        constraint = detector.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(5, 5, 11))
        for obj, c in (_dipole(_pulse_profile()), _device(), (detector, constraint)):
            objects.append(obj)
            constraints.append(c)
        return _place(objects, constraints, _config(_PULSE_TIME, jnp.float32, method))

    _assert_match(_gradients(build, mode="exact"), rtol=1e-5)


def test_exact_mode_rejects_cw_sources():
    objects, constraints = [], []
    _box(objects, constraints)
    for obj, constraint in (_dipole(fdtdx.SingleFrequencyProfile()), _device(), _phasor((800e-9,), True)):
        objects.append(obj)
        constraints.append(constraint)
    obj, arrays, params, config = _place(objects, constraints, _config(_CW_TIME, jnp.float32, "exact"))
    with pytest.raises(ValueError, match="pulsed sources"):
        _grad_of(_phasor_power, obj, arrays, params, config)


def test_frequencies_sharing_one_adjoint_run():
    """Well separated frequencies are driven by one pre-mixed adjoint pulse."""
    from fdtdx.fdtd.adjoint import _build_spec

    build = _dipole_scene(False, jnp.float32, wavelengths=(600e-9, 800e-9))
    obj, _, _, config = build("adjoint")
    assert len(_build_spec(obj, config).groups) == 1
    _assert_match(_gradients(build), rtol=1e-4)


def test_adjoint_frequencies_override():
    """An explicit frequency list restricts the adjoint runs; a loss on those frequencies is unaffected."""
    scene = _dipole_scene(False, jnp.float32)

    def build(method):
        obj, arrays, params, config = scene(method)
        if method == "adjoint":
            grad_cfg = config.gradient_config.aset("adjoint_frequencies", (fdtdx.WaveCharacter(wavelength=800e-9),))
            config = config.aset("gradient_config", grad_cfg)
        return obj, arrays, params, config

    def loss(arrays, objects):
        del objects
        return jnp.sum(jnp.abs(arrays.detector_states["ph"]["phasor"][:, 1]) ** 2)

    _assert_match(_gradients(build, loss), rtol=1e-4)


@_MODES
def test_periodic_plane_source_and_flux(mode):
    """A TFSF plane source in a periodic cell, a device spanning the periodic axes and a phasor flux loss.

    The device and the detector touch the periodic edges, so the design DFT and the adjoint source cover
    the full periodic axes and the detector reads the full-domain co-location.
    """

    def build(method):
        objects, constraints = [], []
        volume = _box(
            objects,
            constraints,
            shape=(8, 8, 24),
            override_types={"min_x": "periodic", "max_x": "periodic", "min_y": "periodic", "max_y": "periodic"},
        )
        source = fdtdx.UniformPlaneSource(
            partial_grid_shape=(None, None, 1),
            wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
            direction="+",
            fixed_E_polarization_vector=(1, 0, 0),
            temporal_profile=_pulse_profile(),
        )
        constraints += [
            source.same_size(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_PML + 2,)),
        ]
        device, _ = _device(shape=(8, 8, 3))
        constraints += [
            device.same_size(volume, axes=(0, 1)),
            device.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(10,)),
        ]
        detector = fdtdx.PhasorPoyntingFluxDetector(
            name="flux",
            partial_grid_shape=(None, None, 1),
            wave_characters=[fdtdx.WaveCharacter(wavelength=w) for w in (760e-9, 850e-9)],
            direction="+",
            scaling_mode="pulse",
        )
        constraints += [
            detector.same_size(volume, axes=(0, 1)),
            detector.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(16,)),
        ]
        objects += [source, device, detector]
        return _place(objects, constraints, _config(_PULSE_TIME, jnp.float32, method))

    def loss(arrays, objects):
        return jnp.sum(objects["flux"].compute_poynting_flux(arrays.detector_states["flux"]))

    _assert_match(_gradients(build, loss, mode), rtol=1e-4)


@_MODES
def test_mode_source_and_mode_overlap(mode):
    """Waveguide with a mode source and a mode-overlap objective through a device."""

    def build(method):
        objects, constraints = [], []
        volume = _box(objects, constraints, shape=(28, 16, 16))
        core = fdtdx.UniformMaterialObject(
            name="core", partial_grid_shape=(None, 4, 4), material=fdtdx.Material(permittivity=6.0)
        )
        constraints += [core.same_size(volume, axes=(0,)), core.place_at_center(volume, axes=(1, 2))]
        source = fdtdx.ModePlaneSource(
            partial_grid_shape=(1, None, None),
            wave_character=fdtdx.WaveCharacter(wavelength=1.2e-6),
            direction="+",
            temporal_profile=fdtdx.GaussianPulseProfile(
                spectral_width=fdtdx.WaveCharacter(frequency=0.1 * c0 / 1.2e-6),
                center_wave=fdtdx.WaveCharacter(wavelength=1.2e-6),
            ),
        )
        constraints += [
            source.same_size(volume, axes=(1, 2)),
            source.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_PML + 1,)),
        ]
        device, _ = _device(shape=(6, 8, 8), permittivity=6.0)
        constraints += [
            device.place_at_center(volume, axes=(1, 2)),
            device.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(11,)),
        ]
        detector = fdtdx.ModeOverlapDetector(
            name="mode",
            partial_grid_shape=(1, None, None),
            wave_characters=[fdtdx.WaveCharacter(wavelength=w) for w in (1.15e-6, 1.25e-6)],
            direction="+",
            scaling_mode="pulse",
        )
        constraints += [
            detector.same_size(volume, axes=(1, 2)),
            detector.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(21,)),
        ]
        objects += [core, source, device, detector]
        return _place(objects, constraints, _config(150e-15, jnp.float32, method))

    def loss(arrays, objects):
        return -jnp.sum(jnp.abs(objects["mode"].compute_overlap(arrays.detector_states["mode"])) ** 2)

    _assert_match(_gradients(build, loss, mode), rtol=1e-4)


@_MODES
def test_mirror_symmetry(mode):
    """With config.symmetry both methods run on the reduced domain and must agree there.

    An x-polarized plane wave along z is mirror symmetric with an electric plane normal to x (E_x normal)
    and a magnetic plane normal to y (E_x tangential); the device and the detector are centred.
    """

    def build(method):
        objects, constraints = [], []
        volume = _box(objects, constraints)
        source = fdtdx.UniformPlaneSource(
            partial_grid_shape=(None, None, 1),
            wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
            direction="+",
            fixed_E_polarization_vector=(1, 0, 0),
            temporal_profile=_pulse_profile(),
        )
        constraints += [
            source.same_size(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_PML + 1,)),
        ]
        objects.append(source)
        for obj, constraint in (
            _device(corner=(6, 6, 7)),
            _phasor((760e-9, 850e-9), False, corner=(4, 4, 11), shape=(8, 8, 1), components=("Ex", "Hy")),
        ):
            objects.append(obj)
            constraints.append(constraint)
        return _place(objects, constraints, _config(_PULSE_TIME, jnp.float32, method, symmetry=(-1, 1, 0)))

    _assert_match(_gradients(build, mode=mode), rtol=1e-4)


@_MODES
def test_anisotropic_and_conductive_background(mode):
    def build(method):
        objects, constraints = [], []
        volume = _box(objects, constraints)
        lossy = fdtdx.UniformMaterialObject(
            name="lossy",
            partial_grid_shape=(None, None, 2),
            material=fdtdx.Material(permittivity=(2.0, 3.0, 2.5), electric_conductivity=2e4),
        )
        constraints += [
            lossy.same_size(volume, axes=(0, 1)),
            lossy.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(10,)),
        ]
        objects.append(lossy)
        for obj, constraint in (
            _dipole(_pulse_profile()),
            _device(corner=(6, 6, 7)),
            _phasor((760e-9, 850e-9), False, corner=(5, 5, 12)),
        ):
            objects.append(obj)
            constraints.append(constraint)
        return _place(objects, constraints, _config(_PULSE_TIME, jnp.float32, method))

    _assert_match(_gradients(build, mode=mode), rtol=1e-4)


@_MODES
def test_inverse_permeability_gradient_inside_devices(mode):
    """With magnetic materials the inverse permeability is an array; its gradient is returned inside devices."""
    results = {}
    for method in ("checkpointed", mode):
        objects, constraints = [], []
        volume = _box(objects, constraints)
        magnetic = fdtdx.UniformMaterialObject(
            name="magnetic", partial_grid_shape=(None, None, 2), material=fdtdx.Material(permeability=1.5)
        )
        constraints += [
            magnetic.same_size(volume, axes=(0, 1)),
            magnetic.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(12,)),
        ]
        objects.append(magnetic)
        for obj, constraint in (_dipole(_pulse_profile()), _device(), _phasor((800e-9,), False, corner=(5, 5, 13))):
            objects.append(obj)
            constraints.append(constraint)
        obj, arrays, params, config = _place(objects, constraints, _config(_PULSE_TIME, jnp.float32, method))
        key = jax.random.PRNGKey(1)
        arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)
        assert isinstance(arrays.inv_permeabilities, jax.Array) and arrays.inv_permeabilities.ndim == 4

        def loss_fn(inv_mu, arrays=arrays, obj=obj, config=config, key=key):
            _, out = fdtdx.run_fdtd(arrays.aset("inv_permeabilities", inv_mu), obj, config, key, show_progress=False)
            return _phasor_power(out, obj)

        grad = jax.jit(jax.grad(loss_fn))(arrays.inv_permeabilities)
        results[method] = np.asarray(grad[:, *obj["dev"].grid_slice]).ravel()
    rel = np.linalg.norm(results[mode] - results["checkpointed"]) / np.linalg.norm(results["checkpointed"])
    assert rel < 1e-4, f"inverse permeability gradient: relative error {rel:.3e}"


@_MODES
def test_arrays_as_jit_arguments(mode):
    """Parameters, arrays and key passed through jit as arguments, as in the optimization examples."""
    obj, arrays, params, config = _dipole_scene(False, jnp.float32, wavelengths=(800e-9,))(mode)
    reference = _gradients(_dipole_scene(False, jnp.float32, wavelengths=(800e-9,)))["checkpointed"][1]

    def loss_fn(params, arrays, key):
        arr, objs, _ = fdtdx.apply_params(arrays, obj, params, key)
        _, out = fdtdx.run_fdtd(arr, objs, config, key, show_progress=False)
        return _phasor_power(out, objs)

    grad = jax.jit(jax.grad(loss_fn), donate_argnames=["arrays"])(params, arrays, jax.random.PRNGKey(1))
    grad = np.concatenate([np.ravel(np.asarray(g)) for g in jax.tree.leaves(grad)])
    assert np.linalg.norm(grad - reference) / np.linalg.norm(reference) < 1e-4


def _error_scene(extra=None, device_in_pml=False, detector=None, cw_and_pulse=False, method="adjoint"):
    objects, constraints = [], []
    _box(objects, constraints)
    entries = [_dipole(_pulse_profile()), _device(corner=(1, 6, 7) if device_in_pml else (6, 6, 7))]
    entries.append(detector or _phasor((800e-9,), False))
    if cw_and_pulse:
        entries.append(_dipole(fdtdx.SingleFrequencyProfile(), corner=(7, 7, 4), name="cw_dip"))
    for obj, constraint in entries:
        objects.append(obj)
        constraints.append(constraint)
    for obj, constraint in extra or ():
        objects.append(obj)
        constraints.append(constraint)
    return _place(objects, constraints, _config(30e-15, jnp.float32, method))


def _grad_of(loss, obj, arrays, params, config):
    key = jax.random.PRNGKey(1)

    def loss_fn(params):
        arr, objs, _ = fdtdx.apply_params(arrays, obj, params, key)
        _, out = fdtdx.run_fdtd(arr, objs, config, key, show_progress=False)
        return loss(out, objs)

    return jax.grad(loss_fn)(params)


def test_nonlinear_detector_in_loss_raises():
    flux = fdtdx.PoyntingFluxDetector(name="flux", partial_grid_shape=(6, 6, 1), direction="+")
    obj, arrays, params, config = _error_scene(
        extra=[(flux, flux.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(5, 5, 12)))]
    )
    with pytest.raises(NotImplementedError, match="PhasorPoyntingFluxDetector"):
        _grad_of(lambda out, _: jnp.sum(out.detector_states["flux"]["poynting_flux"]), obj, arrays, params, config)


@_MODES
def test_final_field_in_loss_raises(mode):
    obj, arrays, params, config = _error_scene(method=mode)
    with pytest.raises(NotImplementedError, match="end of the simulation"):
        _grad_of(lambda out, _: jnp.sum(out.fields.E**2), obj, arrays, params, config)


def test_unused_nonlinear_detector_is_fine():
    """Detectors the loss does not depend on carry symbolic-zero cotangents and are ignored."""
    flux = fdtdx.PoyntingFluxDetector(name="flux", partial_grid_shape=(6, 6, 1), direction="+")
    obj, arrays, params, config = _error_scene(
        extra=[(flux, flux.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(5, 5, 12)))]
    )
    grad = _grad_of(_phasor_power, obj, arrays, params, config)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grad))


def test_device_in_pml_raises():
    obj, arrays, params, config = _error_scene(device_in_pml=True)
    with pytest.raises(ValueError, match="overlaps the PML"):
        _grad_of(_phasor_power, obj, arrays, params, config)


def test_mixed_cw_and_pulsed_sources_raise():
    obj, arrays, params, config = _error_scene(cw_and_pulse=True)
    with pytest.raises(ValueError, match="mix"):
        _grad_of(_phasor_power, obj, arrays, params, config)
