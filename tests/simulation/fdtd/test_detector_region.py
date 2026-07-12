"""Simulation-level checks for region-restricted detector recording.

- Autodiff safety: a loss built on a detector's recorded output must produce a finite, nonzero
  gradient w.r.t. ``inv_permittivities`` through ``run_fdtd`` (region recording is the path inverse
  design relies on).
- DFT subsampling physics: an auto-subsampled phasor must match every-step recording in a full
  simulation (the deterministic single-tone version of this check lives in
  tests/unit/objects/detectors/test_phasor_subsample.py::test_subsample_recovers_amplitude).
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.core.grid import UniformGrid
from fdtdx.fdtd.container import ArrayContainer, FieldState
from fdtdx.fdtd.fdtd import reversible_fdtd
from fdtdx.interfaces.recorder import Recorder

_RES = 50e-9
_SIM_TIME = 60e-15
_VOL = 8
_PML = 2
_FREQ = c0 / 800e-9


def _build(dft_subsample=1, with_gradient=True):
    config = SimulationConfig(
        time=_SIM_TIME,
        grid=UniformGrid(spacing=_RES),
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=None,
    )
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(_VOL, _VOL, _VOL))
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML,
        override_types={f: "periodic" for f in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")},
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    source = fdtdx.PointDipoleSource(
        name="dip",
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(frequency=_FREQ),
        polarization=0,
        amplitude=1.0,
    )
    constraints.append(
        source.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(_VOL // 2,) * 3)
    )
    objects.append(source)

    det = fdtdx.PhasorDetector(
        name="det",
        partial_grid_shape=(4, 4, 4),  # centered → interior on the 8-cell domain
        wave_characters=(fdtdx.WaveCharacter(frequency=_FREQ),),
        dft_subsample=dft_subsample,
        plot=False,
    )
    constraints.append(det.place_at_center(volume, axes=(0, 1, 2)))
    objects.append(det)

    key = jax.random.PRNGKey(0)
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)

    if with_gradient:
        recorder = Recorder(modules=[])
        recorder, recording_state = recorder.init_state(
            input_shape_dtypes={}, max_time_steps=config.time_steps_total, backend="cpu"
        )
        config = config.aset("gradient_config", GradientConfig(method="reversible", recorder=recorder))
        arrays = arrays.aset("recording_state", recording_state)
    return obj, arrays, config


def test_dft_subsample_matches_every_step_recording():
    """An auto-subsampled phasor matches every-step recording in a full FDTD run."""
    key = jax.random.PRNGKey(1)

    def _run(dft_subsample):
        obj, arrays, config = _build(dft_subsample=dft_subsample, with_gradient=False)
        det = next(d for d in obj.forward_detectors if d.name == "det")
        _, out = fdtdx.run_fdtd(arrays=arrays, objects=obj, config=config, key=key, show_progress=False)
        return det, np.asarray(out.detector_states["det"]["phasor"][0])

    _, exact = _run(1)
    det_auto, subsampled = _run("auto")
    assert det_auto._dft_stride > 1  # sanity: subsampling is actually active

    scale = np.max(np.abs(exact))
    assert scale > 0
    assert np.max(np.abs(subsampled - exact)) / scale < 0.05


def test_detector_loss_gradient_is_finite_and_nonzero():
    obj, arrays, config = _build(dft_subsample=1, with_gradient=True)
    key = jax.random.PRNGKey(1)

    def loss_fn(inv_permittivities):
        a = ArrayContainer(
            fields=FieldState(
                E=arrays.fields.E, H=arrays.fields.H, psi_E=arrays.fields.psi_E, psi_H=arrays.fields.psi_H
            ),
            inv_permittivities=inv_permittivities,
            inv_permeabilities=arrays.inv_permeabilities,
            detector_states=arrays.detector_states,
            recording_state=arrays.recording_state,
            electric_conductivity=arrays.electric_conductivity,
            magnetic_conductivity=arrays.magnetic_conductivity,
        )
        _, out = reversible_fdtd(a, obj, config, key, show_progress=False)
        phasor = out.detector_states["det"]["phasor"]
        return jnp.sum(jnp.abs(phasor) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(arrays.inv_permittivities)
    assert jnp.isfinite(loss)
    assert loss > 0, "Detector-based loss should be nonzero with a driven source"
    assert jnp.all(jnp.isfinite(grads)), "Gradient through region detector recording must be finite"
    assert jnp.any(grads != 0), "Gradient must flow through the region-restricted recording path"
