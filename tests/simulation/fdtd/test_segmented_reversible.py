"""Tests for ``GradientConfig(recording_mode="segmented")`` (not part of the branch; optional for the PR).

Intended location: ``tests/simulation/fdtd/test_segmented_reversible.py``. Runs on CPU in about a minute.

Segmented mode regenerates the PML interface record slice by slice during the backward pass instead of recording it
for the whole run. It replays exactly what full mode records, so its gradient must equal the full-mode gradient with
the same ``num_checkpoints_reversible`` bit for bit, and match the exact (checkpointed) autodiff gradient.
"""

from contextlib import contextmanager
from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.core.grid import UniformGrid
from fdtdx.fdtd.fdtd import _reversible_slice_boundaries
from fdtdx.interfaces.recorder import Recorder

_RESOLUTION = 50e-9
_SIM_TIME = 30e-15
_PML_CELLS = 4
_VOLUME_CELLS = 8


@contextmanager
def _x64_enabled():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _gradient_config(method, num_checkpoints_reversible=0, recording_mode="full"):
    if method == "checkpointed":
        return GradientConfig(method="checkpointed", num_checkpoints=16)
    return GradientConfig(
        method="reversible",
        recorder=Recorder(modules=[]),
        num_checkpoints_reversible=num_checkpoints_reversible,
        recording_mode=recording_mode,
    )


def _build(gradient_config, sim_time=_SIM_TIME):
    """PML on all faces (so the interface record is not empty), a lossy slab, a dipole and a phasor detector."""
    config = SimulationConfig(
        time=sim_time,
        grid=UniformGrid(spacing=_RESOLUTION),
        backend="cpu",
        dtype=jnp.float64,
        courant_factor=0.99,
        gradient_config=gradient_config,
    )
    n = _VOLUME_CELLS + 2 * _PML_CELLS
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(n, n, n))
    objects.append(volume)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML_CELLS), volume
    )
    objects.extend(bound_dict.values())
    constraints.extend(c_list)
    slab = fdtdx.UniformMaterialObject(
        name="slab",
        partial_grid_shape=(None, None, 2),
        material=fdtdx.Material(permittivity=2.0, electric_conductivity=1e3),
    )
    constraints += [
        slab.same_size(volume, axes=(0, 1)),
        slab.place_at_center(volume, axes=(0, 1)),
        slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(n // 2,)),
    ]
    objects.append(slab)
    wave = fdtdx.WaveCharacter(frequency=c0 / 800e-9)
    source = fdtdx.PointDipoleSource(
        name="dip", partial_grid_shape=(1, 1, 1), wave_character=wave, polarization=0, amplitude=1.0
    )
    constraints.append(
        source.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(n // 2, n // 2, _PML_CELLS + 1))
    )
    objects.append(source)
    detector = fdtdx.PhasorDetector(
        name="phasor", partial_grid_shape=(None, None, 1), wave_characters=(wave,), components=("Ex", "Ey"), plot=False
    )
    constraints += [
        detector.same_size(volume, axes=(0, 1)),
        detector.place_at_center(volume, axes=(0, 1)),
        detector.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(n - _PML_CELLS - 2,)),
    ]
    objects.append(detector)
    key = jax.random.PRNGKey(0)
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)
    return obj, arrays, config, key


def _grad(gradient_config, sim_time=_SIM_TIME):
    obj, arrays, config, key = _build(gradient_config, sim_time)

    def loss(inv_eps):
        _, out = fdtdx.run_fdtd(arrays.aset("inv_permittivities", inv_eps), obj, config, key, show_progress=False)
        return jnp.sum(jnp.abs(out.detector_states["phasor"]["phasor"]) ** 2) + jnp.sum(out.fields.E**2)

    value, grad = jax.value_and_grad(loss)(arrays.inv_permittivities)
    # the reversible method does not reconstruct the gradient inside the PML; compare outside it
    p = _PML_CELLS
    return value, np.asarray(grad)[..., p:-p, p:-p, p:-p], config


class TestGradientConfig:
    def test_default_is_full(self):
        assert GradientConfig(method="reversible", recorder=Recorder(modules=[])).recording_mode == "full"

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(method="reversible", recording_mode="segmented"), "num_checkpoints_reversible >= 1"),
            (
                dict(method="reversible", recording_mode="partial", num_checkpoints_reversible=2),
                "recording_mode must be",
            ),
            (
                dict(
                    method="checkpointed", num_checkpoints=4, recording_mode="segmented", num_checkpoints_reversible=2
                ),
                "requires method='reversible'",
            ),
        ],
    )
    def test_invalid(self, kwargs, match):
        if kwargs["method"] == "reversible":
            kwargs = dict(kwargs, recorder=Recorder(modules=[]))
        with pytest.raises(Exception, match=match):
            GradientConfig(**kwargs)

    def test_time_step_filter_rejected(self):
        with pytest.raises(Exception, match="time-step filters"):
            GradientConfig(
                method="reversible",
                recorder=Recorder(modules=[fdtdx.LinearReconstructEveryK(k=2)]),
                num_checkpoints_reversible=3,
                recording_mode="segmented",
            )

    def test_recorder_time_steps_is_longest_slice(self):
        for total in range(1, 300):
            for n in range(1, total):
                cfg = _gradient_config("reversible", n, "segmented")
                b = _reversible_slice_boundaries(total, n + 1)
                assert cfg.recorder_time_steps(total) == max(hi - lo for lo, hi in pairwise(b))
        assert _gradient_config("reversible", 5).recorder_time_steps(123) == 123


class TestSegmentedGradient:
    def test_recorder_holds_one_slice(self):
        _, arrays, config, _ = _build(_gradient_config("reversible", 7, "segmented"))
        expected = -(-config.time_steps_total // 8)
        assert {v.shape[0] for v in arrays.recording_state.data.values()} == {expected}

    @pytest.mark.parametrize("num_checkpoints", [1, 4])
    def test_equals_full_mode_and_autodiff(self, num_checkpoints):
        with _x64_enabled():
            v_full, g_full, _ = _grad(_gradient_config("reversible", num_checkpoints, "full"))
            v_seg, g_seg, _ = _grad(_gradient_config("reversible", num_checkpoints, "segmented"))
            _, g_ref, _ = _grad(_gradient_config("checkpointed"))
            assert v_seg == v_full
            np.testing.assert_array_equal(g_seg, g_full)
            assert np.max(np.abs(g_seg - g_ref)) <= 1e-6 * np.max(np.abs(g_ref))

    def test_one_step_slices(self):
        with _x64_enabled():
            sim_time = 2e-15
            _, _, config = _grad(_gradient_config("reversible", 0, "full"), sim_time)
            n = config.time_steps_total - 1
            _, g_full, _ = _grad(_gradient_config("reversible", n, "full"), sim_time)
            _, g_seg, _ = _grad(_gradient_config("reversible", n, "segmented"), sim_time)
            np.testing.assert_array_equal(g_seg, g_full)


class TestMisuse:
    def test_recorder_too_short_raises(self):
        obj, arrays, config, key = _build(_gradient_config("reversible", 7, "segmented"))
        full = config.aset(
            "gradient_config",
            _gradient_config("reversible", 0, "full").aset("recorder", config.gradient_config.recorder),
        )
        with pytest.raises(Exception, match="The recorder holds"):
            fdtdx.run_fdtd(arrays, obj, full, key, show_progress=False)

    def test_full_backward_raises(self):
        obj, arrays, config, key = _build(_gradient_config("reversible", 3, "segmented"))
        state = fdtdx.run_fdtd(arrays, obj, config, key, show_progress=False)
        with pytest.raises(Exception, match="full_backward replays"):
            fdtdx.full_backward(state, obj, config, key)
