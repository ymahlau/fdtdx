"""Integration tests for ``GradientConfig(recording_mode="segmented")`` on a small scene with a PML.

Segmented mode regenerates the PML interface record slice by slice during the backward pass, so it replays exactly
what full mode records: its gradient must equal the full-mode gradient with the same ``num_checkpoints_reversible``
bit for bit. ``place_objects`` sizes the recorder to one slice and resolves ``num_checkpoints_reversible="auto"``.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.core.grid import UniformGrid
from fdtdx.interfaces.recorder import Recorder

_PML_CELLS = 4
_VOLUME_CELLS = 8


def _gradient_config(num_checkpoints_reversible, recording_mode):
    return GradientConfig(
        method="reversible",
        recorder=Recorder(modules=[]),
        num_checkpoints_reversible=num_checkpoints_reversible,
        recording_mode=recording_mode,
    )


def _build(gradient_config):
    """PML on all faces (so the interface record is not empty), a dipole and a phasor detector."""
    config = SimulationConfig(
        time=15e-15,
        grid=UniformGrid(spacing=50e-9),
        backend="cpu",
        dtype=jnp.float32,
        gradient_config=gradient_config,
    )
    n = _VOLUME_CELLS + 2 * _PML_CELLS
    volume = fdtdx.SimulationVolume(partial_grid_shape=(n, n, n))
    bound_dict, constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML_CELLS), volume
    )
    wave = fdtdx.WaveCharacter(frequency=c0 / 800e-9)
    source = fdtdx.PointDipoleSource(
        name="dip", partial_grid_shape=(1, 1, 1), wave_character=wave, polarization=0, amplitude=1.0
    )
    detector = fdtdx.PhasorDetector(
        name="phasor", partial_grid_shape=(None, None, 1), wave_characters=(wave,), components=("Ex",), plot=False
    )
    constraints += [
        source.set_grid_coordinates(
            axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(n // 2, n // 2, _PML_CELLS + 1)
        ),
        detector.same_size(volume, axes=(0, 1)),
        detector.place_at_center(volume, axes=(0, 1)),
        detector.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(n - _PML_CELLS - 2,)),
    ]
    key = jax.random.PRNGKey(0)
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, *bound_dict.values(), source, detector], config=config, constraints=constraints, key=key
    )
    arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)
    return obj, arrays, config, key


def _grad(gradient_config):
    obj, arrays, config, key = _build(gradient_config)

    def loss(inv_eps):
        _, out = fdtdx.run_fdtd(arrays.aset("inv_permittivities", inv_eps), obj, config, key, show_progress=False)
        return jnp.sum(jnp.abs(out.detector_states["phasor"]["phasor"]) ** 2)

    value, grad = jax.value_and_grad(loss)(arrays.inv_permittivities)
    return value, np.asarray(grad), config


def _record_lengths(arrays):
    return {v.shape[0] for v in arrays.recording_state.data.values()}


def test_recorder_holds_one_slice():
    _, arrays, config, _ = _build(_gradient_config(7, "segmented"))
    assert _record_lengths(arrays) == {math.ceil(config.time_steps_total / 8)}
    _, full_arrays, full_config, _ = _build(_gradient_config(7, "full"))
    assert _record_lengths(full_arrays) == {full_config.time_steps_total}


def test_auto_resolves_to_the_memory_optimum():
    _, arrays, config, _ = _build(_gradient_config("auto", "segmented"))
    num_ckpt = config.gradient_config.num_checkpoints_reversible
    assert isinstance(num_ckpt, int)
    record_bytes_per_step = sum(v[0].nbytes for v in arrays.recording_state.data.values())
    field_state_bytes = sum(x.nbytes for x in jax.tree.leaves(arrays.fields))
    expected = round(math.sqrt(2 * config.time_steps_total * record_bytes_per_step / field_state_bytes)) - 1
    assert num_ckpt == max(1, min(expected, config.time_steps_total - 1))
    assert _record_lengths(arrays) == {math.ceil(config.time_steps_total / (num_ckpt + 1))}


def test_gradient_equals_full_mode():
    v_seg, g_seg, _ = _grad(_gradient_config(3, "segmented"))
    v_full, g_full, _ = _grad(_gradient_config(3, "full"))
    assert np.any(g_full != 0)
    assert v_seg == v_full
    np.testing.assert_array_equal(g_seg, g_full)


def test_auto_gradient_equals_full_mode_with_the_resolved_count():
    v_auto, g_auto, config = _grad(_gradient_config("auto", "segmented"))
    v_full, g_full, _ = _grad(_gradient_config(config.gradient_config.num_checkpoints_reversible, "full"))
    assert v_auto == v_full
    np.testing.assert_array_equal(g_auto, g_full)
