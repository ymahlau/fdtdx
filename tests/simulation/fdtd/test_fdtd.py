"""Simulation tests for reversible_fdtd gradient (VJP) paths.

These tests exercise the custom VJP closures (fdtd_fwd, fdtd_bwd, body_fn, cond_fun)
inside reversible_fdtd that only execute during jax.grad backpropagation.
"""

import jax
import jax.numpy as jnp
import pytest

import fdtdx
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.fdtd.container import ArrayContainer
from fdtdx.fdtd.fdtd import reversible_fdtd
from fdtdx.interfaces.recorder import Recorder

_RESOLUTION = 50e-9
_SIM_TIME = 40e-15
_PML_CELLS = 4
_VOLUME_CELLS = 8


def _build_scene():
    """Build a periodic-boundary scene with a CW dipole source driving the run.

    ``reversible_fdtd`` resets E/H on entry, so a test that wants nonzero
    fields (and therefore a nonzero gradient w.r.t. ``inv_permittivities``)
    has to drive them with a source. Periodic boundaries keep the setup
    minimal — no PML bookkeeping, empty recorder interface shapes.
    """
    config = SimulationConfig(
        time=_SIM_TIME,
        resolution=_RESOLUTION,
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=None,
    )
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(
        partial_grid_shape=(_VOLUME_CELLS, _VOLUME_CELLS, _VOLUME_CELLS),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            face: "periodic"
            for face in (
                "min_x",
                "max_x",
                "min_y",
                "max_y",
                "min_z",
                "max_z",
            )
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    source = fdtdx.PointDipoleSource(
        name="dip",
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(frequency=3e8 / 800e-9),
        polarization=0,
        amplitude=1.0,
    )
    constraints.append(
        source.set_grid_coordinates(
            axes=(0, 1, 2),
            sides=("-", "-", "-"),
            coordinates=(
                _VOLUME_CELLS // 2,
                _VOLUME_CELLS // 2,
                _VOLUME_CELLS // 2,
            ),
        )
    )
    objects.append(source)

    key = jax.random.PRNGKey(0)
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)

    recorder = Recorder(modules=[])
    recorder, recording_state = recorder.init_state(
        input_shape_dtypes={},
        max_time_steps=config.time_steps_total,
        backend="cpu",
    )
    config = config.aset(
        "gradient_config",
        GradientConfig(method="reversible", recorder=recorder),
    )
    arrays = arrays.aset("recording_state", recording_state)
    return obj, arrays, config


@pytest.fixture
def scene():
    return _build_scene()


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


class TestReversibleFdtdGradients:
    """Test that jax.grad flows through the custom VJP of reversible_fdtd."""

    def _loss_fn(self, inv_permittivities, arrays, objects, config, key):
        arrays = ArrayContainer(
            E=arrays.E,
            H=arrays.H,
            psi_E=arrays.psi_E,
            psi_H=arrays.psi_H,
            alpha=arrays.alpha,
            kappa=arrays.kappa,
            sigma=arrays.sigma,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=arrays.inv_permeabilities,
            detector_states=arrays.detector_states,
            recording_state=arrays.recording_state,
            electric_conductivity=arrays.electric_conductivity,
            magnetic_conductivity=arrays.magnetic_conductivity,
        )
        _, out = reversible_fdtd(arrays, objects, config, key, show_progress=False)
        return jnp.sum(out.E**2) + jnp.sum(out.H**2)

    def test_gradients_are_finite(self, scene, key):
        obj, arrays, config = scene
        loss, grads = jax.value_and_grad(self._loss_fn)(arrays.inv_permittivities, arrays, obj, config, key)
        assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
        assert jnp.all(jnp.isfinite(grads)), "Gradients contain NaN or Inf"

    def test_loss_is_finite(self, scene, key):
        obj, arrays, config = scene
        loss = self._loss_fn(arrays.inv_permittivities, arrays, obj, config, key)
        assert jnp.isfinite(loss)

    def test_gradients_are_non_trivial(self, scene, key):
        """Gradient of an E-field loss w.r.t. inv_permittivities must be nonzero.

        A loss built only from ``out.inv_permittivities`` would produce
        ``2 * inv_eps`` regardless of whether the time-step VJP fires, so
        this test targets an E-field loss instead: a nonzero gradient proves
        the backward pass genuinely propagates through the Maxwell update.
        """
        obj, arrays, config = scene
        loss, grads = jax.value_and_grad(self._loss_fn)(arrays.inv_permittivities, arrays, obj, config, key)
        assert jnp.isfinite(loss)
        assert jnp.all(jnp.isfinite(grads))
        assert loss > 0, "Loss should be positive with nonzero propagating fields"
        assert jnp.any(grads != 0), "Gradients should be nonzero — VJP must flow through Maxwell updates"
