"""Simulation tests for reversible_fdtd gradient (VJP) paths.

These tests exercise the custom VJP closures (fdtd_fwd, fdtd_bwd, body_fn, cond_fun)
inside reversible_fdtd that only execute during jax.grad backpropagation.
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.fdtd import reversible_fdtd
from fdtdx.interfaces.recorder import Recorder


@pytest.fixture
def field_shape():
    return (3, 2, 2, 2)


@pytest.fixture
def sim_config(field_shape):
    """Config with ~5 time steps and reversible gradient mode.

    A Recorder is required for the backward pass (add_interfaces).
    With no PML objects, the recorder processes empty boundary data.
    """
    base_config = SimulationConfig(
        time=400e-15,
        resolution=40e-6,
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=None,
    )
    recorder = Recorder(modules=[])
    recorder, _ = recorder.init_state(
        input_shape_dtypes={},
        max_time_steps=base_config.time_steps_total,
        backend="cpu",
    )
    config = base_config.aset(
        "gradient_config",
        GradientConfig(method="reversible", recorder=recorder),
    )
    return config


@pytest.fixture
def recording_state(sim_config):
    """Initialize an empty recording state matching the config's recorder."""
    _, state = sim_config.gradient_config.recorder.init_state(
        input_shape_dtypes={},
        max_time_steps=sim_config.time_steps_total,
        backend="cpu",
    )
    return state


@pytest.fixture
def sim_arrays(field_shape, recording_state):
    auxiliary_field_shape = (6, 2, 2, 2)
    return ArrayContainer(
        E=jnp.zeros(field_shape),
        H=jnp.zeros(field_shape),
        psi_E=jnp.zeros(auxiliary_field_shape),
        psi_H=jnp.zeros(auxiliary_field_shape),
        alpha=jnp.zeros(field_shape),
        kappa=jnp.ones(field_shape),
        sigma=jnp.zeros(field_shape),
        inv_permittivities=jnp.ones(field_shape),
        inv_permeabilities=jnp.ones(field_shape),
        detector_states={},
        recording_state=recording_state,
        electric_conductivity=None,
        magnetic_conductivity=None,
    )


@pytest.fixture
def sim_objects():
    return ObjectContainer(object_list=[], volume_idx=0)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


class TestReversibleFdtdGradients:
    """Test that jax.grad flows through the custom VJP of reversible_fdtd."""

    def _loss_fn(self, inv_permittivities, arrays, objects, config, key):
        """L2 loss on output inv_permittivities through a full forward+backward pass.

        Uses inv_permittivities (not E) because reset_array_container zeros E/H,
        and without sources E stays zero. inv_permittivities passes through the
        simulation unchanged, giving a non-trivial gradient through the custom VJP.
        """
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
        _, out = reversible_fdtd(arrays, objects, config, key)
        return jnp.sum(out.inv_permittivities**2)

    def test_gradients_are_finite(self, sim_arrays, sim_objects, sim_config, key):
        """Verify that gradients contain no NaN or Inf values."""
        loss, grads = jax.value_and_grad(self._loss_fn)(
            sim_arrays.inv_permittivities, sim_arrays, sim_objects, sim_config, key
        )
        assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
        assert jnp.all(jnp.isfinite(grads)), "Gradients contain NaN or Inf"

    def test_loss_is_finite(self, sim_arrays, sim_objects, sim_config, key):
        """Verify the forward pass produces a finite loss value."""
        loss = self._loss_fn(sim_arrays.inv_permittivities, sim_arrays, sim_objects, sim_config, key)
        assert jnp.isfinite(loss)

    def test_gradients_with_nonzero_fields(self, sim_objects, sim_config, key, field_shape, recording_state):
        """With nonzero initial E fields, gradients should be non-zero."""
        auxiliary_field_shape = (6, 2, 2, 2)
        arrays = ArrayContainer(
            E=jnp.ones(field_shape) * 0.5,
            H=jnp.ones(field_shape) * 0.3,
            psi_E=jnp.zeros(auxiliary_field_shape),
            psi_H=jnp.zeros(auxiliary_field_shape),
            alpha=jnp.zeros(field_shape),
            kappa=jnp.ones(field_shape),
            sigma=jnp.zeros(field_shape),
            inv_permittivities=jnp.ones(field_shape) * 2.0,
            inv_permeabilities=jnp.ones(field_shape),
            detector_states={},
            recording_state=recording_state,
            electric_conductivity=None,
            magnetic_conductivity=None,
        )
        loss, grads = jax.value_and_grad(self._loss_fn)(arrays.inv_permittivities, arrays, sim_objects, sim_config, key)
        assert jnp.isfinite(loss)
        assert jnp.all(jnp.isfinite(grads))
        assert loss > 0, "Loss should be positive with nonzero initial fields"
        assert jnp.any(grads != 0), "Gradients should be non-zero with nonzero fields"
