"""Simulation tests for time-reversal reconstruction with each boundary type.

For each boundary condition, verifies that one forward step followed by one
backward step exactly reconstructs the original E and H fields.  PML boundaries
break time-reversal symmetry, so they require interface recording/restoration.

Also includes an end-to-end gradient test combining PML with Bloch boundaries
(non-zero wave vector → complex-valued fields) to exercise the recorder with
complex dtypes.
"""

import jax
import jax.numpy as jnp

import fdtdx
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.fdtd.backward import backward
from fdtdx.fdtd.container import ArrayContainer
from fdtdx.fdtd.fdtd import reversible_fdtd
from fdtdx.fdtd.forward import forward
from fdtdx.interfaces.recorder import Recorder

# ── Constants ───────────────────────────────────────────────────────────────────

_RESOLUTION = 50e-9
_SIM_TIME = 40e-15  # very short — only need a few time steps
_PML_CELLS = 4  # thin PML to keep the domain small
_VOLUME_CELLS = 8  # inner domain per axis


# ── Helpers ─────────────────────────────────────────────────────────────────────


def _build_simulation(boundary_types, bloch_vector=(0.0, 0.0, 0.0)):
    """Set up a small simulation with the given boundary types on all faces.

    Args:
        boundary_types: dict mapping face names to boundary type strings,
            e.g. {"min_x": "pml", "max_x": "pml", ...}
        bloch_vector: Bloch wave vector for bloch-type boundaries.

    Returns:
        (obj_container, arrays, config) ready to run forward/backward.
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
        override_types=boundary_types,
        bloch_vector=bloch_vector,
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    return obj_container, arrays, config


def _add_gradient_config(arrays, config, obj_container):
    """Attach a gradient recorder to the config and arrays."""
    input_shape_dtypes = {}
    field_dtype = arrays.E.dtype
    for boundary in obj_container.pml_objects:
        cur_shape = boundary.interface_grid_shape()
        extended_shape = (3, *cur_shape)
        input_shape_dtypes[f"{boundary.name}_E"] = jax.ShapeDtypeStruct(
            shape=extended_shape,
            dtype=field_dtype,
        )
        input_shape_dtypes[f"{boundary.name}_H"] = jax.ShapeDtypeStruct(
            shape=extended_shape,
            dtype=field_dtype,
        )
    recorder = Recorder(modules=[])
    recorder, recording_state = recorder.init_state(
        input_shape_dtypes=input_shape_dtypes,
        max_time_steps=config.time_steps_total,
        backend="cpu",
    )
    grad_cfg = GradientConfig(method="reversible", recorder=recorder)
    config = config.aset("gradient_config", grad_cfg)
    arrays = arrays.aset("recording_state", recording_state)
    return arrays, config


def _seed_fields(arrays, key, obj_container=None):
    """Set E and H to small random values so the step is non-trivial.

    If obj_container is provided, boundary conditions are enforced on
    the seeded fields so that PEC/PMC tangential zeroing doesn't
    destroy information during the forward step.
    """
    k1, k2 = jax.random.split(key)
    E = jax.random.normal(k1, arrays.E.shape, dtype=arrays.E.dtype) * 1e-3
    H = jax.random.normal(k2, arrays.H.shape, dtype=arrays.H.dtype) * 1e-3
    if obj_container is not None:
        for b in obj_container.boundary_objects:
            E = b.apply_post_E_update(E)
            H = b.apply_post_H_update(H)
    arrays = arrays.aset("E", E)
    arrays = arrays.aset("H", H)
    return arrays


def _forward_backward_roundtrip(obj_container, arrays, config, has_pml):
    """Run one forward step then one backward step; return (original, reconstructed)."""
    key = jax.random.PRNGKey(42)
    arrays = _seed_fields(arrays, key, obj_container)

    # backward() always calls add_interfaces, so a recorder is always needed
    arrays, config = _add_gradient_config(arrays, config, obj_container)

    # Save original fields
    E_original = arrays.E.copy()
    H_original = arrays.H.copy()

    # Forward one step
    state_fwd = forward(
        state=(jnp.asarray(0, dtype=jnp.int32), arrays),
        config=config,
        objects=obj_container,
        key=key,
        record_detectors=False,
        record_boundaries=has_pml,
        simulate_boundaries=True,
    )

    # Backward one step
    state_bwd = backward(
        state=state_fwd,
        config=config,
        objects=obj_container,
        key=key,
        record_detectors=False,
        reset_fields=False,
    )

    _, arrays_bwd = state_bwd
    return E_original, H_original, arrays_bwd.E, arrays_bwd.H


# ── Boundary type definitions ──────────────────────────────────────────────────

_ALL_FACES = ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")


def _uniform_boundaries(btype):
    return {face: btype for face in _ALL_FACES}


# ── Tests: time-reversal per boundary type ──────────────────────────────────────


class TestTimeReversalPeriodicBoundary:
    """Periodic (Bloch k=0) boundaries are time-reversible without recording."""

    def test_fields_reconstructed_exactly(self):
        obj, arrays, config = _build_simulation(_uniform_boundaries("periodic"))
        E_orig, H_orig, E_rec, H_rec = _forward_backward_roundtrip(
            obj,
            arrays,
            config,
            has_pml=False,
        )
        assert jnp.allclose(E_rec, E_orig, atol=1e-5), f"E max error: {jnp.max(jnp.abs(E_rec - E_orig))}"
        assert jnp.allclose(H_rec, H_orig, atol=1e-5), f"H max error: {jnp.max(jnp.abs(H_rec - H_orig))}"


class TestTimeReversalPECBoundary:
    """PEC boundaries are time-reversible without recording."""

    def test_fields_reconstructed_exactly(self):
        obj, arrays, config = _build_simulation(_uniform_boundaries("pec"))
        E_orig, H_orig, E_rec, H_rec = _forward_backward_roundtrip(
            obj,
            arrays,
            config,
            has_pml=False,
        )
        assert jnp.allclose(E_rec, E_orig, atol=1e-5), f"E max error: {jnp.max(jnp.abs(E_rec - E_orig))}"
        assert jnp.allclose(H_rec, H_orig, atol=1e-5), f"H max error: {jnp.max(jnp.abs(H_rec - H_orig))}"


class TestTimeReversalPMCBoundary:
    """PMC boundaries are time-reversible without recording."""

    def test_fields_reconstructed_exactly(self):
        obj, arrays, config = _build_simulation(_uniform_boundaries("pmc"))
        E_orig, H_orig, E_rec, H_rec = _forward_backward_roundtrip(
            obj,
            arrays,
            config,
            has_pml=False,
        )
        assert jnp.allclose(E_rec, E_orig, atol=1e-5), f"E max error: {jnp.max(jnp.abs(E_rec - E_orig))}"
        assert jnp.allclose(H_rec, H_orig, atol=1e-5), f"H max error: {jnp.max(jnp.abs(H_rec - H_orig))}"


class TestTimeReversalPMLBoundary:
    """PML boundaries break time-reversal; recording restores it."""

    def test_fields_reconstructed_with_recording(self):
        obj, arrays, config = _build_simulation(_uniform_boundaries("pml"))
        E_orig, H_orig, E_rec, H_rec = _forward_backward_roundtrip(
            obj,
            arrays,
            config,
            has_pml=True,
        )
        assert jnp.allclose(E_rec, E_orig, atol=1e-5), f"E max error: {jnp.max(jnp.abs(E_rec - E_orig))}"
        assert jnp.allclose(H_rec, H_orig, atol=1e-5), f"H max error: {jnp.max(jnp.abs(H_rec - H_orig))}"


class TestTimeReversalBlochBoundary:
    """Bloch boundaries with non-zero k (complex fields) are time-reversible."""

    def test_fields_reconstructed_exactly(self):
        k_bloch = (1e6, 0.0, 0.0)  # non-zero → complex fields
        obj, arrays, config = _build_simulation(
            _uniform_boundaries("bloch"),
            bloch_vector=k_bloch,
        )
        E_orig, H_orig, E_rec, H_rec = _forward_backward_roundtrip(
            obj,
            arrays,
            config,
            has_pml=False,
        )
        assert jnp.allclose(E_rec, E_orig, atol=1e-5), f"E max error: {jnp.max(jnp.abs(E_rec - E_orig))}"
        assert jnp.allclose(H_rec, H_orig, atol=1e-5), f"H max error: {jnp.max(jnp.abs(H_rec - H_orig))}"


class TestTimeReversalMixedPMLPeriodic:
    """Mixed PML (z) + periodic (x, y) — the most common real-world config."""

    def test_fields_reconstructed_with_recording(self):
        boundary_types = {
            "min_x": "periodic",
            "max_x": "periodic",
            "min_y": "periodic",
            "max_y": "periodic",
            "min_z": "pml",
            "max_z": "pml",
        }
        obj, arrays, config = _build_simulation(boundary_types)
        E_orig, H_orig, E_rec, H_rec = _forward_backward_roundtrip(
            obj,
            arrays,
            config,
            has_pml=True,
        )
        assert jnp.allclose(E_rec, E_orig, atol=1e-5), f"E max error: {jnp.max(jnp.abs(E_rec - E_orig))}"
        assert jnp.allclose(H_rec, H_orig, atol=1e-5), f"H max error: {jnp.max(jnp.abs(H_rec - H_orig))}"


class TestTimeReversalMixedPMLBloch:
    """Mixed PML (z) + Bloch (x, y) — complex fields with interface recording."""

    def test_fields_reconstructed_with_recording(self):
        boundary_types = {
            "min_x": "bloch",
            "max_x": "bloch",
            "min_y": "bloch",
            "max_y": "bloch",
            "min_z": "pml",
            "max_z": "pml",
        }
        k_bloch = (1e6, 1e6, 0.0)
        obj, arrays, config = _build_simulation(boundary_types, bloch_vector=k_bloch)
        E_orig, H_orig, E_rec, H_rec = _forward_backward_roundtrip(
            obj,
            arrays,
            config,
            has_pml=True,
        )
        assert jnp.allclose(E_rec, E_orig, atol=1e-5), f"E max error: {jnp.max(jnp.abs(E_rec - E_orig))}"
        assert jnp.allclose(H_rec, H_orig, atol=1e-5), f"H max error: {jnp.max(jnp.abs(H_rec - H_orig))}"

    def test_recorded_interfaces_are_complex(self):
        """Verify the recorder stores complex-valued interface data."""
        boundary_types = {
            "min_x": "bloch",
            "max_x": "bloch",
            "min_y": "bloch",
            "max_y": "bloch",
            "min_z": "pml",
            "max_z": "pml",
        }
        k_bloch = (1e6, 1e6, 0.0)
        obj, arrays, config = _build_simulation(boundary_types, bloch_vector=k_bloch)
        arrays, config = _add_gradient_config(arrays, config, obj)
        key = jax.random.PRNGKey(42)
        arrays = _seed_fields(arrays, key, obj)

        # One forward step with boundary recording
        state_fwd = forward(
            state=(jnp.asarray(0, dtype=jnp.int32), arrays),
            config=config,
            objects=obj,
            key=key,
            record_detectors=False,
            record_boundaries=True,
            simulate_boundaries=True,
        )
        _, arrays_fwd = state_fwd
        # Check that the recorder's stored data has complex dtype
        for k, v in arrays_fwd.recording_state.data.items():
            assert jnp.issubdtype(v.dtype, jnp.complexfloating), (
                f"Recorded interface '{k}' has dtype {v.dtype}, expected complex"
            )


# ── Test: gradient with PML + Bloch (complex fields) ───────────────────────────


class TestGradientPMLBlochComplex:
    """End-to-end gradient test with PML + Bloch (complex fields).

    Verifies that the reversible_fdtd custom VJP works when the recorder
    stores complex-valued field interfaces.
    """

    @staticmethod
    def _build():
        """Build a small PML+Bloch simulation with gradient support."""
        boundary_types = {
            "min_x": "bloch",
            "max_x": "bloch",
            "min_y": "bloch",
            "max_y": "bloch",
            "min_z": "pml",
            "max_z": "pml",
        }
        k_bloch = (1e6, 1e6, 0.0)
        obj, arrays, config = _build_simulation(boundary_types, bloch_vector=k_bloch)
        arrays, config = _add_gradient_config(arrays, config, obj)

        # Seed non-zero fields so gradients are non-trivial
        key = jax.random.PRNGKey(7)
        arrays = _seed_fields(arrays, key, obj)

        return obj, arrays, config

    @staticmethod
    def _loss_fn(inv_permittivities, arrays, objects, config, key):
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
        return jnp.sum(jnp.real(out.inv_permittivities) ** 2)

    def test_gradients_are_finite(self):
        obj, arrays, config = self._build()
        key = jax.random.PRNGKey(99)
        loss, grads = jax.value_and_grad(self._loss_fn)(
            arrays.inv_permittivities,
            arrays,
            obj,
            config,
            key,
        )
        assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
        assert jnp.all(jnp.isfinite(grads)), "Gradients contain NaN or Inf"

    def test_loss_is_positive(self):
        obj, arrays, config = self._build()
        key = jax.random.PRNGKey(99)
        loss = self._loss_fn(arrays.inv_permittivities, arrays, obj, config, key)
        assert loss > 0, "Loss should be positive with non-zero inv_permittivities"

    def test_fields_are_complex(self):
        """Verify the simulation actually uses complex-valued fields."""
        _, arrays, _ = self._build()
        assert jnp.issubdtype(arrays.E.dtype, jnp.complexfloating)
        assert jnp.issubdtype(arrays.H.dtype, jnp.complexfloating)
