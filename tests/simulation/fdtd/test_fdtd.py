"""Simulation tests for reversible_fdtd gradient (VJP) paths.

These tests exercise the custom VJP closures (fdtd_fwd, fdtd_bwd, body_fn, cond_fun)
inside reversible_fdtd that only execute during jax.grad backpropagation.

The cross-validation tests at the bottom of this file compare gradients from
``reversible_fdtd`` (custom VJP) against ``checkpointed_fdtd`` (standard
autodiff). Float64 sharpens the precision floor so an algebraic bug in either
path produces visible disagreement.
"""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import pytest

import fdtdx
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, FieldState
from fdtdx.fdtd.fdtd import checkpointed_fdtd, reversible_fdtd
from fdtdx.interfaces.recorder import Recorder


@contextmanager
def _x64_enabled():
    """Scoped float64 enable; restores the prior state on exit to avoid
    contaminating downstream float32-only tests in the same pytest session."""
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


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
        """L2 loss on the evolved E/H fields after a full forward+backward pass.

        The ``scene`` fixture drives the fields with a CW dipole source so that
        E/H are nonzero by the end of the run — a nonzero gradient w.r.t.
        ``inv_permittivities`` then proves the custom VJP propagates through
        the Maxwell updates rather than passing input ``inv_permittivities``
        straight through to the output.
        """
        arrays = ArrayContainer(
            fields=FieldState(
                E=arrays.fields.E, H=arrays.fields.H, psi_E=arrays.fields.psi_E, psi_H=arrays.fields.psi_H
            ),
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
        return jnp.sum(out.fields.E**2) + jnp.sum(out.fields.H**2)

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


# ──────────────────────────────────────────────────────────────────────────────
# Cross-path gradient validation: reversible vs checkpointed
# ──────────────────────────────────────────────────────────────────────────────


def _build_lossy_dispersive_scene(dtype):
    """Lorentz + σ_E = 100 S/m + dipole, periodic. The scene where the recent
    conductivity-not-forwarded bug actually mattered.
    """
    config = SimulationConfig(
        time=_SIM_TIME,
        resolution=_RESOLUTION,
        backend="cpu",
        dtype=dtype,
        courant_factor=0.99,
        gradient_config=None,
    )
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(_VOLUME_CELLS, _VOLUME_CELLS, _VOLUME_CELLS))
    objects.append(volume)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            "min_x": "periodic",
            "max_x": "periodic",
            "min_y": "periodic",
            "max_y": "periodic",
            "min_z": "periodic",
            "max_z": "periodic",
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    material = fdtdx.Material(
        permittivity=2.0,
        electric_conductivity=1e2,
        dispersion=fdtdx.DispersionModel(
            poles=(fdtdx.LorentzPole(resonance_frequency=2e15, damping=1e13, delta_epsilon=1.5),),
        ),
    )
    slab_cells = _VOLUME_CELLS // 2
    slab = fdtdx.UniformMaterialObject(
        name="slab",
        partial_grid_shape=(None, None, slab_cells),
        material=material,
    )
    constraints.extend(
        [
            slab.same_size(volume, axes=(0, 1)),
            slab.place_at_center(volume, axes=(0, 1)),
            slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_VOLUME_CELLS // 4,)),
        ]
    )
    objects.append(slab)

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
            coordinates=(_VOLUME_CELLS // 2, _VOLUME_CELLS // 2, _VOLUME_CELLS // 2),
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
    return obj, arrays, config


def _build_magnetic_lossy_scene(dtype):
    """Periodic + magnetic_conductivity slab + dipole. Same shape as the
    dispersive-lossy scene but on the magnetic branch."""
    config = SimulationConfig(
        time=_SIM_TIME,
        resolution=_RESOLUTION,
        backend="cpu",
        dtype=dtype,
        courant_factor=0.99,
        gradient_config=None,
    )
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(_VOLUME_CELLS, _VOLUME_CELLS, _VOLUME_CELLS))
    objects.append(volume)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            "min_x": "periodic",
            "max_x": "periodic",
            "min_y": "periodic",
            "max_y": "periodic",
            "min_z": "periodic",
            "max_z": "periodic",
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    material = fdtdx.Material(permittivity=1.0, permeability=2.0, magnetic_conductivity=1e2)
    slab_cells = _VOLUME_CELLS // 2
    slab = fdtdx.UniformMaterialObject(
        name="mag_slab",
        partial_grid_shape=(None, None, slab_cells),
        material=material,
    )
    constraints.extend(
        [
            slab.same_size(volume, axes=(0, 1)),
            slab.place_at_center(volume, axes=(0, 1)),
            slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_VOLUME_CELLS // 4,)),
        ]
    )
    objects.append(slab)

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
            coordinates=(_VOLUME_CELLS // 2, _VOLUME_CELLS // 2, _VOLUME_CELLS // 2),
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
    return obj, arrays, config


def _attach_gradient(arrays, config, obj, method, num_checkpoints=8):
    """Attach a fresh gradient_config + recording_state to the scene."""
    input_shape_dtypes = {}
    field_dtype = arrays.fields.E.dtype
    for boundary in obj.pml_objects:
        cur_shape = boundary.interface_grid_shape()
        extended_shape = (3, *cur_shape)
        input_shape_dtypes[f"{boundary.name}_E"] = jax.ShapeDtypeStruct(shape=extended_shape, dtype=field_dtype)
        input_shape_dtypes[f"{boundary.name}_H"] = jax.ShapeDtypeStruct(shape=extended_shape, dtype=field_dtype)
    recorder = Recorder(modules=[])
    recorder, recording_state = recorder.init_state(
        input_shape_dtypes=input_shape_dtypes,
        max_time_steps=config.time_steps_total,
        backend="cpu",
    )
    if method == "reversible":
        grad_cfg = GradientConfig(method="reversible", recorder=recorder)
    else:
        grad_cfg = GradientConfig(method="checkpointed", num_checkpoints=num_checkpoints)
    config = config.aset("gradient_config", grad_cfg)
    arrays = arrays.aset("recording_state", recording_state)
    return arrays, config


class TestReversibleVsCheckpointedGradient:
    """Gradients from ``reversible_fdtd`` and ``checkpointed_fdtd`` must agree.

    These are independent implementations of the same mathematical operation:
    - reversible: custom VJP that reconstructs forward states via the recorder
    - checkpointed: standard autodiff through ``eqxi.while_loop``

    Any algebraic mistake that lives in one path but not the other (e.g. the
    recent ``forward_single_args_wrapper`` not receiving conductivity, which
    made the reversible-path VJP linearization run lossless while the
    checkpointed path stayed correct) makes the two gradients diverge. The
    cross-check is the single best correctness oracle the codebase has.
    """

    @staticmethod
    def _loss_inv_eps(inv_permittivities, arrays, objects, config, key):
        arrays = ArrayContainer(
            fields=FieldState(
                E=arrays.fields.E,
                H=arrays.fields.H,
                psi_E=arrays.fields.psi_E,
                psi_H=arrays.fields.psi_H,
            ),
            alpha=arrays.alpha,
            kappa=arrays.kappa,
            sigma=arrays.sigma,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=arrays.inv_permeabilities,
            detector_states=arrays.detector_states,
            recording_state=arrays.recording_state,
            electric_conductivity=arrays.electric_conductivity,
            magnetic_conductivity=arrays.magnetic_conductivity,
            dispersive_P_curr=arrays.dispersive_P_curr,
            dispersive_P_prev=arrays.dispersive_P_prev,
            dispersive_c1=arrays.dispersive_c1,
            dispersive_c2=arrays.dispersive_c2,
            dispersive_c3=arrays.dispersive_c3,
            dispersive_inv_c2=arrays.dispersive_inv_c2,
        )
        return arrays

    @staticmethod
    def _run_inv_eps(method, dtype):
        obj, arrays, config = _build_lossy_dispersive_scene(dtype=dtype)
        arrays, config = _attach_gradient(arrays, config, obj, method=method)

        def loss_fn(inv_eps):
            arr = TestReversibleVsCheckpointedGradient._loss_inv_eps(inv_eps, arrays, obj, config, None)
            fdtd_impl = reversible_fdtd if method == "reversible" else checkpointed_fdtd
            _, out = fdtd_impl(arr, obj, config, jax.random.PRNGKey(99), show_progress=False)
            return jnp.sum(jnp.real(out.fields.E) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(arrays.inv_permittivities)
        return loss, grads

    def test_gradients_agree_inv_permittivities_float64(self):
        with _x64_enabled():
            loss_rev, grad_rev = self._run_inv_eps(method="reversible", dtype=jnp.float64)
            loss_chk, grad_chk = self._run_inv_eps(method="checkpointed", dtype=jnp.float64)
            assert jnp.isfinite(loss_rev) and jnp.isfinite(loss_chk)
            # Float64 cross-path agreement: both paths solve the same linearised
            # adjoint problem on the same primal trajectory — any disagreement
            # above ~1e-6 indicates an algebraic mismatch.
            assert jnp.allclose(loss_rev, loss_chk, rtol=1e-9, atol=1e-12), (
                f"Forward loss disagrees: rev={float(loss_rev):.9e}, chk={float(loss_chk):.9e}"
            )
            max_abs = float(jnp.max(jnp.abs(grad_rev) + jnp.abs(grad_chk)))
            max_diff = float(jnp.max(jnp.abs(grad_rev - grad_chk)))
            rel = max_diff / (max_abs + 1e-30)
            assert rel < 1e-5, (
                f"Reversible vs checkpointed gradient disagrees: max|Δ|={max_diff:.3e}, "
                f"max|grad|={max_abs:.3e}, rel={rel:.3e}"
            )

    def test_gradients_agree_inv_permeabilities_magnetic_lossy_float64(self):
        """Cross-validate gradient w.r.t. inv_permeabilities in a slab with
        magnetic_conductivity=1e2. The magnetic-loss branch of update_H is
        the symmetric counterpart of the electric-loss branch and has no
        other gradient cross-check."""

        def run(method):
            obj, arrays, config = _build_magnetic_lossy_scene(dtype=jnp.float64)
            arrays, config = _attach_gradient(arrays, config, obj, method=method)

            def loss_fn(inv_mu):
                arr = ArrayContainer(
                    fields=FieldState(
                        E=arrays.fields.E,
                        H=arrays.fields.H,
                        psi_E=arrays.fields.psi_E,
                        psi_H=arrays.fields.psi_H,
                    ),
                    alpha=arrays.alpha,
                    kappa=arrays.kappa,
                    sigma=arrays.sigma,
                    inv_permittivities=arrays.inv_permittivities,
                    inv_permeabilities=inv_mu,
                    detector_states=arrays.detector_states,
                    recording_state=arrays.recording_state,
                    electric_conductivity=arrays.electric_conductivity,
                    magnetic_conductivity=arrays.magnetic_conductivity,
                )
                fdtd_impl = reversible_fdtd if method == "reversible" else checkpointed_fdtd
                _, out = fdtd_impl(arr, obj, config, jax.random.PRNGKey(99), show_progress=False)
                return jnp.sum(jnp.real(out.fields.H) ** 2)

            return jax.value_and_grad(loss_fn)(arrays.inv_permeabilities)

        with _x64_enabled():
            loss_rev, grad_rev = run("reversible")
            loss_chk, grad_chk = run("checkpointed")
            assert jnp.isfinite(loss_rev) and jnp.isfinite(loss_chk)
            assert jnp.allclose(loss_rev, loss_chk, rtol=1e-9, atol=1e-12)
            max_abs = float(jnp.max(jnp.abs(grad_rev) + jnp.abs(grad_chk)))
            max_diff = float(jnp.max(jnp.abs(grad_rev - grad_chk)))
            rel = max_diff / (max_abs + 1e-30)
            assert rel < 1e-5, f"μ⁻¹ gradient mismatch: max|Δ|={max_diff:.3e}, max|grad|={max_abs:.3e}, rel={rel:.3e}"

    def test_gradients_agree_dispersive_c3_float64(self):
        """Cross-validate gradient w.r.t. dispersive_c3 — the ADE coefficient
        whose VJP path differs most between reversible (algebraic inversion
        of the recurrence in update_E_reverse) and checkpointed (forward
        recurrence differentiated by jax)."""

        def run(method):
            obj, arrays, config = _build_lossy_dispersive_scene(dtype=jnp.float64)
            arrays, config = _attach_gradient(arrays, config, obj, method=method)

            def loss_fn(c3):
                arr = ArrayContainer(
                    fields=FieldState(
                        E=arrays.fields.E,
                        H=arrays.fields.H,
                        psi_E=arrays.fields.psi_E,
                        psi_H=arrays.fields.psi_H,
                    ),
                    alpha=arrays.alpha,
                    kappa=arrays.kappa,
                    sigma=arrays.sigma,
                    inv_permittivities=arrays.inv_permittivities,
                    inv_permeabilities=arrays.inv_permeabilities,
                    detector_states=arrays.detector_states,
                    recording_state=arrays.recording_state,
                    electric_conductivity=arrays.electric_conductivity,
                    magnetic_conductivity=arrays.magnetic_conductivity,
                    dispersive_P_curr=arrays.dispersive_P_curr,
                    dispersive_P_prev=arrays.dispersive_P_prev,
                    dispersive_c1=arrays.dispersive_c1,
                    dispersive_c2=arrays.dispersive_c2,
                    dispersive_c3=c3,
                    dispersive_inv_c2=arrays.dispersive_inv_c2,
                )
                fdtd_impl = reversible_fdtd if method == "reversible" else checkpointed_fdtd
                _, out = fdtd_impl(arr, obj, config, jax.random.PRNGKey(99), show_progress=False)
                return jnp.sum(jnp.real(out.fields.E) ** 2)

            return jax.value_and_grad(loss_fn)(arrays.dispersive_c3)

        with _x64_enabled():
            loss_rev, grad_rev = run("reversible")
            loss_chk, grad_chk = run("checkpointed")
            assert jnp.isfinite(loss_rev) and jnp.isfinite(loss_chk)
            assert jnp.allclose(loss_rev, loss_chk, rtol=1e-9, atol=1e-12)
            max_abs = float(jnp.max(jnp.abs(grad_rev) + jnp.abs(grad_chk)))
            max_diff = float(jnp.max(jnp.abs(grad_rev - grad_chk)))
            rel = max_diff / (max_abs + 1e-30)
            assert rel < 1e-5, f"c3 gradient mismatch: max|Δ|={max_diff:.3e}, max|grad|={max_abs:.3e}, rel={rel:.3e}"
