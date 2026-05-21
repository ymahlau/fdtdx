"""Simulation tests for time-reversal reconstruction with each boundary type.

For each boundary condition, verifies that one forward step followed by one
backward step exactly reconstructs the original E and H fields.  PML boundaries
break time-reversal symmetry, so they require interface recording/restoration.

Also includes an end-to-end gradient test combining PML with Bloch boundaries
(non-zero wave vector → complex-valued fields) to exercise the recorder with
complex dtypes.
"""

from contextlib import contextmanager

import jax
import jax.numpy as jnp

import fdtdx
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.fdtd.backward import backward
from fdtdx.fdtd.container import ArrayContainer, FieldState
from fdtdx.fdtd.fdtd import checkpointed_fdtd, reversible_fdtd
from fdtdx.fdtd.forward import forward
from fdtdx.interfaces.recorder import Recorder


@contextmanager
def _x64_enabled():
    """Scoped float64 enable. Strict reversibility / FD-gradient tests need
    x64 to push the precision floor below ~1e-12; restoring the prior state
    on exit prevents global x64 contamination of float32-only tests."""
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# ── Constants ───────────────────────────────────────────────────────────────────

_RESOLUTION = 50e-9
_SIM_TIME = 40e-15  # very short — only need a few time steps
_PML_CELLS = 4  # thin PML to keep the domain small
_VOLUME_CELLS = 8  # inner domain per axis
_STRICT_N_STEPS = 10  # multi-step reversibility horizon
_STRICT_REL_TOL = 1e-9  # float64 reversibility tolerance (machine precision floor)


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
    field_dtype = arrays.fields.E.dtype
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
    E = jax.random.normal(k1, arrays.fields.E.shape, dtype=arrays.fields.E.dtype) * 1e-3
    H = jax.random.normal(k2, arrays.fields.H.shape, dtype=arrays.fields.H.dtype) * 1e-3
    if obj_container is not None:
        for b in obj_container.boundary_objects:
            E = b.apply_post_E_update(E)
            H = b.apply_post_H_update(H)
    arrays = arrays.aset("fields->E", E)
    arrays = arrays.aset("fields->H", H)
    return arrays


def _forward_backward_roundtrip(obj_container, arrays, config, has_pml):
    """Run one forward step then one backward step; return (original, reconstructed)."""
    key = jax.random.PRNGKey(42)
    arrays = _seed_fields(arrays, key, obj_container)

    # backward() always calls add_interfaces, so a recorder is always needed
    arrays, config = _add_gradient_config(arrays, config, obj_container)

    # Save original fields
    E_original = arrays.fields.E.copy()
    H_original = arrays.fields.H.copy()

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
    return E_original, H_original, arrays_bwd.fields.E, arrays_bwd.fields.H


def _multi_step_roundtrip(
    obj_container,
    arrays,
    config,
    has_pml,
    n_steps,
    seed_dispersive=False,
):
    """Run ``n_steps`` forward steps then ``n_steps`` backward steps.

    Per-step factor errors in the reverse pass accumulate over a multi-step
    horizon — a 1-step roundtrip has no headroom to expose drift that stays
    below atol. Returns ``(originals, reconstructed)`` as dicts containing
    ``E``, ``H``, and optionally ``P_curr`` / ``P_prev`` when
    ``seed_dispersive=True``.
    """
    key = jax.random.PRNGKey(42)
    arrays = _seed_fields(arrays, key, obj_container)

    originals = {"E": arrays.fields.E, "H": arrays.fields.H}
    if seed_dispersive and arrays.dispersive_P_curr is not None:
        # Mask polarization seeds to dispersive cells only — vacuum cells have
        # c1=c2=c3=0 and any nonzero P there would be killed by the forward
        # recurrence and unrecoverable by the reverse.
        k_pc, k_pp = jax.random.split(key)
        disp_mask = (arrays.dispersive_c3 != 0).astype(arrays.dispersive_P_curr.dtype)
        P_curr = jax.random.normal(k_pc, arrays.dispersive_P_curr.shape, dtype=arrays.dispersive_P_curr.dtype)
        P_curr = P_curr * 1e-3 * disp_mask
        P_prev = jax.random.normal(k_pp, arrays.dispersive_P_prev.shape, dtype=arrays.dispersive_P_prev.dtype)
        P_prev = P_prev * 1e-3 * disp_mask
        arrays = arrays.aset("dispersive_P_curr", P_curr)
        arrays = arrays.aset("dispersive_P_prev", P_prev)
        originals["P_curr"] = P_curr
        originals["P_prev"] = P_prev

    arrays, config = _add_gradient_config(arrays, config, obj_container)

    state = (jnp.asarray(0, dtype=jnp.int32), arrays)
    for _ in range(n_steps):
        state = forward(
            state=state,
            config=config,
            objects=obj_container,
            key=key,
            record_detectors=False,
            record_boundaries=has_pml,
            simulate_boundaries=True,
        )
    for _ in range(n_steps):
        state = backward(
            state=state,
            config=config,
            objects=obj_container,
            key=key,
            record_detectors=False,
            reset_fields=False,
        )

    _, arr_rec = state
    reconstructed = {"E": arr_rec.fields.E, "H": arr_rec.fields.H}
    if seed_dispersive and arr_rec.dispersive_P_curr is not None:
        reconstructed["P_curr"] = arr_rec.dispersive_P_curr
        reconstructed["P_prev"] = arr_rec.dispersive_P_prev
    return originals, reconstructed


def _max_relative_error(reconstructed, original):
    """max|rec - orig| / (max|orig| + tiny). Magnitude-relative — exposes drift
    in low-amplitude regions that a fixed-atol assertion masks."""
    err = jnp.max(jnp.abs(reconstructed - original))
    scale = jnp.max(jnp.abs(original)) + 1e-30
    return float(err / scale)


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


class TestTimeReversalDispersiveLorentz:
    """Dispersive (Lorentz) material round-trip reconstructs both E/H and
    the ADE polarization state.

    Periodic boundaries + a Lorentz slab filling the middle of the z axis.
    P_curr and P_prev are seeded with non-zero random values masked to the
    dispersive slab (vacuum cells have c1=c2=c3=0, and information in those
    cells would be lost through the zero-valued recurrence). One forward step
    followed by one reverse step must exactly reconstruct E, H, P_curr and
    P_prev — the reverse is the algebraic inverse of the ADE recurrence.
    """

    @staticmethod
    def _build():
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
            override_types=_uniform_boundaries("periodic"),
        )
        bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
        objects.extend(bound_dict.values())
        constraints.extend(c_list)

        material = fdtdx.Material(
            permittivity=2.0,
            dispersion=fdtdx.DispersionModel(
                poles=(fdtdx.LorentzPole(resonance_frequency=2e15, damping=1e13, delta_epsilon=1.5),),
            ),
        )
        slab_cells = _VOLUME_CELLS // 2
        slab = fdtdx.UniformMaterialObject(
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

        key = jax.random.PRNGKey(0)
        obj_container, arrays, params, config, _ = fdtdx.place_objects(
            object_list=objects,
            config=config,
            constraints=constraints,
            key=key,
        )
        arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
        return obj_container, arrays, config

    def test_fields_and_polarization_reconstructed_exactly(self):
        obj, arrays, config = self._build()
        # Allocation sanity: the slab should have triggered allocation.
        assert arrays.dispersive_P_curr is not None
        assert arrays.dispersive_P_prev is not None
        assert arrays.dispersive_c3 is not None

        key = jax.random.PRNGKey(42)
        k_E, k_H, k_Pc, k_Pp = jax.random.split(key, 4)

        E = jax.random.normal(k_E, arrays.fields.E.shape, dtype=arrays.fields.E.dtype) * 1e-3
        H = jax.random.normal(k_H, arrays.fields.H.shape, dtype=arrays.fields.H.dtype) * 1e-3
        for b in obj.boundary_objects:
            E = b.apply_post_E_update(E)
            H = b.apply_post_H_update(H)

        # Mask polarization seeds to dispersive cells only. In vacuum cells the
        # recurrence coefficients are zero, so any nonzero P there gets dropped
        # by the forward step and the reverse cannot recover it. Keeping P=0 in
        # those cells is consistent with both directions.
        disp_mask = (arrays.dispersive_c3 != 0).astype(arrays.dispersive_P_curr.dtype)
        P_curr = (
            jax.random.normal(k_Pc, arrays.dispersive_P_curr.shape, dtype=arrays.dispersive_P_curr.dtype)
            * 1e-3
            * disp_mask
        )
        P_prev = (
            jax.random.normal(k_Pp, arrays.dispersive_P_prev.shape, dtype=arrays.dispersive_P_prev.dtype)
            * 1e-3
            * disp_mask
        )

        arrays = arrays.aset("fields->E", E)
        arrays = arrays.aset("fields->H", H)
        arrays = arrays.aset("dispersive_P_curr", P_curr)
        arrays = arrays.aset("dispersive_P_prev", P_prev)

        # Recorder setup: there are no PML objects, so the recorder ends up
        # with an empty interface dict — but backward() still needs a
        # gradient_config and recording_state to be present.
        arrays, config = _add_gradient_config(arrays, config, obj)

        E_orig = E
        H_orig = H
        Pc_orig = P_curr
        Pp_orig = P_prev

        state_fwd = forward(
            state=(jnp.asarray(0, dtype=jnp.int32), arrays),
            config=config,
            objects=obj,
            key=key,
            record_detectors=False,
            record_boundaries=False,
            simulate_boundaries=True,
        )
        state_bwd = backward(
            state=state_fwd,
            config=config,
            objects=obj,
            key=key,
            record_detectors=False,
            reset_fields=False,
        )
        _, arrays_bwd = state_bwd

        assert arrays_bwd.dispersive_P_curr is not None
        assert arrays_bwd.dispersive_P_prev is not None

        assert jnp.allclose(arrays_bwd.fields.E, E_orig, atol=1e-5), (
            f"E max err: {jnp.max(jnp.abs(arrays_bwd.fields.E - E_orig))}"
        )
        assert jnp.allclose(arrays_bwd.fields.H, H_orig, atol=1e-5), (
            f"H max err: {jnp.max(jnp.abs(arrays_bwd.fields.H - H_orig))}"
        )
        assert jnp.allclose(arrays_bwd.dispersive_P_curr, Pc_orig, atol=1e-5), (
            f"P_curr max err: {jnp.max(jnp.abs(arrays_bwd.dispersive_P_curr - Pc_orig))}"
        )
        assert jnp.allclose(arrays_bwd.dispersive_P_prev, Pp_orig, atol=1e-5), (
            f"P_prev max err: {jnp.max(jnp.abs(arrays_bwd.dispersive_P_prev - Pp_orig))}"
        )


class TestTimeReversalDispersiveLossy:
    """Time-reversal for a Lorentz-dispersive material *with nonzero electric
    conductivity*.

    This combination activates both the lossy factor ``(1 ± c σ η₀ inv_eps/2)``
    and the dispersive polarization recurrence in the forward/reverse E update.
    Getting the reverse E update right requires the entire RHS (curl and
    polarization delta) to be divided by the lossy factor — a subtle point
    that a pure-dispersive or pure-lossy test alone does not exercise.
    """

    @staticmethod
    def _build():
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
            override_types=_uniform_boundaries("periodic"),
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

        key = jax.random.PRNGKey(0)
        obj_container, arrays, params, config, _ = fdtdx.place_objects(
            object_list=objects,
            config=config,
            constraints=constraints,
            key=key,
        )
        arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
        return obj_container, arrays, config

    def test_fields_and_polarization_reconstructed_exactly(self):
        obj, arrays, config = self._build()
        # Both the dispersive state AND an electric_conductivity map must be
        # populated for this test to mean what we claim.
        assert arrays.dispersive_P_curr is not None
        assert arrays.dispersive_c3 is not None
        assert arrays.electric_conductivity is not None
        assert jnp.any(arrays.electric_conductivity != 0), "Slab must have nonzero σ_E"

        key = jax.random.PRNGKey(42)
        k_E, k_H, k_Pc, k_Pp = jax.random.split(key, 4)

        E = jax.random.normal(k_E, arrays.fields.E.shape, dtype=arrays.fields.E.dtype) * 1e-3
        H = jax.random.normal(k_H, arrays.fields.H.shape, dtype=arrays.fields.H.dtype) * 1e-3
        for b in obj.boundary_objects:
            E = b.apply_post_E_update(E)
            H = b.apply_post_H_update(H)

        disp_mask = (arrays.dispersive_c3 != 0).astype(arrays.dispersive_P_curr.dtype)
        P_curr = (
            jax.random.normal(k_Pc, arrays.dispersive_P_curr.shape, dtype=arrays.dispersive_P_curr.dtype)
            * 1e-3
            * disp_mask
        )
        P_prev = (
            jax.random.normal(k_Pp, arrays.dispersive_P_prev.shape, dtype=arrays.dispersive_P_prev.dtype)
            * 1e-3
            * disp_mask
        )

        arrays = arrays.aset("fields->E", E)
        arrays = arrays.aset("fields->H", H)
        arrays = arrays.aset("dispersive_P_curr", P_curr)
        arrays = arrays.aset("dispersive_P_prev", P_prev)

        arrays, config = _add_gradient_config(arrays, config, obj)

        E_orig = E
        H_orig = H
        Pc_orig = P_curr
        Pp_orig = P_prev

        state_fwd = forward(
            state=(jnp.asarray(0, dtype=jnp.int32), arrays),
            config=config,
            objects=obj,
            key=key,
            record_detectors=False,
            record_boundaries=False,
            simulate_boundaries=True,
        )
        state_bwd = backward(
            state=state_fwd,
            config=config,
            objects=obj,
            key=key,
            record_detectors=False,
            reset_fields=False,
        )
        _, arrays_bwd = state_bwd

        # Magnitude-relative tolerance: exposes drift in low-amplitude regions
        # that a fixed atol=1e-5 vs |E|~1e-3 (1% absolute) silently masks.
        # Float32 single-step floor is ~1e-5 relative; 1e-4 leaves clear margin
        # but catches an O(0.1%) algebraic factor error.
        rel_tol = 1e-4
        rel_E = _max_relative_error(arrays_bwd.fields.E, E_orig)
        rel_H = _max_relative_error(arrays_bwd.fields.H, H_orig)
        rel_Pc = _max_relative_error(arrays_bwd.dispersive_P_curr, Pc_orig)
        rel_Pp = _max_relative_error(arrays_bwd.dispersive_P_prev, Pp_orig)
        assert rel_E < rel_tol, f"E rel err: {rel_E:.3e}"
        assert rel_H < rel_tol, f"H rel err: {rel_H:.3e}"
        assert rel_Pc < rel_tol, f"P_curr rel err: {rel_Pc:.3e}"
        assert rel_Pp < rel_tol, f"P_prev rel err: {rel_Pp:.3e}"


class TestStrictReversibilityLossy:
    """High-precision multi-step reversibility for a Lorentz-dispersive slab
    with electric conductivity.

    Float64 + 10 steps + magnitude-relative tolerance pushes the precision
    floor to ~1e-12. Any algebraic factor mis-distribution in the reverse
    E update — e.g. dropping the lossy factor on the polarization-delta term
    — shows up as ``rel_err`` orders of magnitude above this floor.
    """

    @staticmethod
    def _build():
        config = SimulationConfig(
            time=_SIM_TIME,
            resolution=_RESOLUTION,
            backend="cpu",
            dtype=jnp.float64,
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
            override_types=_uniform_boundaries("periodic"),
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

        key = jax.random.PRNGKey(0)
        obj_container, arrays, params, config, _ = fdtdx.place_objects(
            object_list=objects,
            config=config,
            constraints=constraints,
            key=key,
        )
        arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
        return obj_container, arrays, config

    def test_fields_and_polarization_reconstructed_to_machine_precision(self):
        with _x64_enabled():
            obj, arrays, config = self._build()
            assert arrays.fields.E.dtype == jnp.float64, "Strict test requires float64 fields"
            originals, reconstructed = _multi_step_roundtrip(
                obj, arrays, config, has_pml=False, n_steps=_STRICT_N_STEPS, seed_dispersive=True
            )
            for name in ("E", "H", "P_curr", "P_prev"):
                rel = _max_relative_error(reconstructed[name], originals[name])
                assert rel < _STRICT_REL_TOL, f"{name} rel err over {_STRICT_N_STEPS} steps: {rel:.3e}"


class TestTimeReversalMagneticConductivity:
    """Reversibility with nonzero magnetic conductivity (σ_H) — the symmetric
    branch of the lossy update. Currently no other test exercises the
    ``magnetic_conductivity`` path as a reversibility target.
    """

    @staticmethod
    def _build():
        config = SimulationConfig(
            time=_SIM_TIME,
            resolution=_RESOLUTION,
            backend="cpu",
            dtype=jnp.float64,
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
            override_types=_uniform_boundaries("periodic"),
        )
        bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
        objects.extend(bound_dict.values())
        constraints.extend(c_list)

        material = fdtdx.Material(
            permittivity=1.0,
            permeability=2.0,
            magnetic_conductivity=1e2,
        )
        slab_cells = _VOLUME_CELLS // 2
        slab = fdtdx.UniformMaterialObject(
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

        key = jax.random.PRNGKey(0)
        obj_container, arrays, params, config, _ = fdtdx.place_objects(
            object_list=objects,
            config=config,
            constraints=constraints,
            key=key,
        )
        arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
        return obj_container, arrays, config

    def test_fields_reconstructed_to_machine_precision(self):
        with _x64_enabled():
            obj, arrays, config = self._build()
            assert arrays.magnetic_conductivity is not None
            assert jnp.any(arrays.magnetic_conductivity != 0), "Slab must have nonzero σ_H"
            originals, reconstructed = _multi_step_roundtrip(
                obj, arrays, config, has_pml=False, n_steps=_STRICT_N_STEPS, seed_dispersive=False
            )
            for name in ("E", "H"):
                rel = _max_relative_error(reconstructed[name], originals[name])
                assert rel < _STRICT_REL_TOL, f"{name} rel err over {_STRICT_N_STEPS} steps: {rel:.3e}"


class TestTimeReversalDispersiveDrude:
    """Reversibility for a Drude-pole material (ω₀ = 0).

    Drude exercises a c1 = 2 / (1 + γdt/2) regime distinct from Lorentz; if
    the ADE reverse inversion assumes Lorentz-only parameter ranges it
    silently mis-reconstructs.
    """

    @staticmethod
    def _build():
        config = SimulationConfig(
            time=_SIM_TIME,
            resolution=_RESOLUTION,
            backend="cpu",
            dtype=jnp.float64,
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
            override_types=_uniform_boundaries("periodic"),
        )
        bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
        objects.extend(bound_dict.values())
        constraints.extend(c_list)

        material = fdtdx.Material(
            permittivity=1.0,
            dispersion=fdtdx.DispersionModel(
                poles=(fdtdx.DrudePole(plasma_frequency=2e15, damping=5e13),),
            ),
        )
        slab_cells = _VOLUME_CELLS // 2
        slab = fdtdx.UniformMaterialObject(
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

        key = jax.random.PRNGKey(0)
        obj_container, arrays, params, config, _ = fdtdx.place_objects(
            object_list=objects,
            config=config,
            constraints=constraints,
            key=key,
        )
        arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
        return obj_container, arrays, config

    def test_fields_and_polarization_reconstructed_to_machine_precision(self):
        with _x64_enabled():
            obj, arrays, config = self._build()
            assert arrays.dispersive_c3 is not None
            # Drude has bounded-away-from-zero c2 in the lossy regime — required
            # for the reverse-time inversion of the ADE recurrence to be stable.
            c2_slab = arrays.dispersive_c2[arrays.dispersive_c3 != 0]
            assert jnp.all(jnp.abs(c2_slab) > 1e-3), "Drude c2 must be bounded away from 0 for reversibility"
            originals, reconstructed = _multi_step_roundtrip(
                obj, arrays, config, has_pml=False, n_steps=_STRICT_N_STEPS, seed_dispersive=True
            )
            for name in ("E", "H", "P_curr", "P_prev"):
                rel = _max_relative_error(reconstructed[name], originals[name])
                assert rel < _STRICT_REL_TOL, f"{name} rel err over {_STRICT_N_STEPS} steps: {rel:.3e}"


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
        assert arrays_fwd.recording_state is not None
        for k, v in arrays_fwd.recording_state.data.items():
            assert jnp.issubdtype(v.dtype, jnp.complexfloating), (
                f"Recorded interface '{k}' has dtype {v.dtype}, expected complex"
            )


# ── Test: gradient with PML + Bloch (complex fields) ───────────────────────────


class TestGradientDispersiveLorentz:
    """End-to-end gradient test through a full reversible FDTD run with a
    Lorentz dispersive slab.

    Exercises the custom VJP with the dispersive polarization state threaded
    through forward and backward passes, verifying that ``reversible_fdtd``
    produces finite gradients w.r.t. ``inv_permittivities`` when the ADE
    update path is active.
    """

    @staticmethod
    def _build():
        obj, arrays, config = TestTimeReversalDispersiveLorentz._build()
        arrays, config = _add_gradient_config(arrays, config, obj)
        key = jax.random.PRNGKey(7)
        arrays = _seed_fields(arrays, key, obj)
        return obj, arrays, config

    @staticmethod
    def _loss_fn(inv_permittivities, arrays, objects, config, key):
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
        _, out = reversible_fdtd(arrays, objects, config, key, show_progress=False)
        return jnp.sum(jnp.real(out.fields.E) ** 2)

    def test_dispersive_arrays_allocated(self):
        _, arrays, _ = self._build()
        assert arrays.dispersive_P_curr is not None
        assert arrays.dispersive_P_prev is not None
        assert arrays.dispersive_c1 is not None

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


class TestGradientDispersiveLossy:
    """End-to-end gradient test through a reversible FDTD run with a
    Lorentz-dispersive slab that *also* has nonzero ``electric_conductivity``.

    This combination exercises the reverse E update's lossy + ADE branch — the
    exact code path that the reverse-factoring bug would silently poison.
    The scene injects fields through a ``PointDipoleSource`` (reversible_fdtd
    resets the seeded E/H at the start, so nonzero fields have to be driven
    by a source). We verify (a) finite gradients and (b) agreement with a
    central-difference finite-difference estimate at a single interior voxel:
    if the backward pass drifts (e.g. because the reverse factor is incorrectly
    distributed), AD and FD disagree even when both are finite.
    """

    @staticmethod
    def _build():
        config = SimulationConfig(
            time=_SIM_TIME,
            resolution=_RESOLUTION,
            backend="cpu",
            dtype=jnp.float32,
            courant_factor=0.99,
            gradient_config=None,
        )
        objects, constraints = [], []
        volume = fdtdx.SimulationVolume(partial_grid_shape=(_VOLUME_CELLS, _VOLUME_CELLS, _VOLUME_CELLS))
        objects.append(volume)

        bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
            thickness=_PML_CELLS,
            override_types=_uniform_boundaries("periodic"),
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

        # CW dipole source at the slab center — drives nonzero fields into the
        # dispersive/lossy medium throughout the run.
        omega = 2.0 * jnp.pi * 3e8 / 800e-9
        source = fdtdx.PointDipoleSource(
            name="dip",
            partial_grid_shape=(1, 1, 1),
            wave_character=fdtdx.WaveCharacter(frequency=float(omega) / (2.0 * float(jnp.pi))),
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
        obj_container, arrays, params, config, _ = fdtdx.place_objects(
            object_list=objects,
            config=config,
            constraints=constraints,
            key=key,
        )
        arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
        arrays, config = _add_gradient_config(arrays, config, obj_container)
        return obj_container, arrays, config

    @staticmethod
    def _loss_fn(inv_permittivities, arrays, objects, config, key):
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
        _, out = reversible_fdtd(arrays, objects, config, key, show_progress=False)
        return jnp.sum(jnp.real(out.fields.E) ** 2)

    def test_scene_has_active_loss_and_dispersion(self):
        """Sanity: the medium must actually carry both σ_E and pole coefficients."""
        _, arrays, _ = self._build()
        assert arrays.electric_conductivity is not None
        assert jnp.any(arrays.electric_conductivity != 0), "Slab must have nonzero σ_E"
        assert arrays.dispersive_c3 is not None
        assert jnp.any(arrays.dispersive_c3 != 0), "Slab must have nonzero pole coefficients"

    def test_gradients_are_finite_and_nonzero(self):
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
        assert loss > 0, "Loss must be positive — the source must drive nonzero fields"
        assert jnp.any(grads != 0), "Gradient must be nonzero — reverse pass must propagate through Maxwell"

    def test_gradient_matches_finite_difference(self):
        """AD gradient for a single interior voxel matches central FD.

        This is the sharpest test of reverse-pass correctness: AD and FD must
        produce the same number (within tolerance) regardless of any
        algebraic rearrangement inside the reverse step. A wrong factor
        distribution in the reverse E update leaves AD finite but numerically
        skewed relative to FD.
        """
        obj, arrays, config = self._build()
        key = jax.random.PRNGKey(99)
        inv_eps = arrays.inv_permittivities

        _, grads = jax.value_and_grad(self._loss_fn)(inv_eps, arrays, obj, config, key)

        # Find an interior voxel where both σ_E and the pole coefficients are
        # active — that's where the lossy+dispersive reverse branch actually
        # fires. Dispersive c3 is (num_poles, 1, Nx, Ny, Nz); a nonzero c3
        # implies a slab voxel.
        c3 = arrays.dispersive_c3[0, 0]
        xs, ys, zs = jnp.where(c3 != 0, size=1, fill_value=0)
        voxel = (int(xs[0]), int(ys[0]), int(zs[0]))
        # pick the polarization component the dipole drives (axis 0)
        idx = (0, *voxel)

        # Scale the step to the magnitude of the voxel being probed so that
        # float32 round-off doesn't swamp the finite-difference numerator.
        h = 1e-3 * float(jnp.abs(inv_eps[idx])) + 1e-5
        inv_eps_plus = inv_eps.at[idx].add(h)
        inv_eps_minus = inv_eps.at[idx].add(-h)
        loss_plus = self._loss_fn(inv_eps_plus, arrays, obj, config, key)
        loss_minus = self._loss_fn(inv_eps_minus, arrays, obj, config, key)
        fd = (loss_plus - loss_minus) / (2.0 * h)

        ad = grads[idx]
        # Tolerance wide enough for float32 FD noise over the full reversible
        # pipeline but tight enough to catch a dropped algebraic factor.
        diff = jnp.abs(ad - fd)
        scale = jnp.abs(fd) + jnp.abs(ad) + 1e-12
        rel_err = diff / scale
        assert rel_err < 0.1 or diff < 1e-5, (
            f"AD vs FD mismatch at {idx}: AD={float(ad):.6e}, FD={float(fd):.6e}, rel_err={float(rel_err):.3e}"
        )

    @staticmethod
    def _build_float64():
        """Identical scene to ``_build`` but in float64 so FD noise drops to
        ~1e-12 and the tolerance can tighten by three orders of magnitude."""
        config = SimulationConfig(
            time=_SIM_TIME,
            resolution=_RESOLUTION,
            backend="cpu",
            dtype=jnp.float64,
            courant_factor=0.99,
            gradient_config=None,
        )
        objects, constraints = [], []
        volume = fdtdx.SimulationVolume(partial_grid_shape=(_VOLUME_CELLS, _VOLUME_CELLS, _VOLUME_CELLS))
        objects.append(volume)
        bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
            thickness=_PML_CELLS,
            override_types=_uniform_boundaries("periodic"),
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

        omega = 2.0 * jnp.pi * 3e8 / 800e-9
        source = fdtdx.PointDipoleSource(
            name="dip",
            partial_grid_shape=(1, 1, 1),
            wave_character=fdtdx.WaveCharacter(frequency=float(omega) / (2.0 * float(jnp.pi))),
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
        obj_container, arrays, params, config, _ = fdtdx.place_objects(
            object_list=objects,
            config=config,
            constraints=constraints,
            key=key,
        )
        arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
        arrays, config = _add_gradient_config(arrays, config, obj_container)
        return obj_container, arrays, config

    def test_gradient_matches_finite_difference_float64_multi_voxel(self):
        """Multi-voxel float64 FD vs AD agreement.

        Tightens the single-voxel float32 test in two independent directions:
        (1) float64 drops FD noise to ~1e-12, allowing ``rel_err < 1e-3``
        instead of 0.1; (2) sweeping every lossy+dispersive voxel catches a
        bug that cancels at one voxel but not another (e.g. a sign or
        component-index error in the reverse pass).
        """
        with _x64_enabled():
            obj, arrays, config = self._build_float64()
            assert arrays.fields.E.dtype == jnp.float64
            key = jax.random.PRNGKey(99)
            inv_eps = arrays.inv_permittivities
            _, grads = jax.value_and_grad(TestGradientDispersiveLossy._loss_fn)(inv_eps, arrays, obj, config, key)

            # Find every voxel where BOTH σ_E and c3 are active — the lossy+dispersive
            # reverse branch fires only here. Use polarization axis 0 (the dipole).
            c3 = arrays.dispersive_c3[0, 0]  # (Nx, Ny, Nz)
            # electric_conductivity is shape (9, Nx, Ny, Nz) (full 9-tuple) or
            # (3, ...) or (1, ...). For isotropic the diagonal component is at index 0.
            sigma_diag = arrays.electric_conductivity[0]
            active = (c3 != 0) & (sigma_diag != 0)
            xs, ys, zs = jnp.where(active, size=8, fill_value=-1)

            checked = 0
            for xi, yi, zi in zip(xs.tolist(), ys.tolist(), zs.tolist(), strict=True):
                if xi < 0:
                    continue  # padding from size=8 fill_value
                idx = (0, int(xi), int(yi), int(zi))
                h = 1e-4 * float(jnp.abs(inv_eps[idx])) + 1e-7
                inv_eps_plus = inv_eps.at[idx].add(h)
                inv_eps_minus = inv_eps.at[idx].add(-h)
                loss_plus = TestGradientDispersiveLossy._loss_fn(inv_eps_plus, arrays, obj, config, key)
                loss_minus = TestGradientDispersiveLossy._loss_fn(inv_eps_minus, arrays, obj, config, key)
                fd = (loss_plus - loss_minus) / (2.0 * h)
                ad = grads[idx]
                diff = float(jnp.abs(ad - fd))
                scale = float(jnp.abs(fd) + jnp.abs(ad)) + 1e-12
                rel_err = diff / scale
                assert rel_err < 1e-3 or diff < 1e-9, (
                    f"AD vs FD mismatch at {idx}: AD={float(ad):.6e}, FD={float(fd):.6e}, "
                    f"rel_err={rel_err:.3e}, diff={diff:.3e}"
                )
                checked += 1
            assert checked >= 1, "No active (σ_E≠0 ∧ c3≠0) voxels found — scene misconfigured"


class TestGradientMagneticConductivity:
    """Float64 AD vs FD for ``inv_permeabilities`` in a magnetically lossy slab.

    Symmetric to ``TestGradientDispersiveLossy`` but on the magnetic side.
    The reverse H update has its own ``(1 ± c σ_H inv_μ / (2 η₀))`` factor —
    a bug here would not show up in any electric-conductivity test.
    """

    @staticmethod
    def _build():
        config = SimulationConfig(
            time=_SIM_TIME,
            resolution=_RESOLUTION,
            backend="cpu",
            dtype=jnp.float64,
            courant_factor=0.99,
            gradient_config=None,
        )
        objects, constraints = [], []
        volume = fdtdx.SimulationVolume(partial_grid_shape=(_VOLUME_CELLS, _VOLUME_CELLS, _VOLUME_CELLS))
        objects.append(volume)
        bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
            thickness=_PML_CELLS,
            override_types=_uniform_boundaries("periodic"),
        )
        bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
        objects.extend(bound_dict.values())
        constraints.extend(c_list)

        material = fdtdx.Material(
            permittivity=1.0,
            permeability=2.0,
            magnetic_conductivity=1e2,
        )
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

        omega = 2.0 * jnp.pi * 3e8 / 800e-9
        source = fdtdx.PointDipoleSource(
            name="dip",
            partial_grid_shape=(1, 1, 1),
            wave_character=fdtdx.WaveCharacter(frequency=float(omega) / (2.0 * float(jnp.pi))),
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
        obj_container, arrays, params, config, _ = fdtdx.place_objects(
            object_list=objects,
            config=config,
            constraints=constraints,
            key=key,
        )
        arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
        arrays, config = _add_gradient_config(arrays, config, obj_container)
        return obj_container, arrays, config

    @staticmethod
    def _loss_fn(inv_permeabilities, arrays, objects, config, key):
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
            inv_permittivities=arrays.inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            detector_states=arrays.detector_states,
            recording_state=arrays.recording_state,
            electric_conductivity=arrays.electric_conductivity,
            magnetic_conductivity=arrays.magnetic_conductivity,
        )
        _, out = reversible_fdtd(arrays, objects, config, key, show_progress=False)
        return jnp.sum(jnp.real(out.fields.H) ** 2)

    def test_gradient_matches_finite_difference_multi_voxel(self):
        with _x64_enabled():
            obj, arrays, config = self._build()
            assert arrays.magnetic_conductivity is not None
            assert jnp.any(arrays.magnetic_conductivity != 0)
            key = jax.random.PRNGKey(99)
            inv_mu = arrays.inv_permeabilities
            _, grads = jax.value_and_grad(self._loss_fn)(inv_mu, arrays, obj, config, key)

            # Active voxels: σ_H ≠ 0. Component index 0 of the 9-tuple is the diagonal.
            sigma_diag = arrays.magnetic_conductivity[0]
            active = sigma_diag != 0
            xs, ys, zs = jnp.where(active, size=8, fill_value=-1)
            checked = 0
            for xi, yi, zi in zip(xs.tolist(), ys.tolist(), zs.tolist(), strict=True):
                if xi < 0:
                    continue
                idx = (0, int(xi), int(yi), int(zi))
                h = 1e-4 * float(jnp.abs(inv_mu[idx])) + 1e-7
                inv_mu_plus = inv_mu.at[idx].add(h)
                inv_mu_minus = inv_mu.at[idx].add(-h)
                loss_plus = self._loss_fn(inv_mu_plus, arrays, obj, config, key)
                loss_minus = self._loss_fn(inv_mu_minus, arrays, obj, config, key)
                fd = (loss_plus - loss_minus) / (2.0 * h)
                ad = grads[idx]
                diff = float(jnp.abs(ad - fd))
                scale = float(jnp.abs(fd) + jnp.abs(ad)) + 1e-12
                rel_err = diff / scale
                assert rel_err < 1e-3 or diff < 1e-9, (
                    f"AD vs FD mismatch (μ⁻¹) at {idx}: AD={float(ad):.6e}, FD={float(fd):.6e}, rel_err={rel_err:.3e}"
                )
                checked += 1
            assert checked >= 1, "No active σ_H voxels found — scene misconfigured"


class TestGradientPMLBlochComplex:
    """End-to-end gradient test with PML + Bloch (complex fields).

    Verifies that the reversible_fdtd custom VJP works when the recorder
    stores complex-valued field interfaces.
    """

    @staticmethod
    def _build():
        """Build a small PML+Bloch simulation with a dipole source.

        ``reversible_fdtd`` resets the field arrays on entry, so a seeded E/H
        alone would leave the simulation at zero the whole run and the
        gradient would be trivially zero. A CW ``PointDipoleSource`` at the
        volume center drives nonzero complex fields throughout the run so
        the loss on ``out.fields.E`` genuinely depends on ``inv_permittivities``.
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
                "min_x": "bloch",
                "max_x": "bloch",
                "min_y": "bloch",
                "max_y": "bloch",
                "min_z": "pml",
                "max_z": "pml",
            },
            bloch_vector=(1e6, 1e6, 0.0),
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
        arrays, config = _add_gradient_config(arrays, config, obj)
        return obj, arrays, config

    @staticmethod
    def _loss_fn(inv_permittivities, arrays, objects, config, key):
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
        # Loss on the evolved E field (not inv_permittivities) ensures the
        # gradient flows through the Maxwell update via curl * inv_eps.
        # Using |E|^2 handles complex-valued fields (Bloch k ≠ 0).
        return jnp.sum(jnp.abs(out.fields.E) ** 2)

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
        assert jnp.any(grads != 0), "Gradients should be nonzero — VJP must flow through Maxwell updates"

    def test_loss_is_positive(self):
        obj, arrays, config = self._build()
        key = jax.random.PRNGKey(99)
        loss = self._loss_fn(arrays.inv_permittivities, arrays, obj, config, key)
        assert loss > 0, "Loss should be positive with non-zero inv_permittivities"

    def test_fields_are_complex(self):
        """Verify the simulation actually uses complex-valued fields."""
        _, arrays, _ = self._build()
        assert jnp.issubdtype(arrays.fields.E.dtype, jnp.complexfloating)
        assert jnp.issubdtype(arrays.fields.H.dtype, jnp.complexfloating)


# ── Tests: dispersion coefficient gradients ────────────────────────────────────


def _make_disp_loss_fn(coef_name):
    """Build a loss-fn closure that differentiates w.r.t. one of c1/c2/c3.

    Returns a static function compatible with ``jax.value_and_grad`` whose
    first positional argument is the chosen coefficient array. The remaining
    coefficients are taken from ``arrays``.
    """

    def loss_fn(coef_value, arrays, objects, config, key, _fdtd):
        kwargs = {
            "dispersive_c1": arrays.dispersive_c1,
            "dispersive_c2": arrays.dispersive_c2,
            "dispersive_c3": arrays.dispersive_c3,
        }
        kwargs[f"dispersive_{coef_name}"] = coef_value
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
            dispersive_inv_c2=arrays.dispersive_inv_c2,
            **kwargs,
        )
        _, out = _fdtd(arr, objects, config, key, show_progress=False)
        return jnp.sum(jnp.real(out.fields.E) ** 2)

    return loss_fn


def _coef_fd_check(coef_arr, loss_fn, arrays, obj, config, key, h_scale, h_floor, fdtd_impl):
    """Central-FD vs AD agreement at the first nonzero voxel of ``coef_arr[0,0]``.

    Returns ``(ad, fd, rel_err, diff, idx)`` for the assertion site to format.
    """
    _, grads = jax.value_and_grad(loss_fn)(coef_arr, arrays, obj, config, key, fdtd_impl)
    coef_mid = coef_arr[0, 0]
    xs, ys, zs = jnp.where(coef_mid != 0, size=1, fill_value=0)
    idx = (0, 0, int(xs[0]), int(ys[0]), int(zs[0]))
    h = h_scale * float(jnp.abs(coef_arr[idx])) + h_floor
    coef_plus = coef_arr.at[idx].add(h)
    coef_minus = coef_arr.at[idx].add(-h)
    loss_plus = loss_fn(coef_plus, arrays, obj, config, key, fdtd_impl)
    loss_minus = loss_fn(coef_minus, arrays, obj, config, key, fdtd_impl)
    fd = (loss_plus - loss_minus) / (2.0 * h)
    ad = grads[idx]
    diff = float(jnp.abs(ad - fd))
    scale = float(jnp.abs(fd) + jnp.abs(ad)) + 1e-12
    rel_err = diff / scale
    return ad, fd, rel_err, diff, idx


class TestDispersiveCoefficientGradientReversible:
    """Verify that ``dispersive_c1/c2/c3`` are primal VJP inputs of the
    reversible FDTD path, i.e. the AD gradient w.r.t. each coefficient at
    an interior dispersive voxel matches a central finite-difference estimate.
    """

    @staticmethod
    def _build():
        return TestGradientDispersiveLossy._build()

    @staticmethod
    def _loss_fn(dispersive_c3, arrays, objects, config, key):
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
            dispersive_c3=dispersive_c3,
            dispersive_inv_c2=arrays.dispersive_inv_c2,
        )
        _, out = reversible_fdtd(arrays, objects, config, key, show_progress=False)
        return jnp.sum(jnp.real(out.fields.E) ** 2)

    def test_gradient_through_c3_matches_finite_difference(self):
        obj, arrays, config = self._build()
        key = jax.random.PRNGKey(99)
        c3 = arrays.dispersive_c3
        assert c3 is not None

        _, grads = jax.value_and_grad(self._loss_fn)(c3, arrays, obj, config, key)

        c3_mid = c3[0, 0]
        xs, ys, zs = jnp.where(c3_mid != 0, size=1, fill_value=0)
        idx = (0, 0, int(xs[0]), int(ys[0]), int(zs[0]))

        # c3 ~ K * dt^2 is small (O(1e-2) here). Scale FD step to magnitude so
        # float32 round-off doesn't dominate the numerator.
        h = 1e-2 * float(jnp.abs(c3[idx])) + 1e-6
        c3_plus = c3.at[idx].add(h)
        c3_minus = c3.at[idx].add(-h)
        loss_plus = self._loss_fn(c3_plus, arrays, obj, config, key)
        loss_minus = self._loss_fn(c3_minus, arrays, obj, config, key)
        fd = (loss_plus - loss_minus) / (2.0 * h)

        ad = grads[idx]
        diff = jnp.abs(ad - fd)
        scale = jnp.abs(fd) + jnp.abs(ad) + 1e-12
        rel_err = diff / scale
        assert rel_err < 0.1 or diff < 1e-5, (
            f"AD vs FD mismatch at {idx}: AD={float(ad):.6e}, FD={float(fd):.6e}, rel_err={float(rel_err):.3e}"
        )

    def test_gradient_through_c1_matches_finite_difference(self):
        """c1 appears in both ``update_E`` and ``update_E_reverse`` — a primal
        of the reversible VJP. Currently only c3 was tested; an algebraic
        error specific to c1 (e.g. wrong sign in the reverse recurrence
        inversion) would slip past."""
        obj, arrays, config = self._build()
        key = jax.random.PRNGKey(99)
        c1 = arrays.dispersive_c1
        assert c1 is not None
        loss_fn = _make_disp_loss_fn("c1")
        # c1 ~ O(1) in the physical regime, so a multiplicative step ~1e-3
        # gives well-resolved FD without truncation error.
        ad, fd, rel_err, diff, idx = _coef_fd_check(
            c1, loss_fn, arrays, obj, config, key, h_scale=1e-3, h_floor=1e-6, fdtd_impl=reversible_fdtd
        )
        assert rel_err < 0.1 or diff < 1e-5, (
            f"AD vs FD mismatch (c1) at {idx}: AD={float(ad):.6e}, FD={float(fd):.6e}, rel_err={float(rel_err):.3e}"
        )

    def test_gradient_through_c2_matches_finite_difference(self):
        """c2 is the inverted coefficient in the reverse recurrence:
        ``P^(n-1) = (P^(n+1) - c1 P^n - c3 E^n) / c2``. ``dispersive_inv_c2``
        is the cached 1/c2 (stop_gradient'd), so gradients must flow through
        c2 only. A wrong-coefficient-derivative bug would only show here."""
        obj, arrays, config = self._build()
        key = jax.random.PRNGKey(99)
        c2 = arrays.dispersive_c2
        assert c2 is not None
        # Physical c2 ~ -1; perturb a few percent.
        loss_fn = _make_disp_loss_fn("c2")
        ad, fd, rel_err, diff, idx = _coef_fd_check(
            c2, loss_fn, arrays, obj, config, key, h_scale=1e-3, h_floor=1e-6, fdtd_impl=reversible_fdtd
        )
        assert rel_err < 0.1 or diff < 1e-5, (
            f"AD vs FD mismatch (c2) at {idx}: AD={float(ad):.6e}, FD={float(fd):.6e}, rel_err={float(rel_err):.3e}"
        )


class TestDispersiveCoefficientGradientCheckpointed:
    """Same verification as ``TestDispersiveCoefficientGradientReversible``
    but for the checkpointed path, which uses standard autodiff through
    ``eqxi.while_loop`` so gradients flow naturally.
    """

    @staticmethod
    def _build():
        obj, arrays, config = TestGradientDispersiveLossy._build()
        grad_cfg = GradientConfig(method="checkpointed", num_checkpoints=8)
        config = config.aset("gradient_config", grad_cfg)
        return obj, arrays, config

    @staticmethod
    def _loss_fn(dispersive_c3, arrays, objects, config, key):
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
            dispersive_c3=dispersive_c3,
            dispersive_inv_c2=arrays.dispersive_inv_c2,
        )
        _, out = checkpointed_fdtd(arrays, objects, config, key, show_progress=False)
        return jnp.sum(jnp.real(out.fields.E) ** 2)

    def test_gradient_through_c3_matches_finite_difference(self):
        obj, arrays, config = self._build()
        key = jax.random.PRNGKey(99)
        c3 = arrays.dispersive_c3
        assert c3 is not None

        _, grads = jax.value_and_grad(self._loss_fn)(c3, arrays, obj, config, key)

        c3_mid = c3[0, 0]
        xs, ys, zs = jnp.where(c3_mid != 0, size=1, fill_value=0)
        idx = (0, 0, int(xs[0]), int(ys[0]), int(zs[0]))

        h = 1e-2 * float(jnp.abs(c3[idx])) + 1e-6
        c3_plus = c3.at[idx].add(h)
        c3_minus = c3.at[idx].add(-h)
        loss_plus = self._loss_fn(c3_plus, arrays, obj, config, key)
        loss_minus = self._loss_fn(c3_minus, arrays, obj, config, key)
        fd = (loss_plus - loss_minus) / (2.0 * h)

        ad = grads[idx]
        diff = jnp.abs(ad - fd)
        scale = jnp.abs(fd) + jnp.abs(ad) + 1e-12
        rel_err = diff / scale
        assert rel_err < 0.1 or diff < 1e-5, (
            f"AD vs FD mismatch at {idx}: AD={float(ad):.6e}, FD={float(fd):.6e}, rel_err={float(rel_err):.3e}"
        )

    def test_gradient_through_c1_matches_finite_difference(self):
        """Checkpointed path uses standard autodiff through ``eqxi.while_loop``
        so c1 gradient flows naturally. Mirrors the reversible-path test."""
        obj, arrays, config = self._build()
        key = jax.random.PRNGKey(99)
        c1 = arrays.dispersive_c1
        assert c1 is not None
        loss_fn = _make_disp_loss_fn("c1")
        ad, fd, rel_err, diff, idx = _coef_fd_check(
            c1, loss_fn, arrays, obj, config, key, h_scale=1e-3, h_floor=1e-6, fdtd_impl=checkpointed_fdtd
        )
        assert rel_err < 0.1 or diff < 1e-5, (
            f"AD vs FD mismatch (c1) at {idx}: AD={float(ad):.6e}, FD={float(fd):.6e}, rel_err={float(rel_err):.3e}"
        )

    def test_gradient_through_c2_matches_finite_difference(self):
        """Checkpointed-path c2 gradient via standard autodiff."""
        obj, arrays, config = self._build()
        key = jax.random.PRNGKey(99)
        c2 = arrays.dispersive_c2
        assert c2 is not None
        loss_fn = _make_disp_loss_fn("c2")
        ad, fd, rel_err, diff, idx = _coef_fd_check(
            c2, loss_fn, arrays, obj, config, key, h_scale=1e-3, h_floor=1e-6, fdtd_impl=checkpointed_fdtd
        )
        assert rel_err < 0.1 or diff < 1e-5, (
            f"AD vs FD mismatch (c2) at {idx}: AD={float(ad):.6e}, FD={float(fd):.6e}, rel_err={float(rel_err):.3e}"
        )
