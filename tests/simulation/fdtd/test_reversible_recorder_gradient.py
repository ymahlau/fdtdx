"""The reversible adjoint's recorder path, exercised through an actual gradient.

:class:`~fdtdx.interfaces.recorder.Recorder` exists to store the PML interface values that the
reverse reconstruction reads back, but every gradient test in ``test_fdtd.py`` runs a *fully
periodic* scene. A periodic scene has no PML objects, so ``input_shape_dtypes`` is empty, nothing
is ever recorded, and the recorder's reconstruction is never called -- the machinery under test is
inert. The tests here put a PML on all six faces so the interface record is real, and then take a
gradient with both an exact recorder and a compressing one.
"""

import jax
import jax.numpy as jnp
import pytest

import fdtdx
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.core.grid import UniformGrid
from fdtdx.fdtd.fdtd import checkpointed_fdtd, reversible_fdtd
from fdtdx.interfaces.recorder import Recorder
from fdtdx.interfaces.time_filter import LinearReconstructEveryK

_RESOLUTION = 50e-9
_SIM_TIME = 40e-15
_PML_CELLS = 4
_VOLUME_CELLS = 16  # 4 + 4 cells of PML per axis still leaves an 8-cell interior


def _build_pml_scene(dtype):
    """Dielectric slab + dipole inside a box that is absorbing on all six faces."""
    config = SimulationConfig(
        time=_SIM_TIME,
        grid=UniformGrid(spacing=_RESOLUTION),
        backend="cpu",
        dtype=dtype,
        courant_factor=0.99,
        gradient_config=None,
    )
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(_VOLUME_CELLS,) * 3)
    objects.append(volume)
    # The whole point: real PML on every face, so there is an interface record to reconstruct.
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML_CELLS)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    # Non-dispersive: reversible_fdtd rejects the ADE polarisation recurrence.
    slab = fdtdx.UniformMaterialObject(
        name="slab",
        partial_grid_shape=(None, None, _VOLUME_CELLS // 4),
        material=fdtdx.Material(permittivity=2.0),
    )
    constraints.extend(
        [
            slab.same_size(volume, axes=(0, 1)),
            slab.place_at_center(volume, axes=(0, 1)),
            slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_VOLUME_CELLS // 2,)),
        ]
    )
    objects.append(slab)

    source = fdtdx.PointDipoleSource(
        name="dip",
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(frequency=c0 / 800e-9),
        polarization=0,
        amplitude=1.0,
    )
    constraints.append(
        source.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(_VOLUME_CELLS // 2,) * 3)
    )
    objects.append(source)

    key = jax.random.PRNGKey(0)
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)
    assert obj.pml_objects, "scene must have PML objects or this file tests nothing"
    return obj, arrays, config


def _attach(arrays, config, obj, method, modules):
    input_shape_dtypes = {}
    field_dtype = arrays.fields.E.dtype
    for boundary in obj.pml_objects:
        extended_shape = (3, *boundary.interface_grid_shape())
        input_shape_dtypes[f"{boundary.name}_E"] = jax.ShapeDtypeStruct(extended_shape, field_dtype)
        input_shape_dtypes[f"{boundary.name}_H"] = jax.ShapeDtypeStruct(extended_shape, field_dtype)
    assert input_shape_dtypes, "no interface to record"
    recorder, recording_state = Recorder(modules=modules).init_state(
        input_shape_dtypes=input_shape_dtypes,
        max_time_steps=config.time_steps_total,
        backend="cpu",
    )
    if method == "reversible":
        grad_cfg = GradientConfig(method="reversible", recorder=recorder)
    else:
        grad_cfg = GradientConfig(method="checkpointed", num_checkpoints=8)
    return arrays.aset("recording_state", recording_state), config.aset("gradient_config", grad_cfg)


def _grad(method, modules, dtype=jnp.float32):
    obj, arrays, config = _build_pml_scene(dtype)
    arrays, config = _attach(arrays, config, obj, method, modules)

    def loss_fn(inv_eps):
        arr = arrays.aset("inv_permittivities", inv_eps)
        impl = reversible_fdtd if method == "reversible" else checkpointed_fdtd
        _, out = impl(arr, obj, config, jax.random.PRNGKey(99), show_progress=False)
        return jnp.sum(jnp.real(out.fields.E) ** 2)

    return jax.value_and_grad(loss_fn)(arrays.inv_permittivities)


@pytest.mark.simulation
def test_gradient_is_finite_with_compressing_recorder():
    """A lossy recorder must still produce a finite gradient.

    ``body_fn`` steps the reconstruction back *before* taking the VJP, so entering it at
    ``time_step = k`` back-propagates forward step ``k - 1``. The forward pass runs steps
    ``0 .. time_steps_total - 1``, so the reverse loop has to stop once it reaches step 0. Running
    one call further asks the recorder for the interface at ``t = -1``. With the exact recorder that
    is an out-of-range read contributing ~1e-7; :class:`LinearReconstructEveryK` instead
    interpolates between two saved frames, and at the degenerate index its weight
    ``(t - prev) / (next - prev)`` becomes ``0 / 0`` -- which turns the entire parameter gradient
    into NaN.
    """
    loss, grad = _grad("reversible", [LinearReconstructEveryK(k=8)])
    assert jnp.isfinite(loss), f"forward loss is not finite: {loss}"
    nonfinite = float(jnp.mean(~jnp.isfinite(grad)))
    assert nonfinite == 0.0, (
        f"{100 * nonfinite:.1f}% of d(loss)/d(inv_permittivities) is non-finite with a "
        f"compressing recorder; the reverse loop stepped past time step 0"
    )
    assert float(jnp.max(jnp.abs(grad))) > 0.0, "gradient is identically zero"
