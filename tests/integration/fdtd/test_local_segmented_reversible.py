import jax
import jax.numpy as jnp
import numpy as np

import fdtdx
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.core.grid import UniformGrid
from fdtdx.materials import Material
from fdtdx.objects.device.device import Device
from fdtdx.objects.object import GridCoordinateConstraint

_RESOLUTION = 50e-9
_SIM_TIME = 15e-15
_PML_CELLS = 3
_VOLUME_CELLS = 14


def _build_test_scene(gradient_config, with_lossy=False, with_source_overlap=False):
    config = SimulationConfig(
        time=_SIM_TIME,
        grid=UniformGrid(spacing=_RESOLUTION),
        backend="cpu",
        dtype=jnp.float64,
        courant_factor=0.99,
        gradient_config=gradient_config,
    )
    n = _VOLUME_CELLS
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(n, n, n))
    objects.append(volume)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML_CELLS), volume
    )
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    if with_lossy:
        materials = {
            "air": Material(permittivity=1.0),
            "si": Material(permittivity=2.25, electric_conductivity=1e3),
        }
    else:
        materials = {
            "air": Material(permittivity=1.0),
            "si": Material(permittivity=2.25),
        }
    dev = Device(
        name="dev",
        partial_grid_shape=(4, 4, 4),
        partial_voxel_grid_shape=(1, 1, 1),
        materials=materials,
        param_transforms=[],
    )
    c_dev = GridCoordinateConstraint(object="dev", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[5, 5, 5])
    objects.append(dev)
    constraints.append(c_dev)

    wave = fdtdx.WaveCharacter(frequency=c0 / 800e-9)
    src_coord = [4, 5, 5] if with_source_overlap else [3, 3, 3]
    source = fdtdx.PointDipoleSource(
        name="dip", partial_grid_shape=(1, 1, 1), wave_character=wave, polarization=0, amplitude=1.0
    )
    c_source = GridCoordinateConstraint(object="dip", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=src_coord)
    objects.append(source)
    constraints.append(c_source)

    det = fdtdx.PhasorDetector(
        name="det", partial_grid_shape=(4, 4, 1), wave_characters=(wave,), components=("Ex",), plot=False
    )
    c_det = GridCoordinateConstraint(object="det", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[5, 5, 10])
    objects.append(det)
    constraints.append(c_det)

    key = jax.random.PRNGKey(0)
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)
    return obj, arrays, config, key


def _compute_grad(gradient_config):
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        obj, arrays, config, key = _build_test_scene(gradient_config)

        def loss(inv_eps):
            _, out = fdtdx.run_fdtd(arrays.aset("inv_permittivities", inv_eps), obj, config, key, show_progress=False)
            return jnp.sum(jnp.abs(out.detector_states["det"]["phasor"]) ** 2)

        value, grad = jax.value_and_grad(loss)(arrays.inv_permittivities)
        return float(value), np.asarray(grad), obj
    finally:
        jax.config.update("jax_enable_x64", prev)


class TestLocalSegmentedGradient:
    def test_gradient_agreement_with_checkpointed(self):
        # 1. Reference: checkpointed autodiff
        cfg_chk = GradientConfig(method="checkpointed", num_checkpoints=16)
        v_chk, g_chk, obj = _compute_grad(cfg_chk)

        # 2. Local segmented with whole adjoint
        cfg_whole = GradientConfig(
            method="reversible",
            recording_mode="local_segmented",
            num_checkpoints_reversible=1,
            split_detector_adjoint=False,
            host_offload=False,
        )
        v_whole, g_whole, _ = _compute_grad(cfg_whole)

        # 3. Local segmented with split detector adjoint
        cfg_split = GradientConfig(
            method="reversible",
            recording_mode="local_segmented",
            num_checkpoints_reversible=1,
            split_detector_adjoint=True,
            host_offload=False,
        )
        v_split, g_split, _ = _compute_grad(cfg_split)

        # 4. Local segmented with host offload
        cfg_offload = GradientConfig(
            method="reversible",
            recording_mode="local_segmented",
            num_checkpoints_reversible=2,
            split_detector_adjoint=True,
            host_offload=True,
        )
        v_offload, g_offload, _ = _compute_grad(cfg_offload)

        # Losses should be identical
        assert np.isclose(v_whole, v_chk, rtol=1e-10)
        assert np.isclose(v_split, v_chk, rtol=1e-10)
        assert np.isclose(v_offload, v_chk, rtol=1e-10)

        # Gradients on device region should match
        dev_slice = (slice(None), *obj.devices[0].grid_slice)
        g_chk_dev = g_chk[dev_slice]
        g_whole_dev = g_whole[dev_slice]
        g_split_dev = g_split[dev_slice]
        g_offload_dev = g_offload[dev_slice]

        assert np.max(np.abs(g_whole_dev)) > 1e-8, "Gradient should be non-zero"
        # Split detector adjoint vs whole adjoint should be bit-identical or machine precision
        diff_split = np.max(np.abs(g_split_dev - g_whole_dev))
        diff_offload = np.max(np.abs(g_offload_dev - g_whole_dev))
        diff_chk = np.max(np.abs(g_split_dev - g_chk_dev))
        rel_diff = diff_chk / np.max(np.abs(g_chk_dev))
        print(f"diff_split: {diff_split}, diff_offload: {diff_offload}, rel_diff: {rel_diff}")
        np.testing.assert_allclose(g_split_dev, g_whole_dev, rtol=1e-7, atol=1e-12)
        np.testing.assert_allclose(g_offload_dev, g_whole_dev, rtol=1e-7, atol=1e-12)

        # Agreement against exact checkpointed autodiff
        assert rel_diff < 1e-5, f"Relative gradient difference {rel_diff} exceeds threshold"
