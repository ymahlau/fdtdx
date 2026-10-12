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


def _build_test_scene(
    gradient_config,
    with_lossy=False,
    with_source_overlap=False,
    detector_type="phasor",
):
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

    if detector_type == "field":
        det = fdtdx.FieldDetector(name="det", partial_grid_shape=(4, 4, 1), components=("Ex",), plot=False)
    else:
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


def _compute_grad(gradient_config, detector_type="phasor"):
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        obj, arrays, config, key = _build_test_scene(gradient_config, detector_type=detector_type)

        def loss(inv_eps):
            _, out = fdtdx.run_fdtd(arrays.aset("inv_permittivities", inv_eps), obj, config, key, show_progress=False)
            if detector_type == "field":
                return jnp.sum(out.detector_states["det"]["fields"] ** 2)
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

        # 5. Local segmented with 0 checkpoints (default single-slice reversible)
        cfg_zero = GradientConfig(
            method="reversible",
            recording_mode="local_segmented",
        )
        assert cfg_zero.num_checkpoints_reversible == 0
        v_zero, g_zero, _ = _compute_grad(cfg_zero)

        # Losses should be identical
        assert np.isclose(v_whole, v_chk, rtol=1e-10)
        assert np.isclose(v_split, v_chk, rtol=1e-10)
        assert np.isclose(v_offload, v_chk, rtol=1e-10)
        assert np.isclose(v_zero, v_chk, rtol=1e-10)

        # Gradients on device region should match
        dev_slice = (slice(None), *obj.devices[0].grid_slice)
        g_chk_dev = g_chk[dev_slice]
        g_whole_dev = g_whole[dev_slice]
        g_split_dev = g_split[dev_slice]
        g_offload_dev = g_offload[dev_slice]
        g_zero_dev = g_zero[dev_slice]

        assert np.max(np.abs(g_whole_dev)) > 1e-8, "Gradient should be non-zero"
        # Split detector adjoint vs whole adjoint should be bit-identical or machine precision
        diff_split = np.max(np.abs(g_split_dev - g_whole_dev))
        diff_offload = np.max(np.abs(g_offload_dev - g_whole_dev))
        diff_zero = np.max(np.abs(g_zero_dev - g_whole_dev))
        diff_chk = np.max(np.abs(g_split_dev - g_chk_dev))
        rel_diff = diff_chk / np.max(np.abs(g_chk_dev))
        print(f"diff_split: {diff_split}, diff_offload: {diff_offload}, diff_zero: {diff_zero}, rel_diff: {rel_diff}")
        np.testing.assert_allclose(g_split_dev, g_whole_dev, rtol=1e-7, atol=1e-12)
        np.testing.assert_allclose(g_offload_dev, g_whole_dev, rtol=1e-7, atol=1e-12)
        np.testing.assert_allclose(g_zero_dev, g_whole_dev, rtol=1e-7, atol=1e-12)

        # Agreement against exact checkpointed autodiff
        assert rel_diff < 1e-5, f"Relative gradient difference {rel_diff} exceeds threshold"

    def test_field_detector_split_adjoint(self):
        cfg_chk = GradientConfig(method="checkpointed", num_checkpoints=16)
        v_chk, g_chk, obj = _compute_grad(cfg_chk, detector_type="field")

        cfg_whole = GradientConfig(
            method="reversible",
            recording_mode="local_segmented",
            num_checkpoints_reversible=1,
            split_detector_adjoint=False,
        )
        v_whole, g_whole, _ = _compute_grad(cfg_whole, detector_type="field")

        cfg_split = GradientConfig(
            method="reversible",
            recording_mode="local_segmented",
            num_checkpoints_reversible=1,
            split_detector_adjoint=True,
        )
        v_split, g_split, _ = _compute_grad(cfg_split, detector_type="field")

        assert np.isclose(v_whole, v_chk, rtol=1e-10)
        assert np.isclose(v_split, v_chk, rtol=1e-10)

        dev_slice = (slice(None), *obj.devices[0].grid_slice)
        g_chk_dev = g_chk[dev_slice]
        g_whole_dev = g_whole[dev_slice]
        g_split_dev = g_split[dev_slice]

        assert np.max(np.abs(g_whole_dev)) > 1e-8
        np.testing.assert_allclose(g_split_dev, g_whole_dev, rtol=1e-7, atol=1e-12)
        diff_chk = np.max(np.abs(g_split_dev - g_chk_dev))
        rel_diff = diff_chk / np.max(np.abs(g_chk_dev))
        assert rel_diff < 1e-5

    def test_multiple_devices_gradient_agreement(self):
        prev = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        try:

            def build(gcfg):
                cfg = SimulationConfig(
                    time=_SIM_TIME,
                    grid=UniformGrid(spacing=_RESOLUTION),
                    backend="cpu",
                    dtype=jnp.float64,
                    courant_factor=0.99,
                    gradient_config=gcfg,
                )
                volume = fdtdx.SimulationVolume(partial_grid_shape=(18, 18, 18))
                bound_dict, c_list = fdtdx.boundary_objects_from_config(
                    fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML_CELLS), volume
                )
                objects = [volume, *bound_dict.values()]
                constraints = [*c_list]
                materials = {"air": Material(permittivity=1.0), "si": Material(permittivity=2.25)}
                dev1 = Device(
                    name="dev1",
                    partial_grid_shape=(3, 3, 3),
                    partial_voxel_grid_shape=(1, 1, 1),
                    materials=materials,
                    param_transforms=[],
                )
                c_dev1 = GridCoordinateConstraint(
                    object="dev1", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[5, 5, 5]
                )
                dev2 = Device(
                    name="dev2",
                    partial_grid_shape=(3, 3, 3),
                    partial_voxel_grid_shape=(1, 1, 1),
                    materials=materials,
                    param_transforms=[],
                )
                c_dev2 = GridCoordinateConstraint(
                    object="dev2", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[9, 9, 9]
                )
                objects.extend([dev1, dev2])
                constraints.extend([c_dev1, c_dev2])

                wave = fdtdx.WaveCharacter(frequency=c0 / 800e-9)
                source = fdtdx.PointDipoleSource(
                    name="dip", partial_grid_shape=(1, 1, 1), wave_character=wave, polarization=0, amplitude=1.0
                )
                c_source = GridCoordinateConstraint(
                    object="dip", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[3, 3, 3]
                )
                objects.append(source)
                constraints.append(c_source)

                det = fdtdx.PhasorDetector(
                    name="det", partial_grid_shape=(3, 3, 1), wave_characters=(wave,), components=("Ex",), plot=False
                )
                c_det = GridCoordinateConstraint(
                    object="det", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[9, 9, 13]
                )
                objects.append(det)
                constraints.append(c_det)

                key = jax.random.PRNGKey(0)
                obj, arrays, params, cfg, _ = fdtdx.place_objects(
                    object_list=objects, config=cfg, constraints=constraints, key=key
                )
                arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)
                return obj, arrays, cfg, key

            def get_g(gcfg):
                obj, arrays, cfg, key = build(gcfg)

                def loss(inv_eps):
                    _, out = fdtdx.run_fdtd(
                        arrays.aset("inv_permittivities", inv_eps), obj, cfg, key, show_progress=False
                    )
                    return jnp.sum(jnp.abs(out.detector_states["det"]["phasor"]) ** 2)

                val, grad = jax.value_and_grad(loss)(arrays.inv_permittivities)
                s1 = (slice(None), *obj.devices[0].grid_slice)
                s2 = (slice(None), *obj.devices[1].grid_slice)
                return float(val), np.asarray(grad[s1]), np.asarray(grad[s2])

            cfg_chk = GradientConfig(method="checkpointed", num_checkpoints=16)
            v_chk, g1_chk, g2_chk = get_g(cfg_chk)

            cfg_local = GradientConfig(
                method="reversible",
                recording_mode="local_segmented",
                num_checkpoints_reversible=1,
                split_detector_adjoint=True,
            )
            v_loc, g1_loc, g2_loc = get_g(cfg_local)

            assert np.isclose(v_loc, v_chk, rtol=1e-10)
            diff1 = np.max(np.abs(g1_loc - g1_chk)) / np.max(np.abs(g1_chk))
            diff2 = np.max(np.abs(g2_loc - g2_chk)) / np.max(np.abs(g2_chk))
            assert diff1 < 1e-5
            assert diff2 < 1e-5
        finally:
            jax.config.update("jax_enable_x64", prev)

    def test_etched_device_gradient_agreement(self):
        prev = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        try:

            def build(gcfg):
                cfg = SimulationConfig(
                    time=_SIM_TIME,
                    grid=UniformGrid(spacing=_RESOLUTION),
                    backend="cpu",
                    dtype=jnp.float64,
                    courant_factor=0.99,
                    gradient_config=gcfg,
                )
                volume = fdtdx.SimulationVolume(partial_grid_shape=(14, 14, 14), material=Material(permittivity=2.25))
                bound_dict, c_list = fdtdx.boundary_objects_from_config(
                    fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML_CELLS), volume
                )
                objects = [volume, *bound_dict.values()]
                constraints = [*c_list]
                materials = {"air": Material(permittivity=1.0)}
                dev = Device(
                    name="dev",
                    partial_grid_shape=(4, 4, 4),
                    partial_voxel_grid_shape=(1, 1, 1),
                    materials=materials,
                    param_transforms=[],
                    use_etching=True,
                )
                c_dev = GridCoordinateConstraint(
                    object="dev", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[5, 5, 5]
                )
                objects.append(dev)
                constraints.append(c_dev)

                wave = fdtdx.WaveCharacter(frequency=c0 / 800e-9)
                source = fdtdx.PointDipoleSource(
                    name="dip", partial_grid_shape=(1, 1, 1), wave_character=wave, polarization=0, amplitude=1.0
                )
                c_source = GridCoordinateConstraint(
                    object="dip", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[3, 3, 3]
                )
                objects.append(source)
                constraints.append(c_source)

                det = fdtdx.PhasorDetector(
                    name="det", partial_grid_shape=(4, 4, 1), wave_characters=(wave,), components=("Ex",), plot=False
                )
                c_det = GridCoordinateConstraint(
                    object="det", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=[5, 5, 10]
                )
                objects.append(det)
                constraints.append(c_det)

                key = jax.random.PRNGKey(0)
                obj, arrays, params, cfg, _ = fdtdx.place_objects(
                    object_list=objects, config=cfg, constraints=constraints, key=key
                )
                return obj, arrays, params, cfg, key

            def get_g(gcfg):
                obj, arrays, params, cfg, key = build(gcfg)

                def loss(p):
                    arr, o, _ = fdtdx.apply_params(arrays, obj, p, key)
                    _, out = fdtdx.run_fdtd(arr, o, cfg, key, show_progress=False)
                    return jnp.sum(jnp.abs(out.detector_states["det"]["phasor"]) ** 2)

                val, grad = jax.value_and_grad(loss)(params)
                return float(val), np.asarray(grad["dev"])

            cfg_chk = GradientConfig(method="checkpointed", num_checkpoints=16)
            v_chk, g_chk = get_g(cfg_chk)

            cfg_loc = GradientConfig(
                method="reversible",
                recording_mode="local_segmented",
                num_checkpoints_reversible=1,
                split_detector_adjoint=True,
            )
            v_loc, g_loc = get_g(cfg_loc)

            assert np.isclose(v_loc, v_chk, rtol=1e-10)
            diff = np.max(np.abs(g_loc - g_chk))
            rel_diff = diff / np.max(np.abs(g_chk))
            assert rel_diff < 1e-5
        finally:
            jax.config.update("jax_enable_x64", prev)
