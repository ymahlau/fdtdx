from pathlib import Path

import jax
import jax.numpy as jnp

import fdtdx


def test_ssp_intricate_grad_nan_bug():
    path = Path(__file__).parent.parent.parent / "data" / "example_device_params.npy"
    arr = jnp.load(path)

    config = fdtdx.SimulationConfig(
        time=200e-15,
        resolution=20e-9,
        dtype=jnp.float32,
        courant_factor=0.99,
    )
    material_config = {
        "sio2": fdtdx.Material(permittivity=3.9),
        "si": fdtdx.Material(permittivity=12.25),
    }
    height = 220e-9
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(7e-6, 6e-6, height),
        material=material_config["sio2"],
    )
    device = fdtdx.Device(
        name="Device",
        partial_real_shape=(7e-6, 6e-6, height),
        materials=material_config,
        param_transforms=[
            fdtdx.GaussianSmoothing2D(std_discrete=3),
            fdtdx.SubpixelSmoothedProjection(),
        ],
        partial_voxel_real_shape=(config.resolution, config.resolution, height),
    )
    key = jax.random.PRNGKey(42)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, device],
        config=config,
        constraints=[device.place_at_center(volume)],
        key=key,
    )
    params["Device"] = arr
    arrays, new_objects, _ = fdtdx.apply_params(arrays, objects, params, key, beta=5.0)

    def fn(p):
        cur_material_indices = new_objects["Device"](p[device.name], expand_to_sim_grid=False, beta=5.0)  # type: ignore
        return jnp.sum(cur_material_indices)

    value, grad = jax.value_and_grad(fn)(params)
    assert not jnp.isnan(value) and not jnp.isinf(value)
    assert not jnp.any(jnp.isnan(grad["Device"]))
