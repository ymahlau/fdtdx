"""Integration tests for fdtdx.conversion.vti module."""

import jax

import fdtdx
from fdtdx import export_arrays_snapshot_to_vti


def test_export_arrays_snapshot_to_vti(tmp_path):
    key = jax.random.PRNGKey(seed=42)

    config = fdtdx.SimulationConfig(
        time=100e-15,
        resolution=100e-9,
    )
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(2.0e-6, 2.0e-6, 2.0e-6),
        material=fdtdx.Material(
            permittivity=2.5, permeability=1.5, electric_conductivity=0.1, magnetic_conductivity=0.2
        ),
    )

    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume],
        config=config,
        constraints=[],
        key=subkey,
    )
    arrays, new_objects, _ = fdtdx.apply_params(arrays, objects, params, key)

    output_path = tmp_path / "snapshot.vti"

    # Run the snapshot export
    export_arrays_snapshot_to_vti(arrays, output_path, config.resolution)

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Read file to verify specific physics fields were exported
    with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # The snapshot function maps 'inv_permittivities' to 'permittivity'
    assert 'Name="permittivity"' in content
    assert 'Name="E"' in content
    assert 'Name="H"' in content

    # Check spacing/resolution was passed correctly
    res_str = f"{config.resolution} {config.resolution} {config.resolution}"
    assert res_str in content
