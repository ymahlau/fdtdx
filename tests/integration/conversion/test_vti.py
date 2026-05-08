"""Integration tests for fdtdx.conversion.vti module."""

import jax
import jax.numpy as jnp

import fdtdx
from fdtdx import export_arrays_snapshot_to_vti, export_vtr
from fdtdx.core.grid import RectilinearGrid


def test_export_arrays_snapshot_to_vti(tmp_path):
    key = jax.random.PRNGKey(seed=42)

    config = fdtdx.SimulationConfig(
        time=100e-15,
        grid=fdtdx.UniformGrid(spacing=100e-9),
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
    arrays, _new_objects, _ = fdtdx.apply_params(arrays, objects, params, key)

    output_path = tmp_path / "snapshot.vti"

    # Run the snapshot export
    export_arrays_snapshot_to_vti(arrays, output_path, config.uniform_spacing())

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
    res_str = f"{config.uniform_spacing()} {config.uniform_spacing()} {config.uniform_spacing()}"
    assert res_str in content


def test_export_vtr_uniform_grid(tmp_path):
    grid = RectilinearGrid.uniform(shape=(4, 5, 6), spacing=50e-9)
    cell_data = {"field": jnp.ones((4, 5, 6), dtype=jnp.float32)}
    output_path = tmp_path / "uniform.vtr"

    export_vtr(cell_data, output_path, grid)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    assert 'Name="field"' in content
    assert "RectilinearGrid" in content


def test_export_vtr_nonuniform_grid(tmp_path):
    x_edges = jnp.array([0.0, 50e-9, 150e-9, 350e-9, 750e-9])
    y_edges = jnp.linspace(0.0, 500e-9, 6)
    z_edges = jnp.linspace(0.0, 300e-9, 4)
    grid = RectilinearGrid(x_edges=x_edges, y_edges=y_edges, z_edges=z_edges)
    cell_data = {"E": jnp.zeros((3, 4, 5, 3), dtype=jnp.float32)}
    output_path = tmp_path / "nonuniform.vtr"

    export_vtr(cell_data, output_path, grid)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    assert 'Name="E"' in content
    assert "X_COORDINATES" in content
