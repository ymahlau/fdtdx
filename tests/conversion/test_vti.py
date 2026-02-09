import pytest

import jax
import jax.numpy as jnp

import fdtdx
from fdtdx import (
    export_vti, 
    export_arrays_snapshot_to_vti, 
)

def test_export_vti_file_creation(tmp_path):
    """Test that export_vti creates a valid file with expected XML tags."""
    # Create dummy data
    nx, ny, nz = 10, 10, 10
    scalar_field = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    vector_field = jnp.zeros((3, nx, ny, nz), dtype=jnp.float32)
    
    cell_data = {
        "pressure": scalar_field,
        "velocity": vector_field
    }
    
    output_path = tmp_path / "test_output.vti"
    export_vti(cell_data, output_path, resolution=1e-9)
    
    assert output_path.exists()
    
    # Basic validation of file content
    with open(output_path, "rb") as f:
        content = f.read()
        
    # Check XML structure basics
    assert b'<VTKFile type="ImageData"' in content
    assert b'WholeExtent="0 10 0 10 0 10"' in content
    # Check that data arrays are referenced
    assert b'Name="pressure"' in content
    assert b'Name="velocity"' in content
    assert b'NumberOfComponents="3"' in content  # For velocity
    assert b'AppendedData encoding="raw"' in content


def test_export_vti_with_offset(tmp_path):
    """Test that providing an offset correctly shifts the WholeExtent."""
    nx, ny, nz = 5, 5, 5
    data = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    cell_data = {"field": data}
    
    offset = (10, 20, 30)
    output_path = tmp_path / "offset_test.vti"
    
    export_vti(cell_data, output_path, resolution=1.0, offset=offset)
    
    with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        
    # Extent should be: x_start, x_end, y_start, y_end, z_start, z_end
    # x: 10 to 10+5=15, y: 20 to 20+5=25, z: 30 to 30+5=35
    expected_extent = 'WholeExtent="10 15 20 25 30 35"'
    assert expected_extent in content


def test_export_vti_validation_errors(tmp_path):
    """Test assertions for mismatched shapes and unsupported dimensions."""
    # Mismatched shapes
    data1 = jnp.zeros((10, 10, 10), dtype=jnp.float32)
    data2 = jnp.zeros((5, 5, 5), dtype=jnp.float32)
    
    with pytest.raises(AssertionError, match="same underlying grid"):
        export_vti({"a": data1, "b": data2}, tmp_path / "fail.vti", 1.0)

    # Unsupported dimension (2D)
    data_2d = jnp.zeros((10, 10), dtype=jnp.float32)
    with pytest.raises(AssertionError, match="Only 3d scalar fields"):
        export_vti({"a": data_2d}, tmp_path / "fail.vti", 1.0)


def test_export_arrays_snapshot_to_vti(tmp_path):
    key = jax.random.PRNGKey(seed=42)

    config = fdtdx.SimulationConfig(
        time=100e-15,
        resolution=100e-9,
    )
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(2.0e-6, 2.0e-6, 2.0e-6),
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
    res_str = f'{config.resolution} {config.resolution} {config.resolution}'
    assert res_str in content
