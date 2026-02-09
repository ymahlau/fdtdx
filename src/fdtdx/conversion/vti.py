import struct
import zlib
from pathlib import Path

import jax

from fdtdx.fdtd.container import ArrayContainer
from fdtdx.typing import Slice3D

NUMPY_TO_VTK_DTYPE = {
    "int8": "Int8",
    "uint8": "UInt8",
    "int16": "Int16",
    "uint16": "UInt16",
    "int32": "Int32",
    "uint32": "UInt32",
    "int64": "Int64",
    "uint64": "UInt64",
    "float32": "Float32",
    "float64": "Float64",
}


def export_arrays_snapshot_to_vti(arrays: ArrayContainer, path: Path | str, resolution: float):
    """Convenience function to export a snapshot of FDTD simulation arrays to a VTI file.

    Extracts electromagnetic fields (E, H) and material properties (permittivity, permeability,
    conductivity) from the container. Inverse parameters are converted back to standard values
    (e.g., 1/inv_permittivity) for visualization.

    Args:
        arrays (ArrayContainer): Container holding simulation state and material arrays.
        path (Path | str): Output file path.
        resolution (float): Spatial resolution of the grid from the SimulationConfig.
    """
    cell_data = {
        "permittivity": 1 / arrays.inv_permittivities,
        "E": arrays.E,
        "H": arrays.H,
    }

    if isinstance(arrays.inv_permeabilities, jax.Array) and arrays.inv_permeabilities.shape != ():
        cell_data["permeabilities"] = 1 / arrays.inv_permeabilities
    if arrays.electric_conductivity is not None:
        cell_data["electric_conductivity"] = arrays.electric_conductivity
    if arrays.magnetic_conductivity is not None:
        cell_data["magnetic_conductivity"] = arrays.magnetic_conductivity

    export_vti(cell_data, path, resolution)


def encode_array(array: jax.Array, compression_level: int = -1) -> bytes:
    """Generate raw, compressed byte encoding for a numpy array compatible with VTK.

    Flattens the array in Fortran order and applies zlib compression. Prepends a 16-byte
    header (4 unsigned integers) containing block count and size information required by
    the VTK binary format.
    Implicitly works with both scalar (x, y, z) and vector (n, x, y, z) fields.

    Args:
        array (jax.Array): Input array (scalar or vector field).
        compression_level (int, optional): zlib compression level (0-9). Defaults to -1 (default).

    Returns:
        bytes: The binary header followed by the compressed data.
    """
    # flatten in fortran order
    flat_array = array.flatten(order="F")

    raw_bytes = flat_array.tobytes()
    uncompressed_size = len(raw_bytes)
    compressed_bytes = zlib.compress(raw_bytes, level=compression_level)
    compressed_size = len(compressed_bytes)

    # header: [1 block][uncompressed_size][last_block_size][compressed_size]
    header = struct.pack("<4I", 1, uncompressed_size, uncompressed_size, compressed_size)

    full_data_chunk = header + compressed_bytes
    return full_data_chunk


def export_vti(
    cell_data: dict[str, jax.Array],
    filename: Path | str,
    resolution: float,
    offset: tuple[int, int, int] = (0, 0, 0),
    grid_slice: Slice3D | None = None,
    compression_level: int = -1,
):
    """Export a dictionary of arrays to a VTI (VTK ImageData) file.

    Writes an XML-formatted VTI file with appended binary data. Supports both 3D scalar fields
    (x, y, z) and 4D vector fields (n, x, y, z). All arrays must share the same spatial dimensions.

    Args:
        cell_data (dict[str, jax.Array]): Dictionary mapping field names to numpy arrays.
        filename (Path | str): Output file path.
        resolution (float): Voxel spacing for the grid.
        offset (tuple[int, int, int], optional): Global grid index offset (x, y, z).
            Useful when aligning multiple VTI files. Defaults to (0, 0, 0).
        grid_slice (Slice3D | None, optional): Slice for defining the offset, if provided.
            Useful when aligning multiple VTI files. Overrides `offset`. Defaults to None.
        compression_level (int, optional): zlib compression level. Use level 0 for no compression (fastest)
            and level 9 for highest compression (smallest file size).
            Defaults to -1, which currently corresponds to level 6 compression.

    Raises:
        AssertionError: If arrays have mismatched shapes, invalid dimensions, or unsupported dtypes.
    """
    if grid_slice is not None:
        assert offset == (0, 0, 0)
        offset = (grid_slice[0].start, grid_slice[1].start, grid_slice[2].start)

    shape = next(iter(cell_data.values())).shape[-3:]
    nx, ny, nz = shape

    assert all(a.shape[-3:] == shape for a in cell_data.values()), (
        "All arrays in a VTI file need to be defined over the same underlying grid."
        "Use multiple vti files with offset or grid_slice options when creating visualizations of heterogeneous arrays."
    )
    assert all(a.ndim == 3 or a.ndim == 4 for a in cell_data.values()), (
        "Only 3d scalar fields (x, y, z) and vector fields (n, x, y, z) are supported."
    )
    assert all(str(a.dtype) in NUMPY_TO_VTK_DTYPE for a in cell_data.values()), (
        f"VTI files only support dtypes {list(NUMPY_TO_VTK_DTYPE.keys())}."
    )

    all_encoded_data = []
    current_offset = 0
    data_xml_parts = []

    for name, array in cell_data.items():
        n_comp = 1
        if array.ndim == 4:
            # vector field
            n_comp = array.shape[0]

        vtk_type = NUMPY_TO_VTK_DTYPE[str(array.dtype)]

        data_xml_parts.append(
            f'<DataArray type="{vtk_type}" Name="{name}" NumberOfComponents="{n_comp}" format="appended" offset="{current_offset}"/>'
        )
        encoded_data = encode_array(array, compression_level=compression_level)
        all_encoded_data.append(encoded_data)
        current_offset += len(encoded_data)

    origin = "0 0 0"
    extent = f"{offset[0]} {nx + offset[0]} {offset[1]} {ny + offset[1]} {offset[2]} {nz + offset[2]}"
    spacing = f"{resolution} {resolution} {resolution}"

    xml = f"""<?xml version="1.0"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <ImageData WholeExtent="{extent}" Origin="{origin}" Spacing="{spacing}">
    <Piece Extent="{extent}">
      <CellData>
        {chr(10).join(data_xml_parts)}
      </CellData>
    </Piece>
  </ImageData>
  <AppendedData encoding="raw">
    _"""

    with open(filename, "wb") as f:
        f.write(xml.encode("utf-8"))
        for d in all_encoded_data:
            f.write(d)
        f.write(b"\n</AppendedData>\n</VTKFile>")
