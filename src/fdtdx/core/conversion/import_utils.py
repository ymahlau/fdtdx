"""Utility functions for importing external file formats into FDTD simulations.

This module provides functions to convert various file formats (like GDSII) into
numpy arrays that can be used in FDTD simulations. The conversion functions handle
the transformation of geometric data into grid-based representations suitable for
electromagnetic simulations.
"""

from pathlib import Path

import gdspy
import numpy as np


def gds_to_numpy(
    file_path: str | Path,
    resolution: float,
    layer: int,
    datatype: int | None = None,
) -> np.ndarray[np.bool_]:
    """
    Converts GDSII geometry on a specific layer to a grid-based mask.

    Args:
        file_path: Path to the GDSII file
        resolution: The size of each grid cell in the same units as the GDSII coordinates
        layer: The GDSII layer number to extract
        datatype: The GDSII datatype number to extract. Defaults to layer value.

    Returns:
        np.ndarray[np.bool_]: A 2D NumPy array mask where True indicates geometry presence
    """
    if datatype is None:
        datatype = layer
    gdsii_lib = gdspy.GdsLibrary()
    gdsii_lib.read_gds(file_path)

    cell = gdsii_lib.top_level()[0]  # Assuming one top-level cell

    # Determine bounding box for grid dimensions
    bbox = cell.get_bounding_box()
    x_min, y_min, x_max, y_max = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]

    # Create grid coordinates
    x_coords = np.arange(x_min, x_max + resolution, resolution)
    y_coords = np.arange(y_min, y_max + resolution, resolution)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    # Check each grid point for geometry presence
    polygons = cell.get_polygons(by_spec=(layer, datatype))
    points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    result_bool_list = gdspy.inside(
        points=points,
        polygons=polygons,
    )
    mask = np.asarray(result_bool_list).reshape(grid_x.shape).T
    return mask
