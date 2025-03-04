import numpy as np

from fdtdx.conversion import export_stl


def test_arr_to_gds():
    arr = np.asarray([
        [
            [0, 1, 0,],
            [0, 0, 0,],
            [0, 1, 0,],
        ],
        [
            [0, 0, 0,],
            [0, 1, 0,],
            [0, 0, 0,],
        ],
    ]).astype(bool)
    
    mesh = export_stl(arr, "example.stl")
    assert mesh.vertices.shape == (20, 3)
    
    

