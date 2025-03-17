import numpy as np

from fdtdx.conversion import export_stl


def test_arr_to_stl():
    # fmt: off
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
    # fmt: on

    mesh = export_stl(arr)
    assert mesh.vertices.shape == (20, 3)
