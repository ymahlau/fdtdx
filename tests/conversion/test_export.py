import numpy as np

import fdtdx


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

    mesh = fdtdx.export_stl(arr)
    assert mesh.vertices.shape == (20, 3)
