"""``import fdtdx`` leaves the slow optional imports to the functions that use them."""

import subprocess
import sys

# Each takes about 0.5-2 s to import and is only needed by a few functions:
# tidy3d by the mode solver, seaborn by the line plots and the Logger, trimesh by export_stl.
_IMPORTED_ON_FIRST_USE = ("tidy3d", "seaborn", "trimesh")


def _modules_loaded_after(code: str) -> list[str]:
    script = f"import sys\n{code}\nprint(' '.join(m for m in {_IMPORTED_ON_FIRST_USE!r} if m in sys.modules))"
    # A fresh interpreter: the test session itself has imported most of these already.
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    return result.stdout.split()


def test_import_fdtdx_does_not_import_heavy_dependencies():
    assert _modules_loaded_after("import fdtdx") == []


def test_heavy_dependencies_are_imported_on_first_use():
    code = (
        "import numpy as np\n"
        "import fdtdx\n"
        "from fdtdx.conversion.stl import export_stl\n"
        "from fdtdx.core.physics.modes import _compute_modes\n"
        "from fdtdx.objects.detectors.plotting.line_plot import plot_line_over_time\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "export_stl(np.ones((2, 2, 2), dtype=bool))\n"
        "plot_line_over_time(np.arange(3.0), np.arange(3.0), 'metric')\n"
        "try:\n"
        "    _compute_modes()\n"
        "except TypeError:\n"
        "    pass  # missing arguments: the solver was imported, which is what is checked here\n"
    )
    assert sorted(_modules_loaded_after(code)) == sorted(_IMPORTED_ON_FIRST_USE)
