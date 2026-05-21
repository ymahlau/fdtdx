import jax.numpy as jnp
import numpy as np
import pytest
from matplotlib.figure import Figure

from fdtdx.core.plotting.device_permittivity_index_utils import (
    device_matrix_index_figure,
    index_matrix_to_str,
)
from fdtdx.materials import Material
from fdtdx.typing import ParameterType

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def two_materials():
    """Air (permittivity=1) and Silicon (permittivity=11.7)."""
    return {
        "Air": Material(permittivity=1.0),
        "Silicon": Material(permittivity=11.7),
    }


@pytest.fixture
def three_materials():
    """Air, Glass, Silicon - three materials sorted by permittivity."""
    return {
        "Air": Material(permittivity=1.0),
        "Glass": Material(permittivity=2.25),
        "Silicon": Material(permittivity=11.7),
    }


# ──────────────────────────────────────────────
# index_matrix_to_str
# ──────────────────────────────────────────────


class TestIndexMatrixToStr:
    def test_2x2_matrix(self):
        arr = jnp.array([[0, 1], [2, 3]])
        result = index_matrix_to_str(arr)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert "0" in lines[0] and "1" in lines[0]
        assert "2" in lines[1] and "3" in lines[1]
        assert result.endswith("\n")


# ──────────────────────────────────────────────
# device_matrix_index_figure
# ──────────────────────────────────────────────


class TestDeviceMatrixIndexFigure:
    def test_asserts_3d_input(self, two_materials):
        indices_2d = jnp.zeros((4, 4), dtype=jnp.int32)
        with pytest.raises(AssertionError):
            device_matrix_index_figure(indices_2d, two_materials, ParameterType.DISCRETE)

    def test_continuous_parameter_type(self, two_materials):
        """CONTINUOUS path takes mean over last axis and uses arange indices."""
        indices = jnp.ones((3, 3, 2), dtype=jnp.float32) * 0.5
        fig = device_matrix_index_figure(indices, two_materials, ParameterType.CONTINUOUS)
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_discrete_single_channel_small_matrix(self, two_materials):
        """Small discrete matrix has text annotations and legend with material names."""
        indices = jnp.array([[[0], [1]], [[1], [0]]], dtype=jnp.int32)
        fig = device_matrix_index_figure(indices, two_materials, ParameterType.DISCRETE)
        ax = fig.axes[0]
        assert len(ax.texts) > 0
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert any("Air" in i for i in labels)
        assert any("Silicon" in i for i in labels)
        # Check axis properties
        assert ax.get_xlabel() == "X Axis"
        assert ax.get_ylabel() == "Y Axis"
        assert ax.get_aspect() in ("equal", 1.0)
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            assert line.get_alpha() == 0.0
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_discrete_single_channel_large_matrix(self, two_materials):
        """Large matrix (h*w >= 1500) has no text annotations."""
        indices = jnp.zeros((50, 30, 1), dtype=jnp.int32)
        fig = device_matrix_index_figure(indices, two_materials, ParameterType.DISCRETE)
        ax = fig.axes[0]
        assert len(ax.texts) == 0
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_discrete_multi_channel(self, two_materials):
        """Multi-channel discrete path uses pipe-separated legend labels."""
        indices = jnp.array(
            [
                [[0, 0], [0, 1]],
                [[1, 0], [1, 1]],
            ],
            dtype=jnp.int32,
        )
        fig = device_matrix_index_figure(indices, two_materials, ParameterType.DISCRETE)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert any("|" in i for i in labels)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_three_materials_single_channel(self, three_materials):
        indices = jnp.array([[[0], [1], [2]], [[2], [1], [0]]], dtype=jnp.int32)
        fig = device_matrix_index_figure(indices, three_materials, ParameterType.DISCRETE)
        ax = fig.axes[0]
        legend = ax.get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        assert len(labels) == 3
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_legend_suppressed_when_over_100_elements(self):
        """When there are >= 100 unique legend elements, legend is not shown."""
        materials = {}
        for i in range(101):
            materials[f"Mat{i}"] = Material(permittivity=float(i + 1))
        flat_indices = np.arange(101, dtype=np.int32).reshape(-1, 1, 1)
        indices = jnp.array(flat_indices)
        fig = device_matrix_index_figure(indices, materials, ParameterType.DISCRETE)
        ax = fig.axes[0]
        assert ax.get_legend() is None
        import matplotlib.pyplot as plt

        plt.close(fig)
