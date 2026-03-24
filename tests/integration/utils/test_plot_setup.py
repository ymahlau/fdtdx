"""Integration tests for utils/plot_setup.py - requires place_objects simulation setup."""

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import jax
import matplotlib.pyplot as plt

import fdtdx
from fdtdx import SimulationVolume
from fdtdx.config import SimulationConfig
from fdtdx.objects.object import GridCoordinateConstraint, OrderableObject
from fdtdx.utils.plot_setup import plot_setup, plot_setup_from_side

# Ensure test output directory exists
TEST_OUTPUT_DIR = Path("tests/generated")
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def simulation_setup():
    """Fixture that creates a simple simulation setup with various objects."""
    config = SimulationConfig(
        resolution=20e-9,
        time=100e-15,
    )

    volume = SimulationVolume(
        partial_real_shape=(5e-6, 5e-6, 5e-6),
        name="simulation_volume",
    )

    large_obj = OrderableObject.EmptySimulationObject.LargeObject()
    center_obj = OrderableObject.EmptySimulationObject.CenterObject()
    vertical_obj = OrderableObject.EmptySimulationObject.VerticalObject()
    horizontal_obj = OrderableObject.EmptySimulationObject.HorizontalSlab()

    object_list = [volume, large_obj, center_obj, vertical_obj, horizontal_obj]

    constraints = [
        GridCoordinateConstraint(object="large_background", axes=(0,), sides=("-",), coordinates=(5,)),
        GridCoordinateConstraint(object="large_background", axes=(0,), sides=("+",), coordinates=(245,)),
        GridCoordinateConstraint(object="large_background", axes=(1,), sides=("-",), coordinates=(5,)),
        GridCoordinateConstraint(object="large_background", axes=(1,), sides=("+",), coordinates=(245,)),
        GridCoordinateConstraint(object="large_background", axes=(2,), sides=("-",), coordinates=(5,)),
        GridCoordinateConstraint(object="large_background", axes=(2,), sides=("+",), coordinates=(245,)),
        GridCoordinateConstraint(object="center_object", axes=(0,), sides=("-",), coordinates=(100,)),
        GridCoordinateConstraint(object="center_object", axes=(0,), sides=("+",), coordinates=(150,)),
        GridCoordinateConstraint(object="center_object", axes=(1,), sides=("-",), coordinates=(100,)),
        GridCoordinateConstraint(object="center_object", axes=(1,), sides=("+",), coordinates=(150,)),
        GridCoordinateConstraint(object="center_object", axes=(2,), sides=("-",), coordinates=(100,)),
        GridCoordinateConstraint(object="center_object", axes=(2,), sides=("+",), coordinates=(150,)),
        GridCoordinateConstraint(object="vertical_object", axes=(0,), sides=("-",), coordinates=(175,)),
        GridCoordinateConstraint(object="vertical_object", axes=(0,), sides=("+",), coordinates=(200,)),
        GridCoordinateConstraint(object="vertical_object", axes=(1,), sides=("-",), coordinates=(175,)),
        GridCoordinateConstraint(object="vertical_object", axes=(1,), sides=("+",), coordinates=(200,)),
        GridCoordinateConstraint(object="vertical_object", axes=(2,), sides=("-",), coordinates=(75,)),
        GridCoordinateConstraint(object="vertical_object", axes=(2,), sides=("+",), coordinates=(175,)),
        GridCoordinateConstraint(object="horizontal_slab", axes=(0,), sides=("-",), coordinates=(50,)),
        GridCoordinateConstraint(object="horizontal_slab", axes=(0,), sides=("+",), coordinates=(150,)),
        GridCoordinateConstraint(object="horizontal_slab", axes=(1,), sides=("-",), coordinates=(50,)),
        GridCoordinateConstraint(object="horizontal_slab", axes=(1,), sides=("+",), coordinates=(150,)),
        GridCoordinateConstraint(object="horizontal_slab", axes=(2,), sides=("-",), coordinates=(200,)),
        GridCoordinateConstraint(object="horizontal_slab", axes=(2,), sides=("+",), coordinates=(215,)),
    ]

    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list,
        config=config,
        constraints=constraints,
        key=key,
    )

    return config, objects, large_obj


def test_plot_setup_from_side_xy(simulation_setup):
    """Test plot_setup_from_side with XY plane (viewing from z direction)."""
    config, container, _ = simulation_setup

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    result_fig = plot_setup_from_side(
        config=config,
        objects=container,
        viewing_side="z",
        ax=ax,
        plot_legend=True,
        filename=TEST_OUTPUT_DIR / "test_plot_from_side_xy.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_from_side_xy.png").exists()
    plt.close(fig)


def test_plot_setup_from_side_xz(simulation_setup):
    """Test plot_setup_from_side with XZ plane (viewing from y direction)."""
    config, container, _ = simulation_setup

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    result_fig = plot_setup_from_side(
        config=config,
        objects=container,
        viewing_side="y",
        ax=ax,
        plot_legend=True,
        filename=TEST_OUTPUT_DIR / "test_plot_from_side_xz.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_from_side_xz.png").exists()
    plt.close(fig)


def test_plot_setup_from_side_yz(simulation_setup):
    """Test plot_setup_from_side with YZ plane (viewing from x direction)."""
    config, container, _ = simulation_setup

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    result_fig = plot_setup_from_side(
        config=config,
        objects=container,
        viewing_side="x",
        ax=ax,
        plot_legend=True,
        filename=TEST_OUTPUT_DIR / "test_plot_from_side_yz.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_from_side_yz.png").exists()
    plt.close(fig)


def test_plot_setup(simulation_setup):
    """Test plot_setup function with all three planes."""
    config, container, _ = simulation_setup

    result_fig = plot_setup(
        config=config,
        objects=container,
        plot_legend=True,
        filename=TEST_OUTPUT_DIR / "test_plot_setup_all_planes.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_setup_all_planes.png").exists()
    plt.close("all")


def test_plot_setup_exclude_large_objects(simulation_setup):
    """Test plot_setup with exclude_large_object_ratio to filter out large objects."""
    config, container, _ = simulation_setup

    result_fig = plot_setup(
        config=config,
        objects=container,
        plot_legend=True,
        exclude_large_object_ratio=0.9,
        filename=TEST_OUTPUT_DIR / "test_plot_setup_exclude_large.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_setup_exclude_large.png").exists()

    large_obj = [obj for obj in container.objects if obj.name == "large_background"][0]
    volume = container.volume
    slices = large_obj.grid_slice_tuple
    xy_area = (slices[0][1] - slices[0][0]) * (slices[1][1] - slices[1][0])
    total_xy_area = volume.grid_shape[0] * volume.grid_shape[1]
    coverage_ratio = xy_area / total_xy_area

    assert coverage_ratio > 0.9, "Large object should have coverage > 0.9"

    plt.close("all")


def test_exclude_large_object_ratio_threshold(simulation_setup):
    """Test that exclude_large_object_ratio correctly filters objects at the threshold."""
    config, container, _ = simulation_setup

    large_obj = [obj for obj in container.objects if obj.name == "large_background"][0]
    center_obj = [obj for obj in container.objects if obj.name == "center_object"][0]
    volume = container.volume

    large_slices = large_obj.grid_slice_tuple
    large_xy_area = (large_slices[0][1] - large_slices[0][0]) * (large_slices[1][1] - large_slices[1][0])
    total_xy_area = volume.grid_shape[0] * volume.grid_shape[1]
    large_coverage = large_xy_area / total_xy_area

    center_slices = center_obj.grid_slice_tuple
    center_xy_area = (center_slices[0][1] - center_slices[0][0]) * (center_slices[1][1] - center_slices[1][0])
    center_coverage = center_xy_area / total_xy_area

    assert large_coverage > 0.9, "Large object should have >90% coverage"
    assert center_coverage < 0.9, "Center object should have <90% coverage"

    threshold = (large_coverage + center_coverage) / 2

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    result_fig = plot_setup(
        config=config,
        objects=container,
        axs=axs,
        plot_legend=False,
        exclude_large_object_ratio=threshold,
    )

    assert result_fig is not None
    plt.close("all")
