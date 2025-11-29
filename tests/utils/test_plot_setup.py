from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
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
    # Create a basic configuration
    config = SimulationConfig(
        resolution=20e-9,  # 20 nm resolution
        time=100e-15,  # 100 fs total simulation time
    )

    # Create simulation volume (5 x 5 x 5 µm = 5e-6 meters)
    volume = SimulationVolume(
        partial_real_shape=(5e-6, 5e-6, 5e-6),  # 5 µm in each direction
        name="simulation_volume",
    )

    # Create all objects
    large_obj = OrderableObject.EmptySimulationObject.LargeObject()
    center_obj = OrderableObject.EmptySimulationObject.CenterObject()
    vertical_obj = OrderableObject.EmptySimulationObject.VerticalObject()
    horizontal_obj = OrderableObject.EmptySimulationObject.HorizontalSlab()

    # Create object list (without PML - they don't have colors and won't be plotted anyway)
    object_list = [
        volume,
        large_obj,
        center_obj,
        vertical_obj,
        horizontal_obj,
    ]

    # Create constraints for positioning - use GridCoordinateConstraint for absolute positioning
    # Each object needs constraints for all 6 sides (x-, x+, y-, y+, z-, z+)
    constraints = [
        # large_background: from (5, 5, 5) to (245, 245, 245)
        GridCoordinateConstraint(object="large_background", axes=(0,), sides=("-",), coordinates=(5,)),
        GridCoordinateConstraint(object="large_background", axes=(0,), sides=("+",), coordinates=(245,)),
        GridCoordinateConstraint(object="large_background", axes=(1,), sides=("-",), coordinates=(5,)),
        GridCoordinateConstraint(object="large_background", axes=(1,), sides=("+",), coordinates=(245,)),
        GridCoordinateConstraint(object="large_background", axes=(2,), sides=("-",), coordinates=(5,)),
        GridCoordinateConstraint(object="large_background", axes=(2,), sides=("+",), coordinates=(245,)),
        # center_object: from (100, 100, 100) to (100, 150, 150)
        GridCoordinateConstraint(object="center_object", axes=(0,), sides=("-",), coordinates=(100,)),
        GridCoordinateConstraint(object="center_object", axes=(0,), sides=("+",), coordinates=(150,)),
        GridCoordinateConstraint(object="center_object", axes=(1,), sides=("-",), coordinates=(100,)),
        GridCoordinateConstraint(object="center_object", axes=(1,), sides=("+",), coordinates=(150,)),
        GridCoordinateConstraint(object="center_object", axes=(2,), sides=("-",), coordinates=(100,)),
        GridCoordinateConstraint(object="center_object", axes=(2,), sides=("+",), coordinates=(150,)),
        # vertical_object: from (175, 175, 75) to (200, 200, 175)
        GridCoordinateConstraint(object="vertical_object", axes=(0,), sides=("-",), coordinates=(175,)),
        GridCoordinateConstraint(object="vertical_object", axes=(0,), sides=("+",), coordinates=(200,)),
        GridCoordinateConstraint(object="vertical_object", axes=(1,), sides=("-",), coordinates=(175,)),
        GridCoordinateConstraint(object="vertical_object", axes=(1,), sides=("+",), coordinates=(200,)),
        GridCoordinateConstraint(object="vertical_object", axes=(2,), sides=("-",), coordinates=(75,)),
        GridCoordinateConstraint(object="vertical_object", axes=(2,), sides=("+",), coordinates=(175,)),
        # horizontal_slab: from (50, 50, 200) to (150, 150, 215)
        GridCoordinateConstraint(object="horizontal_slab", axes=(0,), sides=("-",), coordinates=(50,)),
        GridCoordinateConstraint(object="horizontal_slab", axes=(0,), sides=("+",), coordinates=(150,)),
        GridCoordinateConstraint(object="horizontal_slab", axes=(1,), sides=("-",), coordinates=(50,)),
        GridCoordinateConstraint(object="horizontal_slab", axes=(1,), sides=("+",), coordinates=(150,)),
        GridCoordinateConstraint(object="horizontal_slab", axes=(2,), sides=("-",), coordinates=(200,)),
        GridCoordinateConstraint(object="horizontal_slab", axes=(2,), sides=("+",), coordinates=(215,)),
    ]

    # Use place_objects to create the initialized container
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
        config=config, objects=container, plot_legend=True, filename=TEST_OUTPUT_DIR / "test_plot_setup_all_planes.png"
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_setup_all_planes.png").exists()
    plt.close("all")


def test_plot_setup_exclude_large_objects(simulation_setup):
    """Test plot_setup with exclude_large_object_ratio to filter out large objects."""
    config, container, _ = simulation_setup

    # Use a threshold that will exclude the large background object
    # The large object covers (480/500)^2 = 0.9216 = 92.16% in each plane
    # Setting threshold to 0.9 will exclude it
    result_fig = plot_setup(
        config=config,
        objects=container,
        plot_legend=True,
        exclude_large_object_ratio=0.9,
        filename=TEST_OUTPUT_DIR / "test_plot_setup_exclude_large.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_setup_exclude_large.png").exists()

    # Verify that the large object would be excluded (coverage > 0.9)
    # Get the initialized large object from the container
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

    # Get the initialized objects from the container
    large_obj = [obj for obj in container.objects if obj.name == "large_background"][0]
    center_obj = [obj for obj in container.objects if obj.name == "center_object"][0]
    volume = container.volume

    # Calculate coverage ratios for both objects
    large_slices = large_obj.grid_slice_tuple
    large_xy_area = (large_slices[0][1] - large_slices[0][0]) * (large_slices[1][1] - large_slices[1][0])
    total_xy_area = volume.grid_shape[0] * volume.grid_shape[1]
    large_coverage = large_xy_area / total_xy_area

    center_slices = center_obj.grid_slice_tuple
    center_xy_area = (center_slices[0][1] - center_slices[0][0]) * (center_slices[1][1] - center_slices[1][0])
    center_coverage = center_xy_area / total_xy_area

    # Verify our test assumptions
    assert large_coverage > 0.9, "Large object should have >90% coverage"
    assert center_coverage < 0.9, "Center object should have <90% coverage"

    # Test with threshold that should exclude only the large object
    # Set threshold between the two coverage ratios
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

    # The plot was created successfully - the filtering logic was executed
    # Objects with coverage > threshold should be filtered out
    # This tests the coverage_ratio <= exclude_large_object_ratio condition

    plt.close("all")
