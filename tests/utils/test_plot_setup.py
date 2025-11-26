from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import jax
import matplotlib.pyplot as plt

import fdtdx
from fdtdx import SimulationVolume
from fdtdx.config import SimulationConfig
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.object import GridCoordinateConstraint, SimulationObject
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

    # Create simulation volume (10 x 10 x 10 µm = 10e-6 meters)
    volume = SimulationVolume(
        partial_real_shape=(10e-6, 10e-6, 10e-6),  # 10 µm in each direction
        name="simulation_volume",
    )

    # Add a large background object (covers most of the volume)
    class LargeObject(SimulationObject):
        def __init__(self):
            super().__init__(
                partial_real_shape=(480 * 20e-9, 480 * 20e-9, 480 * 20e-9),  # 480 grid points * resolution
                partial_grid_shape=(10, 10, 10),
                name="large_background",
            )

    # Add a smaller centered object
    class CenterObject(SimulationObject):
        def __init__(self):
            super().__init__(
                partial_real_shape=(100 * 20e-9, 100 * 20e-9, 100 * 20e-9),  # 100 grid points * resolution
                partial_grid_shape=(200, 200, 200),
                name="center_object",
            )

    # Add a thin vertical object
    class VerticalObject(SimulationObject):
        def __init__(self):
            super().__init__(
                partial_real_shape=(50 * 20e-9, 50 * 20e-9, 200 * 20e-9),  # grid points * resolution
                partial_grid_shape=(350, 350, 150),
                name="vertical_object",
            )

    # Add a horizontal slab
    class HorizontalSlab(SimulationObject):
        def __init__(self):
            super().__init__(
                partial_real_shape=(200 * 20e-9, 200 * 20e-9, 30 * 20e-9),  # grid points * resolution
                partial_grid_shape=(100, 100, 400),
                name="horizontal_slab",
            )

    # Add PML boundaries (they don't have color attribute, so won't be plotted)
    pml_thickness = 20

    # X boundaries
    PerfectlyMatchedLayer(partial_grid_shape=(pml_thickness, 500, 500), axis=0, direction="low", name="pml_x_low")

    PerfectlyMatchedLayer(partial_grid_shape=(pml_thickness, 500, 500), axis=0, direction="high", name="pml_x_high")

    # Y boundaries
    PerfectlyMatchedLayer(partial_grid_shape=(500, pml_thickness, 500), axis=1, direction="low", name="pml_y_low")

    PerfectlyMatchedLayer(partial_grid_shape=(500, pml_thickness, 500), axis=1, direction="high", name="pml_y_high")

    # Z boundaries
    PerfectlyMatchedLayer(partial_grid_shape=(500, 500, pml_thickness), axis=2, direction="low", name="pml_z_low")

    PerfectlyMatchedLayer(partial_grid_shape=(500, 500, pml_thickness), axis=2, direction="high", name="pml_z_high")

    # Create all objects
    large_obj = LargeObject()
    center_obj = CenterObject()
    vertical_obj = VerticalObject()
    horizontal_obj = HorizontalSlab()

    # Create object list
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
        # large_background: from (10, 10, 10) to (490, 490, 490)
        GridCoordinateConstraint(object="large_background", axes=(0,), sides=("-",), coordinates=(10,)),
        GridCoordinateConstraint(object="large_background", axes=(0,), sides=("+",), coordinates=(490,)),
        GridCoordinateConstraint(object="large_background", axes=(1,), sides=("-",), coordinates=(10,)),
        GridCoordinateConstraint(object="large_background", axes=(1,), sides=("+",), coordinates=(490,)),
        GridCoordinateConstraint(object="large_background", axes=(2,), sides=("-",), coordinates=(10,)),
        GridCoordinateConstraint(object="large_background", axes=(2,), sides=("+",), coordinates=(490,)),
        # center_object: from (200, 200, 200) to (300, 300, 300)
        GridCoordinateConstraint(object="center_object", axes=(0,), sides=("-",), coordinates=(200,)),
        GridCoordinateConstraint(object="center_object", axes=(0,), sides=("+",), coordinates=(300,)),
        GridCoordinateConstraint(object="center_object", axes=(1,), sides=("-",), coordinates=(200,)),
        GridCoordinateConstraint(object="center_object", axes=(1,), sides=("+",), coordinates=(300,)),
        GridCoordinateConstraint(object="center_object", axes=(2,), sides=("-",), coordinates=(200,)),
        GridCoordinateConstraint(object="center_object", axes=(2,), sides=("+",), coordinates=(300,)),
        # vertical_object: from (350, 350, 150) to (400, 400, 350)
        GridCoordinateConstraint(object="vertical_object", axes=(0,), sides=("-",), coordinates=(350,)),
        GridCoordinateConstraint(object="vertical_object", axes=(0,), sides=("+",), coordinates=(400,)),
        GridCoordinateConstraint(object="vertical_object", axes=(1,), sides=("-",), coordinates=(350,)),
        GridCoordinateConstraint(object="vertical_object", axes=(1,), sides=("+",), coordinates=(400,)),
        GridCoordinateConstraint(object="vertical_object", axes=(2,), sides=("-",), coordinates=(150,)),
        GridCoordinateConstraint(object="vertical_object", axes=(2,), sides=("+",), coordinates=(350,)),
        # horizontal_slab: from (100, 100, 400) to (300, 300, 430)
        GridCoordinateConstraint(object="horizontal_slab", axes=(0,), sides=("-",), coordinates=(100,)),
        GridCoordinateConstraint(object="horizontal_slab", axes=(0,), sides=("+",), coordinates=(300,)),
        GridCoordinateConstraint(object="horizontal_slab", axes=(1,), sides=("-",), coordinates=(100,)),
        GridCoordinateConstraint(object="horizontal_slab", axes=(1,), sides=("+",), coordinates=(300,)),
        GridCoordinateConstraint(object="horizontal_slab", axes=(2,), sides=("-",), coordinates=(400,)),
        GridCoordinateConstraint(object="horizontal_slab", axes=(2,), sides=("+",), coordinates=(430,)),
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
