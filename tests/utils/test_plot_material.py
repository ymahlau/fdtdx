from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import jax
import matplotlib.pyplot as plt

import fdtdx
from fdtdx import Cylinder, SimulationVolume, Sphere
from fdtdx.config import SimulationConfig
from fdtdx.objects.object import GridCoordinateConstraint
from fdtdx.objects.static_material.static import Material, UniformMaterialObject
from fdtdx.utils.plot_material import plot_material, plot_material_from_side

# Ensure test output directory exists
TEST_OUTPUT_DIR = Path("tests/generated")
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def simple_material_setup():
    """A simple setup with basic material objects for testing."""
    # Create a basic configuration
    config = SimulationConfig(
        resolution=50e-9,  # 50 nm resolution
        time=100e-15,  # 100 fs total simulation time
    )

    # Create simulation volume (2 x 2 x 2 µm = 2e-6 meters)
    volume = SimulationVolume(
        partial_real_shape=(2e-6, 2e-6, 2e-6),
        name="simulation_volume",
    )

    # Create a simple material object
    material = Material(permittivity=2.25, permeability=1.0)

    test_object = UniformMaterialObject(
        name="test_object",
        material=material,
    )

    # Create object list
    object_list = [
        volume,
        test_object,
    ]

    # Create constraints for positioning
    constraints = [
        # Volume - fill entire space
        GridCoordinateConstraint(object="simulation_volume", axes=(0,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(0,), sides=("+",), coordinates=(40,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(1,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(1,), sides=("+",), coordinates=(40,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(2,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(2,), sides=("+",), coordinates=(40,)),
        # Test object: centered cube
        GridCoordinateConstraint(object="test_object", axes=(0,), sides=("-",), coordinates=(15,)),
        GridCoordinateConstraint(object="test_object", axes=(0,), sides=("+",), coordinates=(25,)),
        GridCoordinateConstraint(object="test_object", axes=(1,), sides=("-",), coordinates=(15,)),
        GridCoordinateConstraint(object="test_object", axes=(1,), sides=("+",), coordinates=(25,)),
        GridCoordinateConstraint(object="test_object", axes=(2,), sides=("-",), coordinates=(15,)),
        GridCoordinateConstraint(object="test_object", axes=(2,), sides=("+",), coordinates=(25,)),
    ]

    # Use place_objects to create the initialized container
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list,
        config=config,
        constraints=constraints,
        key=key,
    )

    return config, arrays


@pytest.fixture
def cylinder_setup():
    """Setup with a cylinder object."""
    config = SimulationConfig(
        resolution=30e-9,
        time=100e-15,
    )

    volume = SimulationVolume(
        partial_real_shape=(3e-6, 3e-6, 3e-6),
        name="simulation_volume",
    )

    # Create cylinder using the correct API from documentation
    cylinder = Cylinder(
        name="test_cylinder",
        radius=0.5e-6,
        axis=2,  # Along z-axis
        materials={  # Pass materials dictionary
            "dielectric": Material(permittivity=4.0, permeability=1.0)
        },
        material_name="dielectric",
    )

    object_list = [volume, cylinder]

    # Create constraints
    constraints = [
        # Volume
        GridCoordinateConstraint(object="simulation_volume", axes=(0,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(0,), sides=("+",), coordinates=(100,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(1,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(1,), sides=("+",), coordinates=(100,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(2,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(2,), sides=("+",), coordinates=(100,)),
        # Cylinder: centered, extending through Z
        GridCoordinateConstraint(object="test_cylinder", axes=(0,), sides=("-",), coordinates=(40,)),
        GridCoordinateConstraint(object="test_cylinder", axes=(0,), sides=("+",), coordinates=(60,)),
        GridCoordinateConstraint(object="test_cylinder", axes=(1,), sides=("-",), coordinates=(40,)),
        GridCoordinateConstraint(object="test_cylinder", axes=(1,), sides=("+",), coordinates=(60,)),
        GridCoordinateConstraint(object="test_cylinder", axes=(2,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="test_cylinder", axes=(2,), sides=("+",), coordinates=(100,)),
    ]

    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list,
        config=config,
        constraints=constraints,
        key=key,
    )

    return config, arrays


@pytest.fixture
def sphere_setup():
    """Setup with a sphere object."""
    config = SimulationConfig(
        resolution=30e-9,
        time=100e-15,
    )

    volume = SimulationVolume(
        partial_real_shape=(3e-6, 3e-6, 3e-6),
        name="simulation_volume",
    )

    # Create sphere using the correct API from documentation
    sphere = Sphere(
        name="test_sphere",
        radius=0.5e-6,
        materials={  # Pass materials dictionary
            "dielectric": Material(permittivity=9.0, permeability=1.0)
        },
        material_name="dielectric",
    )

    object_list = [volume, sphere]

    # Create constraints
    constraints = [
        # Volume
        GridCoordinateConstraint(object="simulation_volume", axes=(0,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(0,), sides=("+",), coordinates=(100,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(1,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(1,), sides=("+",), coordinates=(100,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(2,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(2,), sides=("+",), coordinates=(100,)),
        # Sphere: centered
        GridCoordinateConstraint(object="test_sphere", axes=(0,), sides=("-",), coordinates=(30,)),
        GridCoordinateConstraint(object="test_sphere", axes=(0,), sides=("+",), coordinates=(70,)),
        GridCoordinateConstraint(object="test_sphere", axes=(1,), sides=("-",), coordinates=(30,)),
        GridCoordinateConstraint(object="test_sphere", axes=(1,), sides=("+",), coordinates=(70,)),
        GridCoordinateConstraint(object="test_sphere", axes=(2,), sides=("-",), coordinates=(30,)),
        GridCoordinateConstraint(object="test_sphere", axes=(2,), sides=("+",), coordinates=(70,)),
    ]

    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list,
        config=config,
        constraints=constraints,
        key=key,
    )

    return config, arrays


def test_plot_material_from_side_xy_permittivity(simple_material_setup):
    """Test plot_material_from_side with XY plane showing permittivity."""
    config, arrays = simple_material_setup

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    result_fig = plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="z",
        ax=ax,
        plot_legend=True,
        position=0.0,  # Center
        type="permittivity",
        filename=TEST_OUTPUT_DIR / "test_plot_material_xy_permittivity.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_material_xy_permittivity.png").exists()
    plt.close(fig)


def test_plot_material_from_side_xz_permittivity(simple_material_setup):
    """Test plot_material_from_side with XZ plane showing permittivity."""
    config, arrays = simple_material_setup

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    result_fig = plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="y",
        ax=ax,
        plot_legend=True,
        position=0.0,  # Center
        type="permittivity",
        filename=TEST_OUTPUT_DIR / "test_plot_material_xz_permittivity.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_material_xz_permittivity.png").exists()
    plt.close(fig)


def test_plot_material_from_side_yz_permittivity(simple_material_setup):
    """Test plot_material_from_side with YZ plane showing permittivity."""
    config, arrays = simple_material_setup

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    result_fig = plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="x",
        ax=ax,
        plot_legend=True,
        position=0.0,  # Center
        type="permittivity",
        filename=TEST_OUTPUT_DIR / "test_plot_material_yz_permittivity.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_material_yz_permittivity.png").exists()
    plt.close(fig)


def test_plot_material_from_side_with_offset(simple_material_setup):
    """Test plot_material_from_side with position offset."""
    config, arrays = simple_material_setup

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    result_fig = plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="z",
        ax=ax,
        plot_legend=True,
        position=0.5e-6,  # 0.5 µm offset from center
        type="permittivity",
        filename=TEST_OUTPUT_DIR / "test_plot_material_offset.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_material_offset.png").exists()
    plt.close(fig)


def test_plot_material_all_planes_permittivity(simple_material_setup):
    """Test plot_material function showing permittivity in all three planes."""
    config, arrays = simple_material_setup

    result_fig = plot_material(
        config=config,
        arrays=arrays,
        plot_legend=True,
        type="permittivity",
        filename=TEST_OUTPUT_DIR / "test_plot_material_all_permittivity.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_material_all_permittivity.png").exists()
    plt.close("all")


def test_plot_material_with_custom_positions(simple_material_setup):
    """Test plot_material with custom slice positions."""
    config, arrays = simple_material_setup

    # Custom positions for each slice (in meters)
    custom_positions = (0.2e-6, -0.2e-6, 0.3e-6)  # x, y, z offsets

    result_fig = plot_material(
        config=config,
        arrays=arrays,
        plot_legend=True,
        positions=custom_positions,
        type="permittivity",
        filename=TEST_OUTPUT_DIR / "test_plot_material_custom_positions.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_material_custom_positions.png").exists()
    plt.close("all")


def test_plot_material_permeability(simple_material_setup):
    """Test plot_material_from_side showing permeability."""
    config, arrays = simple_material_setup

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    result_fig = plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="z",
        ax=ax,
        plot_legend=True,
        position=0.0,
        type="permeability",
        filename=TEST_OUTPUT_DIR / "test_plot_material_permeability.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_material_permeability.png").exists()
    plt.close(fig)


def test_plot_material_verify_values(simple_material_setup):
    """Test that plot_material correctly displays material values and spans the full domain."""
    config, arrays = simple_material_setup

    # Plot XY plane at center
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="z",
        ax=ax,
        plot_legend=False,
        position=0.0,
        type="permittivity",
    )

    # Get the image data from the plot
    im = ax.get_images()[0]
    data = im.get_array()

    # The data should contain material values
    assert data is not None
    assert data.size > 0

    # Verify the plot spans the full simulation domain.
    # Setup uses 40 grid cells at 50nm resolution = 2.0 µm in each direction.
    # If the component axis bug is present, array_shape[0] == num_components (~3)
    # instead of Nx (40), collapsing the x-extent to ~0.15 µm instead of 2.0 µm.
    extent = im.get_extent()  # [xmin, xmax, ymin, ymax] in µm
    expected_size_um = 2.0  # 40 cells * 50nm = 2µm
    assert abs(extent[1] - expected_size_um) < 0.1, (
        f"X-extent should be ~{expected_size_um} µm but got {extent[1]:.4f} µm. "
        f"This likely means array_shape[0] is num_components instead of Nx."
    )
    assert abs(extent[3] - expected_size_um) < 0.1, (
        f"Y-extent should be ~{expected_size_um} µm but got {extent[3]:.4f} µm."
    )

    plt.close(fig)


def test_plot_material_cylinder_slice(cylinder_setup):
    """Test plotting material with cylinder object."""
    config, arrays = cylinder_setup

    # Plot at z position that should show the cylinder cross-section
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    result_fig = plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="z",
        ax=ax,
        plot_legend=True,
        position=0.0,  # Center where cylinder exists
        type="permittivity",
        filename=TEST_OUTPUT_DIR / "test_plot_material_cylinder.png",
    )

    # The plot should show a cylindrical region
    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_material_cylinder.png").exists()

    plt.close(fig)


def test_plot_material_sphere_slice(sphere_setup):
    """Test plotting material with sphere object."""
    config, arrays = sphere_setup

    # Plot at y position that should show the sphere cross-section
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    result_fig = plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="y",
        ax=ax,
        plot_legend=True,
        position=0.0,  # Center where sphere is centered in y
        type="permittivity",
        filename=TEST_OUTPUT_DIR / "test_plot_material_sphere.png",
    )

    # The plot should show a spherical region
    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_material_sphere.png").exists()

    plt.close(fig)


def test_plot_material_custom_axes(simple_material_setup):
    """Test plot_material with custom axes provided."""
    config, arrays = simple_material_setup

    # Create custom figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    result_fig = plot_material(
        config=config,
        arrays=arrays,
        axs=axs,
        plot_legend=True,
        type="permittivity",
    )

    assert result_fig is not None
    assert len(axs) == 3

    # Check each axis has a plot
    for ax in axs:
        assert len(ax.get_images()) > 0
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""

    plt.close(fig)


def test_plot_material_edge_positions(simple_material_setup):
    """Test plot_material with extreme position values."""
    config, arrays = simple_material_setup

    # Test with very large offset (should clamp to edge)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    result_fig = plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="z",
        ax=ax,
        plot_legend=False,
        position=10e-6,  # Way outside the volume
        type="permittivity",
    )

    # Should still work (clamps to edge)
    assert result_fig is not None

    plt.close(fig)


def test_plot_material_no_legend(simple_material_setup):
    """Test plot_material without legend."""
    config, arrays = simple_material_setup

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    result_fig = plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="z",
        ax=ax,
        plot_legend=False,  # No legend
        position=0.0,
        type="permittivity",
    )

    assert result_fig is not None
    # Check that colorbar was not added
    # (colorbar adds a separate axis, so we should only have the main axis)
    assert len(fig.axes) == 1  # Only the main axis, no colorbar axis

    plt.close(fig)


def test_plot_material_with_external_figure(simple_material_setup):
    """Test plot_material with externally created figure."""
    config, arrays = simple_material_setup

    # Create figure externally and pass axis
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    result_fig = plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="z",
        ax=ax,
        plot_legend=True,
        position=0.0,
        type="permittivity",
    )

    assert result_fig is not None
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""
    assert ax.get_title() != ""

    plt.close(fig)


def test_plot_material_all_types_objects():
    """Test plot_material with UniformMaterialObject, Cylinder, and Sphere."""
    config = SimulationConfig(
        resolution=40e-9,
        time=100e-15,
    )

    volume = SimulationVolume(
        partial_real_shape=(4e-6, 4e-6, 4e-6),
        name="simulation_volume",
    )

    # Create different types of objects
    cube = UniformMaterialObject(
        name="cube",
        material=Material(permittivity=2.25, permeability=1.0),
    )

    cylinder = Cylinder(
        name="cylinder",
        radius=0.5e-6,
        axis=2,
        materials={"silicon": Material(permittivity=11.7, permeability=1.0)},
        material_name="silicon",
    )

    sphere = Sphere(
        name="sphere",
        radius=0.5e-6,
        materials={"high_dielectric": Material(permittivity=9.0, permeability=1.0)},
        material_name="high_dielectric",
    )

    object_list = [volume, cube, cylinder, sphere]

    # Create constraints
    constraints = [
        # Volume
        GridCoordinateConstraint(object="simulation_volume", axes=(0,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(0,), sides=("+",), coordinates=(100,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(1,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(1,), sides=("+",), coordinates=(100,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(2,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(2,), sides=("+",), coordinates=(100,)),
        # Cube: at bottom left
        GridCoordinateConstraint(object="cube", axes=(0,), sides=("-",), coordinates=(10,)),
        GridCoordinateConstraint(object="cube", axes=(0,), sides=("+",), coordinates=(30,)),
        GridCoordinateConstraint(object="cube", axes=(1,), sides=("-",), coordinates=(10,)),
        GridCoordinateConstraint(object="cube", axes=(1,), sides=("+",), coordinates=(30,)),
        GridCoordinateConstraint(object="cube", axes=(2,), sides=("-",), coordinates=(10,)),
        GridCoordinateConstraint(object="cube", axes=(2,), sides=("+",), coordinates=(30,)),
        # Cylinder: centered
        GridCoordinateConstraint(object="cylinder", axes=(0,), sides=("-",), coordinates=(40,)),
        GridCoordinateConstraint(object="cylinder", axes=(0,), sides=("+",), coordinates=(60,)),
        GridCoordinateConstraint(object="cylinder", axes=(1,), sides=("-",), coordinates=(40,)),
        GridCoordinateConstraint(object="cylinder", axes=(1,), sides=("+",), coordinates=(60,)),
        GridCoordinateConstraint(object="cylinder", axes=(2,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="cylinder", axes=(2,), sides=("+",), coordinates=(100,)),
        # Sphere: at top right
        GridCoordinateConstraint(object="sphere", axes=(0,), sides=("-",), coordinates=(70,)),
        GridCoordinateConstraint(object="sphere", axes=(0,), sides=("+",), coordinates=(90,)),
        GridCoordinateConstraint(object="sphere", axes=(1,), sides=("-",), coordinates=(70,)),
        GridCoordinateConstraint(object="sphere", axes=(1,), sides=("+",), coordinates=(90,)),
        GridCoordinateConstraint(object="sphere", axes=(2,), sides=("-",), coordinates=(70,)),
        GridCoordinateConstraint(object="sphere", axes=(2,), sides=("+",), coordinates=(90,)),
    ]

    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list,
        config=config,
        constraints=constraints,
        key=key,
    )

    # Test plotting
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    result_fig = plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="z",
        ax=ax,
        plot_legend=True,
        position=0.0,
        type="permittivity",
        filename=TEST_OUTPUT_DIR / "test_plot_material_all_types.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_material_all_types.png").exists()
    plt.close(fig)


def test_plot_material_material_axis(simple_material_setup):
    """Test that material_axis is accepted by plot_material and forwarded to all subplots.

    Before the fix, plot_material had no material_axis parameter and never forwarded
    it to plot_material_from_side, so components 1 and 2 were silently unreachable.
    """
    config, arrays = simple_material_setup

    for axis in [0, 1, 2]:
        result_fig = plot_material(
            config=config,
            arrays=arrays,
            plot_legend=False,
            type="permittivity",
            material_axis=axis,
        )
        assert result_fig is not None, f"plot_material returned None for material_axis={axis}"

        # Verify all three subplots have image data
        axs = result_fig.get_axes()
        # Filter out colorbar axes (they have no images)
        plot_axs = [a for a in axs if len(a.get_images()) > 0]
        assert len(plot_axs) == 3, f"Expected 3 subplots with image data for material_axis={axis}, got {len(plot_axs)}"

        # Verify each subplot spans the full 2µm domain on both axes
        expected_size_um = 2.0  # 40 cells * 50nm
        for ax in plot_axs:
            im = ax.get_images()[0]
            extent = im.get_extent()  # [xmin, xmax, ymin, ymax]
            assert abs(extent[1] - expected_size_um) < 0.1, (
                f"material_axis={axis}: x-extent should be ~{expected_size_um} µm "
                f"but got {extent[1]:.4f} µm in subplot '{ax.get_title()}'"
            )
            assert abs(extent[3] - expected_size_um) < 0.1, (
                f"material_axis={axis}: y-extent should be ~{expected_size_um} µm "
                f"but got {extent[3]:.4f} µm in subplot '{ax.get_title()}'"
            )

        plt.close("all")
