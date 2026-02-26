"""Integration tests for utils/plot_material.py - requires place_objects simulation setup."""

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
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
    config = SimulationConfig(
        resolution=50e-9,
        time=100e-15,
    )

    volume = SimulationVolume(
        partial_real_shape=(2e-6, 2e-6, 2e-6),
        name="simulation_volume",
    )

    material = Material(permittivity=2.25, permeability=1.0)

    test_object = UniformMaterialObject(
        name="test_object",
        material=material,
    )

    object_list = [volume, test_object]

    constraints = [
        GridCoordinateConstraint(object="simulation_volume", axes=(0,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(0,), sides=("+",), coordinates=(40,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(1,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(1,), sides=("+",), coordinates=(40,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(2,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(2,), sides=("+",), coordinates=(40,)),
        GridCoordinateConstraint(object="test_object", axes=(0,), sides=("-",), coordinates=(15,)),
        GridCoordinateConstraint(object="test_object", axes=(0,), sides=("+",), coordinates=(25,)),
        GridCoordinateConstraint(object="test_object", axes=(1,), sides=("-",), coordinates=(15,)),
        GridCoordinateConstraint(object="test_object", axes=(1,), sides=("+",), coordinates=(25,)),
        GridCoordinateConstraint(object="test_object", axes=(2,), sides=("-",), coordinates=(15,)),
        GridCoordinateConstraint(object="test_object", axes=(2,), sides=("+",), coordinates=(25,)),
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

    cylinder = Cylinder(
        name="test_cylinder",
        radius=0.5e-6,
        axis=2,
        materials={"dielectric": Material(permittivity=4.0, permeability=1.0)},
        material_name="dielectric",
    )

    object_list = [volume, cylinder]

    constraints = [
        GridCoordinateConstraint(object="simulation_volume", axes=(0,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(0,), sides=("+",), coordinates=(100,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(1,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(1,), sides=("+",), coordinates=(100,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(2,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(2,), sides=("+",), coordinates=(100,)),
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

    sphere = Sphere(
        name="test_sphere",
        radius=0.5e-6,
        materials={"dielectric": Material(permittivity=9.0, permeability=1.0)},
        material_name="dielectric",
    )

    object_list = [volume, sphere]

    constraints = [
        GridCoordinateConstraint(object="simulation_volume", axes=(0,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(0,), sides=("+",), coordinates=(100,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(1,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(1,), sides=("+",), coordinates=(100,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(2,), sides=("-",), coordinates=(0,)),
        GridCoordinateConstraint(object="simulation_volume", axes=(2,), sides=("+",), coordinates=(100,)),
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


@pytest.mark.integration
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
        position=0.0,
        type="permittivity",
        filename=TEST_OUTPUT_DIR / "test_plot_material_xy_permittivity.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_material_xy_permittivity.png").exists()
    plt.close(fig)


@pytest.mark.integration
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
        position=0.0,
        type="permittivity",
        filename=TEST_OUTPUT_DIR / "test_plot_material_xz_permittivity.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_material_xz_permittivity.png").exists()
    plt.close(fig)


@pytest.mark.integration
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
        position=0.0,
        type="permittivity",
        filename=TEST_OUTPUT_DIR / "test_plot_material_yz_permittivity.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_material_yz_permittivity.png").exists()
    plt.close(fig)


@pytest.mark.integration
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
        position=0.5e-6,
        type="permittivity",
        filename=TEST_OUTPUT_DIR / "test_plot_material_offset.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_material_offset.png").exists()
    plt.close(fig)


@pytest.mark.integration
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


@pytest.mark.integration
def test_plot_material_with_custom_positions(simple_material_setup):
    """Test plot_material with custom slice positions."""
    config, arrays = simple_material_setup

    custom_positions = (0.2e-6, -0.2e-6, 0.3e-6)

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


@pytest.mark.integration
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


@pytest.mark.integration
def test_plot_material_verify_values(simple_material_setup):
    """Test that plot_material correctly displays material values."""
    config, arrays = simple_material_setup

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

    im = ax.get_images()[0]
    data = im.get_array()

    assert data is not None
    assert data.size > 0

    plt.close(fig)


@pytest.mark.integration
def test_plot_material_cylinder_slice(cylinder_setup):
    """Test plotting material with cylinder object."""
    config, arrays = cylinder_setup

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    result_fig = plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="z",
        ax=ax,
        plot_legend=True,
        position=0.0,
        type="permittivity",
        filename=TEST_OUTPUT_DIR / "test_plot_material_cylinder.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_material_cylinder.png").exists()
    plt.close(fig)


@pytest.mark.integration
def test_plot_material_sphere_slice(sphere_setup):
    """Test plotting material with sphere object."""
    config, arrays = sphere_setup

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    result_fig = plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="y",
        ax=ax,
        plot_legend=True,
        position=0.0,
        type="permittivity",
        filename=TEST_OUTPUT_DIR / "test_plot_material_sphere.png",
    )

    assert result_fig is not None
    assert (TEST_OUTPUT_DIR / "test_plot_material_sphere.png").exists()
    plt.close(fig)


@pytest.mark.integration
def test_plot_material_custom_axes(simple_material_setup):
    """Test plot_material with custom axes provided."""
    config, arrays = simple_material_setup

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

    for ax in axs:
        assert len(ax.get_images()) > 0
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""

    plt.close(fig)


@pytest.mark.integration
def test_plot_material_edge_positions(simple_material_setup):
    """Test plot_material with extreme position values (clamped to edge)."""
    config, arrays = simple_material_setup

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    result_fig = plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="z",
        ax=ax,
        plot_legend=False,
        position=10e-6,
        type="permittivity",
    )

    assert result_fig is not None
    plt.close(fig)


@pytest.mark.integration
def test_plot_material_no_legend(simple_material_setup):
    """Test plot_material without legend."""
    config, arrays = simple_material_setup

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    result_fig = plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="z",
        ax=ax,
        plot_legend=False,
        position=0.0,
        type="permittivity",
    )

    assert result_fig is not None
    assert len(fig.axes) == 1

    plt.close(fig)


@pytest.mark.integration
def test_plot_material_with_external_figure(simple_material_setup):
    """Test plot_material with externally created figure."""
    config, arrays = simple_material_setup

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
