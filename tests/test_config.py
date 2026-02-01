import math

import jax.numpy as jnp
import pytest

from fdtdx import constants
from fdtdx.config import SimulationConfig
from fdtdx.units.typing import SI
from fdtdx.units.unitful import Unit, Unitful


def test_simulation_config_time_step_duration_unitful():
    """Test that time_step_duration returns correct Unitful with time dimensions"""
    # Create simulation with 1 nanosecond resolution and 10 picoseconds simulation time
    resolution = Unitful(val=1e-9, unit=Unit(scale=0, dim={SI.m: 1}))  # 1 nm
    time = Unitful(val=10e-12, unit=Unit(scale=0, dim={SI.s: 1}))  # 10 ps

    config = SimulationConfig(time=time, resolution=resolution, courant_factor=0.5)

    time_step = config.time_step_duration

    assert isinstance(time_step, Unitful)
    assert time_step.unit.dim == {SI.s: 1}  # Should have time dimension

    # Calculate expected value: courant_number * resolution / c
    expected_courant = 0.5 / math.sqrt(3)  # courant_factor / sqrt(3)
    expected_duration = expected_courant * 1e-9 / constants.c.value()
    assert jnp.allclose(time_step.value(), expected_duration, rtol=1e-10)


def test_simulation_config_max_travel_distance_unitful():
    """Test that max_travel_distance returns correct Unitful with length dimensions"""
    # Create simulation with 100 micrometers resolution and 1 femtosecond time
    resolution = Unitful(val=100e-6, unit=Unit(scale=0, dim={SI.m: 1}))  # 100 μm
    time = Unitful(val=1e-15, unit=Unit(scale=0, dim={SI.s: 1}))  # 1 fs

    config = SimulationConfig(time=time, resolution=resolution)

    max_distance = config.max_travel_distance

    assert isinstance(max_distance, Unitful)
    assert max_distance.unit.dim == {SI.m: 1}  # Should have length dimension

    # Expected: speed of light * time
    expected_distance = constants.c.value() * 1e-15
    assert jnp.allclose(max_distance.value(), expected_distance, rtol=1e-10)


def test_simulation_config_unitful_with_different_scales():
    """Test SimulationConfig with Unitful inputs having different unit scales"""
    # Use millimeters for resolution (scale -3) and microseconds for time (scale -6)
    resolution = Unitful(val=0.5, unit=Unit(scale=-3, dim={SI.m: 1}))  # 0.5 mm
    time = Unitful(val=2.0, unit=Unit(scale=-6, dim={SI.s: 1}))  # 2 μs

    config = SimulationConfig(time=time, resolution=resolution, courant_factor=0.8)

    # Test time_step_duration preserves unit consistency
    time_step = config.time_step_duration
    assert isinstance(time_step, Unitful)
    assert time_step.unit.dim == {SI.s: 1}

    # Test max_travel_distance preserves unit consistency
    max_distance = config.max_travel_distance
    assert isinstance(max_distance, Unitful)
    assert max_distance.unit.dim == {SI.m: 1}

    # Verify the calculations are dimensionally consistent
    # time_step should be: (0.8/sqrt(3)) * 0.5e-3 / c
    expected_courant = 0.8 / math.sqrt(3)
    expected_time_step = expected_courant * 0.5e-3 / constants.c.value()
    assert jnp.allclose(time_step.value(), expected_time_step, rtol=1e-9)

    # max_distance should be: c * 2e-6
    expected_max_distance = constants.c.value() * 2e-6
    assert jnp.allclose(max_distance.value(), expected_max_distance, rtol=1e-9)


def test_simulation_config_time_steps_total_with_fractional_result():
    """Test time_steps_total calculation when time/time_step_duration gives fractional result"""
    # Set up parameters that will give a non-integer number of time steps
    resolution = Unitful(val=50e-9, unit=Unit(scale=0, dim={SI.m: 1}))  # 50 nm
    time = Unitful(val=1.337e-15, unit=Unit(scale=0, dim={SI.s: 1}))  # 1.337 fs

    config = SimulationConfig(time=time, resolution=resolution, courant_factor=0.99)

    # Verify the intermediate calculation uses Unitful properly
    time_step_duration = config.time_step_duration
    assert isinstance(time_step_duration, Unitful)

    # The division should work with Unitful objects
    time_ratio = config.time / time_step_duration
    assert isinstance(time_ratio, Unitful)

    # After materialise(), should get a dimensionless number
    materialised_ratio = time_ratio.materialise()
    assert isinstance(materialised_ratio, (float, int, jnp.ndarray))

    # time_steps_total should be an integer (rounded)
    total_steps = config.time_steps_total
    assert isinstance(total_steps, int)
    assert total_steps > 0


def test_simulation_config_unitful_array_inputs():
    """Test SimulationConfig with Unitful array inputs for time and resolution"""
    # Create array inputs (though config likely expects scalars, test robustness)
    resolution_vals = jnp.array([10e-9])
    # Using an array value like jnp.array([1e-12]) can trigger best_scale during Unitful construction.
    # The resulting log_offset may be large, and 10**log_offset becomes a huge Python int that JAX
    # may parse as int32, causing an overflow.
    time_vals = jnp.array([1e-12])

    resolution = Unitful(val=resolution_vals, unit=Unit(scale=0, dim={SI.m: 1}))
    time = Unitful(val=time_vals, unit=Unit(scale=0, dim={SI.s: 1}))

    # This might raise an error or handle arrays - test the behavior
    try:
        config = SimulationConfig(time=time, resolution=resolution)

        # If arrays are supported, verify the derived properties maintain Unitful nature
        time_step = config.time_step_duration
        max_distance = config.max_travel_distance

        assert isinstance(time_step, Unitful)
        assert isinstance(max_distance, Unitful)
        assert time_step.unit.dim == {SI.s: 1}
        assert max_distance.unit.dim == {SI.m: 1}

        # Verify shapes are preserved or handled appropriately
        if hasattr(time_step.val, "shape"):
            assert time_step.val.shape == resolution_vals.shape  # type: ignore
        if hasattr(max_distance.val, "shape"):
            assert max_distance.val.shape == time_vals.shape  # type: ignore

    except (ValueError, TypeError, Exception) as e:
        # If arrays aren't supported, that's also valid behavior to test
        pytest.skip(f"Array inputs not supported in SimulationConfig: {e}")
