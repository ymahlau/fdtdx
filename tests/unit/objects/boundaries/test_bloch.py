"""Unit tests for objects/boundaries/bloch.py.

Tests BlochBoundary initialization, properties, and Bloch phase computation.
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.objects.boundaries.bloch import BlochBoundary


@pytest.fixture
def micro_config():
    return SimulationConfig(
        time=100e-15,
        resolution=50e-9,
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=None,
    )


@pytest.fixture
def jax_key():
    return jax.random.PRNGKey(0)


def make_bloch(axis=0, direction="-", bloch_vector=(1e6, 0.0, 0.0)):
    """Create a BlochBoundary with given axis/direction/bloch_vector."""
    shape_list = [None, None, None]
    shape_list[axis] = 1
    return BlochBoundary(
        axis=axis,
        partial_grid_shape=tuple(shape_list),
        direction=direction,
        bloch_vector=bloch_vector,
    )


def place_bloch(bb, micro_config, jax_key, volume_shape=(30, 20, 20)):
    """Place a BlochBoundary on a grid matching the given volume shape."""
    axis = bb.axis
    direction = bb.direction

    slices = [[0, volume_shape[i]] for i in range(3)]
    if direction == "-":
        slices[axis] = [0, 1]
    else:
        slices[axis] = [volume_shape[axis] - 1, volume_shape[axis]]

    grid_slice_tuple = tuple(tuple(s) for s in slices)
    return bb.place_on_grid(grid_slice_tuple=grid_slice_tuple, config=micro_config, key=jax_key)


class TestBlochThickness:
    """Tests for BlochBoundary.thickness."""

    def test_thickness_always_one(self):
        bb = make_bloch(axis=0, direction="-")
        assert bb.thickness == 1

    def test_thickness_one_for_all_axes(self):
        for axis in range(3):
            for direction in ("-", "+"):
                bb = make_bloch(axis=axis, direction=direction)
                assert bb.thickness == 1


class TestBlochUsesWrapPadding:
    """Tests for BlochBoundary.uses_wrap_padding."""

    def test_uses_wrap_padding_true(self):
        bb = make_bloch(axis=0, direction="-")
        assert bb.uses_wrap_padding is True

    def test_uses_wrap_padding_for_all_axes(self):
        for axis in range(3):
            bb = make_bloch(axis=axis, direction="+")
            assert bb.uses_wrap_padding is True


class TestBlochDescriptiveName:
    """Tests for BlochBoundary.descriptive_name."""

    @pytest.mark.parametrize(
        "axis,direction,expected",
        [
            (0, "-", "min_x"),
            (0, "+", "max_x"),
            (1, "-", "min_y"),
            (1, "+", "max_y"),
            (2, "-", "min_z"),
            (2, "+", "max_z"),
        ],
    )
    def test_descriptive_name(self, micro_config, jax_key, axis, direction, expected):
        bb = make_bloch(axis=axis, direction=direction)
        placed = place_bloch(bb, micro_config, jax_key, volume_shape=(30, 30, 30))
        assert placed.descriptive_name == expected


class TestBlochVectorStorage:
    """Tests that bloch_vector is stored and accessible."""

    def test_bloch_vector_stored(self):
        vec = (1e6, 2e6, 3e6)
        bb = make_bloch(axis=0, direction="-", bloch_vector=vec)
        assert bb.bloch_vector == vec

    def test_zero_bloch_vector(self):
        vec = (0.0, 0.0, 0.0)
        bb = make_bloch(axis=1, direction="+", bloch_vector=vec)
        assert bb.bloch_vector == vec


class TestBlochGetBlochPhase:
    """Tests for BlochBoundary.get_bloch_phase."""

    def test_zero_bloch_vector_gives_unit_phase(self, micro_config, jax_key):
        """k=0 => exp(i*0*L) = 1+0j."""
        bb = make_bloch(axis=0, direction="-", bloch_vector=(0.0, 0.0, 0.0))
        placed = place_bloch(bb, micro_config, jax_key, volume_shape=(10, 10, 10))
        phase = placed.get_bloch_phase(
            volume_shape=(10, 10, 10),
            resolution=micro_config.resolution,
        )
        assert jnp.abs(phase - 1.0) < 1e-6

    def test_phase_is_complex(self, micro_config, jax_key):
        """Phase returned is a complex scalar."""
        bb = make_bloch(axis=0, direction="-", bloch_vector=(1e6, 0.0, 0.0))
        placed = place_bloch(bb, micro_config, jax_key, volume_shape=(10, 10, 10))
        phase = placed.get_bloch_phase(
            volume_shape=(10, 10, 10),
            resolution=micro_config.resolution,
        )
        assert jnp.issubdtype(phase.dtype, jnp.complexfloating)

    def test_phase_magnitude_is_one(self, micro_config, jax_key):
        """|exp(i*k*L)| = 1 for any real k, L."""
        bb = make_bloch(axis=1, direction="+", bloch_vector=(0.0, 5e6, 0.0))
        placed = place_bloch(bb, micro_config, jax_key, volume_shape=(20, 30, 20))
        phase = placed.get_bloch_phase(
            volume_shape=(20, 30, 20),
            resolution=micro_config.resolution,
        )
        assert jnp.abs(jnp.abs(phase) - 1.0) < 1e-6

    def test_phase_formula(self, micro_config):
        """Phase equals exp(i * k_axis * Naxis * resolution)."""
        k = 1e7
        axis = 2
        Nz = 25
        resolution = micro_config.resolution
        bb = make_bloch(axis=axis, direction="-", bloch_vector=(0.0, 0.0, k))
        # We can call get_bloch_phase without placing (it only uses bloch_vector and axis)
        phase = bb.get_bloch_phase(
            volume_shape=(10, 10, Nz),
            resolution=resolution,
        )
        expected = jnp.exp(1j * k * Nz * resolution)
        assert jnp.abs(phase - expected) < 1e-6

    def test_uses_correct_axis_component(self, micro_config):
        """Only the k component for this boundary's axis affects the phase."""
        k_x, k_y, k_z = 1e6, 2e6, 3e6
        resolution = micro_config.resolution
        volume_shape = (10, 20, 30)

        for axis, k_axis in enumerate([k_x, k_y, k_z]):
            vec_full = (k_x, k_y, k_z)
            bb = make_bloch(axis=axis, direction="-", bloch_vector=vec_full)
            phase = bb.get_bloch_phase(volume_shape, resolution)
            expected = jnp.exp(1j * k_axis * volume_shape[axis] * resolution)
            assert jnp.abs(phase - expected) < 1e-6
