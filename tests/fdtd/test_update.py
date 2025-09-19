from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.fdtd.update import (
    add_interfaces,
    collect_interfaces,
    get_periodic_axes,
    update_detector_states,
    update_E,
    update_E_reverse,
    update_H,
    update_H_reverse,
)
from fdtdx.objects.boundaries.periodic import PeriodicBoundary


class MockArrayContainer:
    """Mock ArrayContainer that supports the at['field'].set(value) syntax"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def at(self, field_name):
        return MockArrayContainerSetter(self, field_name)

    def aset(self, field_name, value):
        setattr(self, field_name, value)
        return self


class MockArrayContainerSetter:
    """Helper class to support the at['field'].set(value) syntax"""

    def __init__(self, container, field_name):
        self._container = container
        self._field_name = field_name

    def set(self, value):
        setattr(self._container, self._field_name, value)
        return self._container


class TestPeriodicAxes:
    """Test get_periodic_axes function"""

    def test_no_periodic_boundaries(self):
        """Test with no periodic boundaries"""
        mock_objects = Mock()
        mock_objects.boundary_objects = []

        result = get_periodic_axes(mock_objects)
        assert result == (False, False, False)

    def test_single_periodic_boundary(self):
        """Test with a single periodic boundary"""
        mock_boundary = Mock(spec=PeriodicBoundary)
        mock_boundary.axis = 0  # x-axis

        mock_objects = Mock()
        mock_objects.boundary_objects = [mock_boundary]

        result = get_periodic_axes(mock_objects)
        assert result == (True, False, False)

    def test_multiple_periodic_boundaries(self):
        """Test with multiple periodic boundaries"""
        mock_boundary1 = Mock(spec=PeriodicBoundary)
        mock_boundary1.axis = 0  # x-axis

        mock_boundary2 = Mock(spec=PeriodicBoundary)
        mock_boundary2.axis = 2  # z-axis

        mock_objects = Mock()
        mock_objects.boundary_objects = [mock_boundary1, mock_boundary2]

        result = get_periodic_axes(mock_objects)
        assert result == (True, False, True)

    def test_mixed_boundary_types(self):
        """Test with mixed boundary types (only PeriodicBoundary should be counted)"""
        mock_periodic = Mock(spec=PeriodicBoundary)
        mock_periodic.axis = 1  # y-axis

        mock_other = Mock()  # Not a PeriodicBoundary

        mock_objects = Mock()
        mock_objects.boundary_objects = [mock_periodic, mock_other]

        result = get_periodic_axes(mock_objects)
        assert result == (False, True, False)


class TestUpdateE:
    """Test update_E function"""

    @pytest.fixture
    def setup(self):
        """Setup common test data"""
        # Create mock arrays with compatible shapes
        arrays = MockArrayContainer(
            E=jnp.ones((10, 10, 10, 3)),
            H=jnp.zeros((10, 10, 10, 3)),
            inv_permittivities=jnp.ones((10, 10, 10, 3)),
            electric_conductivity=None,
            boundary_states={},
            inv_permeabilities=jnp.ones((10, 10, 10, 3)),
        )

        # Mock the curl_H function to return compatible shape
        with patch("fdtdx.fdtd.update.curl_H") as mock_curl:
            mock_curl.return_value = jnp.zeros((10, 10, 10, 3))

            # Create mock objects
            mock_objects = Mock()
            mock_objects.boundary_objects = []
            mock_objects.sources = []

            # Create mock config
            mock_config = Mock()
            mock_config.courant_number = 0.5

            yield arrays, mock_objects, mock_config

    def test_basic_update(self, setup):
        """Test basic E field update without boundaries or sources"""
        arrays, objects, config = setup

        # Patch the at method to avoid the subscript issue
        with patch.object(arrays, "at") as mock_at:
            mock_setter = Mock()
            mock_setter.set.return_value = arrays
            mock_at.return_value = mock_setter

            result = update_E(
                time_step=jnp.array(0), arrays=arrays, objects=objects, config=config, simulate_boundaries=False
            )

        # Should return updated arrays
        assert result is not None

    def test_with_boundaries(self, setup):
        """Test E field update with boundaries"""
        arrays, objects, config = setup

        # Mock boundary object
        mock_boundary = Mock()
        mock_boundary.name = "test_boundary"
        mock_boundary.update_E_boundary_state.return_value = "test_state"
        mock_boundary.update_E.return_value = jnp.ones((10, 10, 10, 3))

        objects.boundary_objects = [mock_boundary]
        arrays.boundary_states = {"test_boundary": "initial_state"}

        # Patch the at method to avoid the subscript issue
        with patch.object(arrays, "at") as mock_at:
            mock_setter = Mock()
            mock_setter.set.return_value = arrays
            mock_at.return_value = mock_setter

            update_E(time_step=jnp.array(0), arrays=arrays, objects=objects, config=config, simulate_boundaries=True)

        # Verify boundary methods were called
        mock_boundary.update_E_boundary_state.assert_called_once()
        mock_boundary.update_E.assert_called_once()

    def test_with_sources(self, setup):
        """Test E field update with sources"""
        arrays, objects, config = setup

        # Mock source object
        mock_source = Mock()
        mock_source.is_on_at_time_step.return_value = True
        mock_source.adjust_time_step_by_on_off.return_value = jnp.array(0)
        mock_source.update_E.return_value = jnp.ones((10, 10, 10, 3))

        objects.sources = [mock_source]

        # Patch the at method to avoid the subscript issue
        with patch.object(arrays, "at") as mock_at:
            mock_setter = Mock()
            mock_setter.set.return_value = arrays
            mock_at.return_value = mock_setter

            update_E(time_step=jnp.array(0), arrays=arrays, objects=objects, config=config, simulate_boundaries=False)

        # Verify source methods were called
        mock_source.is_on_at_time_step.assert_called_once()
        mock_source.update_E.assert_called_once()


class TestUpdateEReverse:
    """Test update_E_reverse function"""

    @pytest.fixture
    def setup(self):
        """Setup common test data"""
        # Create mock arrays with compatible shapes
        arrays = MockArrayContainer(
            E=jnp.ones((10, 10, 10, 3)),
            H=jnp.zeros((10, 10, 10, 3)),
            inv_permittivities=jnp.ones((10, 10, 10, 3)),
            inv_permeabilities=jnp.ones((10, 10, 10, 3)),
            electric_conductivity=None,
        )

        # Mock the curl_H function to return compatible shape
        with patch("fdtdx.fdtd.update.curl_H") as mock_curl:
            mock_curl.return_value = jnp.zeros((10, 10, 10, 3))

            # Create mock objects
            mock_objects = Mock()
            mock_objects.boundary_objects = []
            mock_objects.sources = []

            # Create mock config
            mock_config = Mock()
            mock_config.courant_number = 0.5

            yield arrays, mock_objects, mock_config

    def test_basic_reverse_update(self, setup):
        """Test basic reverse E field update"""
        arrays, objects, config = setup

        # Patch the at method to avoid the subscript issue
        with patch.object(arrays, "at") as mock_at:
            mock_setter = Mock()
            mock_setter.set.return_value = arrays
            mock_at.return_value = mock_setter

            result = update_E_reverse(time_step=jnp.array(0), arrays=arrays, objects=objects, config=config)

        # Should return updated arrays
        assert result is not None

    def test_with_sources(self, setup):
        """Test reverse E field update with sources"""
        arrays, objects, config = setup

        # Mock source object
        mock_source = Mock()
        mock_source.is_on_at_time_step.return_value = True
        mock_source.adjust_time_step_by_on_off.return_value = jnp.array(0)
        mock_source.update_E.return_value = jnp.ones((10, 10, 10, 3))

        objects.sources = [mock_source]

        # Patch the at method to avoid the subscript issue
        with patch.object(arrays, "at") as mock_at:
            mock_setter = Mock()
            mock_setter.set.return_value = arrays
            mock_at.return_value = mock_setter

            update_E_reverse(time_step=jnp.array(0), arrays=arrays, objects=objects, config=config)

        # Verify source methods were called
        mock_source.is_on_at_time_step.assert_called_once()
        mock_source.update_E.assert_called_once()


class TestUpdateH:
    """Test update_H function"""

    @pytest.fixture
    def setup(self):
        """Setup common test data"""
        # Create mock arrays with compatible shapes
        arrays = MockArrayContainer(
            H=jnp.ones((10, 10, 10, 3)),
            E=jnp.zeros((10, 10, 10, 3)),
            inv_permeabilities=jnp.ones((10, 10, 10, 3)),
            magnetic_conductivity=None,
            boundary_states={},
            inv_permittivities=jnp.ones((10, 10, 10, 3)),
        )

        # Mock the curl_E function to return compatible shape
        with patch("fdtdx.fdtd.update.curl_E") as mock_curl:
            mock_curl.return_value = jnp.zeros((10, 10, 10, 3))

            # Create mock objects
            mock_objects = Mock()
            mock_objects.boundary_objects = []
            mock_objects.sources = []

            # Create mock config
            mock_config = Mock()
            mock_config.courant_number = 0.5

            yield arrays, mock_objects, mock_config

    def test_basic_update(self, setup):
        """Test basic H field update without boundaries or sources"""
        arrays, objects, config = setup

        # Patch the at method to avoid the subscript issue
        with patch.object(arrays, "at") as mock_at:
            mock_setter = Mock()
            mock_setter.set.return_value = arrays
            mock_at.return_value = mock_setter

            result = update_H(
                time_step=jnp.array(0), arrays=arrays, objects=objects, config=config, simulate_boundaries=False
            )

        # Should return updated arrays
        assert result is not None

    def test_with_boundaries(self, setup):
        """Test H field update with boundaries"""
        arrays, objects, config = setup

        # Mock boundary object
        mock_boundary = Mock()
        mock_boundary.name = "test_boundary"
        mock_boundary.update_H_boundary_state.return_value = "test_state"
        mock_boundary.update_H.return_value = jnp.ones((10, 10, 10, 3))

        objects.boundary_objects = [mock_boundary]
        arrays.boundary_states = {"test_boundary": "initial_state"}

        # Patch the at method to avoid the subscript issue
        with patch.object(arrays, "at") as mock_at:
            mock_setter = Mock()
            mock_setter.set.return_value = arrays
            mock_at.return_value = mock_setter

            update_H(time_step=jnp.array(0), arrays=arrays, objects=objects, config=config, simulate_boundaries=True)

        # Verify boundary methods were called
        mock_boundary.update_H_boundary_state.assert_called_once()
        mock_boundary.update_H.assert_called_once()

    def test_with_sources(self, setup):
        """Test H field update with sources"""
        arrays, objects, config = setup

        # Mock source object
        mock_source = Mock()
        mock_source.is_on_at_time_step.return_value = True
        mock_source.adjust_time_step_by_on_off.return_value = jnp.array(0)
        mock_source.update_H.return_value = jnp.ones((10, 10, 10, 3))

        objects.sources = [mock_source]

        # Patch the at method to avoid the subscript issue
        with patch.object(arrays, "at") as mock_at:
            mock_setter = Mock()
            mock_setter.set.return_value = arrays
            mock_at.return_value = mock_setter

            update_H(time_step=jnp.array(0), arrays=arrays, objects=objects, config=config, simulate_boundaries=False)

        # Verify source methods were called
        mock_source.is_on_at_time_step.assert_called_once()
        mock_source.update_H.assert_called_once()


class TestUpdateHReverse:
    """Test update_H_reverse function"""

    @pytest.fixture
    def setup(self):
        """Setup common test data"""
        # Create mock arrays with compatible shapes
        arrays = MockArrayContainer(
            H=jnp.ones((10, 10, 10, 3)),
            E=jnp.zeros((10, 10, 10, 3)),
            inv_permittivities=jnp.ones((10, 10, 10, 3)),
            inv_permeabilities=jnp.ones((10, 10, 10, 3)),
            magnetic_conductivity=None,
        )

        # Mock the curl_E function to return compatible shape
        with patch("fdtdx.fdtd.update.curl_E") as mock_curl:
            mock_curl.return_value = jnp.zeros((10, 10, 10, 3))

            # Create mock objects
            mock_objects = Mock()
            mock_objects.boundary_objects = []
            mock_objects.sources = []

            # Create mock config
            mock_config = Mock()
            mock_config.courant_number = 0.5

            yield arrays, mock_objects, mock_config

    def test_basic_reverse_update(self, setup):
        """Test basic reverse H field update"""
        arrays, objects, config = setup

        # Patch the at method to avoid the subscript issue
        with patch.object(arrays, "at") as mock_at:
            mock_setter = Mock()
            mock_setter.set.return_value = arrays
            mock_at.return_value = mock_setter

            result = update_H_reverse(time_step=jnp.array(0), arrays=arrays, objects=objects, config=config)

        # Should return updated arrays
        assert result is not None

    def test_with_sources(self, setup):
        """Test reverse H field update with sources"""
        arrays, objects, config = setup

        # Mock source object
        mock_source = Mock()
        mock_source.is_on_at_time_step.return_value = True
        mock_source.adjust_time_step_by_on_off.return_value = jnp.array(0)
        mock_source.update_H.return_value = jnp.ones((10, 10, 10, 3))

        objects.sources = [mock_source]

        # Patch the at method to avoid the subscript issue
        with patch.object(arrays, "at") as mock_at:
            mock_setter = Mock()
            mock_setter.set.return_value = arrays
            mock_at.return_value = mock_setter

            update_H_reverse(time_step=jnp.array(0), arrays=arrays, objects=objects, config=config)

        # Verify source methods were called
        mock_source.is_on_at_time_step.assert_called_once()
        mock_source.update_H.assert_called_once()


class TestUpdateDetectorStates:
    """Test update_detector_states function"""

    @pytest.fixture
    def setup(self):
        """Setup common test data"""
        # Create mock arrays
        arrays = MockArrayContainer(
            E=jnp.ones((10, 10, 10, 3)),
            H=jnp.zeros((10, 10, 10, 3)),
            inv_permittivities=jnp.ones((10, 10, 10, 3)),
            inv_permeabilities=jnp.ones((10, 10, 10, 3)),
            detector_states={},
        )

        # Mock the interpolate_fields function
        with patch("fdtdx.fdtd.update.interpolate_fields") as mock_interpolate:
            mock_interpolate.return_value = (jnp.ones((10, 10, 10, 3)), jnp.zeros((10, 10, 10, 3)))

            # Create mock objects
            mock_objects = Mock()
            mock_objects.boundary_objects = []

            # Create H_prev array
            H_prev = jnp.zeros((10, 10, 10, 3))

            yield arrays, mock_objects, H_prev

    def test_forward_detectors(self, setup):
        """Test detector update in forward direction"""
        arrays, objects, H_prev = setup

        # Mock detector with proper JAX-compatible attributes
        mock_detector = Mock()
        mock_detector.name = "test_detector"
        mock_detector.exact_interpolation = False
        # Mock the specific method used in the detector update
        mock_detector._is_on_at_time_step_arr = [True] * 100
        mock_detector.update.return_value = "updated_state"

        objects.forward_detectors = [mock_detector]
        objects.backward_detectors = []
        arrays.detector_states = {"test_detector": "initial_state"}

        # Patch the jax.lax.cond to avoid JAX type issues
        with patch("fdtdx.fdtd.update.jax.lax.cond") as mock_cond:
            mock_cond.return_value = "updated_state"

            result = update_detector_states(
                time_step=jnp.array(0), arrays=arrays, objects=objects, H_prev=H_prev, inverse=False
            )

        # Verify detector update was considered
        assert result is not None

    def test_backward_detectors(self, setup):
        """Test detector update in backward direction"""
        arrays, objects, H_prev = setup

        # Mock detector with proper JAX-compatible attributes
        mock_detector = Mock()
        mock_detector.name = "test_detector"
        mock_detector.exact_interpolation = False
        # Mock the specific method used in the detector update
        mock_detector._is_on_at_time_step_arr = [True] * 100
        mock_detector.update.return_value = "updated_state"

        objects.forward_detectors = []
        objects.backward_detectors = [mock_detector]
        arrays.detector_states = {"test_detector": "initial_state"}

        # Patch the jax.lax.cond to avoid JAX type issues
        with patch("fdtdx.fdtd.update.jax.lax.cond") as mock_cond:
            mock_cond.return_value = "updated_state"

            result = update_detector_states(
                time_step=jnp.array(0), arrays=arrays, objects=objects, H_prev=H_prev, inverse=True
            )

        # Verify detector update was considered
        assert result is not None


class TestInterfaceFunctions:
    """Test collect_interfaces and add_interfaces functions"""

    @pytest.fixture
    def setup(self):
        """Setup common test data"""
        # Create mock arrays
        arrays = MockArrayContainer(recording_state="test_state")

        # Mock the collect_boundary_interfaces function
        with patch("fdtdx.fdtd.update.collect_boundary_interfaces") as mock_collect:
            mock_collect.return_value = {"test_interface": jnp.zeros((5, 5, 5, 3))}

            # Create mock objects
            mock_objects = Mock()
            mock_objects.pml_objects = []

            # Create mock config with gradient config
            mock_gradient_config = Mock()
            mock_recorder = Mock()
            mock_recorder.compress.return_value = "compressed_state"
            mock_recorder.decompress.return_value = ({"test_interface": jnp.zeros((5, 5, 5, 3))}, "decompressed_state")
            mock_gradient_config.recorder = mock_recorder

            mock_config = Mock()
            mock_config.gradient_config = mock_gradient_config

            # Create random key
            key = jax.random.PRNGKey(0)

            yield arrays, mock_objects, mock_config, key

    def test_collect_interfaces(self, setup):
        """Test collect_interfaces function"""
        arrays, objects, config, key = setup

        result = collect_interfaces(time_step=jnp.array(0), arrays=arrays, objects=objects, config=config, key=key)

        # Should return updated arrays
        assert result is not None
        # Verify recorder compress was called
        config.gradient_config.recorder.compress.assert_called_once()

    def test_add_interfaces(self, setup):
        """Test add_interfaces function"""
        arrays, objects, config, key = setup

        # Mock the add_boundary_interfaces function
        with patch("fdtdx.fdtd.update.add_boundary_interfaces") as mock_add:
            mock_add.return_value = arrays

            result = add_interfaces(time_step=jnp.array(0), arrays=arrays, objects=objects, config=config, key=key)

        # Should return updated arrays
        assert result is not None
        # Verify recorder decompress was called
        config.gradient_config.recorder.decompress.assert_called_once()

    def test_collect_interfaces_no_recorder(self, setup):
        """Test collect_interfaces without recorder raises exception"""
        arrays, objects, config, key = setup
        config.gradient_config.recorder = None

        with pytest.raises(Exception, match="Need recorder to record boundaries"):
            collect_interfaces(time_step=jnp.array(0), arrays=arrays, objects=objects, config=config, key=key)

    def test_collect_interfaces_no_recording_state(self, setup):
        """Test collect_interfaces without recording state raises exception"""
        arrays, objects, config, key = setup
        arrays.recording_state = None

        with pytest.raises(Exception, match="Need recording state to record boundaries"):
            collect_interfaces(time_step=jnp.array(0), arrays=arrays, objects=objects, config=config, key=key)
