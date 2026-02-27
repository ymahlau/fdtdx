from unittest.mock import Mock

import jax.numpy as jnp

from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, SimulationState, reset_array_container
from fdtdx.interfaces.state import RecordingState
from fdtdx.materials import Material
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.boundaries.periodic import PeriodicBoundary
from fdtdx.objects.detectors.detector import Detector, DetectorState
from fdtdx.objects.device.device import Device
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.sources.source import Source
from fdtdx.objects.static_material.static import StaticMultiMaterialObject, UniformMaterialObject


class TestObjectContainer:
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock objects with proper name attributes
        self.mock_source = Mock(spec=Source)
        self.mock_source.name = "source1"

        self.mock_detector = Mock(spec=Detector)
        self.mock_detector.name = "detector1"
        self.mock_detector.inverse = False

        self.mock_inverse_detector = Mock(spec=Detector)
        self.mock_inverse_detector.name = "detector2"
        self.mock_inverse_detector.inverse = True

        self.mock_pml = Mock(spec=PerfectlyMatchedLayer)
        self.mock_pml.name = "pml1"

        self.mock_periodic = Mock(spec=PeriodicBoundary)
        self.mock_periodic.name = "periodic1"

        self.mock_device = Mock(spec=Device)
        self.mock_device.name = "device1"

        self.mock_uniform_material = Mock(spec=UniformMaterialObject)
        self.mock_uniform_material.name = "uniform_mat"

        self.mock_static_multi_material = Mock(spec=StaticMultiMaterialObject)
        self.mock_static_multi_material.name = "multi_mat"

        # Mock volume object
        self.mock_volume = Mock(spec=SimulationObject)
        self.mock_volume.name = "volume"

        # Create object list with volume at index 0
        self.object_list = [
            self.mock_volume,
            self.mock_source,
            self.mock_detector,
            self.mock_inverse_detector,
            self.mock_pml,
            self.mock_periodic,
            self.mock_device,
            self.mock_uniform_material,
            self.mock_static_multi_material,
        ]

        self.container = ObjectContainer(object_list=self.object_list, volume_idx=0)

    def test_volume_property(self):
        """Test volume property returns correct object."""
        assert self.container.volume == self.mock_volume

    def test_objects_property(self):
        """Test objects property returns all objects."""
        assert self.container.objects == self.object_list

    def test_sources_property(self):
        """Test sources property filters correctly."""
        sources = self.container.sources
        assert len(sources) == 1
        assert sources[0] == self.mock_source

    def test_detectors_property(self):
        """Test detectors property filters correctly."""
        detectors = self.container.detectors
        assert len(detectors) == 2
        assert self.mock_detector in detectors
        assert self.mock_inverse_detector in detectors

    def test_forward_detectors_property(self):
        """Test forward_detectors property filters correctly."""
        forward_detectors = self.container.forward_detectors
        assert len(forward_detectors) == 1
        assert forward_detectors[0] == self.mock_detector

    def test_backward_detectors_property(self):
        """Test backward_detectors property filters correctly."""
        backward_detectors = self.container.backward_detectors
        assert len(backward_detectors) == 1
        assert backward_detectors[0] == self.mock_inverse_detector

    def test_pml_objects_property(self):
        """Test pml_objects property filters correctly."""
        pml_objects = self.container.pml_objects
        assert len(pml_objects) == 1
        assert pml_objects[0] == self.mock_pml

    def test_periodic_objects_property(self):
        """Test periodic_objects property filters correctly."""
        periodic_objects = self.container.periodic_objects
        assert len(periodic_objects) == 1
        assert periodic_objects[0] == self.mock_periodic

    def test_boundary_objects_property(self):
        """Test boundary_objects property filters correctly."""
        boundary_objects = self.container.boundary_objects
        assert len(boundary_objects) == 2
        assert self.mock_pml in boundary_objects
        assert self.mock_periodic in boundary_objects

    def test_static_material_objects_property(self):
        """Test static_material_objects property filters correctly."""
        static_materials = self.container.static_material_objects
        assert len(static_materials) == 2
        assert self.mock_uniform_material in static_materials
        assert self.mock_static_multi_material in static_materials

    def test_devices_property(self):
        """Test devices property filters correctly."""
        devices = self.container.devices
        assert len(devices) == 1
        assert devices[0] == self.mock_device

    def test_all_objects_non_magnetic(self):
        """Test all_objects_non_magnetic property."""
        # Mock materials to be non-magnetic
        mock_material = Mock(spec=Material)
        mock_material.is_magnetic = False
        self.mock_uniform_material.material = mock_material
        self.mock_device.materials = {"mat1": mock_material}
        self.mock_static_multi_material.materials = {"mat1": mock_material}

        assert self.container.all_objects_non_magnetic is True

        # Test with magnetic material
        mock_magnetic_material = Mock(spec=Material)
        mock_magnetic_material.is_magnetic = True
        self.mock_uniform_material.material = mock_magnetic_material

        assert self.container.all_objects_non_magnetic is False

    def test_all_objects_non_electrically_conductive(self):
        """Test all_objects_non_electrically_conductive property."""
        # Mock materials to be non-conductive
        mock_material = Mock(spec=Material)
        mock_material.is_electrically_conductive = False
        self.mock_uniform_material.material = mock_material

        assert self.container.all_objects_non_electrically_conductive is True

        # Test with conductive material
        mock_conductive_material = Mock(spec=Material)
        mock_conductive_material.is_electrically_conductive = True
        self.mock_uniform_material.material = mock_conductive_material

        assert self.container.all_objects_non_electrically_conductive is False

    def test_all_objects_isotropic_permittivity(self):
        """Test all_objects_isotropic_permittivity property."""
        # Mock materials to be isotropic
        mock_material = Mock(spec=Material)
        mock_material.is_isotropic_permittivity = True
        self.mock_uniform_material.material = mock_material
        self.mock_device.materials = {"mat1": mock_material}
        self.mock_static_multi_material.materials = {"mat1": mock_material}

        assert self.container.all_objects_isotropic_permittivity is True

        # Test with anisotropic material
        mock_anisotropic_material = Mock(spec=Material)
        mock_anisotropic_material.is_isotropic_permittivity = False
        self.mock_uniform_material.material = mock_anisotropic_material

        assert self.container.all_objects_isotropic_permittivity is False

    def test_all_objects_isotropic_permeability(self):
        """Test all_objects_isotropic_permeability property."""
        # Mock materials to have isotropic permeability
        mock_material = Mock(spec=Material)
        mock_material.is_isotropic_permeability = True
        self.mock_uniform_material.material = mock_material
        self.mock_device.materials = {"mat1": mock_material}
        self.mock_static_multi_material.materials = {"mat1": mock_material}

        assert self.container.all_objects_isotropic_permeability is True

        # Test with anisotropic permeability
        mock_anisotropic_material = Mock(spec=Material)
        mock_anisotropic_material.is_isotropic_permeability = False
        self.mock_uniform_material.material = mock_anisotropic_material

        assert self.container.all_objects_isotropic_permeability is False

    def test_all_objects_isotropic_electric_conductivity(self):
        """Test all_objects_isotropic_electric_conductivity property."""
        # Mock materials to have isotropic electric conductivity
        mock_material = Mock(spec=Material)
        mock_material.is_isotropic_electric_conductivity = True
        self.mock_uniform_material.material = mock_material
        self.mock_device.materials = {"mat1": mock_material}
        self.mock_static_multi_material.materials = {"mat1": mock_material}

        assert self.container.all_objects_isotropic_electric_conductivity is True

        # Test with anisotropic electric conductivity
        mock_anisotropic_material = Mock(spec=Material)
        mock_anisotropic_material.is_isotropic_electric_conductivity = False
        self.mock_uniform_material.material = mock_anisotropic_material

        assert self.container.all_objects_isotropic_electric_conductivity is False

    def test_all_objects_isotropic_magnetic_conductivity(self):
        """Test all_objects_isotropic_magnetic_conductivity property."""
        # Mock materials to have isotropic magnetic conductivity
        mock_material = Mock(spec=Material)
        mock_material.is_isotropic_magnetic_conductivity = True
        self.mock_uniform_material.material = mock_material
        self.mock_device.materials = {"mat1": mock_material}
        self.mock_static_multi_material.materials = {"mat1": mock_material}

        assert self.container.all_objects_isotropic_magnetic_conductivity is True

        # Test with anisotropic magnetic conductivity
        mock_anisotropic_material = Mock(spec=Material)
        mock_anisotropic_material.is_isotropic_magnetic_conductivity = False
        self.mock_uniform_material.material = mock_anisotropic_material

        assert self.container.all_objects_isotropic_magnetic_conductivity is False

    def test_iteration(self):
        """Test container iteration."""
        objects = list(self.container)
        assert objects == self.object_list

    def test_getitem_by_name(self):
        """Test __getitem__ with object name."""
        obj = self.container["source1"]
        assert obj == self.mock_source

    def test_contains(self):
        """Test __contains__ method."""
        assert "source1" in self.container
        assert "nonexistent" not in self.container

    def test_setitem(self):
        """Test __setitem__ method."""
        new_source = Mock(spec=Source)
        new_source.name = "source1"
        self.container["source1"] = new_source
        assert self.container["source1"] == new_source

    def test_copy(self):
        """Test copy method."""
        copied = self.container.copy()
        assert copied.object_list == self.container.object_list
        assert copied.volume_idx == self.container.volume_idx
        # Verify it's a different list object
        assert copied.object_list is not self.container.object_list

    def test_replace_sources(self):
        """Test replace_sources method."""
        new_source = Mock(spec=Source)
        new_source.name = "new_source"
        new_sources = [new_source]

        new_container = self.container.replace_sources(new_sources)

        # Verify old sources are gone
        assert len(new_container.sources) == 1
        assert new_container.sources[0] == new_source
        # Verify other objects are preserved
        assert len(new_container.detectors) == 2


class TestArrayContainer:
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock arrays
        self.magnetic_conductivity = None
        self.electric_conductivity = None
        self.E = jnp.ones((3, 10, 10, 10))
        self.H = jnp.ones((3, 10, 10, 10))
        self.psi_E = jnp.zeros((6, 10, 10, 10))
        self.psi_H = jnp.zeros((6, 10, 10, 10))
        self.alpha = jnp.zeros((3, 10, 10, 10))
        self.kappa = jnp.ones((3, 10, 10, 10))
        self.sigma = jnp.zeros((3, 10, 10, 10))
        self.inv_permittivities = jnp.ones((3, 10, 10, 10))
        self.inv_permeabilities = jnp.ones((3, 10, 10, 10))

        # Create mock states
        self.detector_states = {"detector1": Mock(spec=DetectorState)}
        self.recording_state = Mock(spec=RecordingState)

        self.array_container = ArrayContainer(
            E=self.E,
            H=self.H,
            psi_E=self.psi_E,
            psi_H=self.psi_H,
            alpha=self.alpha,
            kappa=self.kappa,
            sigma=self.sigma,
            inv_permittivities=self.inv_permittivities,
            inv_permeabilities=self.inv_permeabilities,
            detector_states=self.detector_states,
            recording_state=self.recording_state,
        )

    def test_array_container_creation(self):
        """Test ArrayContainer creation with all fields."""
        assert jnp.array_equal(self.array_container.E, self.E)
        assert jnp.array_equal(self.array_container.H, self.H)
        assert jnp.array_equal(self.array_container.inv_permittivities, self.inv_permittivities)
        assert jnp.array_equal(self.array_container.inv_permeabilities, self.inv_permeabilities)
        assert self.array_container.detector_states == self.detector_states
        assert self.array_container.recording_state == self.recording_state

    def test_array_container_optional_fields(self):
        """Test ArrayContainer creation with optional fields."""
        electric_conductivity = jnp.ones((3, 10, 10, 10))
        magnetic_conductivity = jnp.ones((3, 10, 10, 10))

        container = ArrayContainer(
            E=self.E,
            H=self.H,
            psi_E=self.psi_E,
            psi_H=self.psi_H,
            alpha=self.alpha,
            kappa=self.kappa,
            sigma=self.sigma,
            inv_permittivities=self.inv_permittivities,
            inv_permeabilities=self.inv_permeabilities,
            detector_states=self.detector_states,
            recording_state=self.recording_state,
            electric_conductivity=electric_conductivity,
            magnetic_conductivity=magnetic_conductivity,
        )

        assert container.electric_conductivity is not None
        assert container.magnetic_conductivity is not None

        assert jnp.array_equal(container.electric_conductivity, electric_conductivity)
        assert jnp.array_equal(container.magnetic_conductivity, magnetic_conductivity)

    def test_array_container_tree_class_properties(self):
        """Test that ArrayContainer inherits TreeClass properties."""
        # TreeClass methods should be available on the class, not instance
        assert hasattr(ArrayContainer, "__tree_flatten__")
        assert hasattr(ArrayContainer, "tree_unflatten")
        # aset should be available on instance
        assert hasattr(self.array_container, "aset")


class TestResetArrayContainer:
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock arrays with non-zero values
        self.E = jnp.ones((3, 5, 5, 5))
        self.H = jnp.ones((3, 5, 5, 5))
        self.psi_E = jnp.zeros((6, 5, 5, 5))
        self.psi_H = jnp.zeros((6, 5, 5, 5))
        self.alpha = jnp.zeros((3, 5, 5, 5))
        self.kappa = jnp.ones((3, 5, 5, 5))
        self.sigma = jnp.zeros((3, 5, 5, 5))
        self.inv_permittivities = jnp.ones((3, 5, 5, 5))
        self.inv_permeabilities = jnp.ones((3, 5, 5, 5))

        # Create mock states
        self.detector_states = {"detector1": {"data": jnp.ones(10)}}
        self.recording_state = RecordingState(data={"recording": jnp.ones(10)}, state={"state": jnp.ones(5)})

        self.array_container = ArrayContainer(
            E=self.E,
            H=self.H,
            psi_E=self.psi_E,
            psi_H=self.psi_H,
            alpha=self.alpha,
            kappa=self.kappa,
            sigma=self.sigma,
            inv_permittivities=self.inv_permittivities,
            inv_permeabilities=self.inv_permeabilities,
            detector_states=self.detector_states,
            recording_state=self.recording_state,
        )

        # Create mock objects
        self.mock_boundary = Mock(spec=PerfectlyMatchedLayer)
        self.mock_boundary.name = "boundary1"

        self.objects = ObjectContainer(object_list=[self.mock_boundary], volume_idx=0)

    def test_reset_array_container_with_detector_reset(self):
        """Test reset_array_container with detector state reset."""
        result = reset_array_container(
            arrays=self.array_container, objects=self.objects, reset_detector_states=True, reset_recording_state=False
        )

        # Detector states should be zeroed
        assert jnp.all(result.detector_states["detector1"]["data"] == 0)

        # Recording state should NOT be reset
        assert result.recording_state == self.recording_state

    def test_reset_array_container_with_recording_reset(self):
        """Test reset_array_container with recording state reset."""
        result = reset_array_container(
            arrays=self.array_container, objects=self.objects, reset_detector_states=False, reset_recording_state=True
        )

        # Detector states should NOT be reset - compare individual elements
        assert jnp.array_equal(result.detector_states["detector1"]["data"], self.detector_states["detector1"]["data"])

        # Recording state should be reset (zeroed)
        assert result.recording_state is not None
        assert jnp.all(result.recording_state.data["recording"] == 0)
        assert jnp.all(result.recording_state.state["state"] == 0)

    def test_reset_array_container_no_recording_state(self):
        """Test reset_array_container with no recording state."""
        array_container_no_recording = self.array_container.aset("recording_state", None)

        result = reset_array_container(
            arrays=array_container_no_recording,
            objects=self.objects,
            reset_detector_states=True,
            reset_recording_state=True,
        )

        # Should handle None recording state gracefully
        assert result.recording_state is None


class TestSimulationState:
    def test_simulation_state_type_alias(self):
        """Test that SimulationState is properly defined as a type alias."""
        # Create a mock ArrayContainer
        array_container = Mock(spec=ArrayContainer)
        time_step = jnp.array(5)

        # Test that the tuple format works
        state: SimulationState = (time_step, array_container)

        assert isinstance(state, tuple)
        assert len(state) == 2
        assert state[0] == time_step
        assert state[1] == array_container
