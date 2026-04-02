from unittest.mock import Mock

import jax.numpy as jnp
import pytest

from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, reset_array_container
from fdtdx.interfaces.state import RecordingState
from fdtdx.materials import Material
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.detectors.detector import Detector, DetectorState
from fdtdx.objects.device.device import Device
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.sources.source import Source
from fdtdx.objects.static_material.static import StaticMultiMaterialObject, UniformMaterialObject


class TestObjectContainer:
    def setup_method(self):
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

        self.mock_periodic = Mock(spec=BlochBoundary)
        self.mock_periodic.name = "periodic1"
        self.mock_periodic.needs_complex_fields = False

        self.mock_device = Mock(spec=Device)
        self.mock_device.name = "device1"

        self.mock_uniform_material = Mock(spec=UniformMaterialObject)
        self.mock_uniform_material.name = "uniform_mat"

        self.mock_static_multi_material = Mock(spec=StaticMultiMaterialObject)
        self.mock_static_multi_material.name = "multi_mat"

        self.mock_volume = Mock(spec=SimulationObject)
        self.mock_volume.name = "volume"

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
        assert self.container.volume == self.mock_volume

    def test_objects_property(self):
        assert self.container.objects == self.object_list

    def test_sources_property(self):
        sources = self.container.sources
        assert len(sources) == 1
        assert sources[0] == self.mock_source

    def test_detectors_property(self):
        detectors = self.container.detectors
        assert len(detectors) == 2
        assert self.mock_detector in detectors
        assert self.mock_inverse_detector in detectors

    def test_forward_detectors_property(self):
        forward_detectors = self.container.forward_detectors
        assert len(forward_detectors) == 1
        assert forward_detectors[0] == self.mock_detector

    def test_backward_detectors_property(self):
        backward_detectors = self.container.backward_detectors
        assert len(backward_detectors) == 1
        assert backward_detectors[0] == self.mock_inverse_detector

    def test_pml_objects_property(self):
        pml_objects = self.container.pml_objects
        assert len(pml_objects) == 1
        assert pml_objects[0] == self.mock_pml

    def test_periodic_objects_property(self):
        periodic_objects = self.container.periodic_objects
        assert len(periodic_objects) == 1
        assert periodic_objects[0] == self.mock_periodic

    def test_boundary_objects_property(self):
        boundary_objects = self.container.boundary_objects
        assert len(boundary_objects) == 2
        assert self.mock_pml in boundary_objects
        assert self.mock_periodic in boundary_objects

    def test_static_material_objects_property(self):
        static_materials = self.container.static_material_objects
        assert len(static_materials) == 2
        assert self.mock_uniform_material in static_materials
        assert self.mock_static_multi_material in static_materials

    def test_devices_property(self):
        devices = self.container.devices
        assert len(devices) == 1
        assert devices[0] == self.mock_device

    def _setup_all_materials(self, **attrs):
        """Helper to set material attributes on all material-bearing objects."""
        mock_material = Mock(spec=Material)
        for attr, val in attrs.items():
            setattr(mock_material, attr, val)
        self.mock_uniform_material.material = mock_material
        self.mock_device.materials = {"mat1": mock_material}
        self.mock_static_multi_material.materials = {"mat1": mock_material}
        return mock_material

    @pytest.mark.parametrize(
        "property_name,attr_name,true_val",
        [
            ("all_objects_non_magnetic", "is_magnetic", False),
            ("all_objects_non_electrically_conductive", "is_electrically_conductive", False),
            ("all_objects_non_magnetically_conductive", "is_magnetically_conductive", False),
            ("all_objects_isotropic_permittivity", "is_isotropic_permittivity", True),
            ("all_objects_isotropic_permeability", "is_isotropic_permeability", True),
            ("all_objects_isotropic_electric_conductivity", "is_isotropic_electric_conductivity", True),
            ("all_objects_isotropic_magnetic_conductivity", "is_isotropic_magnetic_conductivity", True),
            ("all_objects_diagonally_anisotropic_permittivity", "is_diagonally_anisotropic_permittivity", True),
            ("all_objects_diagonally_anisotropic_permeability", "is_diagonally_anisotropic_permeability", True),
            (
                "all_objects_diagonally_anisotropic_electric_conductivity",
                "is_diagonally_anisotropic_electric_conductivity",
                True,
            ),
            (
                "all_objects_diagonally_anisotropic_magnetic_conductivity",
                "is_diagonally_anisotropic_magnetic_conductivity",
                True,
            ),
        ],
    )
    def test_material_property_all_true(self, property_name, attr_name, true_val):
        """Test each material property returns True when all materials satisfy it."""
        self._setup_all_materials(**{attr_name: true_val})
        assert getattr(self.container, property_name) is True

    def test_material_property_false_on_uniform(self):
        """Test material property returns False when UniformMaterialObject fails."""
        self._setup_all_materials(is_magnetic=False)
        bad = Mock(spec=Material)
        bad.is_magnetic = True
        self.mock_uniform_material.material = bad
        assert self.container.all_objects_non_magnetic is False

    def test_material_fn_device_with_single_material(self):
        """Test _is_material_fn_true_for_all when Device.materials is a single Material."""
        mock_material = Mock(spec=Material)
        mock_material.is_magnetic = False
        self.mock_uniform_material.material = mock_material
        self.mock_device.materials = mock_material
        self.mock_static_multi_material.materials = mock_material
        assert self.container.all_objects_non_magnetic is True

    def test_material_fn_dict_value_fails(self):
        """Test _is_material_fn_true_for_all returns False when one dict value fails."""
        good = Mock(spec=Material)
        good.is_magnetic = False
        bad = Mock(spec=Material)
        bad.is_magnetic = True
        self.mock_uniform_material.material = good
        self.mock_device.materials = {"good": good, "bad": bad}
        self.mock_static_multi_material.materials = {"m": good}
        assert self.container.all_objects_non_magnetic is False

    def test_material_fn_no_material_objects(self):
        """Test _is_material_fn_true_for_all with only non-material objects."""
        container = ObjectContainer(
            object_list=[self.mock_volume, self.mock_source, self.mock_pml],
            volume_idx=0,
        )
        assert container.all_objects_non_magnetic is True

    def test_iteration(self):
        objects = list(self.container)
        assert objects == self.object_list

    def test_getitem_by_name(self):
        obj = self.container["source1"]
        assert obj == self.mock_source

    def test_getitem_nonexistent_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            self.container["nonexistent"]

    def test_contains(self):
        assert "source1" in self.container
        assert "nonexistent" not in self.container

    def test_setitem(self):
        new_source = Mock(spec=Source)
        new_source.name = "source1"
        self.container["source1"] = new_source
        assert self.container["source1"] == new_source

    def test_setitem_nonexistent_raises(self):
        """Test __setitem__ with nonexistent key raises ValueError."""
        new_obj = Mock(spec=SimulationObject)
        new_obj.name = "nonexistent"
        with pytest.raises(ValueError, match="nonexistent"):
            self.container["nonexistent"] = new_obj

    def test_copy(self):
        copied = self.container.copy()
        assert copied.object_list == self.container.object_list
        assert copied.volume_idx == self.container.volume_idx
        assert copied.object_list is not self.container.object_list

    def test_replace_sources(self):
        new_source = Mock(spec=Source)
        new_source.name = "new_source"

        new_container = self.container.replace_sources([new_source])

        assert len(new_container.sources) == 1
        assert new_container.sources[0] == new_source
        assert len(new_container.detectors) == 2


class TestArrayContainer:
    def setup_method(self):
        self.E = jnp.ones((3, 10, 10, 10))
        self.H = jnp.ones((3, 10, 10, 10))
        self.psi_E = jnp.zeros((6, 10, 10, 10))
        self.psi_H = jnp.zeros((6, 10, 10, 10))
        self.alpha = jnp.zeros((3, 10, 10, 10))
        self.kappa = jnp.ones((3, 10, 10, 10))
        self.sigma = jnp.zeros((3, 10, 10, 10))
        self.inv_permittivities = jnp.ones((3, 10, 10, 10))
        self.inv_permeabilities = jnp.ones((3, 10, 10, 10))

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

    def test_creation(self):
        assert jnp.array_equal(self.array_container.E, self.E)
        assert jnp.array_equal(self.array_container.H, self.H)
        assert jnp.array_equal(self.array_container.inv_permittivities, self.inv_permittivities)
        assert jnp.array_equal(self.array_container.inv_permeabilities, self.inv_permeabilities)
        assert self.array_container.detector_states == self.detector_states
        assert self.array_container.recording_state == self.recording_state

    def test_optional_conductivity_fields(self):
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

        assert jnp.array_equal(container.electric_conductivity, electric_conductivity)
        assert jnp.array_equal(container.magnetic_conductivity, magnetic_conductivity)

    def test_default_conductivity_is_none(self):
        assert self.array_container.electric_conductivity is None
        assert self.array_container.magnetic_conductivity is None

    def test_inv_permeabilities_float(self):
        """Test that inv_permeabilities can be a float."""
        container = ArrayContainer(
            E=self.E,
            H=self.H,
            psi_E=self.psi_E,
            psi_H=self.psi_H,
            alpha=self.alpha,
            kappa=self.kappa,
            sigma=self.sigma,
            inv_permittivities=self.inv_permittivities,
            inv_permeabilities=1.0,
            detector_states=self.detector_states,
            recording_state=self.recording_state,
        )
        assert container.inv_permeabilities == 1.0


class TestResetArrayContainer:
    def setup_method(self):
        self.E = jnp.ones((3, 5, 5, 5))
        self.H = jnp.ones((3, 5, 5, 5))
        self.psi_E = jnp.zeros((6, 5, 5, 5))
        self.psi_H = jnp.zeros((6, 5, 5, 5))
        self.alpha = jnp.zeros((3, 5, 5, 5))
        self.kappa = jnp.ones((3, 5, 5, 5))
        self.sigma = jnp.zeros((3, 5, 5, 5))
        self.inv_permittivities = jnp.ones((3, 5, 5, 5))
        self.inv_permeabilities = jnp.ones((3, 5, 5, 5))

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

        self.mock_boundary = Mock(spec=PerfectlyMatchedLayer)
        self.mock_boundary.name = "boundary1"

        self.objects = ObjectContainer(object_list=[self.mock_boundary], volume_idx=0)

    def test_e_and_h_fields_zeroed(self):
        result = reset_array_container(arrays=self.array_container, objects=self.objects)
        assert jnp.all(result.E == 0)
        assert jnp.all(result.H == 0)

    def test_material_properties_preserved(self):
        result = reset_array_container(arrays=self.array_container, objects=self.objects)
        assert jnp.array_equal(result.inv_permittivities, self.inv_permittivities)
        assert jnp.array_equal(result.inv_permeabilities, self.inv_permeabilities)

    def test_detector_states_reset_by_default(self):
        result = reset_array_container(arrays=self.array_container, objects=self.objects)
        assert jnp.all(result.detector_states["detector1"]["data"] == 0)

    def test_detector_states_preserved_when_not_reset(self):
        result = reset_array_container(arrays=self.array_container, objects=self.objects, reset_detector_states=False)
        assert jnp.array_equal(
            result.detector_states["detector1"]["data"],
            self.detector_states["detector1"]["data"],
        )

    def test_recording_state_not_reset_by_default(self):
        result = reset_array_container(arrays=self.array_container, objects=self.objects)
        assert result.recording_state == self.recording_state

    def test_recording_state_reset(self):
        result = reset_array_container(arrays=self.array_container, objects=self.objects, reset_recording_state=True)
        assert jnp.all(result.recording_state.data["recording"] == 0)
        assert jnp.all(result.recording_state.state["state"] == 0)

    def test_recording_state_none_handled(self):
        array_container_no_recording = self.array_container.aset("recording_state", None)
        result = reset_array_container(
            arrays=array_container_no_recording,
            objects=self.objects,
            reset_recording_state=True,
        )
        assert result.recording_state is None
