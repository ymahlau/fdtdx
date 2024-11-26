from typing import Self
import jax
import pytreeclass as tc

from fdtdx.core.jax.pytrees import ExtendedTreeClass
from fdtdx.interfaces.state import RecordingState
from fdtdx.objects.boundaries.perfectly_matched_layer import BoundaryState, PerfectlyMatchedLayer
from fdtdx.objects.multi_material.device import Device
from fdtdx.objects.detectors.detector import Detector, DetectorState
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.sources.source import Source


ParameterContainer = dict[str, dict[str, jax.Array]]


@tc.autoinit
class ObjectContainer(ExtendedTreeClass):
    object_list: list[SimulationObject]
    volume_idx: int
    
    @property
    def volume(self) -> SimulationObject:
        return self.object_list[self.volume_idx]
    
    @property
    def objects(self) -> list[SimulationObject]:
        return self.object_list
    
    @property
    def static_material_objects(self) -> list[SimulationObject]:
        return [
            o for o in self.objects
            if not isinstance(o, Device)
        ]
    
    @property
    def sources(self) -> list[Source]:
        return [o for o in self.objects if isinstance(o, Source)]

    @property
    def devices(self) -> list[Device]:
        return [o for o in self.objects if isinstance(o, Device)]
    
    @property
    def detectors(self) -> list[Detector]:
        return [o for o in self.objects if isinstance(o, Detector)]
    
    @property
    def forward_detectors(self) -> list[Detector]:
        return [o for o in self.detectors if not o.inverse]
    
    @property
    def backward_detectors(self) -> list[Detector]:
        return [o for o in self.detectors if o.inverse]
    
    @property
    def pml_objects(self) -> list[PerfectlyMatchedLayer]:
        return [o for o in self.objects if isinstance(o, PerfectlyMatchedLayer)]
    
    def __iter__(self):
        return iter(self.object_list)
    
    def __getitem__(
        self,
        key: str,
    ) -> SimulationObject:
        for o in self.objects:
            if o.name == key:
                return o
        raise ValueError(
            f"Key {key} does not exist in object list: {[o.name for o in self.objects]}"
        )
    
    
    def replace_sources(
        self,
        sources: list[Source],
    ) -> Self:
        new_objects = [
            o for o in self.objects
            if not o in self.sources
        ] + sources
        self = self.aset("object_list", new_objects)
        return self
    


@tc.autoinit
class ArrayContainer(ExtendedTreeClass):
    E: jax.Array
    H: jax.Array
    inv_permittivities: jax.Array
    inv_permeabilities: jax.Array
    boundary_states: dict[str, BoundaryState]
    detector_states: dict[str, DetectorState]
    recording_state: RecordingState | None

# time step and arrays
SimulationState = tuple[jax.Array, ArrayContainer]


def reset_array_container(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    reset_detector_states: bool = False,
    reset_recording_state: bool = False,
) -> ArrayContainer:
    E = arrays.E * 0
    H = arrays.H * 0
    
    boundary_states = {}
    for pml in objects.pml_objects:
        boundary_states[pml.name] = pml.reset_state(
            state=arrays.boundary_states[pml.name]
        )
    
    detector_states = arrays.detector_states
    if reset_detector_states:
        detector_states = {
            k: {
                k2: v2 * 0
                for k2, v2 in v.items()
            }
            for k, v in detector_states.items()
        }
    
    recording_state = arrays.recording_state
    if reset_recording_state and arrays.recording_state is not None:
        recording_state = RecordingState(
            data={
                k: v * 0 
                for k, v in arrays.recording_state.data.items()
            },
            state={
                k: v * 0 
                for k, v in arrays.recording_state.state.items()
            }
        )
    
    return ArrayContainer(
        E=E,
        H=H,
        inv_permittivities=arrays.inv_permittivities,
        inv_permeabilities=arrays.inv_permeabilities,
        boundary_states=boundary_states,
        detector_states=detector_states,
        recording_state=recording_state,
    )
    