"""Container module for managing collections of simulation objects and arrays.

This module provides container classes for organizing and managing simulation objects
and array data within FDTD simulations. It includes support for different object types
like sources, detectors, PML boundaries, and devices.
"""

from typing import Self

import jax
import pytreeclass as tc

from fdtdx.core.jax.pytrees import ExtendedTreeClass
from fdtdx.interfaces.state import RecordingState
from fdtdx.objects.boundaries.perfectly_matched_layer import BoundaryState, PerfectlyMatchedLayer
from fdtdx.objects.detectors.detector import Detector, DetectorState
from fdtdx.objects.multi_material.device import Device
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.sources.source import Source

# Type alias for parameter dictionaries containing JAX arrays
ParameterContainer = dict[str, dict[str, jax.Array]]


@tc.autoinit
class ObjectContainer(ExtendedTreeClass):
    """Container for managing simulation objects and their relationships.

    This class provides a structured way to organize and access different types of simulation
    objects like sources, detectors, PML boundaries and devices. It maintains object lists
    and provides filtered access to specific object types.

    Attributes:
        object_list: List of all simulation objects in the container.
        volume_idx: Index of the volume object in the object list.
    """

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
        return [o for o in self.objects if not isinstance(o, Device)]

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
        raise ValueError(f"Key {key} does not exist in object list: {[o.name for o in self.objects]}")

    def replace_sources(
        self,
        sources: list[Source],
    ) -> Self:
        new_objects = [o for o in self.objects if o not in self.sources] + sources
        self = self.aset("object_list", new_objects)
        return self


@tc.autoinit
class ArrayContainer(ExtendedTreeClass):
    """Container for simulation field arrays and states.

    This class holds the electromagnetic field arrays and various state information
    needed during FDTD simulation. It includes the E and H fields, material properties,
    and states for boundaries, detectors and recordings.

    Attributes:
        E: Electric field array.
        H: Magnetic field array.
        inv_permittivities: Inverse permittivity values array.
        inv_permeabilities: Inverse permeability values array.
        boundary_states: Dictionary mapping boundary names to their states.
        detector_states: Dictionary mapping detector names to their states.
        recording_state: Optional state for recording simulation data.
    """

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
    """Reset an ArrayContainer's fields and optionally its states.

    This function creates a new ArrayContainer with zeroed E and H fields while preserving
    material properties. It can optionally reset detector and recording states.

    Args:
        arrays: The ArrayContainer to reset.
        objects: ObjectContainer with simulation objects.
        reset_detector_states: Whether to zero detector states.
        reset_recording_state: Whether to zero recording state.

    Returns:
        A new ArrayContainer with reset fields and optionally reset states.
    """
    E = arrays.E * 0
    H = arrays.H * 0

    boundary_states = {}
    for pml in objects.pml_objects:
        boundary_states[pml.name] = pml.reset_state(state=arrays.boundary_states[pml.name])

    detector_states = arrays.detector_states
    if reset_detector_states:
        detector_states = {k: {k2: v2 * 0 for k2, v2 in v.items()} for k, v in detector_states.items()}

    recording_state = arrays.recording_state
    if reset_recording_state and arrays.recording_state is not None:
        recording_state = RecordingState(
            data={k: v * 0 for k, v in arrays.recording_state.data.items()},
            state={k: v * 0 for k, v in arrays.recording_state.state.items()},
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
