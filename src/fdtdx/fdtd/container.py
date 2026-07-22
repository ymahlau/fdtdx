"""Container module for managing collections of simulation objects and arrays.

This module provides container classes for organizing and managing simulation objects
and array data within FDTD simulations. It includes support for different object types
like sources, detectors, PML boundaries, Bloch/periodic boundaries, and devices.
"""

from typing import Callable, Iterator, Self, TypeVar

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field
from fdtdx.interfaces.state import RecordingState
from fdtdx.materials import Material
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.boundaries.boundary import BaseBoundary
from fdtdx.objects.boundaries.pec import PerfectElectricConductor
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.boundaries.pmc import PerfectMagneticConductor
from fdtdx.objects.detectors.detector import Detector, DetectorState
from fdtdx.objects.device.device import Device
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.sources.source import Source
from fdtdx.objects.static_material.static import StaticMultiMaterialObject, UniformMaterialObject

# Type alias for parameter dictionaries containing JAX arrays
ParameterContainer = dict[str, dict[str, jax.Array] | jax.Array]

_ObjT = TypeVar("_ObjT", bound=SimulationObject)


@autoinit
class ObjectContainer(TreeClass):
    """Container for managing simulation objects and their relationships.

    This class provides a structured way to organize and access different types of simulation
    objects like sources, detectors, PML/periodic boundaries and devices. It maintains object lists
    and provides filtered access to specific object types.
    """

    #: List of all simulation objects in the container.
    object_list: list[SimulationObject]

    #: Index of the volume object in the object list.
    volume_idx: int = frozen_field()

    def filter_objects(
        self,
        object_type: type[_ObjT] | tuple[type[_ObjT], ...],
        predicate: Callable[[_ObjT], bool] | None = None,
    ) -> list[_ObjT]:
        """Refactor filtering functions of object lists using proper filtering."""
        result = [o for o in self.object_list if isinstance(o, object_type)]
        if predicate is not None:
            result = [o for o in result if predicate(o)]
        return result

    @property
    def volume(self) -> SimulationObject:
        return self.object_list[self.volume_idx]

    @property
    def objects(self) -> list[SimulationObject]:
        return self.object_list

    @property
    def static_material_objects(self) -> list[UniformMaterialObject | StaticMultiMaterialObject]:
        return self.filter_objects((UniformMaterialObject, StaticMultiMaterialObject))

    @property
    def sources(self) -> list[Source]:
        return self.filter_objects(Source)

    @property
    def devices(self) -> list[Device]:
        return self.filter_objects(Device)

    @property
    def detectors(self) -> list[Detector]:
        return self.filter_objects(Detector)

    @property
    def forward_detectors(self) -> list[Detector]:
        return self.filter_objects(Detector, predicate=lambda o: not o.inverse)

    @property
    def backward_detectors(self) -> list[Detector]:
        return self.filter_objects(Detector, predicate=lambda o: o.inverse)

    @property
    def pml_objects(self) -> list[PerfectlyMatchedLayer]:
        return self.filter_objects(PerfectlyMatchedLayer)

    @property
    def periodic_objects(self) -> list[BlochBoundary]:
        return self.filter_objects(BlochBoundary, predicate=lambda o: not o.needs_complex_fields)

    @property
    def pec_objects(self) -> list[PerfectElectricConductor]:
        return self.filter_objects(PerfectElectricConductor)

    @property
    def pmc_objects(self) -> list[PerfectMagneticConductor]:
        return self.filter_objects(PerfectMagneticConductor)

    @property
    def bloch_objects(self) -> list[BlochBoundary]:
        return self.filter_objects(BlochBoundary)

    @property
    def boundary_objects(self) -> list[BaseBoundary]:
        return self.filter_objects(BaseBoundary)

    @property
    def any_object_subpixel_smoothing(self) -> bool:
        """True if any static multi-material object requests sub-pixel dielectric smoothing.

        When True the permittivity must be allocated as an anisotropic effective permittivity at
        interface cells even though every underlying material is isotropic. The default (diagonal)
        variant uses a 3-component allocation; a full 9-component tensor is only used when
        ``any_object_subpixel_full_tensor`` is also True.
        """
        return any(getattr(o, "subpixel_smoothing", False) for o in self.static_material_objects)

    @property
    def any_object_subpixel_full_tensor(self) -> bool:
        """True if any smoothed object requests the full 9-component tensor (vs the cheap 3-comp diagonal).

        When True the permittivity is allocated as a full 9-component tensor and the anisotropic update
        kernel is used; when False (all smoothed objects diagonal) a 3-component diagonal allocation runs on
        the cheaper elementwise update, which is exact for axis-aligned interfaces.
        """
        return any(
            getattr(o, "subpixel_smoothing", False) and getattr(o, "subpixel_full_tensor", False)
            for o in self.static_material_objects
        )

    @property
    def all_objects_non_magnetic(self) -> bool:
        def _fn(m: Material):
            return not m.is_magnetic

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_non_electrically_conductive(self) -> bool:
        def _fn(m: Material):
            return not m.is_electrically_conductive

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_non_magnetically_conductive(self) -> bool:
        def _fn(m: Material):
            return not m.is_magnetically_conductive

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_isotropic_permittivity(self) -> bool:
        def _fn(m: Material):
            return m.is_isotropic_permittivity

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_isotropic_permeability(self) -> bool:
        def _fn(m: Material):
            return m.is_isotropic_permeability

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_isotropic_electric_conductivity(self) -> bool:
        def _fn(m: Material):
            return m.is_isotropic_electric_conductivity

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_isotropic_magnetic_conductivity(self) -> bool:
        def _fn(m: Material):
            return m.is_isotropic_magnetic_conductivity

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_diagonally_anisotropic_permittivity(self) -> bool:
        def _fn(m: Material):
            return m.is_diagonally_anisotropic_permittivity

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_diagonally_anisotropic_permeability(self) -> bool:
        def _fn(m: Material):
            return m.is_diagonally_anisotropic_permeability

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_diagonally_anisotropic_electric_conductivity(self) -> bool:
        def _fn(m: Material):
            return m.is_diagonally_anisotropic_electric_conductivity

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_diagonally_anisotropic_magnetic_conductivity(self) -> bool:
        def _fn(m: Material):
            return m.is_diagonally_anisotropic_magnetic_conductivity

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_non_dispersive(self) -> bool:
        def _fn(m: Material):
            return not m.is_dispersive

        return self._is_material_fn_true_for_all(_fn)

    @staticmethod
    def _as_materials(m: "Material | dict[str, Material]") -> Iterator[Material]:
        if isinstance(m, Material):
            yield m
        elif isinstance(m, dict):
            yield from m.values()

    def _iter_materials(self) -> Iterator[Material]:
        for o in self.static_material_objects:  # UniformMaterialObject | StaticMultiMaterialObject
            if isinstance(o, UniformMaterialObject):
                yield o.material
            else:
                yield from self._as_materials(o.materials)
        for o in self.devices:
            yield from self._as_materials(o.materials)

    @property
    def all_objects_isotropic_dispersion(self) -> bool:
        """Whether every dispersive material applies the same poles to all three axes.

        Drives the size of the material-component axis of the dispersive
        recurrence coefficients ``c1``/``c2``: 1 (broadcast) when ``True``, 3
        (per-axis, diagonally anisotropic dispersion) when ``False``.
        """

        def _fn(m: Material):
            return m.has_isotropic_dispersion

        return self._is_material_fn_true_for_all(_fn)

    @property
    def all_objects_axis_aligned_dispersion(self) -> bool:
        """Whether no dispersive material carries oriented poles.

        When ``False``, at least one material has an off-diagonal coupling
        tensor: the field couplings ``c3``/``c4`` are widened to 9 components
        and the simulation runs through the fully anisotropic update path.
        """

        def _fn(m: Material):
            return m.has_axis_aligned_dispersion

        return self._is_material_fn_true_for_all(_fn)

    @property
    def max_num_dispersive_poles(self) -> int:
        """Maximum number of dispersive poles required across all objects.

        Walks every object (UniformMaterialObject, Device, StaticMultiMaterialObject)
        and returns the largest pole count of any Material attached to them.
        Drives the leading dimension of the per-cell dispersive coefficient and
        polarization arrays, which are zero-padded for materials with fewer
        poles.
        """
        return max(
            (m.dispersion.num_poles for m in self._iter_materials() if m.dispersion is not None),
            default=0,
        )

    @property
    def has_dispersive_edot(self) -> bool:
        """Whether any object uses a CCPR pole with a non-zero ``dE/dt`` coupling.

        This gates allocation of the ``dispersive_c4`` coefficient array: when
        ``False`` (all poles are Lorentz/Drude, or there is no dispersion) the
        ADE update takes the classic ``c4``-free path and stays bit-identical to
        pre-CCPR behaviour.
        """

        def _material_has_edot(m: Material) -> bool:
            if m.dispersion is None:
                return False
            return any(b != 0.0 for p in m.dispersion.poles for b in p.coupling_edot_axes)

        return any(_material_has_edot(m) for m in self._iter_materials())

    def _is_material_fn_true_for_all(
        self,
        fn: Callable[[Material], bool],
    ) -> bool:
        return all(fn(m) for m in self._iter_materials())

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

    def __contains__(
        self,
        key: str,
    ) -> bool:
        for o in self.objects:
            if o.name == key:
                return True
        return False

    def __setitem__(
        self,
        key: str,
        val: SimulationObject,
    ):
        idx = self.index(key)
        self.object_list[idx] = val

    def index(self, name: str) -> int:
        for idx, o in enumerate(self.object_list):
            if o.name == name:
                return idx
        raise ValueError(f"Object '{name}' does not exist in object list: {[o.name for o in self.objects]}")

    def copy(
        self,
    ) -> "ObjectContainer":
        new_list = self.object_list.copy()
        return ObjectContainer(
            object_list=new_list,
            volume_idx=self.volume_idx,
        )

    def replace_sources(
        self,
        sources: list[Source],
    ) -> Self:
        new_objects = [o for o in self.objects if o not in self.sources] + sources
        self = self.aset("object_list", new_objects)
        return self


PmlAuxField = dict[str, tuple[jax.Array, jax.Array]]


@autoinit
class FieldState(TreeClass):
    """Dynamic electromagnetic field state that evolves each time step.

    Grouping these together makes it impossible to forget a field when resetting
    simulation state — ArrayContainer.reset() zeroes this entire struct at once.
    """

    #: Electric field array.
    E: jax.Array

    #: Magnetic field array.
    H: jax.Array

    #: PML auxiliary electric field, stored as a dictionary mapping each PML
    #: object's name to a tuple of two arrays (each of shape ``pml.grid_shape``).
    psi_E: PmlAuxField

    #: PML auxiliary magnetic field, stored as a dictionary mapping each PML
    #: object's name to a tuple of two arrays (each of shape ``pml.grid_shape``).
    psi_H: PmlAuxField

    #: Dispersive ADE polarization state at time step ``n``. Shape
    #: ``(num_poles, 3, Nx, Ny, Nz)``. ``None`` for non-dispersive simulations.
    dispersive_P_curr: jax.Array | None = None

    #: Dispersive ADE polarization state at time step ``n-1``. Shape
    #: ``(num_poles, 3, Nx, Ny, Nz)``. ``None`` for non-dispersive simulations.
    dispersive_P_prev: jax.Array | None = None


@autoinit
class ArrayContainer(TreeClass):
    """Container for simulation field arrays and states.

    This class holds the electromagnetic field arrays and various state information
    needed during FDTD simulation. It includes the E and H fields, material properties,
    and states for boundaries, detectors and recordings.
    """

    #: Dynamic electromagnetic fields (E, H and PML auxiliaries).
    fields: FieldState

    #: Inverse permittivity values array.
    inv_permittivities: jax.Array

    #: Inverse permeability values array.
    inv_permeabilities: jax.Array | float

    #: Dictionary mapping detector names to their states.
    detector_states: dict[str, DetectorState]

    #: Optional state for recording simulation data.
    recording_state: RecordingState | None

    #: field for electric conductivity terms. Defaults to None.
    electric_conductivity: jax.Array | None = None

    #: field for magnetic conductivity terms. Defaults to None.
    magnetic_conductivity: jax.Array | None = None

    #: Per-cell dispersive recurrence coefficient c1. Shape
    #: ``(num_poles, num_components, Nx, Ny, Nz)`` with ``num_components`` 1
    #: (isotropic dispersion, broadcast over the field components) or 3
    #: (per-axis / diagonally anisotropic dispersion). ``None`` for
    #: non-dispersive simulations.
    dispersive_c1: jax.Array | None = None

    #: Per-cell dispersive recurrence coefficient c2. Shape
    #: ``(num_poles, num_components, Nx, Ny, Nz)``, ``num_components in (1, 3)``.
    #: ``None`` for non-dispersive simulations.
    dispersive_c2: jax.Array | None = None

    #: Per-cell dispersive field coupling c3. Shape
    #: ``(num_poles, num_components, Nx, Ny, Nz)``, ``num_components in (1, 3, 9)``
    #: — 9 encodes a row-major 3x3 coupling tensor per pole (oriented poles,
    #: off-diagonal dispersion). ``None`` for non-dispersive simulations.
    dispersive_c3: jax.Array | None = None

    #: Per-cell dispersive recurrence coefficient c4 (the ``dE/dt`` / CCPR
    #: coupling to ``E^{n+1}``). Shape ``(num_poles, num_components, Nx, Ny, Nz)``,
    #: ``num_components in (1, 3)``. ``None`` unless at least one CCPR pole with
    #: non-zero ``coupling_edot`` is present; Lorentz/Drude-only sims leave it
    #: ``None`` and skip the CCPR update path.
    dispersive_c4: jax.Array | None = None

    #: Per-cell cached ``1 / c2`` with non-dispersive cells set to 0. Lets the
    #: reverse-time ADE update avoid a ``jnp.where`` + division per step.
    #: Derived from ``dispersive_c2``; never differentiated independently.
    #: Shape ``(num_poles, num_components, Nx, Ny, Nz)``, ``num_components in
    #: (1, 3)``. ``None`` for non-dispersive simulations.
    dispersive_inv_c2: jax.Array | None = None

    #: Backup of inverse permittivity values array.
    #: Only used when etching a device.
    initial_inv_permittivities: jax.Array | None = None

    def reset(
        self,
        reset_detector_states: bool = True,
        reset_recording_state: bool = False,
    ) -> "ArrayContainer":
        """Return a reset copy of this array container.

        Dynamic field arrays are zeroed while material arrays and conductivity
        arrays are preserved. Detector states are reset by default because they
        accumulate time-dependent measurements. Recording state is preserved by
        default so partial simulations can continue writing to the same buffers.

        Args:
            reset_detector_states: Whether to zero all detector state arrays.
                Defaults to True.
            reset_recording_state: Whether to zero recording data and state
                arrays when a recording state is present. Defaults to False.

        Returns:
            A new ArrayContainer with reset dynamic state.
        """
        # FieldState now holds the dispersive ADE polarization (dispersive_P_curr/prev)
        # alongside E/H/psi, so this single tree.map zeroes all dynamic per-timestep
        # state at once (``None`` leaves stay ``None`` for non-dispersive sims).
        # Coefficient arrays (c1/c2/c3/inv_c2) are material properties and preserved.
        arrays = self.aset("fields", jax.tree.map(jnp.zeros_like, self.fields))

        detector_states = self.detector_states
        if reset_detector_states:
            detector_states = {k: {k2: v2 * 0 for k2, v2 in v.items()} for k, v in detector_states.items()}
        arrays = arrays.aset("detector_states", detector_states)

        recording_state = self.recording_state
        if reset_recording_state and self.recording_state is not None:
            recording_state = RecordingState(
                data={k: v * 0 for k, v in self.recording_state.data.items()},
                state={k: v * 0 for k, v in self.recording_state.state.items()},
            )
        arrays = arrays.aset("recording_state", recording_state)

        return arrays


# time step and arrays
SimulationState = tuple[jax.Array, ArrayContainer]
