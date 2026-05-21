import sys
import warnings

if sys.version_info >= (3, 14):
    warnings.warn(
        "Python 3.14+ is not supported by fdtdx. Expect crashes and unknown errors. "
        "Support for Python 3.14 will be added in the coming months.",
        UserWarning,
        stacklevel=2,
    )

from fdtdx import constants
from fdtdx.colors import Color
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.constants import wavelength_to_period
from fdtdx.conversion.json import export_json, export_json_str, import_from_json
from fdtdx.conversion.stl import export_stl
from fdtdx.conversion.vti import export_arrays_snapshot_to_vti, export_vti
from fdtdx.core.jax.pytrees import (
    TreeClass,
    autoinit,
    field,
    frozen_field,
    frozen_private_field,
    private_field,
)
from fdtdx.core.physics.losses import metric_efficiency
from fdtdx.core.physics.metrics import (
    compute_energy,
    compute_poynting_flux,
    normalize_by_energy,
    normalize_by_poynting_flux,
)
from fdtdx.core.physics.modes import compute_mode
from fdtdx.core.switch import OnOffSwitch
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.dispersion import (
    DispersionModel,
    DrudePole,
    LorentzPole,
    Pole,
    compute_eps_spectrum_from_coefficients,
    compute_impedance_corrected_temporal_profile,
    compute_pole_coefficients,
)
from fdtdx.fdtd.backward import full_backward
from fdtdx.fdtd.container import ArrayContainer, FieldState, ObjectContainer, ParameterContainer, SimulationState
from fdtdx.fdtd.initialization import apply_params, place_objects, resolve_object_constraints
from fdtdx.fdtd.wrapper import run_fdtd
from fdtdx.interfaces.modules import DtypeConversion
from fdtdx.interfaces.recorder import Recorder, RecordingState
from fdtdx.interfaces.time_filter import LinearReconstructEveryK
from fdtdx.materials import Material
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.boundaries.initialization import BoundaryConfig, boundary_objects_from_config
from fdtdx.objects.boundaries.pec import PerfectElectricConductor
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.boundaries.pmc import PerfectMagneticConductor
from fdtdx.objects.detectors.detector import Detector, DetectorState
from fdtdx.objects.detectors.energy import EnergyDetector
from fdtdx.objects.detectors.field import FieldDetector
from fdtdx.objects.detectors.mode import ModeOverlapDetector
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.detectors.poynting_flux import PoyntingFluxDetector
from fdtdx.objects.device.device import Device
from fdtdx.objects.device.parameters.continuous import (
    GaussianSmoothing2D,
    StandardToCustomRange,
    StandardToInversePermittivityRange,
    StandardToPlusOneMinusOneRange,
)
from fdtdx.objects.device.parameters.discrete import (
    BinaryMedianFilterModule,
    ConnectHolesAndStructures,
    RemoveFloatingMaterial,
)
from fdtdx.objects.device.parameters.discretization import (
    BrushConstraint2D,
    ClosestIndex,
    PillarDiscretization,
    circular_brush,
)
from fdtdx.objects.device.parameters.projection import SubpixelSmoothedProjection, TanhProjection
from fdtdx.objects.device.parameters.symmetries import (
    DiagonalSymmetry2D,
    DiagonalSymmetry3D,
    HorizontalSymmetry2D,
    HorizontalSymmetry3D,
    PointSymmetry2D,
    PointSymmetry3D,
    VerticalSymmetry2D,
    VerticalSymmetry3D,
)
from fdtdx.objects.device.parameters.transform import ParameterTransformation
from fdtdx.objects.object import (
    GridCoordinateConstraint,
    PositionConstraint,
    RealCoordinateConstraint,
    SimulationObject,
    SizeConstraint,
    SizeExtensionConstraint,
)
from fdtdx.objects.sources.dipole import PointDipoleSource
from fdtdx.objects.sources.linear_polarization import GaussianPlaneSource, UniformPlaneSource
from fdtdx.objects.sources.mode import ModePlaneSource
from fdtdx.objects.sources.profile import (
    CustomTimeSignalProfile,
    GaussianPulseProfile,
    SingleFrequencyProfile,
    TemporalProfile,
)
from fdtdx.objects.static_material.cylinder import Cylinder
from fdtdx.objects.static_material.polygon import (
    ExtrudedPolygon,
    extruded_polygon_from_gds,
    extruded_polygon_from_gds_path,
)
from fdtdx.objects.static_material.sphere import Sphere
from fdtdx.objects.static_material.static import SimulationVolume, UniformMaterialObject
from fdtdx.utils.extend_pml import extend_material_to_pml
from fdtdx.utils.logger import Logger
from fdtdx.utils.plot_field_slice import plot_field_slice, plot_field_slice_component
from fdtdx.utils.plot_material import plot_material, plot_material_from_side
from fdtdx.utils.plot_setup import plot_setup, plot_setup_from_side
from fdtdx.utils.sparams import PortSpec, calculate_sparam, calculate_sparams, setup_sparams_simulation

# PeriodicBoundary is now an alias for BlochBoundary with bloch_vector=(0,0,0)
PeriodicBoundary = BlochBoundary

__all__ = [
    "ArrayContainer",
    "BinaryMedianFilterModule",
    "BlochBoundary",
    "BoundaryConfig",
    "BrushConstraint2D",
    "ClosestIndex",
    "Color",
    "ConnectHolesAndStructures",
    "CustomTimeSignalProfile",
    "Cylinder",
    "Detector",
    "DetectorState",
    "Device",
    "DiagonalSymmetry2D",
    "DiagonalSymmetry3D",
    "DispersionModel",
    "DrudePole",
    "DtypeConversion",
    "EnergyDetector",
    "ExtrudedPolygon",
    "FieldDetector",
    "FieldState",
    "GaussianPlaneSource",
    "GaussianPulseProfile",
    "GaussianSmoothing2D",
    "GradientConfig",
    "GridCoordinateConstraint",
    "HorizontalSymmetry2D",
    "HorizontalSymmetry3D",
    "LinearReconstructEveryK",
    "Logger",
    "LorentzPole",
    "Material",
    "ModeOverlapDetector",
    "ModePlaneSource",
    "ObjectContainer",
    "OnOffSwitch",
    "ParameterContainer",
    "ParameterTransformation",
    "PerfectElectricConductor",
    "PerfectMagneticConductor",
    "PerfectlyMatchedLayer",
    "PeriodicBoundary",
    "PhasorDetector",
    "PillarDiscretization",
    "PointDipoleSource",
    "PointSymmetry2D",
    "PointSymmetry3D",
    "Pole",
    "PortSpec",
    "PositionConstraint",
    "PoyntingFluxDetector",
    "RealCoordinateConstraint",
    "Recorder",
    "RecordingState",
    "RemoveFloatingMaterial",
    "SimulationConfig",
    "SimulationObject",
    "SimulationState",
    "SimulationVolume",
    "SingleFrequencyProfile",
    "SizeConstraint",
    "SizeExtensionConstraint",
    "Sphere",
    "StandardToCustomRange",
    "StandardToInversePermittivityRange",
    "StandardToPlusOneMinusOneRange",
    "SubpixelSmoothedProjection",
    "TanhProjection",
    "TemporalProfile",
    "TreeClass",
    "UniformMaterialObject",
    "UniformPlaneSource",
    "VerticalSymmetry2D",
    "VerticalSymmetry3D",
    "WaveCharacter",
    "apply_params",
    "autoinit",
    "boundary_objects_from_config",
    "calculate_sparam",
    "calculate_sparams",
    "circular_brush",
    "compute_energy",
    "compute_eps_spectrum_from_coefficients",
    "compute_impedance_corrected_temporal_profile",
    "compute_mode",
    "compute_pole_coefficients",
    "compute_poynting_flux",
    "constants",
    "export_arrays_snapshot_to_vti",
    "export_json",
    "export_json_str",
    "export_stl",
    "export_vti",
    "extend_material_to_pml",
    "extruded_polygon_from_gds",
    "extruded_polygon_from_gds_path",
    "field",
    "frozen_field",
    "frozen_private_field",
    "full_backward",
    "import_from_json",
    "metric_efficiency",
    "normalize_by_energy",
    "normalize_by_poynting_flux",
    "place_objects",
    "plot_field_slice",
    "plot_field_slice_component",
    "plot_material",
    "plot_material_from_side",
    "plot_setup",
    "plot_setup_from_side",
    "private_field",
    "resolve_object_constraints",
    "run_fdtd",
    "setup_sparams_simulation",
    "wavelength_to_period",
]
