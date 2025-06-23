from fdtdx import constants
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.constants import wavelength_to_period
from fdtdx.conversion.export import export_stl
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
from fdtdx.core.plotting import colors
from fdtdx.core.switch import OnOffSwitch
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.fdtd.backward import full_backward
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, ParameterContainer, SimulationState
from fdtdx.fdtd.initialization import apply_params, place_objects
from fdtdx.fdtd.wrapper import run_fdtd
from fdtdx.interfaces.modules import DtypeConversion
from fdtdx.interfaces.recorder import Recorder, RecordingState
from fdtdx.interfaces.time_filter import LinearReconstructEveryK
from fdtdx.materials import Material
from fdtdx.objects.boundaries.initialization import BoundaryConfig, boundary_objects_from_config
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.boundaries.periodic import PeriodicBoundary
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
    BOTTOM_Z_PADDING_CONFIG_REPEAT,
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
from fdtdx.objects.device.parameters.symmetries import DiagonalSymmetry2D
from fdtdx.objects.device.parameters.transform import ParameterTransformation
from fdtdx.objects.object import (
    GridCoordinateConstraint,
    PositionConstraint,
    RealCoordinateConstraint,
    SimulationObject,
    SizeConstraint,
    SizeExtensionConstraint,
)
from fdtdx.objects.sources.linear_polarization import GaussianPlaneSource, UniformPlaneSource
from fdtdx.objects.sources.mode import ModePlaneSource
from fdtdx.objects.sources.profile import GaussianPulseProfile, SingleFrequencyProfile
from fdtdx.objects.static_material.cylinder import Cylinder
from fdtdx.objects.static_material.polygon import ExtrudedPolygon
from fdtdx.objects.static_material.sphere import Sphere
from fdtdx.objects.static_material.static import SimulationVolume, UniformMaterialObject
from fdtdx.utils.logger import Logger
from fdtdx.utils.plot_setup import plot_setup

__all__ = [
    # conversion
    "export_stl",
    # core
    "TreeClass",
    "autoinit",
    "field",
    "private_field",
    "frozen_field",
    "frozen_private_field",
    "metric_efficiency",
    "compute_energy",
    "normalize_by_energy",
    "compute_poynting_flux",
    "normalize_by_poynting_flux",
    "OnOffSwitch",
    "WaveCharacter",
    # fdtd
    "run_fdtd",
    "place_objects",
    "apply_params",
    "full_backward",
    "ArrayContainer",
    "ParameterContainer",
    "ObjectContainer",
    "SimulationState",
    # interfaces
    "Recorder",
    "RecordingState",
    "DtypeConversion",
    "LinearReconstructEveryK",
    # objects:
    "SimulationObject",
    "PositionConstraint",
    "SizeConstraint",
    "SizeExtensionConstraint",
    "GridCoordinateConstraint",
    "RealCoordinateConstraint",
    # boundaries
    "PerfectlyMatchedLayer",
    "PeriodicBoundary",
    "BoundaryConfig",
    "boundary_objects_from_config",
    # detector
    "EnergyDetector",
    "PoyntingFluxDetector",
    "FieldDetector",
    "PhasorDetector",
    "ModeOverlapDetector",
    # device
    "Device",
    "StandardToInversePermittivityRange",
    "StandardToCustomRange",
    "StandardToPlusOneMinusOneRange",
    "GaussianSmoothing2D",
    "RemoveFloatingMaterial",
    "ConnectHolesAndStructures",
    "BinaryMedianFilterModule",
    "ClosestIndex",
    "BrushConstraint2D",
    "PillarDiscretization",
    "TanhProjection",
    "SubpixelSmoothedProjection",
    "DiagonalSymmetry2D",
    "ParameterTransformation",
    "circular_brush",
    "BOTTOM_Z_PADDING_CONFIG_REPEAT",
    # sources
    "GaussianPlaneSource",
    "UniformPlaneSource",
    "ModePlaneSource",
    "SingleFrequencyProfile",
    "GaussianPulseProfile",
    # static material
    "Cylinder",
    "Sphere",
    "ExtrudedPolygon",
    "UniformMaterialObject",
    "SimulationVolume",
    # utils
    "Logger",
    "plot_setup",
    # config
    "SimulationConfig",
    "GradientConfig",
    # other
    "wavelength_to_period",
    "Material",
    "colors",
    "constants",
]
