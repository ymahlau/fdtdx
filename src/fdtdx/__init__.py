from fdtdx.conversion.export import export_stl
from fdtdx.core.physics.losses import metric_efficiency
from fdtdx.core.physics.metrics import compute_energy, normalize_by_energy, poynting_flux, normalize_by_poynting_flux
from fdtdx.core.switch import OnOffSwitch
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.fdtd.wrapper import run_fdtd
from fdtdx.fdtd.initialization import place_objects, apply_params
from fdtdx.fdtd.backward import full_backward
from fdtdx.fdtd.container import ArrayContainer, ParameterContainer
from fdtdx.interfaces.recorder import Recorder, RecordingState
from fdtdx.interfaces.modules import DtypeConversion
from fdtdx.interfaces.time_filter import LinearReconstructEveryK
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.boundaries.periodic import PeriodicBoundary
from fdtdx.objects.boundaries.initialization import BoundaryConfig, boundary_objects_from_config 
from fdtdx.objects.detectors.energy import EnergyDetector
from fdtdx.objects.detectors.poynting_flux import PoyntingFluxDetector
from fdtdx.objects.detectors.field import FieldDetector
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.detectors.mode import ModeOverlapDetector
from fdtdx.objects.device.device import Device
from fdtdx.objects.device.parameters.continuous import (
    StandardToInversePermittivityRange,
    StandardToCustomRange,
    StandardToPlusOneMinusOneRange,
    GaussianSmoothing2D,
)
from fdtdx.objects.device.parameters.discrete import (
    RemoveFloatingMaterial, 
    ConnectHolesAndStructures, 
    BinaryMedianFilterModule,
)
from fdtdx.objects.device.parameters.discretization import ClosestIndex, BrushConstraint2D, PillarDiscretization
from fdtdx.objects.device.parameters.projection import TanhProjection, SubpixelSmoothedProjection
from fdtdx.objects.device.parameters.symmetries import DiagonalSymmetry2D
from fdtdx.objects.device.parameters.transform import ParameterTransformation
from fdtdx.objects.sources.linear_polarization import GaussianPlaneSource, UniformPlaneSource
from fdtdx.objects.sources.mode import ModePlaneSource
from fdtdx.objects.sources.profile import SingleFrequencyProfile, GaussianPulseProfile
from fdtdx.objects.static_material.cylinder import Cylinder
from fdtdx.objects.static_material.sphere import Sphere
from fdtdx.objects.static_material.static import UniformMaterialObject, SimulationVolume
from fdtdx.objects.object import SimulationObject
from fdtdx.utils.logger import Logger
from fdtdx.utils.plot_setup import plot_setup
from fdtdx.config import SimulationConfig, GradientConfig
from fdtdx.constants import wavelength_to_period
from fdtdx.materials import Material
from fdtdx.core.plotting import colors

__all__ = [
    # conversion
    'export_stl',
    # core
    'metric_efficiency',
    'compute_energy',
    'normalize_by_energy',
    'poynting_flux',
    'normalize_by_poynting_flux',
    'OnOffSwitch',
    'WaveCharacter',
    # fdtd
    'run_fdtd',
    'place_objects',
    'apply_params',
    'full_backward',
    'ArrayContainer',
    'ParameterContainer',
    # interfaces
    'Recorder',
    'RecordingState',
    'DtypeConversion',
    'LinearReconstructEveryK',
    # objects:
    'SimulationObject',
    # boundaries
    'PerfectlyMatchedLayer',
    'PeriodicBoundary',
    'BoundaryConfig',
    'boundary_objects_from_config',
    # detector
    'EnergyDetector',
    'PoyntingFluxDetector',
    'FieldDetector',
    'PhasorDetector',
    'ModeOverlapDetector',
    # device
    'Device',
    'StandardToInversePermittivityRange',
    'StandardToCustomRange',
    'StandardToPlusOneMinusOneRange',
    'GaussianSmoothing2D',
    'RemoveFloatingMaterial', 
    'ConnectHolesAndStructures', 
    'BinaryMedianFilterModule',
    'ClosestIndex', 
    'BrushConstraint2D', 
    'PillarDiscretization',
    'TanhProjection',
    'SubpixelSmoothedProjection',
    'DiagonalSymmetry2D',
    'ParameterTransformation',
    # sources
    'GaussianPlaneSource',
    'UniformPlaneSource',
    'ModePlaneSource',
    'SingleFrequencyProfile',
    'GaussianPulseProfile',
    # static material
    'Cylinder',
    'Sphere',
    'UniformMaterialObject',
    'SimulationVolume',
    # utils
    'Logger',
    'plot_setup',
    # config
    'SimulationConfig',
    'GradientConfig',
    # other
    'wavelength_to_period',
    'Material',
    'colors',
]

