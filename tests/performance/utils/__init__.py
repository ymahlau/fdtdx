"""Utilities for FDTDX performance benchmarking and scene building."""

from .env import capture_env
from .export import export_results
from .plotting import plot_field_intensity
from .results import BenchmarkResult, grid_shape_from_config
from .runner import compile_fn, read_memory_stats, run_compiled
from .scene import (
    add_gds_geometry,
    add_mode_detector,
    add_mode_source,
    add_phasor_monitors,
    build_domain,
    build_scene,
    extend_gds_with_port_stubs,
    port_to_sim_coords,
)

__all__ = [
    "BenchmarkResult",
    "add_gds_geometry",
    "add_mode_detector",
    "add_mode_source",
    "add_phasor_monitors",
    "build_domain",
    "build_scene",
    "capture_env",
    "compile_fn",
    "export_results",
    "extend_gds_with_port_stubs",
    "grid_shape_from_config",
    "plot_field_intensity",
    "port_to_sim_coords",
    "read_memory_stats",
    "run_compiled",
]
