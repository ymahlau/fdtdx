"""SOI directional coupler benchmark — geometry from coupler.gds.

Matches the simulation setup from arXiv 2506.16665 (JPPhotonics/fdtd-pipeline):

  GDS file   : coupler.gds  (gdsfactory generic PDK, layer 1/0 = Si core)
  Domain     : 42 x 7 x 4 um^3  (x: coupler + 1 um margins each side, y, z: cladding)
  Layer stack: 220 nm Si core extruded from GDS polygons, SiO2 background
  Source     : Gaussian pulse injected into port o1 (WG1 input, x=-10 um GDS)
  Detectors  : ModeOverlapDetectors at WG1-output (o4) and WG2-output (o3)

Port positions (GDS coordinates):
  o1  (-10, -1.632) um  WG1 input   <- source here
  o2  (-10,  2.368) um  WG2 input
  o3  ( 30,  2.368) um  WG2 output  <- cross detector
  o4  ( 30, -1.632) um  WG1 output  <- thru detector

Two simulation configs (both non-uniform RectilinearGrid):
  cpl10   10 cells/lambda  — dx_si~44 nm, dx_sio2~107 nm, ~8.8 M cells  (fast check)
  cpl15   15 cells/lambda  — dx_si~30 nm, dx_sio2~ 71 nm, ~28 M cells   (paper-accurate)

Environment flags:
  PERF_VISUALIZE=1  Also save phasor field plots to <run_dir>/figs/
  PERF_TRACE=1      Capture Perfetto/XLA trace
  PERF_MEMORY=1     Record peak device memory
  PERF_FINE=1       Run the fine-grid config (skipped by default)
"""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.core.grid import RectilinearGrid
from fdtdx.fdtd.initialization import apply_params, place_objects
from fdtdx.fdtd.wrapper import run_fdtd
from fdtdx.materials import Material
from fdtdx.objects.static_material.gds_layer_stack import GDSLayerSpec

from ..utils import (
    BenchmarkResult,
    build_scene,
    plot_field_intensity,
    run_benchmark,
)

# ---------------------------------------------------------------------------
# Device layout
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CouplerConfig:
    """Physical and layout constants for coupler.gds (all lengths in metres)."""
    gds_path: Path
    cell_name: str
    gds_center: tuple[float, float]
    domain_x: float
    domain_y: float
    domain_z: float
    pml_cells: int
    wavelength: float
    si_thickness: float
    si_z_base: float
    ports: dict
    source_port: str
    detector_ports: list
    source_span_y: float

    @property
    def si_z_center(self) -> float:
        return self.si_z_base + self.si_thickness / 2

    @property
    def domain_shape(self) -> tuple[float, float, float]:
        return (self.domain_x, self.domain_y, self.domain_z)

    def layer_specs(self) -> list[GDSLayerSpec]:
        return [GDSLayerSpec(
            gds_layer=1, material_name="si",
            thickness=self.si_thickness, z_base=self.si_z_base, name="si_core",
        )]


_COUPLER = CouplerConfig(
    gds_path=Path(__file__).parent / "coupler.gds",
    cell_name="coupler",
    gds_center=(10e-6, 0.368e-6),
    domain_x=42e-6,
    domain_y=7e-6,
    domain_z=4e-6,
    pml_cells=12,
    wavelength=1550e-9,
    si_thickness=220e-9,
    si_z_base=4e-6 / 2 - 220e-9 / 2,
    ports={
        "o1": {"x_m": -10e-6, "y_m": -1.632e-6, "width_m": 0.5e-6, "orientation": 180.0},
        "o2": {"x_m": -10e-6, "y_m":  2.368e-6, "width_m": 0.5e-6, "orientation": 180.0},
        "o3": {"x_m":  30e-6, "y_m":  2.368e-6, "width_m": 0.5e-6, "orientation":   0.0},
        "o4": {"x_m":  30e-6, "y_m": -1.632e-6, "width_m": 0.5e-6, "orientation":   0.0},
    },
    source_port="o1",
    detector_ports=[("det_thru", "o4"), ("det_cross", "o3")],
    source_span_y=2e-6,
)


# ---------------------------------------------------------------------------
# Simulation configs — parametrized by cells per wavelength
#
# dx_fine   = wavelength / (n_Si   * cells_per_lambda)   [in waveguide / Si region]
# dx_coarse = wavelength / (n_SiO2 * cells_per_lambda)   [in SiO2 cladding]
#
#  cpl=10:  dx_fine ~44 nm, dx_coarse ~107 nm  ->  ~8.8 M cells  (fast sanity check)
#  cpl=15:  dx_fine ~30 nm, dx_coarse  ~71 nm  -> ~28.4 M cells  (paper-accurate)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SimConfig:
    """Non-uniform grid resolution parametrized by cells per wavelength."""
    cells_per_lambda: int

    @property
    def name(self) -> str:
        return f"cpl{self.cells_per_lambda}"

    @property
    def dx_fine(self) -> float:
        n_si = math.sqrt(fdtdx.constants.relative_permittivity_silicon)
        return _COUPLER.wavelength / (n_si * self.cells_per_lambda)

    @property
    def dx_coarse(self) -> float:
        n_sio2 = math.sqrt(fdtdx.constants.relative_permittivity_silica)
        return _COUPLER.wavelength / (n_sio2 * self.cells_per_lambda)

    def make_grid(self) -> RectilinearGrid:
        """Non-uniform RectilinearGrid with fine cells in waveguide / Si regions."""
        cfg = _COUPLER
        dx_f, dx_c = self.dx_fine, self.dx_coarse

        nx = round(cfg.domain_x / dx_f)
        x_edges = np.linspace(0.0, cfg.domain_x, nx + 1)

        y_fine_lo, y_fine_hi = 0.5e-6, cfg.domain_y - 0.5e-6
        y_edges = np.concatenate([
            np.linspace(0.0, y_fine_lo, round(y_fine_lo / dx_c) + 1),
            np.linspace(y_fine_lo, y_fine_hi, round((y_fine_hi - y_fine_lo) / dx_f) + 1)[1:],
            np.linspace(y_fine_hi, cfg.domain_y, round((cfg.domain_y - y_fine_hi) / dx_c) + 1)[1:],
        ])

        z_fine_lo, z_fine_hi = 1.0e-6, 3.0e-6
        z_edges = np.concatenate([
            np.linspace(0.0, z_fine_lo, round(z_fine_lo / dx_c) + 1),
            np.linspace(z_fine_lo, z_fine_hi, round((z_fine_hi - z_fine_lo) / dx_f) + 1)[1:],
            np.linspace(z_fine_hi, cfg.domain_z, round((cfg.domain_z - z_fine_hi) / dx_c) + 1)[1:],
        ])

        return RectilinearGrid(
            x_edges=jnp.asarray(x_edges),
            y_edges=jnp.asarray(y_edges),
            z_edges=jnp.asarray(z_edges),
        )


_SIM_COARSE = SimConfig(cells_per_lambda=10)   # dx_si~44 nm, dx_sio2~107 nm, ~8.8 M cells
_SIM_FINE   = SimConfig(cells_per_lambda=15)   # dx_si~30 nm, dx_sio2~ 71 nm, ~28 M cells


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

def _build_scene(sim_time: float, sim_cfg: SimConfig, with_phasor_detectors: bool = False) -> tuple:
    wave_char = fdtdx.WaveCharacter(wavelength=_COUPLER.wavelength)
    temporal_profile = fdtdx.GaussianPulseProfile(
        center_wave=wave_char,
        spectral_width=fdtdx.WaveCharacter(wavelength=_COUPLER.wavelength * 10),
    )
    return build_scene(
        gds_path=_COUPLER.gds_path,
        cell_name=_COUPLER.cell_name,
        layer_specs=_COUPLER.layer_specs(),
        materials={"si": Material(permittivity=fdtdx.constants.relative_permittivity_silicon)},
        ports=_COUPLER.ports,
        source_port=_COUPLER.source_port,
        detector_ports=_COUPLER.detector_ports,
        gds_center=_COUPLER.gds_center,
        domain_shape=_COUPLER.domain_shape,
        pml_cells=_COUPLER.pml_cells,
        background_material=Material(permittivity=fdtdx.constants.relative_permittivity_silica),
        wave_char=wave_char,
        temporal_profile=temporal_profile,
        grid=sim_cfg.make_grid(),
        sim_time=sim_time,
        source_span_y=_COUPLER.source_span_y,
        norm_det_dx=sim_cfg.dx_fine,
        with_phasor_monitors=with_phasor_detectors,
        phasor_z_height=_COUPLER.si_z_center,
    )


# ---------------------------------------------------------------------------
# Single benchmark test — tracing and visualisation gated by env flags
# ---------------------------------------------------------------------------

@pytest.mark.performance
@pytest.mark.parametrize("sim_cfg", [
    pytest.param(_SIM_COARSE, id="cpl10"),
    pytest.param(
        _SIM_FINE, id="cpl15",
        marks=pytest.mark.skipif(
            not os.environ.get("PERF_FINE"),
            reason="set PERF_FINE=1 to run the 15 cells/lambda grid (~28 M cells)",
        ),
    ),
])
def test_directional_coupler(sim_cfg: SimConfig, perf_env, perf_sink, perf_run_dir):
    """FDTD benchmark for the GDS-imported SOI directional coupler.

    Runs the simulation, records MCUPS, and optionally saves field plots
    (PERF_VISUALIZE=1), a Perfetto trace (PERF_TRACE=1), or peak memory
    stats (PERF_MEMORY=1).
    """
    with_viz = bool(os.environ.get("PERF_VISUALIZE"))
    dt = 0.99 / math.sqrt(3) * sim_cfg.dx_fine / 3e8
    # Run long enough for the Gaussian pulse to fully traverse the domain and exit:
    # 2 * (n_max * domain_x / c).  This ensures the phasor DFT converges.
    n_si = math.sqrt(fdtdx.constants.relative_permittivity_silicon)
    sim_time = 2.0 * n_si * _COUPLER.domain_x / 3e8
    n_steps = int(sim_time / dt)

    objects, constraints, config, _ = _build_scene(sim_time, sim_cfg, with_phasor_detectors=with_viz)

    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = place_objects(
        object_list=objects, config=config, constraints=constraints, key=key,
    )
    arrays = fdtdx.extend_material_to_pml(objects=obj_container, arrays=arrays)
    arrays, obj_container, _ = apply_params(arrays, obj_container, params, key)

    static_mem: int | None = None
    try:
        stats = jax.devices()[0].memory_stats()
        if stats:
            static_mem = stats.get("bytes_in_use")
    except Exception:
        pass

    _run_jit = jax.jit(run_fdtd, static_argnames=["show_progress"])

    def _run():
        return _run_jit(
            arrays=arrays, objects=obj_container, config=config,
            key=jax.random.PRNGKey(0), show_progress=False,
        )

    bench_name = f"si_coupler_gds_{sim_cfg.name}"
    compile_s, run_s, peak_mem, trace_path, mem_profile = run_benchmark(
        _run, name=bench_name, output_dir=perf_run_dir,
    )

    _, final_arrays = _run()
    jax.block_until_ready(final_arrays)

    # Print S-parameters: mode overlap is bi-orthogonal, so forward power is
    # extracted correctly even in the presence of standing waves in the field.
    states = jax.device_get(final_arrays.detector_states)
    norm = obj_container["det_source"].compute_overlap(states["det_source"])
    norm_power = float(jnp.abs(norm) ** 2)
    for det_name, _port_key in _COUPLER.detector_ports:
        det = obj_container[det_name]
        amp = det.compute_overlap(states[det_name])
        t = float(jnp.abs(amp) ** 2 / norm_power)
        print(f"  T({det_name}): {t:.4f}  ({t * 100:.1f}%)")

    grid = config.grid
    result = BenchmarkResult(
        name=bench_name,
        grid_shape=(len(grid.x_edges) - 1, len(grid.y_edges) - 1, len(grid.z_edges) - 1),
        time_steps=config.time_steps_total,
        compile_seconds=compile_s,
        run_seconds=run_s,
        env=perf_env,
        static_memory_bytes=static_mem,
        peak_memory_bytes=peak_mem,
        trace_path=trace_path,
        memory_profile_path=mem_profile,
    )
    perf_sink.append(result)
    print(f"\n  {result.summary()}")

    if with_viz:
        plot_field_intensity(
            detector_states=jax.device_get(final_arrays.detector_states),
            config=config,
            figs_dir=perf_run_dir / "figs",
            gds_center=_COUPLER.gds_center,
            domain_shape=_COUPLER.domain_shape,
            n_steps=n_steps,
            dx_nm=round(sim_cfg.dx_fine * 1e9),
            y_lim=(-3.0, 3.0),
        )

    assert result.mcups > 0
    assert math.isfinite(result.mcups)
