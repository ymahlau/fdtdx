"""SOI directional coupler benchmark — geometry from coupler.gds.

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

"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.core.grid import RectilinearGrid
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.fdtd import custom_fdtd_forward
from fdtdx.fdtd.initialization import apply_params, place_objects
from fdtdx.materials import Material
from fdtdx.objects.static_material.gds_layer_stack import GDSLayerSpec

from ..utils import (
    BenchmarkResult,
    build_scene,
    compile_fn,
    plot_field_intensity,
    run_compiled,
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
        return [
            GDSLayerSpec(
                gds_layer=1,
                material_name="si",
                thickness=self.si_thickness,
                z_base=self.si_z_base,
                name="si_core",
            )
        ]


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
        "o2": {"x_m": -10e-6, "y_m": 2.368e-6, "width_m": 0.5e-6, "orientation": 180.0},
        "o3": {"x_m": 30e-6, "y_m": 2.368e-6, "width_m": 0.5e-6, "orientation": 0.0},
        "o4": {"x_m": 30e-6, "y_m": -1.632e-6, "width_m": 0.5e-6, "orientation": 0.0},
    },
    source_port="o1",
    detector_ports=[("det_thru", "o4"), ("det_cross", "o3")],
    source_span_y=2e-6,
)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def _dx(cells_per_lambda: int, n: float) -> float:
    return _COUPLER.wavelength / (n * cells_per_lambda)


def _make_grid(cells_per_lambda: int) -> RectilinearGrid:
    n_si = math.sqrt(fdtdx.constants.relative_permittivity_silicon)
    n_sio2 = math.sqrt(fdtdx.constants.relative_permittivity_silica)
    dx_f = _dx(cells_per_lambda, n_si)
    dx_c = _dx(cells_per_lambda, n_sio2)
    cfg = _COUPLER

    x_edges = np.linspace(0.0, cfg.domain_x, round(cfg.domain_x / dx_f) + 1)

    y_lo, y_hi = 0.5e-6, cfg.domain_y - 0.5e-6
    y_edges = np.concatenate(
        [
            np.linspace(0.0, y_lo, round(y_lo / dx_c) + 1),
            np.linspace(y_lo, y_hi, round((y_hi - y_lo) / dx_f) + 1)[1:],
            np.linspace(y_hi, cfg.domain_y, round((cfg.domain_y - y_hi) / dx_c) + 1)[1:],
        ]
    )

    z_lo, z_hi = 1.0e-6, 3.0e-6
    z_edges = np.concatenate(
        [
            np.linspace(0.0, z_lo, round(z_lo / dx_c) + 1),
            np.linspace(z_lo, z_hi, round((z_hi - z_lo) / dx_f) + 1)[1:],
            np.linspace(z_hi, cfg.domain_z, round((cfg.domain_z - z_hi) / dx_c) + 1)[1:],
        ]
    )

    return RectilinearGrid(
        x_edges=jnp.asarray(x_edges),
        y_edges=jnp.asarray(y_edges),
        z_edges=jnp.asarray(z_edges),
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _setup(
    cells_per_lambda: int,
    *,
    figs_dir: Path,
    with_phasor_detectors: bool = False,
) -> tuple[ObjectContainer, ArrayContainer, fdtdx.SimulationConfig]:
    n_si = math.sqrt(fdtdx.constants.relative_permittivity_silicon)
    dx_fine = _dx(cells_per_lambda, n_si)
    sim_time = 2.0 * n_si * _COUPLER.domain_x / 3e8

    wave_char = fdtdx.WaveCharacter(wavelength=_COUPLER.wavelength)
    temporal_profile = fdtdx.GaussianPulseProfile(
        center_wave=wave_char,
        spectral_width=fdtdx.WaveCharacter(wavelength=_COUPLER.wavelength * 10),
    )

    objects, constraints, config, _ = build_scene(
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
        grid=_make_grid(cells_per_lambda),
        sim_time=sim_time,
        source_span_y=_COUPLER.source_span_y,
        norm_det_dx=dx_fine,
        with_phasor_monitors=with_phasor_detectors,
        phasor_z_height=_COUPLER.si_z_center,
    )

    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays = fdtdx.extend_material_to_pml(objects=obj_container, arrays=arrays)
    arrays, obj_container, _ = apply_params(arrays, obj_container, params, key)

    figs_dir.mkdir(exist_ok=True)
    fdtdx.plot_setup(config=config, objects=obj_container, filename=figs_dir / f"setup_cpl{cells_per_lambda}.png")
    fdtdx.plot_material(config=config, arrays=arrays, filename=figs_dir / f"material_cpl{cells_per_lambda}.png")

    return obj_container, arrays, config


def _compile(
    objects: ObjectContainer,
    arrays: ArrayContainer,
    config: fdtdx.SimulationConfig,
    *,
    max_steps: int | None = None,
) -> tuple[any, float, dict]:
    end_time = config.time_steps_total
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        end_time = min(end_time, max_steps)

    fn_kwargs = dict(
        arrays=arrays,
        objects=objects,
        config=config,
        key=jax.random.PRNGKey(0),
        reset_container=True,
        record_detectors=True,
        start_time=0,
        end_time=end_time,
        show_progress=False,
    )
    return compile_fn(
        custom_fdtd_forward,
        fn_kwargs,
        static_argnames=("reset_container", "record_detectors", "start_time", "end_time", "show_progress"),
    )


def _run(
    compiled,
    dynamic_kwargs: dict,
    *,
    name: str,
    n_reps: int,
    do_trace: bool,
    do_memory: bool,
    output_dir: Path,
) -> tuple[list[float], int | None, str | None, str | None, any]:
    return run_compiled(
        compiled,
        dynamic_kwargs,
        name=name,
        n_reps=n_reps,
        do_trace=do_trace,
        do_memory=do_memory,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.parametrize(
    "cells_per_lambda",
    [
        pytest.param(10, id="coarse"),
        pytest.param(15, id="med", marks=pytest.mark.perf_med),
        pytest.param(20, id="fine", marks=pytest.mark.perf_fine),
    ],
)
def test_directional_coupler(cells_per_lambda, perf_env, perf_sink, perf_run_dir, perf_options):
    """FDTD benchmark for the GDS-imported SOI directional coupler."""
    bench_name = f"si_coupler_gds_cpl{cells_per_lambda}"
    figs_dir = perf_run_dir / "figs"

    objects, arrays, config = _setup(
        cells_per_lambda,
        figs_dir=figs_dir,
        with_phasor_detectors=perf_options.visualize,
    )

    static_mem: int | None = None
    try:
        stats = jax.devices()[0].memory_stats()
        if stats:
            static_mem = stats.get("bytes_in_use")
    except Exception:
        pass

    compiled, compile_s, dynamic_kwargs = _compile(
        objects,
        arrays,
        config,
        max_steps=perf_options.max_steps,
    )
    run_s, peak_mem, trace_path, mem_profile, final_state = _run(
        compiled,
        dynamic_kwargs,
        name=bench_name,
        n_reps=perf_options.reps,
        do_trace=perf_options.trace,
        do_memory=perf_options.memory,
        output_dir=perf_run_dir,
    )

    final_time_step, final_arrays = final_state
    measured_steps = int(jax.device_get(final_time_step))
    jax.block_until_ready(final_arrays)

    states = jax.device_get(final_arrays.detector_states)
    norm = objects["det_source"].compute_overlap(states["det_source"])
    norm_power = float(jnp.abs(norm).squeeze() ** 2)
    for det_name, _port_key in _COUPLER.detector_ports:
        amp = objects[det_name].compute_overlap(states[det_name])
        t = float((jnp.abs(amp).squeeze() ** 2) / norm_power)
        print(f"  T({det_name}): {t:.4f}  ({t * 100:.1f}%)")

    grid = config.grid
    result = BenchmarkResult(
        name=bench_name,
        grid_shape=(len(grid.x_edges) - 1, len(grid.y_edges) - 1, len(grid.z_edges) - 1),
        time_steps=measured_steps,
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
    if measured_steps != config.time_steps_total:
        print(f"  (max steps={config.time_steps_total}, measured prefix={measured_steps})")

    if perf_options.visualize:
        n_si = math.sqrt(fdtdx.constants.relative_permittivity_silicon)
        dx_fine = _dx(cells_per_lambda, n_si)
        dt = 0.99 / math.sqrt(3) * dx_fine / 3e8
        n_steps = int(2.0 * n_si * _COUPLER.domain_x / 3e8 / dt)
        plot_field_intensity(
            detector_states=jax.device_get(final_arrays.detector_states),
            config=config,
            figs_dir=figs_dir,
            gds_center=_COUPLER.gds_center,
            domain_shape=_COUPLER.domain_shape,
            n_steps=n_steps,
            dx_nm=round(dx_fine * 1e9),
            y_lim=(-3.0, 3.0),
        )

    assert result.mcups > 0
    assert math.isfinite(result.mcups)
