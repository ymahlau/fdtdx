"""Run a standalone SOI directional-coupler speed benchmark.

This script is intentionally not a pytest test. It reports timing and
environment metadata so changes can be compared by humans or dedicated
performance jobs without adding GPU-dependent pass/fail thresholds to CI.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

import fdtdx
from fdtdx.core.grid import RectilinearGrid
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.fdtd import custom_fdtd_forward
from fdtdx.fdtd.initialization import apply_params, place_objects
from fdtdx.materials import Material
from fdtdx.objects.static_material.gds_layer_stack import GDSLayerSpec, gds_layer_stack


@dataclass(frozen=True)
class CouplerConfig:
    gds_path: Path
    cell_name: str = "coupler"
    gds_center: tuple[float, float] = (10e-6, 0.368e-6)
    domain_x: float = 42e-6
    domain_y: float = 7e-6
    domain_z: float = 4e-6
    pml_cells: int = 12
    wavelength: float = 1550e-9
    si_thickness: float = 220e-9
    source_span_y: float = 2e-6

    @property
    def si_z_base(self) -> float:
        return self.domain_z / 2 - self.si_thickness / 2

    @property
    def domain_shape(self) -> tuple[float, float, float]:
        return (self.domain_x, self.domain_y, self.domain_z)

    @property
    def ports(self) -> dict[str, dict[str, float]]:
        return {
            "o1": {"x_m": -10e-6, "y_m": -1.632e-6, "width_m": 0.5e-6, "orientation": 180.0},
            "o2": {"x_m": -10e-6, "y_m": 2.368e-6, "width_m": 0.5e-6, "orientation": 180.0},
            "o3": {"x_m": 30e-6, "y_m": 2.368e-6, "width_m": 0.5e-6, "orientation": 0.0},
            "o4": {"x_m": 30e-6, "y_m": -1.632e-6, "width_m": 0.5e-6, "orientation": 0.0},
        }

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


def dx(cells_per_lambda: int, n: float, cfg: CouplerConfig) -> float:
    return cfg.wavelength / (n * cells_per_lambda)


def make_grid(cells_per_lambda: int, cfg: CouplerConfig) -> RectilinearGrid:
    n_si = math.sqrt(fdtdx.constants.relative_permittivity_silicon)
    n_sio2 = math.sqrt(fdtdx.constants.relative_permittivity_silica)
    dx_f = dx(cells_per_lambda, n_si, cfg)
    dx_c = dx(cells_per_lambda, n_sio2, cfg)

    x_edges = np.linspace(0.0, cfg.domain_x, round(cfg.domain_x / dx_f) + 1)
    y_edges = np.concatenate(
        [
            np.linspace(0.0, 0.5e-6, round(0.5e-6 / dx_c) + 1),
            np.linspace(0.5e-6, cfg.domain_y - 0.5e-6, round((cfg.domain_y - 1e-6) / dx_f) + 1)[1:],
            np.linspace(cfg.domain_y - 0.5e-6, cfg.domain_y, round(0.5e-6 / dx_c) + 1)[1:],
        ]
    )
    z_edges = np.concatenate(
        [
            np.linspace(0.0, 1.0e-6, round(1.0e-6 / dx_c) + 1),
            np.linspace(1.0e-6, 3.0e-6, round(2.0e-6 / dx_f) + 1)[1:],
            np.linspace(3.0e-6, cfg.domain_z, round(1.0e-6 / dx_c) + 1)[1:],
        ]
    )
    return RectilinearGrid(
        x_edges=jnp.asarray(x_edges),
        y_edges=jnp.asarray(y_edges),
        z_edges=jnp.asarray(z_edges),
    )


def port_to_sim_coords(port: dict[str, float], cfg: CouplerConfig) -> tuple[float, float]:
    return (
        port["x_m"] + cfg.domain_x / 2 - cfg.gds_center[0],
        port["y_m"] + cfg.domain_y / 2 - cfg.gds_center[1],
    )


def extend_gds_with_port_stubs(cfg: CouplerConfig) -> Path:
    import gdstk

    lib = gdstk.read_gds(str(cfg.gds_path))
    real_cells = {c.name: c for c in lib.cells if not c.name.startswith("$$$")}
    target = cast(Any, real_cells[cfg.cell_name])
    left_um = (cfg.gds_center[0] - cfg.domain_x / 2) * 1e6
    right_um = (cfg.gds_center[0] + cfg.domain_x / 2) * 1e6

    for spec in cfg.layer_specs():
        for port in cfg.ports.values():
            gds_x = port["x_m"] * 1e6
            gds_y = port["y_m"] * 1e6
            half_w = port["width_m"] * 1e6 / 2
            if abs(port["orientation"] - 180.0) < 1.0:
                rect = gdstk.rectangle((left_um, gds_y - half_w), (gds_x, gds_y + half_w), layer=spec.gds_layer)
            else:
                rect = gdstk.rectangle((gds_x, gds_y - half_w), (right_um, gds_y + half_w), layer=spec.gds_layer)
            target.add(rect)

    tmp = tempfile.NamedTemporaryFile(suffix=".gds", delete=False)
    lib.write_gds(tmp.name)
    return Path(tmp.name)


def place_at_port(obj: Any, name: str, port: dict[str, float], volume: Any, cfg: CouplerConfig) -> list[Any]:
    sim_x, sim_y = port_to_sim_coords(port, cfg)
    return [
        obj.same_size(volume, axes=(2,)),
        obj.place_at_center(volume, axes=(2,)),
        fdtdx.RealCoordinateConstraint(object=name, axes=(0,), sides=("-",), coordinates=(sim_x,)),
        fdtdx.RealCoordinateConstraint(
            object=name,
            axes=(1,),
            sides=("-",),
            coordinates=(sim_y - cfg.source_span_y / 2,),
        ),
    ]


def add_mode_detector(
    objects: list[Any],
    constraints: list[Any],
    *,
    name: str,
    port: dict[str, float],
    wave_char: fdtdx.WaveCharacter,
    volume: Any,
    cfg: CouplerConfig,
) -> None:
    det = fdtdx.ModeOverlapDetector(
        name=name,
        partial_grid_shape=(1, None, None),
        partial_real_shape=(None, cfg.source_span_y, None),
        wave_characters=(wave_char,),
        direction="+",
        mode_index=0,
        filter_pol="te",
        scaling_mode="pulse",
    )
    constraints.extend(place_at_port(det, name, port, volume, cfg))
    objects.append(det)


def setup_scene(
    cells_per_lambda: int, cfg: CouplerConfig
) -> tuple[ObjectContainer, ArrayContainer, fdtdx.SimulationConfig]:
    n_si = math.sqrt(fdtdx.constants.relative_permittivity_silicon)
    dx_fine = dx(cells_per_lambda, n_si, cfg)
    sim_time = 2.0 * n_si * cfg.domain_x / 3e8

    wave_char = fdtdx.WaveCharacter(wavelength=cfg.wavelength)
    temporal_profile = fdtdx.GaussianPulseProfile(
        center_wave=wave_char,
        spectral_width=fdtdx.WaveCharacter(wavelength=cfg.wavelength * 10),
    )
    config = fdtdx.SimulationConfig(grid=make_grid(cells_per_lambda, cfg), time=sim_time, gradient_config=None)

    objects: list[Any] = []
    constraints: list[Any] = []
    volume = fdtdx.SimulationVolume(partial_real_shape=cfg.domain_shape)
    objects.append(volume)

    boundaries, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=cfg.pml_cells),
        volume,
    )
    objects.extend(boundaries.values())
    constraints.extend(boundary_constraints)

    cladding = fdtdx.UniformMaterialObject(
        name="cladding",
        partial_real_shape=(None, None, None),
        material=Material(permittivity=fdtdx.constants.relative_permittivity_silica),
    )
    constraints.extend(cladding.same_position_and_size(volume))
    objects.append(cladding)

    import_gds = extend_gds_with_port_stubs(cfg)
    try:
        gds_objs, gds_constraints = gds_layer_stack(
            gds_source=import_gds,
            cell_name=cfg.cell_name,
            layers=cfg.layer_specs(),
            materials={"si": Material(permittivity=fdtdx.constants.relative_permittivity_silicon)},
            simulation_volume=volume,
            gds_center=cfg.gds_center,
        )
        objects.extend(gds_objs)
        constraints.extend(gds_constraints)
    finally:
        import_gds.unlink(missing_ok=True)

    source = fdtdx.ModePlaneSource(
        name="source",
        partial_grid_shape=(1, None, None),
        partial_real_shape=(None, cfg.source_span_y, None),
        wave_character=wave_char,
        temporal_profile=temporal_profile,
        direction="+",
        mode_index=0,
        filter_pol="te",
    )
    constraints.extend(place_at_port(source, "source", cfg.ports["o1"], volume, cfg))
    objects.append(source)

    norm_port = {**cfg.ports["o1"], "x_m": cfg.ports["o1"]["x_m"] + dx_fine}
    add_mode_detector(
        objects, constraints, name="det_source", port=norm_port, wave_char=wave_char, volume=volume, cfg=cfg
    )
    add_mode_detector(
        objects, constraints, name="det_thru", port=cfg.ports["o4"], wave_char=wave_char, volume=volume, cfg=cfg
    )
    add_mode_detector(
        objects, constraints, name="det_cross", port=cfg.ports["o3"], wave_char=wave_char, volume=volume, cfg=cfg
    )

    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = place_objects(objects, config, constraints, key)
    arrays = fdtdx.extend_material_to_pml(objects=obj_container, arrays=arrays)
    arrays, obj_container, _ = apply_params(arrays, obj_container, params, key)
    return obj_container, arrays, config


def compile_forward(
    objects: ObjectContainer,
    arrays: ArrayContainer,
    config: fdtdx.SimulationConfig,
    max_steps: int | None,
) -> tuple[Any, float, dict[str, Any]]:
    end_time = config.time_steps_total if max_steps is None else min(config.time_steps_total, max_steps)
    kwargs = {
        "arrays": arrays,
        "objects": objects,
        "config": config,
        "key": jax.random.PRNGKey(0),
        "reset_container": True,
        "record_detectors": True,
        "start_time": 0,
        "end_time": end_time,
        "show_progress": False,
    }
    static_argnames = ("reset_container", "record_detectors", "start_time", "end_time", "show_progress")
    t0 = time.perf_counter()
    compiled = jax.jit(custom_fdtd_forward, static_argnames=static_argnames).lower(**kwargs).compile()
    compile_seconds = time.perf_counter() - t0
    return compiled, compile_seconds, {k: v for k, v in kwargs.items() if k not in static_argnames}


def run_compiled(
    compiled: Any,
    kwargs: dict[str, Any],
    *,
    reps: int,
    trace: bool,
    memory: bool,
    run_dir: Path,
) -> tuple[list[float], int | None, str | None, str | None, Any]:
    run_seconds: list[float] = []
    trace_path: str | None = None
    final_state = None

    for i in range(reps):
        if trace and i == 0:
            trace_dir = run_dir / "traces" / "directional_coupler"
            trace_dir.mkdir(parents=True, exist_ok=True)
            with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
                t0 = time.perf_counter()
                final_state = compiled(**kwargs)
                jax.block_until_ready(final_state)
            trace_path = str(trace_dir)
        else:
            t0 = time.perf_counter()
            final_state = compiled(**kwargs)
            jax.block_until_ready(final_state)
        run_seconds.append(time.perf_counter() - t0)

    memory_profile_path: str | None = None
    peak_memory_bytes: int | None = None
    if memory:
        mem_dir = run_dir / "memory_profiles"
        mem_dir.mkdir(parents=True, exist_ok=True)
        memory_profile_path = str(mem_dir / "directional_coupler.pb")
        jax.profiler.save_device_memory_profile(memory_profile_path)
        stats = read_memory_stats()
        peak = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use")
        peak_memory_bytes = int(peak) if peak else None

    return run_seconds, peak_memory_bytes, trace_path, memory_profile_path, final_state


def grid_shape(config: fdtdx.SimulationConfig) -> tuple[int, int, int]:
    grid = cast(RectilinearGrid, config.grid)
    return (len(grid.x_edges) - 1, len(grid.y_edges) - 1, len(grid.z_edges) - 1)


def read_memory_stats() -> dict[str, int]:
    try:
        return jax.devices()[0].memory_stats() or {}
    except Exception:
        return {}


def capture_env() -> dict[str, Any]:
    devices = []
    for d in jax.devices():
        entry: dict[str, Any] = {"id": d.id, "kind": d.device_kind, "platform": d.platform}
        try:
            stats = d.memory_stats()
            if stats and stats.get("bytes_limit"):
                entry["memory_bytes"] = stats["bytes_limit"]
        except Exception:
            pass
        devices.append(entry)
    return {
        "git_sha": git_sha(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "python_version": platform.python_version(),
        "platform": platform.system().lower(),
        "cpu_model": cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "devices": devices,
        "jax_x64_enabled": jax.config.read("jax_enable_x64"),
    }


def git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[1],
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-per-lambda", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None, help="Limit the run to this many FDTD steps.")
    parser.add_argument("--reps", type=int, default=1, help="Number of timed executions after compilation.")
    parser.add_argument("--trace", action="store_true", help="Capture a Perfetto/XLA trace for the first rep.")
    parser.add_argument("--memory", action="store_true", help="Write a JAX device memory profile after the run.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reps <= 0:
        raise ValueError("--reps must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    cfg = CouplerConfig(gds_path=Path(__file__).with_name("coupler.gds"))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    device = jax.devices()[0].device_kind.lower().replace(" ", "_")
    run_dir = args.output_dir / f"{timestamp}_{device}"
    run_dir.mkdir(parents=True, exist_ok=True)

    objects, arrays, config = setup_scene(args.cells_per_lambda, cfg)
    static_memory_bytes = read_memory_stats().get("bytes_in_use")
    compiled, compile_seconds, kwargs = compile_forward(objects, arrays, config, args.max_steps)
    run_seconds, peak_memory_bytes, trace_path, memory_profile_path, final_state = run_compiled(
        compiled,
        kwargs,
        reps=args.reps,
        trace=args.trace,
        memory=args.memory,
        run_dir=run_dir,
    )

    final_time_step, final_arrays = final_state
    measured_steps = int(jax.device_get(final_time_step))
    shape = grid_shape(config)
    total_cell_updates = math.prod(shape) * measured_steps
    mcups = total_cell_updates / min(run_seconds) / 1e6

    states = jax.device_get(final_arrays.detector_states)
    norm = cast(Any, objects["det_source"]).compute_overlap(states["det_source"])
    norm_power = float(jnp.abs(norm).squeeze() ** 2)
    transmissions = {}
    for det_name in ("det_thru", "det_cross"):
        amp = cast(Any, objects[det_name]).compute_overlap(states[det_name])
        transmissions[det_name] = float((jnp.abs(amp).squeeze() ** 2) / norm_power)

    result: dict[str, Any] = {
        "name": "directional_coupler",
        "cells_per_lambda": args.cells_per_lambda,
        "grid_shape": list(shape),
        "time_steps": measured_steps,
        "configured_time_steps": config.time_steps_total,
        "total_cell_updates": total_cell_updates,
        "compile_seconds": compile_seconds,
        "run_seconds_best": min(run_seconds),
        "run_seconds_median": statistics.median(run_seconds),
        "run_seconds_worst": max(run_seconds),
        "mcups": mcups,
        "transmissions": transmissions,
        "static_memory_bytes": static_memory_bytes,
        "peak_memory_bytes": peak_memory_bytes,
        "trace_path": trace_path,
        "memory_profile_path": memory_profile_path,
    }
    payload = {"session_timestamp": timestamp, "env": capture_env(), "result": result}
    result_path = run_dir / "results.json"
    result_path.write_text(json.dumps(payload, indent=2))

    print(f"grid={shape} steps={measured_steps} compile={compile_seconds:.2f}s")
    print(f"run_best={min(run_seconds):.3f}s run_median={statistics.median(run_seconds):.3f}s MCUPS={mcups:.1f}")
    print(f"T(det_thru)={transmissions['det_thru']:.4f} T(det_cross)={transmissions['det_cross']:.4f}")
    if static_memory_bytes is not None:
        print(f"static_mem={static_memory_bytes / 1024**2:.0f}MB")
    if peak_memory_bytes is not None:
        print(f"peak_mem={peak_memory_bytes / 1024**2:.0f}MB")
    if trace_path:
        print(f"trace={trace_path}")
    if memory_profile_path:
        print(f"mem_profile={memory_profile_path}")
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
