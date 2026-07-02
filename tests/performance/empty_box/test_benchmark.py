"""Forward throughput benchmark: uniform air box with PML on all six faces.

This is the simplest possible FDTD scene — no material interfaces, one Gaussian-pulse
dipole source — and produces the peak-throughput (ceiling) MCUPS number.  It serves
as the baseline against which more realistic photonic benchmarks are compared.

The simulation runs until the pulse is fully absorbed by the PML (energy drops below
threshold=1e-6), with a hard cap at 10 diagonal transit times per domain.

Grid sizes
----------
- small:  32^3  interior cells  -> (32 + 2x8)^3  = 48^3   ~110 K cells
- medium: 64^3  interior cells  -> (64 + 2x8)^3  = 80^3   ~512 K cells
- large: 128^3  interior cells  -> (128 + 2x8)^3 = 144^3  ~3.0 M cells  (--perf-large)
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import fdtdx
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.initialization import apply_params, place_objects
from fdtdx.fdtd.stop_conditions import EnergyThresholdCondition
from fdtdx.fdtd.wrapper import run_fdtd

from ..utils import BenchmarkResult, compile_fn, run_compiled

_RESOLUTION = 100e-9
_PML_CELLS = 8
_WAVELENGTH = 1550e-9


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _setup(
    interior_cells: int, *, figs_dir: Path
) -> tuple[ObjectContainer, ArrayContainer, SimulationConfig]:
    c0 = 3e8
    total_cells = interior_cells + 2 * _PML_CELLS
    total_size = total_cells * _RESOLUTION
    sim_time = 10 * math.sqrt(3) * total_size / c0  # hard cap: 10 diagonal transit times

    wave_char = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    temporal_profile = fdtdx.GaussianPulseProfile(
        center_wave=wave_char,
        spectral_width=fdtdx.WaveCharacter(wavelength=_WAVELENGTH * 10),
    )

    config = SimulationConfig(
        time=sim_time,
        grid=UniformGrid(spacing=_RESOLUTION),
        dtype=jnp.float32,
        gradient_config=None,
    )

    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(partial_real_shape=(total_size, total_size, total_size))
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML_CELLS)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    source = fdtdx.PointDipoleSource(
        name="dipole",
        partial_grid_shape=(1, 1, 1),
        wave_character=wave_char,
        temporal_profile=temporal_profile,
        polarization=0,
        amplitude=1.0,
    )
    constraints.append(source.place_at_center(volume, axes=(0, 1, 2)))
    objects.append(source)

    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = place_objects(
        object_list=objects, config=config, constraints=constraints, key=key,
    )
    arrays, obj_container, _ = apply_params(arrays, obj_container, params, key)

    figs_dir.mkdir(exist_ok=True)
    fdtdx.plot_setup(config=config, objects=obj_container,
                     filename=figs_dir / f"setup_box{interior_cells}.png")
    fdtdx.plot_material(config=config, arrays=arrays,
                        filename=figs_dir / f"material_box{interior_cells}.png")

    return obj_container, arrays, config


def _compile(
    objects: ObjectContainer,
    arrays: ArrayContainer,
    config: SimulationConfig,
) -> tuple[any, float, dict]:
    fn_kwargs = dict(
        arrays=arrays,
        objects=objects,
        config=config,
        key=jax.random.PRNGKey(0),
        stopping_condition=EnergyThresholdCondition(threshold=1e-6),
        show_progress=False,
    )
    return compile_fn(run_fdtd, fn_kwargs, static_argnames=("show_progress",))


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
        compiled, dynamic_kwargs,
        name=name, n_reps=n_reps,
        do_trace=do_trace, do_memory=do_memory,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.parametrize("interior_cells", [
    pytest.param(32,  id="coarse"),
    pytest.param(64,  id="med",  marks=pytest.mark.perf_med),
    pytest.param(128, id="fine", marks=pytest.mark.perf_fine),
])
def test_empty_box(interior_cells, perf_env, perf_sink, perf_run_dir, perf_options):
    """Measure forward-only MCUPS for a uniform PML box with early energy stopping."""
    bench_name = f"empty_box_{interior_cells}"
    figs_dir = perf_run_dir / "figs"

    objects, arrays, config = _setup(interior_cells, figs_dir=figs_dir)
    compiled, compile_s, dynamic_kwargs = _compile(objects, arrays, config)
    run_s, peak_mem, trace_path, mem_profile, final_state = _run(
        compiled, dynamic_kwargs,
        name=bench_name,
        n_reps=perf_options.reps,
        do_trace=perf_options.trace,
        do_memory=perf_options.memory,
        output_dir=perf_run_dir,
    )

    actual_steps = int(jax.device_get(final_state[0]))

    total_cells = interior_cells + 2 * _PML_CELLS
    result = BenchmarkResult(
        name=bench_name,
        grid_shape=(total_cells, total_cells, total_cells),
        time_steps=actual_steps,
        compile_seconds=compile_s,
        run_seconds=run_s,
        env=perf_env,
        peak_memory_bytes=peak_mem,
        trace_path=trace_path,
        memory_profile_path=mem_profile,
    )
    perf_sink.append(result)
    print(f"\n  {result.summary()}")
    print(f"  (max steps={config.time_steps_total}, early-stopped at step {actual_steps})")

    assert result.mcups > 0
    assert math.isfinite(result.mcups)
    assert actual_steps < config.time_steps_total, "simulation ran to hard cap — PML not absorbing?"
