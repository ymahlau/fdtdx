"""Forward throughput benchmark: uniform air box with PML on all six faces.

This is the simplest possible FDTD scene — no material interfaces, one dipole
source — and produces the peak-throughput (ceiling) MCUPS number.  It serves
as the baseline against which more realistic photonic benchmarks (directional
coupler, MMI, etc.) will be compared once geometry import is available.

Grid sizes
----------
- small:  32^3  interior cells  -> (32 + 2x8)^3  = 48^3  ~= 110 K total cells
- medium: 64^3  interior cells  -> (64 + 2x8)^3  = 80^3  ~= 512 K total cells
- large: 128^3  interior cells  -> (128 + 2x8)^3 = 144^3 ~= 3.0 M total cells

The large grid is skipped unless PERF_LARGE=1 is set, to keep the default
run time reasonable on CPU.
"""

from __future__ import annotations

import math
import os

import jax
import jax.numpy as jnp
import pytest

import fdtdx
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.fdtd.initialization import apply_params, place_objects
from fdtdx.fdtd.wrapper import run_fdtd

from ..utils import BenchmarkResult, run_benchmark

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RESOLUTION = 100e-9    # 100 nm — arbitrary for throughput test
_PML_CELLS = 8
_N_STEPS = 500          # override via PERF_STEPS env var
_WAVELENGTH = 1550e-9


def _n_steps() -> int:
    return int(os.environ.get("PERF_STEPS", str(_N_STEPS)))


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------
def _build_scene(interior_cells: int) -> tuple:
    """Uniform air box + PML on all faces + CW dipole source at centre."""
    n_steps = _n_steps()
    c0 = 3e8
    courant_factor = 0.99
    # Compute dt from grid spacing so we get exactly n_steps timesteps
    dt = courant_factor / math.sqrt(3) * _RESOLUTION / c0
    sim_time = n_steps * dt

    config = SimulationConfig(
        time=sim_time,
        grid=UniformGrid(spacing=_RESOLUTION),
        dtype=jnp.float32,
        gradient_config=None,
    )

    objects, constraints = [], []

    total_cells = interior_cells + 2 * _PML_CELLS
    total_size = total_cells * _RESOLUTION
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(total_size, total_size, total_size),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML_CELLS)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    source = fdtdx.PointDipoleSource(
        name="dipole",
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        polarization=0,
        amplitude=1.0,
    )
    constraints.append(source.place_at_center(volume, axes=(0, 1, 2)))
    objects.append(source)

    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, obj_container, _ = apply_params(arrays, obj_container, params, key)
    return obj_container, arrays, config


# ---------------------------------------------------------------------------
# Parametrised benchmark
# ---------------------------------------------------------------------------
_GRID_SIZES = [
    pytest.param(32, id="small_32"),
    pytest.param(64, id="medium_64"),
    pytest.param(
        128,
        id="large_128",
        marks=pytest.mark.skipif(
            not os.environ.get("PERF_LARGE"),
            reason="set PERF_LARGE=1 to run the large (128³) benchmark",
        ),
    ),
]


@pytest.mark.performance
@pytest.mark.parametrize("interior_cells", _GRID_SIZES)
def test_forward_throughput(interior_cells, perf_env, perf_sink, perf_run_dir):
    """Measure forward-only MCUPS for a uniform PML box."""
    objects, arrays, config = _build_scene(interior_cells)

    _run_jit = jax.jit(run_fdtd, static_argnames=["show_progress"])

    def _run():
        return _run_jit(
            arrays=arrays,
            objects=objects,
            config=config,
            key=jax.random.PRNGKey(0),
            show_progress=False,
        )

    bench_name = f"empty_box_{interior_cells}"
    compile_s, run_s, peak_mem, trace_path, mem_profile = run_benchmark(
        _run, name=bench_name, output_dir=perf_run_dir
    )

    total_cells = interior_cells + 2 * _PML_CELLS
    grid_shape = (total_cells, total_cells, total_cells)
    result = BenchmarkResult(
        name=bench_name,
        grid_shape=grid_shape,
        time_steps=config.time_steps_total,
        compile_seconds=compile_s,
        run_seconds=run_s,
        env=perf_env,
        peak_memory_bytes=peak_mem,
        trace_path=trace_path,
        memory_profile_path=mem_profile,
    )
    perf_sink.append(result)

    print(f"\n  {result.summary()}")

    # Sanity: throughput must be positive and finite
    assert result.mcups > 0
    assert math.isfinite(result.mcups)
