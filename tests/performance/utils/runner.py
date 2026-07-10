from __future__ import annotations

import time
from pathlib import Path
from typing import Any

_RESULTS_DIR = Path(__file__).parents[2] / "results"


def compile_fn(fn, fn_kwargs: dict, *, static_argnames: tuple[str, ...] = ()) -> tuple[Any, float, dict]:
    """JIT-compile ``fn`` without executing it.

    Static args are baked into the compiled artifact and must not be passed at
    call time, so this function also returns the dynamic-only subset of kwargs.

    Returns:
        ``(compiled, compile_seconds, dynamic_kwargs)``
    """
    import jax

    jitted = jax.jit(fn, static_argnames=static_argnames or None)
    t0 = time.perf_counter()
    compiled = jitted.lower(**fn_kwargs).compile()
    compile_s = time.perf_counter() - t0
    dynamic_kwargs = {k: v for k, v in fn_kwargs.items() if k not in static_argnames}
    return compiled, compile_s, dynamic_kwargs


def run_compiled(
    compiled,
    fn_kwargs: dict,
    *,
    name: str = "benchmark",
    n_reps: int = 1,
    do_trace: bool = False,
    do_memory: bool = False,
    output_dir: Path | None = None,
) -> tuple[list[float], int | None, str | None, str | None, Any]:
    """Time ``n_reps`` executions of a pre-compiled JAX function.

    Args:
        compiled: Result of ``jax.jit(fn).lower(**kwargs).compile()``.
        fn_kwargs: Keyword arguments forwarded to ``compiled`` on each call.
        name: Identifier used for trace/profile file naming.
        n_reps: Number of timed repetitions.
        do_trace: Capture an XLA/Perfetto trace for the first rep.
        do_memory: Record peak device memory and save a pprof memory profile.
        output_dir: Per-session results folder. Falls back to the package-level
            ``results/`` directory if not provided.

    Returns:
        ``(run_seconds, peak_memory_bytes, trace_path, memory_profile_path, last_result)``

    Note:
        ``peak_memory_bytes`` is JAX's ``peak_bytes_in_use``, read after the run.
        JAX has no stable public API to reset that high-water mark before timing
        starts (see https://github.com/jax-ml/jax/issues/8096), so this is the
        peak since process start, not an isolated per-run measurement — earlier
        setup/compilation allocations in this process count too. Treat it as an
        upper bound.
    """
    import jax

    base = output_dir if output_dir is not None else _RESULTS_DIR

    memory_profile_path: str | None = None
    if do_memory:
        try:
            mem_dir = base / "memory_profiles"
            mem_dir.mkdir(parents=True, exist_ok=True)
            jax.profiler.save_device_memory_profile(str(mem_dir / f"{name}.pb"))
            memory_profile_path = str(mem_dir / f"{name}.pb")
        except Exception:
            pass

    run_seconds: list[float] = []
    trace_path: str | None = None
    result = None

    for i in range(n_reps):
        if do_trace and i == 0:
            trace_dir = base / "traces" / name
            trace_dir.mkdir(parents=True, exist_ok=True)
            with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
                t0 = time.perf_counter()
                result = compiled(**fn_kwargs)
                jax.block_until_ready(result)
            run_seconds.append(time.perf_counter() - t0)
            trace_path = str(trace_dir)
        else:
            t0 = time.perf_counter()
            result = compiled(**fn_kwargs)
            jax.block_until_ready(result)
            run_seconds.append(time.perf_counter() - t0)

    peak_memory_bytes: int | None = None
    if do_memory:
        stats = read_memory_stats()
        peak = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use")
        if peak:
            peak_memory_bytes = int(peak)

    return run_seconds, peak_memory_bytes, trace_path, memory_profile_path, result


def read_memory_stats() -> dict[str, int]:
    import jax

    try:
        return jax.devices()[0].memory_stats() or {}
    except Exception:
        return {}
