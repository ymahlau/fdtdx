"""Pytest configuration for performance tests.

All tests in this directory are automatically marked ``performance``.
Run only performance tests:   pytest -m performance
Exclude performance tests:    pytest -m "not performance"

Each session writes to its own folder:
  tests/performance/results/{timestamp}_{device}/
    results.json      — benchmark timings and environment
    figs/             — setup/material/field plots
    traces/           — XLA/Perfetto traces (disable with --no-perf-trace)

Tracing, memory profiling, and visualization are ON by default.
Pass --no-perf-trace / --no-perf-memory / --no-perf-visualize to disable.
"""

from __future__ import annotations

import dataclasses
import os

# tests/conftest.py forces JAX_PLATFORMS=cpu to avoid sharding-mesh mismatches
# in unit/integration tests. Performance benchmarks need the real device (GPU),
# so remove that restriction before JAX is first imported in this session.
os.environ.pop("JAX_PLATFORMS", None)
from datetime import datetime, timezone
from pathlib import Path

import pytest

from .utils import BenchmarkResult, capture_env, export_results

_RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("performance", "FDTDX performance benchmark options")
    g.addoption(
        "--no-perf-trace",
        action="store_false",
        dest="perf_trace",
        default=True,
        help="Disable XLA/Perfetto tracing (enabled by default).",
    )
    g.addoption(
        "--no-perf-memory",
        action="store_false",
        dest="perf_memory",
        default=True,
        help="Disable peak memory recording (enabled by default).",
    )
    g.addoption(
        "--no-perf-visualize",
        action="store_false",
        dest="perf_visualize",
        default=True,
        help="Disable field intensity and setup plots (enabled by default).",
    )
    g.addoption(
        "--perf-reps",
        type=int,
        default=1,
        metavar="N",
        help="Number of timed execution repetitions (default: 1).",
    )
    g.addoption(
        "--perf-med",
        action="store_true",
        default=False,
        help="Include medium-resolution variants of all benchmarks.",
    )
    g.addoption(
        "--perf-fine",
        action="store_true",
        default=False,
        help="Include fine-resolution variants of all benchmarks.",
    )


# ---------------------------------------------------------------------------
# Collection hooks
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if "performance" in Path(str(item.fspath)).parts:
            item.add_marker(pytest.mark.performance)

    skip_med = pytest.mark.skip(reason="pass --perf-med to include medium-resolution variants")
    skip_fine = pytest.mark.skip(reason="pass --perf-fine to include fine-resolution variants")

    for item in items:
        if item.get_closest_marker("perf_med") and not config.getoption("--perf-med"):
            item.add_marker(skip_med)
        if item.get_closest_marker("perf_fine") and not config.getoption("--perf-fine"):
            item.add_marker(skip_fine)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PerfOptions:
    trace: bool
    memory: bool
    visualize: bool
    reps: int


@pytest.fixture(scope="session")
def perf_options(request: pytest.FixtureRequest) -> PerfOptions:
    return PerfOptions(
        trace=request.config.getoption("perf_trace"),
        memory=request.config.getoption("perf_memory"),
        visualize=request.config.getoption("perf_visualize"),
        reps=request.config.getoption("--perf-reps"),
    )


@pytest.fixture(scope="session")
def perf_env() -> dict:
    return capture_env()


@pytest.fixture(scope="session")
def perf_run_dir(perf_env: dict) -> Path:
    """Create and return the per-session output folder."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    devices = perf_env.get("devices", [])
    device_slug = devices[0].get("kind", "cpu").replace(" ", "_").lower() if devices else "cpu"
    run_dir = _RESULTS_DIR / f"{ts}_{device_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figs").mkdir(exist_ok=True)
    (run_dir / "traces").mkdir(exist_ok=True)
    return run_dir


@pytest.fixture(scope="session")
def perf_sink() -> list[BenchmarkResult]:
    """Accumulates BenchmarkResult objects across the session."""
    return []


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    sink: list[BenchmarkResult] = getattr(session, "_perf_sink", None) or []
    run_dir: Path | None = getattr(session, "_perf_run_dir", None)
    if sink and run_dir is not None:
        export_results(sink, run_dir)


@pytest.fixture(autouse=True)
def _register_sink(request: pytest.FixtureRequest, perf_sink: list, perf_run_dir: Path) -> None:
    request.session._perf_sink = perf_sink  # type: ignore[attr-defined]
    request.session._perf_run_dir = perf_run_dir  # type: ignore[attr-defined]
