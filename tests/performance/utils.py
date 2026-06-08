"""Utilities for FDTDX performance benchmarking and scene building.

Benchmark environment flags
---------------------------
PERF_REPS=N       Number of timed repetitions after warmup (default 3).
PERF_STEPS=N      Number of FDTD timesteps per run (default 500).
PERF_LARGE=1      Enable the large (128³) grid benchmark.
PERF_TRACE=1      Emit a Perfetto/XLA trace for the first timed rep.
                  Traces land in <run_dir>/traces/<name>/.
                  Open at https://ui.perfetto.dev
PERF_MEMORY=1     Record peak device memory (bytes) in the JSON output.

Scene builder
-------------
Generalizable primitives for building fdtdx simulation scenes from GDS geometry
and port metadata (gdsfactory .yml or .json format).  Individual jobs import
these utilities and supply GDS paths, port names, domain sizes, and materials;
the scene builder handles constraint construction, object placement, and
phasor-monitor setup.
"""

from __future__ import annotations

import dataclasses
import math
import os
import platform
import subprocess
import time
from pathlib import Path

_RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _git_sha() -> str:
    try:
        repo_root = Path(__file__).parents[2]
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def capture_env() -> dict:
    """Capture hardware and software environment for benchmark reproducibility."""
    import jax
    import jaxlib

    devices = []
    for d in jax.devices():
        entry: dict = {"id": d.id, "kind": d.device_kind, "platform": d.platform}
        try:
            stats = d.memory_stats()
            if stats and stats.get("bytes_limit"):
                entry["memory_bytes"] = stats["bytes_limit"]
        except Exception:
            pass
        devices.append(entry)

    return {
        "git_sha": _git_sha(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "python_version": platform.python_version(),
        "platform": platform.system().lower(),
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "devices": devices,
        "jax_x64_enabled": jax.config.jax_enable_x64,
    }


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _read_memory_stats() -> dict[str, int]:
    """Return memory stats for device 0, or empty dict if unavailable."""
    import jax
    try:
        stats = jax.devices()[0].memory_stats()
        return stats or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BenchmarkResult:
    name: str
    grid_shape: tuple[int, int, int]
    time_steps: int
    compile_seconds: float
    run_seconds: list[float]
    env: dict
    peak_memory_bytes: int | None = None
    static_memory_bytes: int | None = None
    trace_path: str | None = None
    memory_profile_path: str | None = None

    @property
    def total_cell_updates(self) -> int:
        return math.prod(self.grid_shape) * self.time_steps

    @property
    def mcups(self) -> float:
        """Million cell updates per second (best run)."""
        return self.total_cell_updates / min(self.run_seconds) / 1e6

    def to_dict(self) -> dict:
        runs = sorted(self.run_seconds)
        d: dict = {
            "name": self.name,
            "grid_shape": list(self.grid_shape),
            "time_steps": self.time_steps,
            "total_cell_updates": self.total_cell_updates,
            "compile_seconds": self.compile_seconds,
            "run_seconds_best": min(self.run_seconds),
            "run_seconds_median": runs[len(runs) // 2],
            "run_seconds_worst": max(self.run_seconds),
            "mcups": self.mcups,
            "env": self.env,
        }
        if self.static_memory_bytes is not None:
            d["static_memory_bytes"] = self.static_memory_bytes
            d["static_memory_gb"] = round(self.static_memory_bytes / 1024**3, 3)
        if self.peak_memory_bytes is not None:
            d["peak_memory_bytes"] = self.peak_memory_bytes
            d["peak_memory_gb"] = round(self.peak_memory_bytes / 1024**3, 3)
        if self.trace_path is not None:
            d["trace_path"] = self.trace_path
        if self.memory_profile_path is not None:
            d["memory_profile_path"] = self.memory_profile_path
        return d

    def summary(self) -> str:
        parts = [
            f"{self.name}:",
            f"grid={self.grid_shape}",
            f"steps={self.time_steps}",
            f"compile={self.compile_seconds:.2f}s",
            f"run_best={min(self.run_seconds):.3f}s",
            f"MCUPS={self.mcups:.1f}",
        ]
        if self.static_memory_bytes is not None:
            parts.append(f"static_mem={self.static_memory_bytes / 1024**2:.0f}MB")
        if self.peak_memory_bytes is not None:
            parts.append(f"peak_mem={self.peak_memory_bytes / 1024**2:.0f}MB")
        if self.trace_path is not None:
            parts.append(f"trace={self.trace_path}  (open at https://ui.perfetto.dev)")
        if self.memory_profile_path is not None:
            parts.append(f"mem_profile={self.memory_profile_path}  (open with: go tool pprof -http=: <path>)")
        return "\n  ".join(parts)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    run_fn,
    *,
    name: str = "benchmark",
    n_reps: int | None = None,
    output_dir: Path | None = None,
) -> tuple[float, list[float], int | None, str | None, str | None]:
    """Time a jitted FDTD call with optional tracing and memory profiling.

    Args:
        run_fn: Zero-argument callable returning a JAX pytree to block on.
        name: Identifier used for trace directory naming.
        n_reps: Timed repetitions after warmup. Defaults to PERF_REPS (3).
        output_dir: Per-run results folder (from ``perf_run_dir`` fixture).
                    Traces land in ``output_dir/traces/<name>/``.
                    Falls back to ``results/traces/<name>/`` if not provided.

    Returns:
        (compile_seconds, run_seconds_list, peak_memory_bytes_or_None,
         trace_path_or_None, memory_profile_path_or_None)

    Environment flags:
        PERF_TRACE=1    Capture a Perfetto/XLA trace for the first timed rep.
        PERF_MEMORY=1   Record peak device memory bytes and save a pprof
                        memory profile (open with ``go tool pprof -http=: <path>``).
    """
    import jax

    if n_reps is None:
        n_reps = int(os.environ.get("PERF_REPS", "3"))

    do_trace = bool(os.environ.get("PERF_TRACE"))
    do_memory = bool(os.environ.get("PERF_MEMORY"))

    base = output_dir if output_dir is not None else _RESULTS_DIR

    # ------------------------------------------------------------------
    # Warmup — triggers JIT compilation
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    result = run_fn()
    jax.block_until_ready(result)
    compile_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # pprof memory profile — captured right after warmup while compiled
    # buffers are live; shows per-allocation Python stack traces so you
    # can see exactly which arrays (E, H, psi, alpha, kappa, sigma, …)
    # are consuming device memory.
    # Open with:  go tool pprof -http=: <path>
    # ------------------------------------------------------------------
    memory_profile_path: str | None = None
    if do_memory:
        try:
            mem_dir = base / "memory_profiles"
            mem_dir.mkdir(parents=True, exist_ok=True)
            mem_path = mem_dir / f"{name}.pb"
            jax.profiler.save_device_memory_profile(str(mem_path))
            memory_profile_path = str(mem_path)
        except Exception:
            pass
        _read_memory_stats()  # reset peak counter baseline

    # ------------------------------------------------------------------
    # Timed repetitions
    # ------------------------------------------------------------------
    run_seconds: list[float] = []
    trace_path: str | None = None

    for i in range(n_reps):
        if do_trace and i == 0:
            # Capture XLA/Perfetto trace for the first timed rep only.
            trace_dir = base / "traces" / name
            trace_dir.mkdir(parents=True, exist_ok=True)
            with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
                t0 = time.perf_counter()
                result = run_fn()
                jax.block_until_ready(result)
            run_seconds.append(time.perf_counter() - t0)
            trace_path = str(trace_dir)
        else:
            t0 = time.perf_counter()
            result = run_fn()
            jax.block_until_ready(result)
            run_seconds.append(time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Peak memory (high-water mark across all timed reps)
    # ------------------------------------------------------------------
    peak_memory_bytes: int | None = None
    if do_memory:
        stats = _read_memory_stats()
        peak = stats.get("peak_bytes_in_use")
        if peak is not None and peak > 0:
            peak_memory_bytes = int(peak)
        else:
            current = stats.get("bytes_in_use")
            if current is not None:
                peak_memory_bytes = int(current)

    return compile_seconds, run_seconds, peak_memory_bytes, trace_path, memory_profile_path


# =============================================================================
# Scene builder utilities
# =============================================================================
# Generalizable primitives for building fdtdx simulation scenes from GDS
# geometry and port metadata supplied as plain Python dicts.
#
# Port dict format (all quantities in SI metres):
#   {"x_m": float, "y_m": float, "width_m": float, "orientation": float}
#   orientation in degrees: 0=+x, 90=+y, 180=-x, 270=-y  (gdsfactory convention)
#
# Typical usage in a job script:
#
#   ports = {
#       "o1": {"x_m": -10e-6, "y_m": -1.632e-6, "width_m": 0.5e-6, "orientation": 180.0},
#       "o3": {"x_m":  30e-6, "y_m":  2.368e-6, "width_m": 0.5e-6, "orientation":   0.0},
#   }
#   objects, constraints, config, volume = build_domain(...)
#   add_gds_geometry(objects, constraints, ...)
#   add_mode_source(objects, constraints, port=ports["o1"], ...)
#   add_mode_detector(objects, constraints, port=ports["o3"], ...)
#   add_phasor_monitors(objects, constraints, ...)
# =============================================================================


def port_to_sim_coords(
    port: dict,
    domain_shape: tuple[float, float, float],
    gds_center: tuple[float, float],
) -> tuple[float, float]:
    """Convert a GDS port position to simulation coordinates (metres).

    sim_x = port_x + domain_x/2 - gds_center_x
    sim_y = port_y + domain_y/2 - gds_center_y

    Args:
        port: Port dict with keys ``x_m`` and ``y_m`` in metres.
        domain_shape: (x, y, z) total simulation domain in metres.
        gds_center: GDS (x, y) coordinate in metres that maps to the simulation centre.

    Returns:
        (sim_x, sim_y) in metres.
    """
    return (
        port["x_m"] + domain_shape[0] / 2.0 - gds_center[0],
        port["y_m"] + domain_shape[1] / 2.0 - gds_center[1],
    )


def build_domain(
    domain_shape: tuple[float, float, float],
    pml_cells: int,
    background_material,
    grid,
    sim_time: float,
    dtype=None,
) -> tuple:
    """Create SimulationVolume, PML boundaries, background fill, and SimulationConfig.

    Args:
        domain_shape: (x, y, z) total domain extents in metres including PML.
        pml_cells: Number of PML cells on each face.
        background_material: :class:`fdtdx.Material` filling the entire volume
            (e.g. SiO₂ cladding).
        grid: Grid object (:class:`fdtdx.RectilinearGrid` or
            :class:`fdtdx.UniformGrid`).
        sim_time: Total simulation time in seconds.
        dtype: JAX dtype for field arrays; defaults to ``jnp.float32``.

    Returns:
        ``(objects, constraints, config, volume)``
    """
    import jax.numpy as jnp_

    import fdtdx
    from fdtdx.config import SimulationConfig

    if dtype is None:
        dtype = jnp_.float32

    config = SimulationConfig(
        grid=grid,
        time=sim_time,
        dtype=dtype,
        gradient_config=None,
    )

    objects: list = []
    constraints: list = []

    volume = fdtdx.SimulationVolume(partial_real_shape=domain_shape)
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=pml_cells)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    cladding = fdtdx.UniformMaterialObject(
        name="cladding",
        partial_real_shape=(None, None, None),
        material=background_material,
    )
    constraints.extend(cladding.same_position_and_size(volume))
    objects.append(cladding)

    return objects, constraints, config, volume


def add_gds_geometry(
    objects: list,
    constraints: list,
    gds_path,
    cell_name: str,
    layer_specs: list,
    materials: dict,
    volume,
    gds_center: tuple[float, float],
) -> None:
    """Import GDS polygons and add extruded layer objects to the scene in-place.

    Args:
        objects: Scene object list to extend.
        constraints: Scene constraint list to extend.
        gds_path: Path to the ``.gds`` file.
        cell_name: GDS cell containing the device polygons.
        layer_specs: List of :class:`~fdtdx.objects.static_material.gds_layer_stack.GDSLayerSpec`.
        materials: Dict mapping material name strings to :class:`fdtdx.Material`.
        volume: :class:`fdtdx.SimulationVolume` used for size/position constraints.
        gds_center: GDS (x, y) coordinate in metres mapped to the simulation centre.
    """
    from fdtdx.objects.static_material.gds_layer_stack import gds_layer_stack

    gds_objs, gds_cons = gds_layer_stack(
        gds_source=gds_path,
        cell_name=cell_name,
        layers=layer_specs,
        materials=materials,
        simulation_volume=volume,
        gds_center=gds_center,
    )
    objects.extend(gds_objs)
    constraints.extend(gds_cons)


def add_mode_source(
    objects: list,
    constraints: list,
    port: dict,
    name: str,
    wave_character,
    temporal_profile,
    volume,
    domain_shape: tuple[float, float, float],
    gds_center: tuple[float, float],
    span_y: float,
    mode_index: int = 0,
    filter_pol: str | None = "te",
    direction: str = "+",
) -> None:
    """Add a :class:`fdtdx.ModePlaneSource` placed at a port position.

    The source spans ``span_y`` metres in the transverse (y) direction,
    centred on the port's y coordinate, and covers the full z extent of the
    simulation volume.  It is placed one cell thick in x at ``port["x_m"]``
    converted to simulation coordinates.

    Args:
        objects: Scene object list to extend.
        constraints: Scene constraint list to extend.
        port: Port dict with keys ``x_m``, ``y_m`` in metres.
        name: Unique object name for this source.
        wave_character: :class:`fdtdx.WaveCharacter` (wavelength / frequency).
        temporal_profile: Temporal envelope (e.g. :class:`fdtdx.GaussianPulseProfile`).
        volume: :class:`fdtdx.SimulationVolume` reference for constraints.
        domain_shape: (x, y, z) total domain in metres.
        gds_center: GDS (x, y) in metres mapped to the simulation centre.
        span_y: Transverse (y) extent of the source cross-section in metres.
        mode_index: Waveguide mode order (0 = fundamental).
        filter_pol: Polarisation filter ``"te"``, ``"tm"``, or ``None``.
        direction: Propagation direction ``"+"`` or ``"-"``.
    """
    import fdtdx

    sim_x, sim_y = port_to_sim_coords(port, domain_shape, gds_center)
    source = fdtdx.ModePlaneSource(
        name=name,
        partial_grid_shape=(1, None, None),
        partial_real_shape=(None, span_y, None),
        wave_character=wave_character,
        temporal_profile=temporal_profile,
        direction=direction,
        mode_index=mode_index,
        filter_pol=filter_pol,
    )
    constraints.extend([
        source.same_size(volume, axes=(2,)),
        source.place_at_center(volume, axes=(2,)),
        fdtdx.RealCoordinateConstraint(
            object=name, axes=(0,), sides=("-",), coordinates=(sim_x,),
        ),
        fdtdx.RealCoordinateConstraint(
            object=name, axes=(1,), sides=("-",), coordinates=(sim_y - span_y / 2,),
        ),
    ])
    objects.append(source)


def add_mode_detector(
    objects: list,
    constraints: list,
    port: dict,
    name: str,
    wave_characters: tuple,
    volume,
    domain_shape: tuple[float, float, float],
    gds_center: tuple[float, float],
    span_y: float,
    mode_index: int = 0,
    filter_pol: str | None = "te",
    direction: str = "+",
    scaling_mode: str = "pulse",
) -> None:
    """Add a :class:`fdtdx.ModeOverlapDetector` placed at a port position.

    Same placement logic as :func:`add_mode_source`.

    Args:
        objects: Scene object list to extend.
        constraints: Scene constraint list to extend.
        port: Port dict with keys ``x_m``, ``y_m`` in metres.
        name: Unique object name for this detector.
        wave_characters: Tuple of :class:`fdtdx.WaveCharacter` objects.
        volume: :class:`fdtdx.SimulationVolume` reference for constraints.
        domain_shape: (x, y, z) total domain in metres.
        gds_center: GDS (x, y) in metres mapped to the simulation centre.
        span_y: Transverse (y) extent in metres.
        mode_index: Waveguide mode order.
        filter_pol: Polarisation filter or ``None``.
        direction: Propagation direction ``"+"`` or ``"-"``.
        scaling_mode: ``"pulse"`` (default) or ``"continuous"``.
    """
    import fdtdx

    sim_x, sim_y = port_to_sim_coords(port, domain_shape, gds_center)
    det = fdtdx.ModeOverlapDetector(
        name=name,
        partial_grid_shape=(1, None, None),
        partial_real_shape=(None, span_y, None),
        wave_characters=wave_characters,
        direction=direction,
        mode_index=mode_index,
        filter_pol=filter_pol,
        scaling_mode=scaling_mode,
    )
    constraints.extend([
        det.same_size(volume, axes=(2,)),
        det.place_at_center(volume, axes=(2,)),
        fdtdx.RealCoordinateConstraint(
            object=name, axes=(0,), sides=("-",), coordinates=(sim_x,),
        ),
        fdtdx.RealCoordinateConstraint(
            object=name, axes=(1,), sides=("-",), coordinates=(sim_y - span_y / 2,),
        ),
    ])
    objects.append(det)


def extend_gds_with_port_stubs(
    gds_path: Path,
    cell_name: str,
    ports: dict,
    layer_specs: list,
    gds_center: tuple[float, float],
    domain_shape: tuple[float, float, float],
) -> Path:
    """Return path to a temp GDS with waveguide stub rectangles added at each port.

    The original GDS waveguides end at the port positions, leaving a gap between
    the waveguide end and the domain/PML boundary.  This causes the mode source to
    inject into a Si/cladding discontinuity, producing reflections and standing waves.

    This function adds rectangular polygons on every layer in ``layer_specs``,
    extending each port waveguide from the port position to the domain boundary.
    The stubs are then imported by :func:`gds_layer_stack` identically to the main
    device geometry, ensuring correct material assignment and z-extent.

    Args:
        gds_path: Path to the original ``.gds`` file.
        cell_name: GDS cell to modify.
        ports: Port dict ``{port_name: {"x_m", "y_m", "width_m", "orientation"}}``.
        layer_specs: Layer specs from which GDS layer/datatype are extracted.
        gds_center: GDS (x, y) in metres mapped to simulation centre.
        domain_shape: (x, y, z) total domain in metres.

    Returns:
        Path to a temporary ``.gds`` file with stubs added.
    """
    import tempfile

    import gdstk

    lib = gdstk.read_gds(str(gds_path))
    real_cells = {c.name: c for c in lib.cells if not c.name.startswith("$$$")}
    target = real_cells[cell_name]

    gds_cx_um = gds_center[0] * 1e6
    domain_x_um = domain_shape[0] * 1e6
    left_um = gds_cx_um - domain_x_um / 2
    right_um = gds_cx_um + domain_x_um / 2

    for spec in layer_specs:
        layer = spec.gds_layer
        datatype = getattr(spec, "gds_datatype", 0)
        for port in ports.values():
            gds_x = port["x_m"] * 1e6
            gds_y = port["y_m"] * 1e6
            half_w = port["width_m"] * 1e6 / 2
            if abs(port["orientation"] - 180.0) < 1.0:
                rect = gdstk.rectangle(
                    (left_um, gds_y - half_w), (gds_x, gds_y + half_w),
                    layer=layer, datatype=datatype,
                )
            else:
                rect = gdstk.rectangle(
                    (gds_x, gds_y - half_w), (right_um, gds_y + half_w),
                    layer=layer, datatype=datatype,
                )
            target.add(rect)

    tmp = tempfile.NamedTemporaryFile(suffix=".gds", delete=False)
    lib.write_gds(tmp.name)
    return Path(tmp.name)


def add_phasor_monitors(
    objects: list,
    constraints: list,
    volume,
    wave_character,
    z_height: float,
) -> None:
    """Add a top-view XY frequency-domain field monitor to the scene in-place.

    Creates a :class:`fdtdx.PhasorDetector` (``"phasor_xy"``) that accumulates the
    ``Ey`` component at the wavelength defined by ``wave_character``.

    Args:
        objects: Scene object list to extend.
        constraints: Scene constraint list to extend.
        volume: :class:`fdtdx.SimulationVolume` reference.
        wave_character: :class:`fdtdx.WaveCharacter` giving the monitoring wavelength.
        z_height: z coordinate in simulation space (metres) for the XY slice.
            Typically the Si core centre.
    """
    import fdtdx

    det_xy = fdtdx.PhasorDetector(
        name="phasor_xy",
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave_character,),
        components=("Ey",),
        reduce_volume=False,
        plot=False,
        scaling_mode="continuous",
    )
    objects.append(det_xy)
    constraints += [
        det_xy.same_size(volume, axes=(0, 1)),
        det_xy.place_at_center(volume, axes=(0, 1)),
        fdtdx.RealCoordinateConstraint(
            object="phasor_xy", axes=(2,), sides=("-",), coordinates=(z_height,),
        ),
    ]


def plot_field_intensity(
    detector_states: dict,
    config,
    figs_dir: Path,
    *,
    gds_path=None,
    gds_layer: tuple[int, int] = (1, 0),
    gds_center: tuple[float, float] = (0.0, 0.0),
    domain_shape: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_steps: int = 0,
    dx_nm: int = 50,
    y_lim: tuple[float, float] | None = None,
    det_info: dict | None = None,
) -> None:
    """Save inferno |Ey|² intensity plots for phasor detectors.

    Reads ``detector_states["phasor_xy"]``, converts simulation grid edges to GDS
    µm coordinates, and writes a PNG to ``figs_dir``.

    Args:
        detector_states: ``final_arrays.detector_states`` dict from :func:`fdtdx.run_fdtd`.
        config: :class:`fdtdx.SimulationConfig` holding the grid edges.
        figs_dir: Output directory for PNG files.
        gds_path: Path to ``.gds`` file for polygon overlay.  Skipped if ``None``.
        gds_layer: ``(layer, datatype)`` pair for GDS polygon overlay.
        gds_center: GDS (x, y) in metres mapped to the simulation centre.
        domain_shape: (x, y, z) total domain in metres — used for axis conversion.
        n_steps: Total simulation steps (for plot title).
        dx_nm: Grid spacing in nm (for plot title).
        y_lim: Optional (y_min, y_max) in GDS µm to crop the XY plot.
        det_info: Override axis metadata per detector:
            ``{det_name: (title, xc_um, yc_um, ylabel, ylim)}``.
            Defaults to GDS-coordinate axes derived from the grid.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    figs_dir = Path(figs_dir)
    figs_dir.mkdir(exist_ok=True)

    domain_x_m, domain_y_m, _ = domain_shape
    gds_cx_um = gds_center[0] * 1e6
    gds_cy_um = gds_center[1] * 1e6

    # x: GDS coords (same origin as paper).
    # y: centred at the coupler midpoint (gds_cy) so waveguides sit at ±2 µm,
    #    matching the paper's y-axis convention.
    x_edges_um = np.array(config.grid.x_edges) * 1e6 - domain_x_m * 1e6 / 2 + gds_cx_um
    y_edges_um = np.array(config.grid.y_edges) * 1e6 - domain_y_m * 1e6 / 2

    xc = 0.5 * (x_edges_um[:-1] + x_edges_um[1:])
    yc = 0.5 * (y_edges_um[:-1] + y_edges_um[1:])

    default_info: dict = {
        "phasor_xy": (xc, yc, y_lim),
    }
    info = det_info if det_info is not None else default_info

    for det_name, state in detector_states.items():
        if "phasor" not in state or det_name not in info:
            continue
        xvals, yvals, ylim = info[det_name]
        phasor = np.array(state["phasor"])[0, 0, 0].squeeze()
        intensity = np.abs(phasor) ** 2

        # Figure proportions: match the paper (40 µm x-span, ~6 µm y-span → ~6:1).
        x_span = xvals[-1] - xvals[0]
        y_span = (ylim[1] - ylim[0]) if ylim is not None else (yvals[-1] - yvals[0])
        fig_w = 10.0
        fig_h = max(1.5, fig_w * y_span / x_span)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        ax.pcolormesh(xvals, yvals, intensity.T, cmap="turbo",
                      vmin=0, vmax=intensity.max(), shading="nearest")
        ax.set_xlabel("x (µm)", fontsize=11)
        ax.set_ylabel("y (µm)", fontsize=11)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_title(f"fdtdx FDTD  (dx={dx_nm} nm, {n_steps} steps)", fontsize=11)

        out = figs_dir / f"{det_name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def build_scene(
    *,
    gds_path,
    cell_name: str,
    layer_specs: list,
    materials: dict,
    ports: dict,
    source_port: str,
    detector_ports: list[tuple[str, str]],
    gds_center: tuple[float, float],
    domain_shape: tuple[float, float, float],
    pml_cells: int,
    background_material,
    wave_char,
    temporal_profile,
    grid,
    sim_time: float,
    source_span_y: float,
    norm_det_dx: float = 30e-9,
    norm_det_name: str = "det_source",
    with_port_stubs: bool = True,
    with_phasor_monitors: bool = False,
    phasor_z_height: float | None = None,
) -> tuple:
    """Build a complete fdtdx scene for a GDS-imported SOI device.

    Combines :func:`build_domain`, :func:`add_gds_geometry`,
    :func:`add_mode_source`, :func:`add_mode_detector`, and
    :func:`add_phasor_monitors` into a single call.  Device-specific constants
    (GDS path, ports, materials, domain, grid) are passed as keyword arguments;
    the scene topology (source → norm detector → output detectors → optional
    phasor monitors) is fixed and applies to any 2-port or 4-port SOI device.

    Args:
        gds_path: Path to the ``.gds`` file.
        cell_name: GDS cell name.
        layer_specs: List of :class:`~fdtdx.objects.static_material.gds_layer_stack.GDSLayerSpec`.
        materials: Dict mapping material names to :class:`fdtdx.Material`.
        ports: Port dict ``{port_name: {"x_m", "y_m", "width_m", "orientation"}}``.
        source_port: Key in ``ports`` to use as the excitation port.
        detector_ports: ``[(detector_name, port_key), ...]`` — one
            :class:`fdtdx.ModeOverlapDetector` per entry, placed at the given port.
        gds_center: GDS (x, y) in metres mapped to the simulation centre.
        domain_shape: (x, y, z) total domain in metres including PML.
        pml_cells: PML thickness in grid cells on every face.
        background_material: :class:`fdtdx.Material` for the cladding fill.
        wave_char: :class:`fdtdx.WaveCharacter` for the source and all detectors.
        temporal_profile: Source temporal envelope.
        grid: :class:`fdtdx.RectilinearGrid` or :class:`fdtdx.UniformGrid`.
        sim_time: Total simulation time in seconds.
        source_span_y: Transverse (y) span in metres for the source and detectors.
        norm_det_dx: x-offset in metres from the source port for the normalisation
            detector (default 30 nm = one fine grid cell).
        norm_det_name: Name for the normalisation detector (default
            ``"det_source"``).
        with_port_stubs: If ``True`` (default), extend port waveguides to the domain
            boundary via :func:`extend_gds_with_port_stubs`.  Without stubs the
            source injects into a Si/cladding discontinuity causing standing waves.
        with_phasor_monitors: Add an XY :class:`fdtdx.PhasorDetector` monitor.
        phasor_z_height: z coordinate (metres) for the XY phasor slice; typically
            the Si core centre.  Required when ``with_phasor_monitors=True``.

    Returns:
        ``(objects, constraints, config, volume)``
    """
    objects, constraints, config, volume = build_domain(
        domain_shape=domain_shape,
        pml_cells=pml_cells,
        background_material=background_material,
        grid=grid,
        sim_time=sim_time,
    )

    import_gds = gds_path
    if with_port_stubs:
        import_gds = extend_gds_with_port_stubs(
            gds_path=gds_path,
            cell_name=cell_name,
            ports=ports,
            layer_specs=layer_specs,
            gds_center=gds_center,
            domain_shape=domain_shape,
        )

    add_gds_geometry(
        objects, constraints,
        gds_path=import_gds,
        cell_name=cell_name,
        layer_specs=layer_specs,
        materials=materials,
        volume=volume,
        gds_center=gds_center,
    )

    add_mode_source(
        objects, constraints,
        port=ports[source_port],
        name="source",
        wave_character=wave_char,
        temporal_profile=temporal_profile,
        volume=volume,
        domain_shape=domain_shape,
        gds_center=gds_center,
        span_y=source_span_y,
    )

    # Normalisation detector: same cross-section as source, shifted by norm_det_dx in x.
    norm_port = {**ports[source_port], "x_m": ports[source_port]["x_m"] + norm_det_dx}
    add_mode_detector(
        objects, constraints,
        port=norm_port,
        name=norm_det_name,
        wave_characters=(wave_char,),
        volume=volume,
        domain_shape=domain_shape,
        gds_center=gds_center,
        span_y=source_span_y,
    )

    for det_name, port_key in detector_ports:
        add_mode_detector(
            objects, constraints,
            port=ports[port_key],
            name=det_name,
            wave_characters=(wave_char,),
            volume=volume,
            domain_shape=domain_shape,
            gds_center=gds_center,
            span_y=source_span_y,
        )

    if with_phasor_monitors:
        add_phasor_monitors(
            objects, constraints,
            volume=volume,
            wave_character=wave_char,
            z_height=phasor_z_height,
        )

    return objects, constraints, config, volume


def _overlay_gds(ax, gds_path, gds_layer: tuple[int, int], color: str = "#4C8BE0", alpha: float = 0.35) -> None:
    """Overlay GDS polygons (in µm) onto a matplotlib axes.  Silently skips if gdstk is missing."""
    try:
        import gdstk
    except ImportError:
        return
    lib = gdstk.read_gds(str(gds_path))
    layer, datatype = gds_layer
    real_cells = [c for c in lib.cells if not c.name.startswith("$$$")]
    referenced = {ref.cell.name for c in real_cells for ref in c.references if hasattr(ref.cell, "name")}
    tops = [c for c in real_cells if c.name not in referenced] or real_cells
    for top in tops:
        for poly in top.get_polygons(layer=layer, datatype=datatype):
            pts = poly.points  # gdstk units are µm for gdsfactory GDS files
            ax.fill(pts[:, 0], pts[:, 1], facecolor=color, edgecolor="none", alpha=alpha, zorder=1)
