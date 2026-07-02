from __future__ import annotations

import dataclasses
import math


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
