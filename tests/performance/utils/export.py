from __future__ import annotations

import json
from pathlib import Path

from .results import BenchmarkResult


def export_results(results: list[BenchmarkResult], run_dir: Path) -> None:
    """Write benchmark results to ``run_dir/results.json`` and print a summary table."""
    if not results:
        return

    env = results[0].env
    ts = run_dir.name.split("_")[0]
    payload = {
        "session_timestamp": ts,
        "env": env,
        "results": [r.to_dict() for r in results],
    }
    out_path = run_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2))

    header = f"{'benchmark':<35} {'grid':>18} {'steps':>7} {'compile':>9} {'run':>9} {'MCUPS':>9}"
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        grid_str = f"{r.grid_shape[0]}x{r.grid_shape[1]}x{r.grid_shape[2]}"
        rows.append(
            f"{r.name:<35} {grid_str:>18} {r.time_steps:>7} "
            f"{r.compile_seconds:>8.2f}s {min(r.run_seconds):>8.3f}s {r.mcups:>8.1f}"
        )
    rows.append(sep)
    print("\n" + "\n".join(rows))
    print(f"Results written to {out_path}")
