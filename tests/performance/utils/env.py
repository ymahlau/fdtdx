from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path


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
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent,
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
