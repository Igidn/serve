from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

_cache: tuple[float, list[dict[str, Any]]] = (0.0, [])


def gpus(max_age: float = 2.0) -> list[dict[str, Any]]:
    """Return one record per visible GPU, using nvidia-smi.

    Cached briefly because the idle watchdog and startup log both sample it.
    Returns an empty list on a machine without NVIDIA tooling.
    """
    global _cache
    stamp, cached = _cache
    if time.time() - stamp < max_age:
        return cached
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        rows: list[dict[str, Any]] = []
        for line in out.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5:
                continue
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(float(parts[2])),
                    "memory_used_mb": int(float(parts[3])),
                    "utilization": int(float(parts[4])),
                }
            )
        _cache = (time.time(), rows)
        return rows
    except (OSError, subprocess.SubprocessError, ValueError):
        _cache = (time.time(), [])
        return []


def disk(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total": usage.total, "used": usage.used, "free": usage.free}


def driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        line = out.stdout.strip().splitlines()
        return line[0].strip() if line else None
    except (OSError, subprocess.SubprocessError):
        return None


def driver_major() -> int | None:
    version = driver_version()
    if not version:
        return None
    try:
        return int(version.split(".", 1)[0])
    except ValueError:
        return None
