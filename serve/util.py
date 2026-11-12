from __future__ import annotations

import time
from pathlib import Path


def now() -> float:
    return time.time()


def human_bytes(value: float | int | None) -> str:
    if not value:
        return "0 B"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TiB"


def tail(path: Path, lines: int = 200, max_bytes: int = 512 * 1024) -> str:
    """Return the last ``lines`` lines of a file without reading all of it."""
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    if size == 0:
        return ""
    start = max(0, size - max_bytes)
    try:
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    if start > 0:
        # Drop the partially read first line.
        text = text.split("\n", 1)[-1]
    return "\n".join(text.splitlines()[-lines:])


def dir_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def one_line(text: str, limit: int = 240) -> str:
    flat = " ".join(text.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")
