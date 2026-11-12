from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# gcsfuse (Kaggle's /kaggle/input) is fine for big sequential reads and awful
# for the 4 KB page faults safetensors mmap does, which is why loading 20 GB of
# weights straight from the mount takes ~40 minutes. Copying the checkpoint to
# the working disk once, with 16 MB preads from several threads, turns that
# into a few minutes and makes every reload afterwards instant.
CHUNK = 16 * 1024 * 1024
WORKERS = 8
MARKER = ".serve-staged.json"
PARTIAL = ".serve-staging"
SKIP_SUFFIXES = {".gguf", ".onnx", ".msgpack"}
NETWORK_FS = {"nfs", "nfs4", "lustre", "cifs", "smb3"}


def mount_fstype(path: Path) -> str | None:
    """Filesystem type behind ``path``; ``None`` where /proc is unavailable."""
    target = os.path.realpath(path)
    best: str | None = None
    fstype: str | None = None
    try:
        with open("/proc/mounts", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 3:
                    continue
                point = parts[1].replace("\\040", " ")
                if target != point and not target.startswith(point.rstrip("/") + "/"):
                    continue
                if best is None or len(point) > len(best):
                    best, fstype = point, parts[2]
    except OSError:
        return None
    return fstype


def is_network_mount(path: Path) -> bool:
    fstype = mount_fstype(path) or ""
    return fstype.startswith("fuse") or fstype in NETWORK_FS


def checkpoint_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for entry in sorted(root.rglob("*")):
        try:
            if not entry.is_file():
                continue
        except OSError:
            continue
        if entry.name.startswith(".") or entry.suffix.lower() in SKIP_SUFFIXES:
            continue
        files.append(entry)
    return files


def total_bytes(files: list[Path]) -> int:
    total = 0
    for path in files:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def ready(dest: Path) -> bool:
    """True when ``dest`` already holds a complete copy of a checkpoint."""
    marker = dest / MARKER
    if not marker.is_file():
        # A plain Hugging Face download leaves config.json behind and no
        # staging marker; a half-finished copy always leaves PARTIAL.
        return (dest / "config.json").is_file() and not (dest / PARTIAL).exists()
    try:
        recorded = json.loads(marker.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(recorded, dict) or not recorded:
        return False
    for name, size in recorded.items():
        path = dest / name
        try:
            if not path.is_file() or path.stat().st_size != size:
                return False
        except OSError:
            return False
    return True


def stage(src: Path, dest: Path, on_progress=None) -> Path:
    """Copy a checkpoint from a network mount into ``dest`` and mark it done."""
    files = checkpoint_files(src)
    if not files:
        raise RuntimeError(f"{src} has no files to copy")
    total = total_bytes(files)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / PARTIAL).write_text("staging\n")

    counted = [0]
    lock = threading.Lock()

    def advance(count: int) -> None:
        with lock:
            counted[0] += count
            done = counted[0]
        if on_progress:
            on_progress(done, total)

    for path in files:
        target = dest / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        size = path.stat().st_size
        if (
            target.is_file()
            and not target.with_name(target.name + ".part").exists()
            and target.stat().st_size == size
        ):
            advance(size)
            continue
        copy_file(path, target, advance)

    recorded = {}
    for path in files:
        recorded[str(path.relative_to(src))] = path.stat().st_size
    (dest / MARKER).write_text(json.dumps(recorded))
    (dest / PARTIAL).unlink(missing_ok=True)
    if on_progress:
        on_progress(total, total)
    return dest


def copy_file(src: Path, dst: Path, on_chunk=None, workers: int = WORKERS, chunk: int = CHUNK) -> None:
    """Copy one file with parallel ranged reads, then rename into place."""
    size = src.stat().st_size
    tmp = dst.with_name(dst.name + ".part")
    fd_in = os.open(src, os.O_RDONLY)
    try:
        fd_out = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.ftruncate(fd_out, size)

            def work(offset: int) -> None:
                remaining = min(chunk, size - offset)
                position = 0
                while remaining > 0:
                    data = os.pread(fd_in, remaining, offset + position)
                    if not data:
                        break
                    written = 0
                    while written < len(data):
                        written += os.pwrite(fd_out, data[written:], offset + position + written)
                    position += len(data)
                    remaining -= len(data)
                    if on_chunk is not None:
                        on_chunk(len(data))

            if size:
                with ThreadPoolExecutor(max_workers=min(workers, max(1, -(-size // chunk)))) as pool:
                    for _ in pool.map(work, range(0, size, chunk)):
                        pass
            os.fsync(fd_out)
        finally:
            os.close(fd_out)
    finally:
        os.close(fd_in)
    tmp.replace(dst)
