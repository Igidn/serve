"""llama.cpp runtime: prebuilt CUDA binaries and the checks around them.

vLLM needs a working Python import stack and, on Turing, falls back to slow
Triton kernels. llama.cpp ships a single binary that already knows how to use
compute 7.5, quantizes the KV cache, and never touches the 20 GB working-disk
quota with a 16 GB GGUF. The trade-off is that its release assets are not
installed by pip, so this module downloads them and proves CUDA works before
the engine starts.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import httpx

from .config import Config
from .util import one_line

README = "https://github.com/ggml-org/llama.cpp/releases"
BINARY = "llama-{build}-bin-ubuntu-{cuda}-x64.tar.gz"
CUDART = "cudart-llama-{build}-bin-ubuntu-{cuda}-x64.tar.gz"


def release_url(build: str, name: str) -> str:
    return f"{README}/download/{build}/{name}"


def install_dir(cfg: Config) -> Path:
    return cfg.state_dir / "runtime" / "llamacpp" / cfg.llamacpp_build


def binary(cfg: Config) -> Path | None:
    """The ``llama-server`` binary of the pinned build, if it is installed."""
    root = install_dir(cfg)
    if not root.is_dir():
        return None
    for path in sorted(root.rglob("llama-server")):
        try:
            if path.is_file() and os.access(path, os.X_OK):
                return path
        except OSError:
            continue
    return None


def lib_dirs(cfg: Config) -> list[str]:
    """Directories holding the shared objects the binary links against."""
    root = install_dir(cfg)
    if not root.is_dir():
        return []
    dirs: set[str] = set()
    for path in root.rglob("*.so*"):
        dirs.add(str(path.parent))
    return sorted(dirs)


def runtime_env(cfg: Config) -> dict[str, str]:
    dirs = lib_dirs(cfg)
    if not dirs:
        return {}
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return {"LD_LIBRARY_PATH": os.pathsep.join([*dirs, existing] if existing else dirs)}


def system_cuda_runtime() -> bool:
    """Whether the machine already has the CUDA runtime llama.cpp links to.

    Kaggle images ship the CUDA toolkit, so pulling the 600 MB bundled
    ``cudart`` archive is usually wasted quota. If the libraries are missing,
    the archive is fetched instead.
    """
    bases = [Path("/usr/local/cuda/lib64"), Path("/usr/lib/x86_64-linux-gnu")]
    bases += sorted(Path("/usr/local").glob("cuda*/lib64"))
    for base in bases:
        try:
            if not base.is_dir():
                continue
            if list(base.glob("libcudart.so*")) and list(base.glob("libcublas.so*")):
                return True
        except OSError:
            continue
    return False


async def download(url: str, dest: Path, emit) -> Path:
    emit(f"downloading {url}")
    async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
        async with client.stream("GET", url) as response:
            if response.status_code >= 400:
                raise RuntimeError(f"{url} returned HTTP {response.status_code}")
            total = int(response.headers.get("content-length") or 0)
            done = 0
            last_report = 0
            with dest.open("wb") as handle:
                async for chunk in response.aiter_bytes(4 * 1024**2):
                    handle.write(chunk)
                    done += len(chunk)
                    if total and done - last_report >= 256 * 1024**2:
                        last_report = done
                        emit(f"  {done * 100 // total}% of {total // 1024**2} MiB")
    return dest


def extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise RuntimeError(f"refusing to extract unsafe path {member.name}")
        # "tar" keeps the executable bit (the "data" filter would drop it)
        # while still refusing absolute paths and links that escape the target.
        try:
            tar.extractall(target, filter="tar")
        except TypeError:  # Python < 3.12
            tar.extractall(target)


async def install(cfg: Config, emit) -> Path:
    """Fetch the pinned build, verify it, and return the server binary."""
    found = binary(cfg)
    if found is not None:
        return found
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=cfg.state_dir, prefix="llamacpp-dl-"))
    patterns = [BINARY] if system_cuda_runtime() else [BINARY, CUDART]
    try:
        for pattern in patterns:
            name = pattern.format(build=cfg.llamacpp_build, cuda=cfg.llamacpp_cuda)
            archive = await download(release_url(cfg.llamacpp_build, name), staging / name, emit)
            extract(archive, install_dir(cfg))
            archive.unlink(missing_ok=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    found = binary(cfg)
    if found is None or not os.access(found, os.X_OK):
        raise RuntimeError(
            f"the {cfg.llamacpp_build} archive did not contain an executable llama-server"
        )
    emit(f"installed llama.cpp {cfg.llamacpp_build} at {found}")
    return found


async def verify(cfg: Config) -> tuple[bool, str]:
    """Prove the binary runs and sees a CUDA device.

    ``--list-devices`` loads every backend and exits, which is the cheapest way
    to catch a glibc mismatch or a backend that silently fell back to the CPU.
    """
    found = binary(cfg)
    if found is None:
        return False, "llama-server is not installed"
    proc = await asyncio.create_subprocess_exec(
        str(found),
        "--list-devices",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, **runtime_env(cfg)},
    )
    assert proc.stdout is not None
    try:
        raw = await asyncio.wait_for(proc.stdout.read(), timeout=180)
        code = await asyncio.wait_for(proc.wait(), timeout=30)
    except asyncio.TimeoutError:
        proc.kill()
        return False, "llama-server --list-devices did not answer within 180s"
    text = raw.decode("utf-8", errors="replace").strip()
    if code != 0:
        return False, text or f"exit code {code}"
    if "CUDA" not in text:
        return False, (
            "llama-server runs but sees no CUDA device; the CUDA libraries did not load:\n"
            + one_line(text, 800)
        )
    return True, text
