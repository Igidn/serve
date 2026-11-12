from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

import httpx

from .config import Config
from .util import tail


class EngineCrash(RuntimeError):
    pass


REPORT_PACKAGES = ("vllm", "torch", "transformers", "tokenizers", "pydantic", "numpy")


def version_header() -> str:
    """One log line with the versions that decide whether vLLM can import."""
    import importlib.metadata

    report = []
    for package in REPORT_PACKAGES:
        try:
            report.append(f"{package}={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            report.append(f"{package}=missing")
    report.append(f"python={'.'.join(map(str, sys.version_info[:3]))}")
    return " ".join(report)


def engine_env(cfg: Config) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "HF_HOME": str(cfg.hf_home),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "DO_NOT_TRACK": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    if cfg.hf_token:
        env["HF_TOKEN"] = cfg.hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = cfg.hf_token
    return env


class EngineProcess:
    """One ``vllm serve`` child process, in its own session so it can be
    terminated as a group (vLLM spawns worker processes)."""

    def __init__(self, cfg: Config, port: int, log_path: Path, log):
        self.cfg = cfg
        self.port = port
        self.log_path = log_path
        self.log = log
        self.proc: asyncio.subprocess.Process | None = None
        self._handle = None

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc and self.proc.returncode is None else None

    @property
    def alive(self) -> bool:
        return bool(self.proc and self.proc.returncode is None)

    async def start(self, argv: list[str]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.log_path.open("wb", buffering=0)
        self._handle.write(("$ " + " ".join(argv) + "\n").encode())
        self._handle.write((version_header() + "\n").encode())
        self.proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=self._handle,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
            env=engine_env(self.cfg),
            cwd=str(self.cfg.root),
        )
        self.log.info("engine started pid=%s port=%s", self.proc.pid, self.port)

    async def wait_ready(self, timeout: float, cancelled: asyncio.Event | None = None) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        url = f"http://127.0.0.1:{self.port}/health"
        async with httpx.AsyncClient(timeout=3.0) as client:
            while True:
                if cancelled is not None and cancelled.is_set():
                    raise asyncio.CancelledError
                if not self.alive:
                    raise EngineCrash(self._crash_message())
                try:
                    response = await client.get(url)
                    if response.status_code < 500:
                        return
                except httpx.HTTPError:
                    pass
                if asyncio.get_running_loop().time() > deadline:
                    raise EngineCrash(
                        f"engine did not become healthy within {int(timeout)}s\n"
                        + tail(self.log_path, 40)
                    )
                await asyncio.sleep(2.0)

    def _crash_message(self) -> str:
        code = self.proc.returncode if self.proc else None
        return f"engine exited with code {code}\n" + tail(self.log_path, 40)

    async def stop(self, grace: float = 20.0) -> None:
        proc = self.proc
        if proc is None:
            return
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=grace)
            except asyncio.TimeoutError:
                self.log.warning("engine pid=%s did not exit; killing", proc.pid)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    pass
        self.proc = None
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None
        self.log.info("engine stopped")
