from __future__ import annotations

import asyncio
import importlib.metadata
import os
import shutil
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .util import human_bytes, one_line

PACKAGE_FOR_ENGINE = {"vllm": ("vllm",), "vllm-omni": ("vllm", "vllm-omni")}


def installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass
class RuntimeState:
    engine: str
    installed: bool
    version: str | None
    installing: bool = False
    error: str | None = None


class RuntimeManager:
    """Installs vLLM / vLLM-Omni into the running Python environment.

    Installation is deliberately a first-class, visible action instead of
    something the kernel launcher does silently: on Kaggle it takes several
    minutes and several GB, and the dashboard should still be reachable while
    it runs.
    """

    def __init__(self, cfg: Config, log):
        self.cfg = cfg
        self.log = log
        self._tasks: dict[str, asyncio.Task] = {}
        self._errors: dict[str, str] = {}
        self._lines: deque[str] = deque(maxlen=300)
        self.log_path = cfg.logs_dir / "install.log"

    # -- status ------------------------------------------------------------

    def status(self, engine: str) -> RuntimeState:
        packages = PACKAGE_FOR_ENGINE.get(engine, ())
        versions = {p: installed_version(p) for p in packages}
        installed = all(versions.values()) if versions else False
        task = self._tasks.get(engine)
        return RuntimeState(
            engine=engine,
            installed=installed,
            version=versions.get("vllm-omni") or versions.get("vllm"),
            installing=bool(task and not task.done()),
            error=self._errors.get(engine),
        )

    def all_status(self) -> list[dict]:
        return [self.status(engine).__dict__ for engine in PACKAGE_FOR_ENGINE]

    def lines(self, limit: int = 60) -> list[str]:
        return list(self._lines)[-limit:]

    # -- install -----------------------------------------------------------

    async def ensure(self, engine: str) -> None:
        if self.status(engine).installed:
            return
        task = self._tasks.get(engine)
        if task and not task.done():
            await task
            return
        task = asyncio.create_task(self._install(engine))
        self._tasks[engine] = task
        await task

    def start(self, engine: str) -> None:
        """Kick off an install without waiting for it."""
        if self.status(engine).installed:
            return
        task = self._tasks.get(engine)
        if task and not task.done():
            return
        self._tasks[engine] = asyncio.create_task(self._install(engine))

    async def _install(self, engine: str) -> None:
        self._errors.pop(engine, None)
        self._emit(f"preparing {engine}")
        try:
            await self._ensure_uv()
            await self._check_space(engine)
            steps = self._commands(engine)
            for argv in steps:
                self._emit("$ " + " ".join(argv))
                await self._run(argv)
            state = self.status(engine)
            if not state.installed:
                raise RuntimeError(
                    f"{engine} finished installing but the package is still missing"
                )
            self._emit(f"{engine} is ready ({state.version})")
        except Exception as exc:  # surfaced in the dashboard
            message = one_line(str(exc), 500)
            self._errors[engine] = message
            self._emit(f"install failed: {message}")
            self.log.warning("runtime install failed for %s: %s", engine, exc)
            raise

    def _commands(self, engine: str) -> list[list[str]]:
        python = sys.executable
        uv = shutil.which("uv") or str(Path(sys.executable).with_name("uv"))
        if engine == "vllm":
            return [
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    python,
                    "--no-cache",
                    "--torch-backend=auto",
                    f"vllm=={self.cfg.vllm_version}",
                ]
            ]
        return [
            [
                uv,
                "pip",
                "install",
                "--python",
                python,
                "--no-cache",
                "--torch-backend=auto",
                f"vllm=={self.cfg.vllm_version}",
                f"vllm-omni=={self.cfg.omni_version}",
            ]
        ]

    async def _ensure_uv(self) -> None:
        if shutil.which("uv"):
            return
        self._emit("installing uv (used for the vLLM install)")
        await self._run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--quiet", "uv"]
        )
        if not shutil.which("uv"):
            local = Path(sys.executable).with_name("uv")
            if not local.is_file():
                raise RuntimeError("uv was installed but is not on PATH")

    async def _check_space(self, engine: str) -> None:
        need = 12 if engine == "vllm" else 17
        free = shutil.disk_usage(self.cfg.state_dir).free
        if free < need * 1024**3:
            raise RuntimeError(
                f"{engine} needs roughly {need} GB during install, only "
                f"{human_bytes(free)} is free. Delete cached models or use a bigger working disk."
            )

    async def _run(self, argv: list[str]) -> None:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=dict(os.environ),
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                self._emit(line)
        code = await proc.wait()
        if code != 0:
            raise RuntimeError(f"command failed with exit code {code}")

    def _emit(self, line: str) -> None:
        for chunk in line.splitlines():
            self._lines.append(chunk)
        try:
            with self.log_path.open("a") as handle:
                handle.write(line + "\n")
        except OSError:
            pass
        self.log.info("runtime: %s", line)
