from __future__ import annotations

import asyncio
import importlib.metadata
import os
import shutil
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from . import llamacpp
from .catalog import vllm_executable
from .config import Config
from .util import human_bytes, one_line

PACKAGE_FOR_ENGINE = {
    "vllm": ("vllm",),
    "vllm-omni": ("vllm", "vllm-omni"),
    # llama.cpp is a downloaded binary, not a pip package, so it has no
    # distribution version to look up.
    "llamacpp": (),
}

# Import the modules the `vllm serve` CLI touches before it reads any config.
# A version number on disk is not proof that the tree is importable: a stale
# module left by an earlier install (or a dependency downgraded by another
# package) only shows up here.
IMPORT_PROBE = (
    "import vllm, vllm.config.model, vllm.engine.arg_utils, "
    "vllm.entrypoints.serve.utils.api_utils; "
    "print('import ok', vllm.__version__)"
)


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
        self._verified: dict[str, str] = {}
        self.log_path = cfg.logs_dir / "install.log"

    # -- status ------------------------------------------------------------

    def status(self, engine: str) -> RuntimeState:
        task = self._tasks.get(engine)
        installing = bool(task and not task.done())
        if engine == "llamacpp":
            found = llamacpp.binary(self.cfg)
            return RuntimeState(
                engine=engine,
                installed=bool(found),
                version=self.cfg.llamacpp_build if found else None,
                installing=installing,
                error=self._errors.get(engine),
            )
        packages = PACKAGE_FOR_ENGINE.get(engine, ())
        versions = {p: installed_version(p) for p in packages}
        installed = all(versions.values()) if versions else False
        return RuntimeState(
            engine=engine,
            installed=installed,
            version=versions.get("vllm-omni") or versions.get("vllm"),
            installing=installing,
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
        await self._ensure_install(engine)

    async def _ensure_install(self, engine: str, reinstall: bool = False) -> None:
        task = self._tasks.get(engine)
        if task and not task.done():
            await asyncio.gather(task, return_exceptions=True)
            if not reinstall:
                return
        task = asyncio.create_task(self._install(engine, reinstall=reinstall))
        self._tasks[engine] = task
        await task

    async def prepare(self, engine: str) -> None:
        """Make sure the engine is installed *and* can actually start.

        Kaggle images ship their own vLLM, and a half-overwritten install still
        reports the new version number while some modules are old. Both cases
        end as ``engine exited with code 1`` inside ``import vllm``, so check
        the import here where we can reinstall instead of guessing. For
        llama.cpp the equivalent failure is a binary that cannot see CUDA.
        """
        if engine == "llamacpp":
            if llamacpp.binary(self.cfg) is None:
                self._emit(f"llama.cpp {self.cfg.llamacpp_build} is missing; installing")
                await self._ensure_install(engine)
            if self._verified.get(engine) == self.cfg.llamacpp_build:
                return
            ok, output = await llamacpp.verify(self.cfg)
            if not ok:
                for line in output.splitlines()[:20]:
                    self._emit(f"  {line}")
                self._emit("llama.cpp cannot see the GPU; fetching the CUDA runtime it needs")
                await self._ensure_install(engine, reinstall=True)
                ok, output = await llamacpp.verify(self.cfg)
                if not ok:
                    for line in output.splitlines()[:20]:
                        self._emit(f"  {line}")
                    raise RuntimeError(
                        f"llama.cpp {self.cfg.llamacpp_build} is installed but CUDA is not "
                        f"available: {output}"
                    )
            self._verified[engine] = self.cfg.llamacpp_build
            self._emit(f"llama.cpp check passed: {' '.join(output.split())}")
            return

        want = self.cfg.omni_version if engine == "vllm-omni" else self.cfg.vllm_version
        have = installed_version("vllm")
        if have != want:
            self._emit(f"vllm {have or 'is missing'}, want {want}; reinstalling")
            await self._ensure_install(engine, reinstall=True)
        elif self._verified.get(engine) == want:
            return

        ok, output = await self.verify()
        if not ok:
            self._emit("the vllm install cannot be imported; reinstalling cleanly")
            await self._ensure_install(engine, reinstall=True)
            ok, output = await self.verify()
            if not ok:
                raise RuntimeError(
                    f"vllm {installed_version('vllm')} is installed but cannot be imported:\n"
                    + output
                )
        self._verified[engine] = installed_version("vllm") or want

    async def verify(self) -> tuple[bool, str]:
        """Run the engine CLI's import stack in a throwaway process.

        The ``vllm`` console script imports every subcommand before it looks at
        arguments, so ``serve --help`` reproduces exactly what the engine does
        at startup, using the same interpreter the engine will use.
        """
        executable = vllm_executable(sys.executable)
        argv = (
            [executable, "serve", "--help"]
            if executable
            else [sys.executable, "-c", IMPORT_PROBE]
        )
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=dict(os.environ),
        )
        assert proc.stdout is not None
        try:
            raw = await asyncio.wait_for(proc.stdout.read(), timeout=600)
            code = await asyncio.wait_for(proc.wait(), timeout=60)
        except asyncio.TimeoutError:
            proc.kill()
            return False, "the vllm CLI did not answer within 600s"
        text = raw.decode("utf-8", errors="replace").strip()
        if code == 0:
            self._emit("import check passed")
            return True, text
        for line in text.splitlines()[-25:]:
            self._emit(line)
        self._emit(f"import check failed with exit code {code}")
        return False, text

    def start(self, engine: str) -> None:
        """Kick off an install without waiting for it."""
        if self.status(engine).installed:
            return
        task = self._tasks.get(engine)
        if task and not task.done():
            return
        self._tasks[engine] = asyncio.create_task(self._install(engine))

    async def _install(self, engine: str, reinstall: bool = False) -> None:
        self._errors.pop(engine, None)
        self._emit(f"preparing {engine}")
        try:
            await self._check_space(engine)
            if engine == "llamacpp":
                # install() fetches only what is missing, so a retry after a
                # failed CUDA check downloads the bundled runtime instead of
                # the 17 GB model or the binary again.
                found = await llamacpp.install(self.cfg, self._emit, force_cudart=reinstall)
                self._emit(f"llama.cpp {self.cfg.llamacpp_build} is ready at {found}")
                return
            await self._ensure_uv()
            steps = self._commands(engine, reinstall=reinstall)
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

    def _commands(self, engine: str, reinstall: bool = False) -> list[list[str]]:
        python = sys.executable
        uv = shutil.which("uv") or str(Path(sys.executable).with_name("uv"))
        steps: list[list[str]] = []
        if reinstall:
            # Remove the old files first. `uv pip install --reinstall` only
            # replaces what its own RECORD knows about, and Kaggle images ship
            # vLLM installed by pip, so a plain reinstall can leave modules
            # from the image behind and keep the import broken.
            packages = ["vllm"] if engine == "vllm" else ["vllm", "vllm-omni"]
            steps.append([python, "-m", "pip", "uninstall", "-y", *packages])
        install = [
            uv,
            "pip",
            "install",
            "--python",
            python,
            "--no-cache",
            "--torch-backend=auto",
            *(["--reinstall"] if reinstall else []),
            f"vllm=={self.cfg.vllm_version}",
        ]
        if engine != "vllm":
            install.append(f"vllm-omni=={self.cfg.omni_version}")
        steps.append(install)
        return steps

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
        need = {"vllm": 12, "vllm-omni": 17, "llamacpp": 2}[engine]
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
