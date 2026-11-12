from __future__ import annotations

import asyncio
import base64
import io
import shutil
import tarfile
from pathlib import Path

from .config import Config
from .util import tail


class Tunnel:
    """Installs the opentunnel CLI and exposes the dashboard on a public URL.

    The tunnel identity (private key, token, certificate) is written under the
    kernel's working directory. Kaggle throws that away when the run ends, so
    the bundle is also returned as base64: store it in a Kaggle Secret named
    ``OPENTUNNEL_IDENTITY_B64`` and the URL stays the same across kernels.
    """

    def __init__(self, cfg: Config, log):
        self.cfg = cfg
        self.log = log
        self.state = "disabled" if not cfg.tunnel_enabled else "starting"
        self.url: str | None = None
        self.error: str | None = None
        self._task: asyncio.Task | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._pump: asyncio.Task | None = None
        self._stopping = False
        self._lock = asyncio.Lock()
        self.log_path = cfg.logs_dir / "tunnel.log"

    # -- identity ----------------------------------------------------------

    def restore_identity(self) -> None:
        raw = self.cfg.tunnel_identity
        if not raw:
            return
        try:
            blob = base64.b64decode(raw)
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
                try:
                    archive.extractall(self.cfg.opentunnel_data, filter="data")
                except TypeError:  # Python < 3.12 without the PEP 706 backport
                    archive.extractall(self.cfg.opentunnel_data)
            self._emit("restored opentunnel identity from OPENTUNNEL_IDENTITY_B64")
        except Exception as exc:
            self._emit(f"could not restore tunnel identity: {exc}")

    def identity_b64(self) -> str | None:
        base = self.cfg.opentunnel_data / "opentunnel"
        if not base.is_dir():
            return None
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            archive.add(base, arcname="opentunnel")
        return base64.b64encode(buffer.getvalue()).decode()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if not self.cfg.tunnel_enabled:
            return
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._start())

    async def retry(self) -> None:
        task = self._task
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._kill()
        self.start()

    async def _start(self) -> None:
        async with self._lock:
            self.state = "starting"
            self.error = None
            try:
                self.restore_identity()
                binary = await self._install_cli()
                await self._supervise(binary)
            except asyncio.CancelledError:
                await self._kill()
                raise
            except Exception as exc:
                self.state = "error"
                self.error = str(exc) or exc.__class__.__name__
                self._emit(f"tunnel failed: {self.error}")
                self.log.warning("tunnel setup failed: %s", self.error)

    async def _supervise(self, binary: str) -> None:
        """Keep a tunnel process alive for as long as the app runs.

        ``opentunnel route add`` does not detach: the process is the tunnel.
        So a URL on stdout means "ready", and an exit means "reconnect", not
        "broken". Restarting is cheap because the identity is on disk.
        """
        delay = 2.0
        while not self._stopping:
            try:
                url = await self._route(binary)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state = "error"
                self.error = str(exc) or exc.__class__.__name__
                self._emit(f"tunnel route failed: {self.error}")
                self.log.warning("tunnel setup failed: %s", self.error)
            else:
                self.url = url
                self.state = "ready"
                self.error = None
                self.log.info("tunnel ready at %s", url)
                self._emit(f"dashboard: {url}")
                delay = 2.0

            proc = self._proc
            code = await proc.wait() if proc is not None else None
            if self._stopping:
                return
            self.url = None
            self.state = "reconnecting"
            self._emit(f"tunnel exited (code {code}); reconnecting in {delay:.0f}s")
            self.log.warning("tunnel exited (code %s); reconnecting", code)
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, 60.0)

    async def _install_cli(self) -> str:
        binary = shutil.which("opentunnel", path=self.cfg.opentunnel_env()["PATH"])
        if binary:
            return binary
        self._emit("installing the opentunnel CLI")
        proc = await asyncio.create_subprocess_shell(
            "curl -fsSL https://opentunnel.xyz/install | sh",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self.cfg.opentunnel_env(),
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            self._emit(raw.decode("utf-8", errors="replace").rstrip())
        if await proc.wait() != 0:
            raise RuntimeError("opentunnel install failed (see the log)")
        for candidate in (
            Path.home() / ".local" / "bin" / "opentunnel",
            Path("/usr/local/bin/opentunnel"),
        ):
            if candidate.is_file():
                return str(candidate)
        found = shutil.which("opentunnel", path=self.cfg.opentunnel_env()["PATH"])
        if not found:
            raise RuntimeError("opentunnel installed but was not found on PATH")
        return found

    async def _route(self, binary: str) -> str:
        """Start the tunnel and return its URL as soon as it is printed.

        The process is left running on purpose; ``_supervise`` waits on it and
        restarts it when it dies. Waiting for it to exit here is what used to
        turn a working tunnel into ``tunnel failed:`` after thirty seconds.
        """
        argv = [
            binary,
            "route",
            "add",
            str(self.cfg.app_port),
            "--name",
            self.cfg.route_name,
        ]
        self._emit("$ " + " ".join(argv))
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=self.cfg.opentunnel_env(),
            )
        except:
            proc = await asyncio.create_subprocess_exec(
                "opentunnel up",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        self._proc = proc
        assert proc.stdout is not None
        url: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        lines: list[str] = []

        async def pump() -> None:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                lines.append(line)
                if len(lines) > 80:
                    lines.pop(0)
                self._emit(line)
                stripped = line.strip()
                if stripped.startswith("https://") and not url.done():
                    url.set_result(stripped)
            if not url.done():
                detail = " | ".join(lines[-4:]) or "no output"
                url.set_exception(
                    RuntimeError(f"opentunnel exited before printing a URL: {detail}")
                )

        # Kept referenced: cancelling the drain would close the pipe the
        # tunnel process is still writing to.
        self._pump = asyncio.create_task(pump())
        try:
            return await asyncio.wait_for(asyncio.shield(url), timeout=900)
        except asyncio.TimeoutError as exc:
            await self._kill()
            raise RuntimeError("timed out waiting for the tunnel certificate") from exc
        except BaseException:
            await self._kill()
            raise

    async def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=10)
        except (ProcessLookupError, asyncio.TimeoutError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    async def stop(self, keep_state: bool = False) -> None:
        self._stopping = True
        task = self._task
        if task and not task.done() and not keep_state:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._kill()
        binary = shutil.which("opentunnel", path=self.cfg.opentunnel_env()["PATH"])
        if binary:
            try:
                proc = await asyncio.create_subprocess_exec(
                    binary,
                    "down",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    env=self.cfg.opentunnel_env(),
                )
                await asyncio.wait_for(proc.wait(), timeout=30)
            except (OSError, asyncio.TimeoutError):
                pass
        if not keep_state:
            self.state = "stopped"

    # -- reporting ---------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "url": self.url,
            "error": self.error,
            "log": str(self.log_path),
            "has_identity": self.cfg.opentunnel_profile_dir.is_dir(),
        }

    def _emit(self, line: str) -> None:
        try:
            with self.log_path.open("a") as handle:
                handle.write(line + "\n")
        except OSError:
            pass
        self.log.info("tunnel: %s", line)


def tunnel_tail(cfg: Config, lines: int = 120) -> str:
    return tail(cfg.logs_dir / "tunnel.log", lines)
