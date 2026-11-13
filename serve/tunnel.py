from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .config import Config
from .util import tail


class PiTunnel:
    """Expose the Pi's WebSocket relay through a persistent opentunnel.

    The identity lives under ``serve-state/opentunnel`` so restarting the Pi
    process reuses the same route. The public URL is written to
    ``serve-state/pi-tunnel-url.txt`` for pasting into the Kaggle launcher.
    """

    def __init__(self, cfg: Config, log, port: int):
        self.cfg = cfg
        self.log = log
        self.port = port
        self.state = "starting" if cfg.pi_tunnel else "disabled"
        self.url: str | None = None
        self.error: str | None = None
        self._task: asyncio.Task | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._pump: asyncio.Task | None = None
        self._stopping = False
        self._lock = asyncio.Lock()
        self.log_path = cfg.logs_dir / "tunnel.log"

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if not self.cfg.pi_tunnel:
            return
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._start())

    async def _start(self) -> None:
        async with self._lock:
            try:
                binary = await self._install_cli()
                await self._supervise(binary)
            except asyncio.CancelledError:
                await self._kill()
                raise
            except Exception as exc:
                self.state = "error"
                self.error = str(exc) or exc.__class__.__name__
                self._emit(f"opentunnel failed: {self.error}")
                self.log.warning("opentunnel setup failed: %s", self.error)

    async def _supervise(self, binary: str) -> None:
        delay = 2.0
        while not self._stopping:
            try:
                url = await self._route(binary)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state = "error"
                self.error = str(exc) or exc.__class__.__name__
                self._emit(f"opentunnel route failed: {self.error}")
                self.log.warning("opentunnel route failed: %s", self.error)
            else:
                self.url = url
                self.state = "ready"
                self.error = None
                relay_url = self.relay_url()
                self._write_url(relay_url)
                self.log.info("kaggle relay url: %s", relay_url)
                self._emit(f"kaggle relay url: {relay_url}")
                delay = 2.0

            proc = self._proc
            code = await proc.wait() if proc is not None else None
            if self._stopping:
                return
            if code == 0 and self.url is not None:
                self.state = "ready"
                return
            self.url = None
            self.state = "reconnecting"
            self._emit(f"opentunnel exited (code {code}); reconnecting in {delay:.0f}s")
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
            raise RuntimeError("opentunnel install failed (see the tunnel log)")
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
        argv = [
            binary,
            "route",
            "add",
            str(self.port),
            "--name",
            self.cfg.pi_tunnel_name,
        ]
        self._emit("$ " + " ".join(argv))
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self.cfg.opentunnel_env(),
        )
        self._proc = proc
        assert proc.stdout is not None
        found: asyncio.Future[str] = asyncio.get_running_loop().create_future()
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
                if (
                    stripped.startswith(("https://", "wss://", "http://", "ws://"))
                    and not found.done()
                ):
                    found.set_result(stripped)
            if not found.done():
                detail = " | ".join(lines[-4:]) or "no output"
                found.set_exception(
                    RuntimeError(f"opentunnel exited before printing a URL: {detail}")
                )

        self._pump = asyncio.create_task(pump())
        try:
            return await asyncio.wait_for(asyncio.shield(found), timeout=900)
        except asyncio.TimeoutError as exc:
            await self._kill()
            raise RuntimeError("timed out waiting for the opentunnel URL") from exc
        except BaseException:
            await self._kill()
            raise

    def relay_url(self) -> str | None:
        if not self.url:
            return None
        parsed = urlsplit(self.url)
        scheme = "wss" if parsed.scheme in {"https", "wss"} else "ws"
        base = parsed.path.rstrip("/")
        if base.endswith(self.cfg.pi_ws_path):
            path = base
        else:
            path = base + self.cfg.pi_ws_path
        return urlunsplit((scheme, parsed.netloc, path or self.cfg.pi_ws_path, "", ""))

    def _write_url(self, url: str | None) -> None:
        if not url:
            return
        try:
            self.cfg.pi_tunnel_url_file.parent.mkdir(parents=True, exist_ok=True)
            self.cfg.pi_tunnel_url_file.write_text(url + "\n")
        except OSError as exc:
            self.log.warning("could not write the tunnel url: %s", exc)

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

    async def stop(self) -> None:
        if not self.cfg.pi_tunnel:
            return
        self._stopping = True
        task, self._task = self._task, None
        if task and not task.done():
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
        self.state = "stopped"

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "url": self.url,
            "relay_url": self.relay_url(),
            "error": self.error,
            "log": str(self.log_path),
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
