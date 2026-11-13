from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx

from . import relay
from .config import Config

try:  # websockets >= 13
    from websockets.asyncio.client import connect as ws_connect
except ImportError:  # pragma: no cover - older uvicorn[standard] pins
    from websockets.client import connect as ws_connect  # type: ignore

from websockets.exceptions import ConnectionClosed


class RelayClient:
    """Kaggle side of the relay.

    The kernel is behind NAT and has no stable address. It dials the Pi's
    persistent opentunnel WebSocket and then serves the local llama.cpp engine
    through that connection. Nothing listens on a public Kaggle port.
    """

    def __init__(
        self,
        cfg: Config,
        log,
        on_activity: Callable[[], None] | None = None,
        on_open: Callable[[], None] | None = None,
        on_close: Callable[[], None] | None = None,
    ):
        self.cfg = cfg
        self.log = log
        self.on_activity = on_activity
        self.on_open = on_open
        self.on_close = on_close
        self.state = "starting"
        self.error: str | None = None
        self._task: asyncio.Task | None = None
        self._ws = None
        self._stopping = False
        self._send_lock = asyncio.Lock()
        self._streams: dict[int, asyncio.Queue[relay.Frame]] = {}
        self._tasks: dict[int, asyncio.Task] = {}
        self._client: httpx.AsyncClient | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopping = True
        task, self._task = self._task, None
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._close()

    async def _close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass
        self._cancel_streams()

    def _cancel_streams(self) -> None:
        tasks, self._tasks = self._tasks, {}
        self._streams.clear()
        for task in tasks.values():
            task.cancel()

    # -- connection --------------------------------------------------------

    async def _run(self) -> None:
        delay = 1.0
        while not self._stopping:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=15.0, read=None, write=None, pool=None),
                    limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
                )
            try:
                await self._connect()
                delay = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state = "reconnecting"
                self.error = str(exc) or exc.__class__.__name__
                self.log.warning("relay disconnected: %s", self.error)
            finally:
                await self._close()
            if self._stopping:
                break
            self.log.info("reconnecting to %s in %.0fs", self.cfg.relay_url, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, 30.0)
        self.state = "stopped"

    async def _connect(self) -> None:
        self.state = "connecting"
        self.error = None
        self.log.info("connecting to raspi relay %s", self.cfg.relay_url)
        async with ws_connect(
            self.cfg.relay_url,
            max_size=relay.MAX_MESSAGE,
            ping_interval=20,
            ping_timeout=20,
            open_timeout=20,
            compression=None,
        ) as ws:
            self._ws = ws
            self.state = "connected"
            self.log.info("raspi relay connected")
            await ws.send(
                json.dumps({"hello": "serve-node", "token": self.cfg.relay_token or ""})
            )
            try:
                async for raw in ws:
                    if isinstance(raw, str):
                        continue
                    self._dispatch(ws, bytes(raw))
            except ConnectionClosed:
                pass
            finally:
                self._cancel_streams()
                self.state = "reconnecting"

    def _dispatch(self, ws, raw: bytes) -> None:
        try:
            frame = relay.decode(raw)
        except relay.ProtocolError as exc:
            self.log.warning("ignoring bad relay frame: %s", exc)
            return

        if frame.op == relay.OPEN:
            queue: asyncio.Queue[relay.Frame] = asyncio.Queue()
            old = self._tasks.pop(frame.stream, None)
            if old is not None:
                old.cancel()
            self._streams[frame.stream] = queue
            self._tasks[frame.stream] = asyncio.create_task(
                self._serve(ws, frame.stream, frame.payload, queue)
            )
            return

        if frame.op == relay.ABORT:
            task = self._tasks.pop(frame.stream, None)
            if task is not None:
                task.cancel()
            self._streams.pop(frame.stream, None)
            return

        queue = self._streams.get(frame.stream)
        if queue is not None:
            queue.put_nowait(frame)

    # -- one request -------------------------------------------------------

    async def _serve(self, ws, stream: int, payload: bytes, queue: asyncio.Queue) -> None:
        started = False
        try:
            method, path, headers = relay.parse_open(payload)
            body = bytearray()
            while True:
                frame = await queue.get()
                if frame.op == relay.DATA:
                    body.extend(frame.payload)
                elif frame.op == relay.END:
                    break
                elif frame.op == relay.ABORT:
                    return

            if self.on_activity:
                self.on_activity()
            if self.on_open:
                self.on_open()
            started = True

            url = f"http://127.0.0.1:{self.cfg.engine_port}{path}"
            assert self._client is not None
            async with self._client.stream(
                method, url, headers=headers, content=bytes(body)
            ) as response:
                await self._send(
                    ws,
                    relay.Frame(
                        relay.HEAD,
                        stream,
                        relay.head_payload(
                            response.status_code, relay.clean_headers(response.headers.items())
                        ),
                    ),
                )
                async for chunk in response.aiter_raw():
                    for offset in range(0, len(chunk), relay.CHUNK):
                        await self._send(
                            ws,
                            relay.Frame(relay.DATA, stream, chunk[offset : offset + relay.CHUNK]),
                        )
                await self._send(ws, relay.Frame(relay.END, stream))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = f"{exc.__class__.__name__}: {exc}".encode()
            try:
                await self._send(ws, relay.Frame(relay.ERROR, stream, message))
            except Exception:
                pass
            self.log.warning("relay request failed: %s", exc)
        finally:
            self._streams.pop(stream, None)
            self._tasks.pop(stream, None)
            if started and self.on_close:
                self.on_close()

    async def _send(self, ws, frame: relay.Frame) -> None:
        async with self._send_lock:
            if self._ws is not ws:
                raise RuntimeError("relay connection is gone")
            await ws.send(frame.encode())
