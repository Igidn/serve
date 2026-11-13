from __future__ import annotations

import asyncio
import hmac
import itertools
import json
import logging
import sys
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__, relay
from .config import Config
from .tunnel import PiTunnel

try:  # websockets >= 13
    from websockets.asyncio.server import serve as ws_serve
except ImportError:  # pragma: no cover - older uvicorn[standard] pins
    from websockets.legacy.server import serve as ws_serve  # type: ignore

from websockets.exceptions import ConnectionClosed

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class RelayOffline(RuntimeError):
    pass


class RelayHub:
    """The single live Kaggle connection and its in-flight streams."""

    def __init__(self, token: str | None, log):
        self.token = token
        self.log = log
        self._ws = None
        self._send_lock = asyncio.Lock()
        self._streams: dict[int, asyncio.Queue[relay.Frame]] = {}
        self._ids = itertools.count(1)
        self._connected_at: float | None = None

    @property
    def connected(self) -> bool:
        return self._ws is not None

    @property
    def uptime(self) -> float | None:
        return None if self._connected_at is None else time.time() - self._connected_at

    def authorized(self, hello: str) -> bool:
        if not self.token:
            return True
        try:
            data = json.loads(hello)
            candidate = str(data.get("token", ""))
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(candidate.encode(), self.token.encode())

    async def attach(self, ws) -> None:
        old, self._ws = self._ws, ws
        self._connected_at = time.time()
        self.log.info("kaggle node connected")
        if old is not None and old is not ws:
            try:
                await old.close(code=1012, reason="replaced by a newer node")
            except Exception:
                pass

    async def detach(self, ws) -> None:
        if self._ws is not ws:
            return
        self._ws = None
        self._connected_at = None
        self.log.info("kaggle node disconnected")
        for stream, queue in list(self._streams.items()):
            queue.put_nowait(relay.Frame(relay.ERROR, stream, b"kaggle node disconnected"))
        self._streams.clear()

    async def receive(self, ws, raw: bytes) -> None:
        if self._ws is not ws:
            return
        try:
            frame = relay.decode(raw)
        except relay.ProtocolError as exc:
            self.log.warning("ignoring bad relay frame: %s", exc)
            return
        if frame.op == relay.OPEN:
            # Only the Kaggle node opens streams; ignore a stray OPEN.
            return
        queue = self._streams.get(frame.stream)
        if queue is not None:
            queue.put_nowait(frame)

    async def send(self, frame: relay.Frame) -> None:
        ws = self._ws
        if ws is None:
            raise RelayOffline("the kaggle node is not connected")
        async with self._send_lock:
            if self._ws is not ws:
                raise RelayOffline("the kaggle node disconnected")
            await ws.send(frame.encode())

    def open_stream(self) -> int:
        stream = next(self._ids)
        self._streams[stream] = asyncio.Queue()
        return stream

    def close_stream(self, stream: int) -> None:
        self._streams.pop(stream, None)

    async def next_frame(self, stream: int, timeout: float | None = None) -> relay.Frame | None:
        queue = self._streams.get(stream)
        if queue is None:
            return None
        try:
            if timeout is None:
                return await queue.get()
            return await asyncio.wait_for(queue.get(), timeout)
        except asyncio.TimeoutError:
            return None


def offline() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": "The Kaggle node is not connected to the Pi relay.",
                "type": "service_unavailable",
                "code": "relay_offline",
            }
        },
    )


def create_http_app(hub: RelayHub) -> FastAPI:
    app = FastAPI(title="serve-pi", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def index():
        return {
            "name": "serve relay",
            "connected": hub.connected,
            "uptime": hub.uptime,
            "endpoints": {"openai": "/v1", "health": "/health"},
        }

    @app.get("/health")
    async def health():
        if not hub.connected:
            return JSONResponse(
                status_code=503,
                content={"ok": False, "connected": False, "uptime": None},
            )
        return {"ok": True, "connected": True, "uptime": hub.uptime}

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def forward(path: str, request: Request):
        if not hub.connected:
            return offline()

        target = "/v1/" + path
        if request.url.query:
            target += "?" + request.url.query

        stream = hub.open_stream()
        headers = relay.clean_headers(request.headers.items())

        try:
            body = await request.body()
            await hub.send(
                relay.Frame(
                    relay.OPEN,
                    stream,
                    relay.open_payload(request.method, target, headers),
                )
            )
            if body:
                for offset in range(0, len(body), relay.CHUNK):
                    await hub.send(
                        relay.Frame(relay.DATA, stream, body[offset : offset + relay.CHUNK])
                    )
            await hub.send(relay.Frame(relay.END, stream))

            first = await hub.next_frame(stream, timeout=600)
            if first is None:
                raise RelayOffline("the kaggle node did not answer")
            if first.op == relay.ERROR:
                raise RelayOffline(first.payload.decode("utf-8", errors="replace"))
            if first.op != relay.HEAD:
                raise RelayOffline(f"unexpected relay frame {first.op}")

            status, response_headers = relay.parse_head(first.payload)
        except RelayOffline as exc:
            hub.close_stream(stream)
            try:
                await hub.send(relay.Frame(relay.ABORT, stream))
            except Exception:
                pass
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"relay request failed: {exc}",
                        "type": "upstream_error",
                        "code": "bad_gateway",
                    }
                },
            )
        except Exception as exc:
            hub.close_stream(stream)
            try:
                await hub.send(relay.Frame(relay.ABORT, stream))
            except Exception:
                pass
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"relay request failed: {exc}",
                        "type": "upstream_error",
                        "code": "bad_gateway",
                    }
                },
            )

        async def body_stream():
            finished = False
            try:
                while True:
                    frame = await hub.next_frame(stream)
                    if frame is None:
                        break
                    if frame.op == relay.DATA:
                        yield frame.payload
                    elif frame.op in {relay.END, relay.ABORT, relay.ERROR}:
                        finished = True
                        break
            finally:
                hub.close_stream(stream)
                if not finished:
                    try:
                        await hub.send(relay.Frame(relay.ABORT, stream))
                    except Exception:
                        pass

        return StreamingResponse(
            body_stream(), status_code=status, headers=response_headers
        )

    return app


async def websocket_handler(hub: RelayHub, websocket, path: str | None = None) -> None:
    try:
        hello = await asyncio.wait_for(websocket.recv(), timeout=15)
    except (asyncio.TimeoutError, ConnectionClosed):
        try:
            await websocket.close(code=1008, reason="missing hello")
        except Exception:
            pass
        return

    if not isinstance(hello, str) or not hub.authorized(hello):
        hub.log.warning("rejected relay connection with a bad token")
        try:
            await websocket.close(code=1008, reason="bad relay token")
        except Exception:
            pass
        return

    await hub.attach(websocket)
    try:
        async for message in websocket:
            if isinstance(message, str):
                continue
            await hub.receive(websocket, bytes(message))
    except ConnectionClosed:
        pass
    finally:
        await hub.detach(websocket)


def setup_logging(cfg: Config) -> None:
    fmt = logging.Formatter(LOG_FORMAT)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    file = logging.FileHandler(cfg.logs_dir / "pi.log")
    file.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers[:] = [stream, file]
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "uvicorn.error", "uvicorn.access", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def run(cfg: Config) -> None:
    log = logging.getLogger("serve.pi")
    hub = RelayHub(cfg.relay_token, logging.getLogger("serve.relay"))
    tunnel = PiTunnel(cfg, logging.getLogger("serve.tunnel"), cfg.pi_ws_port)

    http_app = create_http_app(hub)
    server = uvicorn.Server(
        uvicorn.Config(
            http_app,
            host=cfg.pi_http_host,
            port=cfg.pi_http_port,
            log_config=None,
            access_log=False,
        )
    )

    ws_kwargs = {
        "host": cfg.pi_ws_host,
        "port": cfg.pi_ws_port,
        "max_size": relay.MAX_MESSAGE,
        "ping_interval": 20,
        "ping_timeout": 20,
        "compression": None,
    }
    async with ws_serve(lambda ws, path=None: websocket_handler(hub, ws, path), **ws_kwargs):
        log.info("harness endpoint: http://%s:%s/v1", cfg.pi_http_host, cfg.pi_http_port)
        log.info("kaggle websocket: ws://%s:%s%s", cfg.pi_ws_host, cfg.pi_ws_port, cfg.pi_ws_path)
        if cfg.pi_tunnel:
            tunnel.start()
        try:
            await server.serve()
        finally:
            await tunnel.stop()


def main() -> None:
    cfg = Config.from_env().ensure()
    setup_logging(cfg)
    try:
        asyncio.run(run(cfg))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
