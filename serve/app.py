from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .catalog import ENGINES, Catalog, ModelSpec
from .config import Config
from .idle import IdleWatchdog
from .manager import ModelManager
from .runtime import RuntimeManager
from .settings import Settings
from .tunnel import Tunnel, tunnel_tail
from .util import tail

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

STATIC_DIR = Path(__file__).parent / "static"


class ShutdownController:
    def __init__(self, cfg: Config, manager: ModelManager, tunnel: Tunnel, log):
        self.cfg = cfg
        self.manager = manager
        self.tunnel = tunnel
        self.log = log
        self._started = False

    async def __call__(self, reason: str) -> None:
        if self._started:
            return
        self._started = True
        self.log.warning("shutting down: %s", reason)
        try:
            await self.manager.shutdown()
        except Exception as exc:
            self.log.warning("error stopping the engine: %s", exc)
        try:
            await self.tunnel.stop()
        except Exception as exc:
            self.log.warning("error stopping the tunnel: %s", exc)
        if self.cfg.allow_exit:
            # Let the HTTP response (if any) flush, then end the Kaggle run.
            await asyncio.sleep(1.0)
            os._exit(0)


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = (cfg or Config.from_env()).ensure()
    # Make Hugging Face keep its metadata inside the kernel's working directory.
    os.environ.setdefault("HF_HOME", str(cfg.hf_home))
    log = logging.getLogger("serve")
    catalog = Catalog(cfg.custom_models_file)
    runtime = RuntimeManager(cfg, logging.getLogger("serve.runtime"))
    manager = ModelManager(cfg, catalog, runtime, logging.getLogger("serve.engine"))
    tunnel = Tunnel(cfg, logging.getLogger("serve.tunnel"))
    settings = Settings(cfg.settings_file, cfg.idle_timeout)
    manager.idle_timeout = settings.idle_timeout
    shutdown = ShutdownController(cfg, manager, tunnel, log)
    watchdog = IdleWatchdog(manager, shutdown, logging.getLogger("serve.idle"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=None, write=None, pool=None),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        )
        tunnel.start()
        watchdog.start()
        try:
            yield
        finally:
            await watchdog.stop()
            await manager.shutdown()
            await tunnel.stop()
            await app.state.client.aclose()

    app = FastAPI(title="serve", version=__version__, lifespan=lifespan)
    app.state.cfg = cfg
    app.state.catalog = catalog
    app.state.manager = manager
    app.state.runtime = runtime
    app.state.tunnel = tunnel
    app.state.settings = settings
    app.state.shutdown = shutdown
    app.state.watchdog = watchdog

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- proxy ---------------------------------------------------------

    def no_model() -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "No model is loaded. Pick one in the serve dashboard.",
                    "type": "invalid_request_error",
                    "code": "no_model",
                }
            },
        )

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy_v1(path: str, request: Request):
        manager.touch()
        if manager.phase != "ready" or manager.engine is None or not manager.engine.alive:
            return no_model()

        url = f"http://127.0.0.1:{cfg.engine_port}/v1/{path}"
        if request.url.query:
            url += "?" + request.url.query
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in HOP_BY_HOP
        }
        client: httpx.AsyncClient = app.state.client
        manager.inflight += 1
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                manager.inflight = max(0, manager.inflight - 1)

        try:
            upstream = await client.send(
                client.build_request(
                    request.method,
                    url,
                    headers=headers,
                    content=request.stream(),
                ),
                stream=True,
            )
        except httpx.HTTPError as exc:
            release()
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"the inference engine is unreachable: {exc}",
                        "type": "upstream_error",
                        "code": "bad_gateway",
                    }
                },
            )

        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in HOP_BY_HOP
        }

        async def body():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                release()
                await upstream.aclose()

        return StreamingResponse(
            body(), status_code=upstream.status_code, headers=response_headers
        )

    # -- control API ---------------------------------------------------

    @app.get("/api/state")
    async def state():
        snapshot = manager.snapshot()
        snapshot["tunnel"] = tunnel.snapshot()
        snapshot["settings"] = {"idle_timeout": manager.idle_timeout}
        snapshot["app"] = {
            "version": __version__,
            "port": cfg.app_port,
            "root": str(cfg.root),
            "allow_exit": cfg.allow_exit,
        }
        return snapshot

    @app.get("/api/tunnel/identity")
    async def tunnel_identity():
        bundle = tunnel.identity_b64()
        if not bundle:
            return JSONResponse(
                status_code=404,
                content={"error": "the tunnel identity does not exist yet"},
            )
        return {"b64": bundle}

    @app.post("/api/tunnel/retry")
    async def tunnel_retry():
        if not cfg.tunnel_enabled:
            return JSONResponse(status_code=400, content={"error": "tunnel is disabled"})
        await tunnel.retry()
        return {"ok": True}

    @app.get("/api/logs")
    async def logs(source: str = "engine", lines: int = 200):
        lines = max(10, min(2000, lines))
        if source == "engine":
            return {"source": source, "text": tail(cfg.logs_dir / "engine.log", lines)}
        if source == "tunnel":
            return {"source": source, "text": tunnel_tail(cfg, lines)}
        if source == "install":
            text = "\n".join(runtime.lines(lines))
            return {"source": source, "text": text or tail(cfg.logs_dir / "install.log", lines)}
        if source == "app":
            return {"source": source, "text": tail(cfg.logs_dir / "app.log", lines)}
        return JSONResponse(status_code=400, content={"error": f"unknown source {source}"})

    @app.post("/api/models/{key}/load")
    async def load_model(key: str, request: Request):
        payload = await _json(request)
        try:
            await manager.load(key, replace=bool(payload.get("replace", False)), actor="dashboard")
        except KeyError:
            return JSONResponse(status_code=404, content={"error": f"unknown model {key}"})
        except ValueError as exc:
            return JSONResponse(status_code=409, content={"error": str(exc)})
        return {"ok": True, "key": key}

    @app.post("/api/models/unload")
    async def unload_model():
        await manager.unload()
        return {"ok": True}

    @app.post("/api/models/{key}/download")
    async def download_model(key: str):
        try:
            await manager.download(key)
        except KeyError:
            return JSONResponse(status_code=404, content={"error": f"unknown model {key}"})
        except ValueError as exc:
            return JSONResponse(status_code=409, content={"error": str(exc)})
        return {"ok": True, "key": key}

    @app.delete("/api/models/{key}/cache")
    async def delete_model(key: str):
        try:
            await manager.delete(key)
        except KeyError:
            return JSONResponse(status_code=404, content={"error": f"unknown model {key}"})
        except ValueError as exc:
            return JSONResponse(status_code=409, content={"error": str(exc)})
        return {"ok": True, "key": key}

    @app.post("/api/models")
    async def add_model(request: Request):
        payload = await _json(request)
        try:
            if not payload.get("key"):
                payload["key"] = _key_from_repo(str(payload.get("repo", "")))
            payload.setdefault("kind", "chat")
            payload.setdefault(
                "engine", "vllm-omni" if payload.get("kind") == "image" else "vllm"
            )
            payload.setdefault("label", payload["key"].replace("-", " ").replace("_", " "))
            spec = ModelSpec.from_dict(payload)
            catalog.add_custom(spec)
        except (TypeError, ValueError) as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        return spec.to_dict()

    @app.delete("/api/models/{key}")
    async def remove_model(key: str):
        try:
            catalog.remove_custom(key)
        except ValueError as exc:
            return JSONResponse(status_code=409, content={"error": str(exc)})
        return {"ok": True}

    @app.post("/api/runtime/{engine}/install")
    async def install_runtime(engine: str):
        if engine not in ENGINES:
            return JSONResponse(status_code=400, content={"error": f"unknown engine {engine}"})
        runtime.start(engine)
        manager.touch()
        return runtime.status(engine).__dict__

    @app.put("/api/settings")
    async def update_settings(request: Request):
        payload = await _json(request)
        try:
            settings.update(idle_timeout=payload.get("idle_timeout"))
        except (TypeError, ValueError) as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        manager.set_idle_timeout(settings.idle_timeout)
        return {"idle_timeout": settings.idle_timeout}

    @app.post("/api/shutdown")
    async def poweroff():
        asyncio.create_task(shutdown("requested from the dashboard"))
        return {"ok": True, "message": "stopping the engine and ending the kernel run"}

    @app.get("/api/health")
    async def health():
        return {"ok": True, "phase": manager.phase}

    # -- dashboard -----------------------------------------------------

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico")
    async def favicon():
        return JSONResponse(status_code=204, content=None)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


async def _json(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _key_from_repo(repo: str) -> str:
    name = repo.split("/", 1)[-1].lower()
    key = "".join(ch if ch.isalnum() else "-" for ch in name)
    while "--" in key:
        key = key.replace("--", "-")
    return key.strip("-")[:64] or "model"
