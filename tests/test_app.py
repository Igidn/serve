from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from serve.app import create_app


class FakeEngine:
    alive = True

    async def stop(self):
        self.alive = False


def start_upstream(port: int) -> uvicorn.Server:
    upstream = FastAPI()

    @upstream.get("/health")
    async def health():
        return {"ok": True}

    @upstream.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()

        async def stream():
            for index in range(3):
                yield f"data: {payload['model']}:{index}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    server = uvicorn.Server(
        uvicorn.Config(upstream, host="127.0.0.1", port=port, log_level="error")
    )
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            return server
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("upstream did not start")


@pytest.fixture
def client(cfg):
    app = create_app(cfg)
    with TestClient(app) as test_client:
        yield test_client, app


def test_state_reports_idle_gateway(client) -> None:
    test_client, _ = client
    payload = test_client.get("/api/state").json()
    assert payload["phase"] == "idle"
    assert payload["endpoints"]["base"] == "/v1"
    assert payload["tunnel"]["state"] == "disabled"


def test_inference_without_a_model_is_a_clean_503(client) -> None:
    test_client, _ = client
    response = test_client.get("/v1/models")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "no_model"


def test_proxy_streams_from_the_engine_and_tracks_activity(client, cfg) -> None:
    test_client, app = client
    manager = app.state.manager
    server = start_upstream(cfg.engine_port)
    try:
        manager.phase = "ready"
        manager.engine = FakeEngine()
        manager.last_activity = 0

        response = test_client.post(
            "/v1/chat/completions",
            json={"model": "demo", "messages": [], "stream": True},
        )

        assert response.status_code == 200
        assert "data: demo:0" in response.text
        assert "data: [DONE]" in response.text
        assert manager.inflight == 0
        assert manager.last_activity > 0
    finally:
        server.should_exit = True


def test_settings_are_applied_and_persisted(client) -> None:
    test_client, app = client
    response = test_client.put("/api/settings", json={"idle_timeout": 1800})
    assert response.status_code == 200
    assert app.state.manager.idle_timeout == 1800
    assert "1800" in app.state.cfg.settings_file.read_text()
    assert test_client.get("/api/state").json()["idle_timeout"] == 1800


def test_custom_model_lifecycle(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/models",
        json={
            "repo": "owner/some-image-model",
            "kind": "image",
            "engine": "vllm-omni",
            "extra_args": ["--enforce-eager"],
        },
    )
    assert response.status_code == 200
    key = response.json()["key"]
    assert key == "some-image-model"

    state = test_client.get("/api/state").json()
    assert any(model["key"] == key for model in state["models"])

    assert test_client.delete(f"/api/models/{key}").status_code == 200
    state = test_client.get("/api/state").json()
    assert all(model["key"] != key for model in state["models"])


def test_unknown_model_is_404(client) -> None:
    test_client, _ = client
    assert test_client.post("/api/models/nope/load").status_code == 404


def test_logs_endpoint_never_500s(client) -> None:
    test_client, _ = client
    for source in ("engine", "install", "tunnel", "app"):
        response = test_client.get(f"/api/logs?source={source}")
        assert response.status_code == 200
        assert "text" in response.json()
