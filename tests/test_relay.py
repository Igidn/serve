from __future__ import annotations

import asyncio
import dataclasses
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastapi.testclient import TestClient

from serve import relay
from serve.pi import RelayHub, create_http_app
from serve.relay_client import RelayClient
from serve.tunnel import PiTunnel

try:
    from websockets.asyncio.server import serve as ws_serve
except ImportError:  # pragma: no cover
    from websockets.legacy.server import serve as ws_serve  # type: ignore


def test_frames_round_trip() -> None:
    frame = relay.Frame(relay.DATA, 4242, b"hello")
    decoded = relay.decode(frame.encode())
    assert decoded == frame


def test_head_and_open_payloads_round_trip() -> None:
    opened = relay.open_payload("POST", "/v1/chat/completions", {"content-type": "application/json"})
    assert relay.parse_open(opened) == (
        "POST",
        "/v1/chat/completions",
        {"content-type": "application/json"},
    )

    head = relay.head_payload(200, {"content-type": "text/event-stream"})
    assert relay.parse_head(head) == (200, {"content-type": "text/event-stream"})


def test_hop_by_hop_headers_are_dropped() -> None:
    headers = relay.clean_headers(
        [
            ("Connection", "keep-alive"),
            ("Transfer-Encoding", "chunked"),
            ("Content-Type", "application/json"),
            ("X-Trace", "abc"),
        ]
    )
    assert headers == {"content-type": "application/json", "x-trace": "abc"}


def test_bad_path_is_rejected() -> None:
    payload = relay.open_payload("GET", "v1/models", {})
    try:
        relay.parse_open(payload)
    except relay.ProtocolError:
        pass
    else:
        raise AssertionError("expected a protocol error")


def test_pi_reports_offline_without_a_node() -> None:
    hub = RelayHub(None, logging.getLogger("test.relay"))
    app = create_http_app(hub)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["connected"] is False

        response = client.get("/v1/models")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "relay_offline"

        root = client.get("/")
        assert root.status_code == 200
        assert root.json()["connected"] is False


def test_token_is_checked_when_configured() -> None:
    hub = RelayHub("s3cret", logging.getLogger("test.relay"))
    assert hub.authorized('{"token": "s3cret"}') is True
    assert hub.authorized('{"token": "wrong"}') is False
    assert hub.authorized("not json") is False

    open_hub = RelayHub(None, logging.getLogger("test.relay"))
    assert open_hub.authorized("not json") is True


def test_opentunnel_url_becomes_a_websocket_url(cfg) -> None:
    tunnel = PiTunnel(cfg, logging.getLogger("test.tunnel"), cfg.pi_ws_port)
    assert tunnel.relay_url() is None
    tunnel.url = "https://abc.opentunnel.xyz"
    assert tunnel.relay_url() == "wss://abc.opentunnel.xyz/kaggle"
    tunnel.url = "https://abc.opentunnel.xyz/serve"
    assert tunnel.relay_url() == "wss://abc.opentunnel.xyz/serve/kaggle"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class EngineHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"data: hello\n\n")
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, *args) -> None:
        pass


def test_node_serves_a_stream_over_the_relay(cfg) -> None:
    engine = ThreadingHTTPServer(("127.0.0.1", cfg.engine_port), EngineHandler)
    threading.Thread(target=engine.serve_forever, daemon=True).start()
    ws_port = free_port()
    node = dataclasses.replace(cfg, relay_url=f"ws://127.0.0.1:{ws_port}/kaggle")
    result: dict = {}

    async def scenario() -> None:
        done = asyncio.Event()

        async def handler(websocket, path=None) -> None:
            hello = await websocket.recv()
            assert "serve-node" in hello
            await websocket.send(
                relay.Frame(
                    relay.OPEN,
                    7,
                    relay.open_payload(
                        "POST",
                        "/v1/chat/completions",
                        {"content-type": "application/json"},
                    ),
                ).encode()
            )
            await websocket.send(relay.Frame(relay.DATA, 7, b'{"model": "qwen"}').encode())
            await websocket.send(relay.Frame(relay.END, 7).encode())

            frames = []
            while True:
                frame = relay.decode(await websocket.recv())
                frames.append(frame)
                if frame.op in {relay.END, relay.ERROR}:
                    break
            result["frames"] = frames
            done.set()

        async with ws_serve(handler, "127.0.0.1", ws_port):
            client = RelayClient(node, logging.getLogger("test.relay_client"))
            client.start()
            try:
                await asyncio.wait_for(done.wait(), timeout=10)
            finally:
                await client.stop()

    try:
        asyncio.run(scenario())
    finally:
        engine.shutdown()

    frames = result["frames"]
    head = next(frame for frame in frames if frame.op == relay.HEAD)
    status, headers = relay.parse_head(head.payload)
    assert status == 200
    assert headers["content-type"] == "text/event-stream"
    body = b"".join(frame.payload for frame in frames if frame.op == relay.DATA)
    assert b"data: hello" in body
    assert frames[-1].op == relay.END

