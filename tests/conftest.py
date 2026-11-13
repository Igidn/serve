from __future__ import annotations

import socket

import pytest

from serve.config import Config


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("SERVE_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("SERVE_PI_TUNNEL", "0")
    monkeypatch.setenv("SERVE_ALLOW_EXIT", "0")
    monkeypatch.setenv("SERVE_ENGINE_PORT", str(free_port()))
    monkeypatch.setenv("SERVE_IDLE_TIMEOUT", "600")
    return Config.from_env()
