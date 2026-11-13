"""A tiny multiplexed HTTP-over-WebSocket protocol.

One WebSocket carries many concurrent requests. Every message starts with a
4-byte stream id and a 1-byte opcode, followed by an optional payload:

    OPEN    Pi -> Kaggle   JSON: {method, path, headers}
    DATA    both ways      body bytes
    END     both ways      no more body for this stream
    HEAD    Kaggle -> Pi   JSON: {status, headers}
    ERROR   Kaggle -> Pi   UTF-8 error text
    ABORT   both ways      cancel the stream

Request bodies are small JSON documents, so they are buffered before the
request is made. Response bodies (including SSE token streams) stay streaming.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

OPEN = 1
DATA = 2
END = 3
ABORT = 4
HEAD = 5
ERROR = 6

STRUCT = struct.Struct(">IB")
CHUNK = 256 * 1024
MAX_MESSAGE = 8 * 1024 * 1024

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
}


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Frame:
    op: int
    stream: int
    payload: bytes = b""

    def encode(self) -> bytes:
        return STRUCT.pack(self.stream, self.op) + self.payload


def decode(raw: bytes | bytearray | memoryview) -> Frame:
    data = bytes(raw)
    if len(data) < STRUCT.size:
        raise ProtocolError("frame is shorter than its header")
    stream, op = STRUCT.unpack_from(data)
    if op not in {OPEN, DATA, END, ABORT, HEAD, ERROR}:
        raise ProtocolError(f"unknown opcode {op}")
    return Frame(op, stream, data[STRUCT.size :])


def clean_headers(items) -> dict[str, str]:
    """Drop hop-by-hop headers while preserving the rest."""
    headers: dict[str, str] = {}
    for key, value in items:
        name = str(key).lower()
        if name in HOP_BY_HOP:
            continue
        headers[name] = str(value)
    return headers


def open_payload(method: str, path: str, headers: dict[str, str]) -> bytes:
    return json.dumps({"method": method, "path": path, "headers": headers}).encode()


def head_payload(status: int, headers: dict[str, str]) -> bytes:
    return json.dumps({"status": int(status), "headers": headers}).encode()


def parse_open(payload: bytes) -> tuple[str, str, dict[str, str]]:
    try:
        data = json.loads(payload)
        method = str(data["method"]).upper()
        path = str(data["path"])
        headers = {str(k).lower(): str(v) for k, v in data.get("headers", {}).items()}
    except (ValueError, KeyError, TypeError) as exc:
        raise ProtocolError(f"bad OPEN payload: {exc}") from exc
    if not path.startswith("/"):
        raise ProtocolError("path must start with '/'")
    return method, path, headers


def parse_head(payload: bytes) -> tuple[int, dict[str, str]]:
    try:
        data = json.loads(payload)
        status = int(data["status"])
        headers = {str(k).lower(): str(v) for k, v in data.get("headers", {}).items()}
    except (ValueError, KeyError, TypeError) as exc:
        raise ProtocolError(f"bad HEAD payload: {exc}") from exc
    return status, headers
