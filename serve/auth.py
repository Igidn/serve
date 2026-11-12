"""The one key that unlocks the dashboard and the ``/v1`` API.

The key is ``SERVE_API_KEY`` when the environment sets one (a Kaggle secret,
for instance). Otherwise it is generated once and kept beside the rest of the
state, so a restarted process hands the same key back to the same clients.

The dashboard trades the key for a signed cookie. The cookie is an HMAC over
its own expiry, keyed with the API key, so there is no session store to keep
and rotating the key invalidates every old session.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from pathlib import Path
from urllib.parse import quote, urlsplit

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

KEY_ENV = "SERVE_API_KEY"
COOKIE = "serve_session"
SESSION_TTL = 30 * 24 * 60 * 60
KEY_FILE = "api-key"

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# Reachable without the key: the login page itself, the static assets it
# needs, the health probe the launcher polls, and a logout link.
PUBLIC_PATHS = {"/login", "/logout", "/api/health", "/favicon.ico"}


def load_key(state_dir: Path, env_key: str | None) -> tuple[str, str]:
    """Return ``(key, source)``, where source is ``env`` or ``file``.

    A generated key is written with mode 600 and must survive a restart:
    changing it on every boot would log the browser out and break every
    client that saved it.
    """
    env_key = (env_key or "").strip()
    if env_key:
        return env_key, "env"

    path = state_dir / KEY_FILE
    try:
        existing = path.read_text(encoding="utf-8").strip()
    except OSError:
        existing = ""
    if existing:
        return existing, "file"

    key = "sk-serve-" + secrets.token_urlsafe(32)
    state_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(key + "\n", encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)
    return key, "file"


class Auth:
    """Compares keys and mints/verifies dashboard session cookies."""

    def __init__(self, key: str, source: str = "file", ttl: int = SESSION_TTL):
        self.key = key
        self.source = source
        self.ttl = ttl

    @classmethod
    def from_config(cls, cfg) -> "Auth":
        key, source = load_key(cfg.state_dir, cfg.api_key)
        return cls(key, source)

    def matches(self, candidate: str | None) -> bool:
        if not candidate:
            return False
        return hmac.compare_digest(candidate.strip().encode(), self.key.encode())

    def from_headers(self, headers) -> bool:
        """``Authorization: Bearer …`` or ``x-api-key: …``."""
        value = headers.get("authorization") or ""
        if value[:7].lower() == "bearer " and self.matches(value[7:]):
            return True
        return self.matches(headers.get("x-api-key"))

    def session(self, now: float | None = None) -> str:
        expires = int((time.time() if now is None else now) + self.ttl)
        return f"{expires}.{self._sign(expires)}"

    def session_ok(self, value: str | None, now: float | None = None) -> bool:
        if not value or "." not in value:
            return False
        raw, _, signature = value.partition(".")
        try:
            expires = int(raw)
        except ValueError:
            return False
        if expires < (time.time() if now is None else now):
            return False
        return hmac.compare_digest(signature, self._sign(expires))

    def _sign(self, expires: int) -> str:
        message = f"v1.{expires}".encode()
        return hmac.new(self.key.encode(), message, hashlib.sha256).hexdigest()


def is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith("/static/")


def safe_next(value: str) -> str:
    """Only ever bounce back to a path on this host."""
    return value if value.startswith("/") and not value.startswith("//") else "/"


def _same_origin(request: Request) -> bool:
    """Whether a cookie-authenticated mutation came from this dashboard.

    A browser attaches cookies to cross-site form posts, so the key alone is
    not enough for state-changing requests that ride on the session cookie.
    """
    site = request.headers.get("sec-fetch-site")
    if site and site not in {"same-origin", "none"}:
        return False
    origin = request.headers.get("origin")
    if not origin:
        return True
    return urlsplit(origin).netloc == request.headers.get("host", "")


class AuthMiddleware(BaseHTTPMiddleware):
    """Require the key for everything except the login and health surfaces.

    ``/v1`` answers in the OpenAI error shape so clients can show something
    useful; a browser asking for the dashboard gets bounced to ``/login``
    and comes back afterwards.
    """

    def __init__(self, app, auth: Auth, log):
        super().__init__(app)
        self.auth = auth
        self.log = log

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if request.method == "OPTIONS" or is_public(path):
            return await call_next(request)
        if self.auth.from_headers(request.headers):
            return await call_next(request)
        if self.auth.session_ok(request.cookies.get(COOKIE)):
            if request.method not in SAFE_METHODS and not _same_origin(request):
                self.log.warning("refused cross-origin %s %s", request.method, path)
                return JSONResponse(
                    status_code=403,
                    content={"error": "cross-origin request refused"},
                )
            return await call_next(request)

        if path.startswith("/v1/"):
            return JSONResponse(
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
                content={
                    "error": {
                        "message": "Missing or invalid API key. Send "
                        "`Authorization: Bearer <key>` (the key is on the "
                        "dashboard's Session panel).",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                },
            )
        if path.startswith("/api/"):
            return JSONResponse(
                status_code=401,
                content={"error": "missing or invalid API key"},
            )
        return RedirectResponse(f"/login?next={quote(path)}", status_code=303)
