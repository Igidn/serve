from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from serve.app import create_app
from serve.auth import Auth, load_key


def test_key_is_generated_once_and_stays_private(tmp_path) -> None:
    key, source = load_key(tmp_path, None)
    assert source == "file"
    assert key.startswith("sk-serve-")

    path = tmp_path / "api-key"
    assert path.read_text().strip() == key
    assert path.stat().st_mode & 0o777 == 0o600

    again, source = load_key(tmp_path, None)
    assert (again, source) == (key, "file")


def test_environment_key_wins_and_writes_nothing(tmp_path) -> None:
    key, source = load_key(tmp_path, "from-the-secret")
    assert (key, source) == ("from-the-secret", "env")
    assert not (tmp_path / "api-key").exists()


def test_bearer_and_x_api_key_headers() -> None:
    auth = Auth("sk-serve-test")
    assert auth.from_headers({"authorization": "Bearer sk-serve-test"})
    assert auth.from_headers({"authorization": "bearer sk-serve-test"})
    assert auth.from_headers({"x-api-key": "sk-serve-test"})
    assert not auth.from_headers({"authorization": "Bearer nope"})
    assert not auth.from_headers({})


def test_session_round_trip_expiry_and_tampering() -> None:
    auth = Auth("sk-serve-test", ttl=60)
    cookie = auth.session()
    assert auth.session_ok(cookie)
    assert not auth.session_ok(cookie, now=time.time() + 120)
    assert not auth.session_ok(None)
    assert not auth.session_ok("garbage")

    last = "0" if cookie[-1] != "0" else "1"
    assert not auth.session_ok(cookie[:-1] + last)

    # A session signed with another key is not accepted either.
    assert not Auth("sk-serve-other").session_ok(cookie)


@pytest.fixture
def app_client(cfg):
    app = create_app(cfg)
    with TestClient(app) as test_client:
        yield test_client, app


def test_dashboard_redirects_to_the_login_page(app_client) -> None:
    client, _ = app_client
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login?next=/"
    assert client.get("/login").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/api/health").status_code == 200


def test_state_requires_a_session_or_key(app_client) -> None:
    client, _ = app_client
    response = client.get("/api/state")
    assert response.status_code == 401
    assert "api key" in response.json()["error"].lower()


def test_v1_keeps_the_openai_error_shape(app_client) -> None:
    client, _ = app_client
    response = client.get("/v1/models")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_bearer_key_opens_the_api(app_client) -> None:
    client, app = app_client
    key = app.state.auth.key
    # 503 is "no model loaded", which means the request got past the door.
    assert client.get("/v1/models", headers={"authorization": f"Bearer {key}"}).status_code == 503
    assert client.get("/api/state", headers={"x-api-key": key}).status_code == 200


def test_login_sets_a_cookie_and_logout_clears_it(app_client) -> None:
    client, app = app_client
    wrong = client.post("/login", data={"key": "nope"}, follow_redirects=False)
    assert wrong.status_code == 303
    assert wrong.headers["location"].startswith("/login?error=1")
    assert "serve_session" not in client.cookies

    ok = client.post("/login", data={"key": app.state.auth.key}, follow_redirects=False)
    assert ok.status_code == 303
    assert ok.headers["location"] == "/"
    assert client.get("/api/state").status_code == 200

    client.get("/logout", follow_redirects=False)
    assert client.get("/api/state").status_code == 401


def test_login_never_redirects_off_the_host(app_client) -> None:
    client, app = app_client
    failed = client.post(
        "/login", data={"key": "nope", "next": "//evil.example"}, follow_redirects=False
    )
    assert failed.headers["location"].startswith("/login?error=1")
    assert "evil" not in failed.headers["location"]

    ok = client.post(
        "/login",
        data={"key": app.state.auth.key, "next": "//evil.example"},
        follow_redirects=False,
    )
    assert ok.headers["location"] == "/"


def test_cookie_mutations_refuse_cross_site_requests(app_client) -> None:
    client, app = app_client
    client.post("/login", data={"key": app.state.auth.key}, follow_redirects=False)

    blocked = client.put(
        "/api/settings",
        json={"idle_timeout": 900},
        headers={"origin": "https://evil.example"},
    )
    assert blocked.status_code == 403

    allowed = client.put(
        "/api/settings",
        json={"idle_timeout": 900},
        headers={"origin": "http://testserver"},
    )
    assert allowed.status_code == 200


def test_the_key_is_visible_once_you_are_in(app_client) -> None:
    client, app = app_client
    client.post("/login", data={"key": app.state.auth.key}, follow_redirects=False)
    payload = client.get("/api/auth").json()
    assert payload["key"] == app.state.auth.key
    assert payload["source"] == "file"
