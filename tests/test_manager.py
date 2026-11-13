from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from serve.catalog import Catalog
from serve.manager import ModelManager


class FakeRuntime:
    def all_status(self):
        return []


def manager_for(cfg) -> ModelManager:
    return ModelManager(cfg, Catalog(cfg.custom_models_file), FakeRuntime(), None)


def test_idle_timer_counts_down(cfg) -> None:
    manager = manager_for(cfg)
    manager.last_activity -= 120
    assert 470 <= manager.idle_remaining() <= 480


def test_inflight_request_pauses_idle(cfg) -> None:
    manager = manager_for(cfg)
    manager.last_activity -= 10_000
    assert manager.idle_remaining() == 0
    assert manager.idle_paused() is False

    manager.inflight = 1
    assert manager.idle_paused() is True
    manager.inflight = 0

    manager.phase = "loading"
    assert manager.idle_paused() is True


def test_running_download_pauses_idle(cfg) -> None:
    manager = manager_for(cfg)
    manager._jobs["x"] = {"state": "running"}
    assert manager.idle_paused() is True
    manager._jobs["x"]["state"] = "done"
    assert manager.idle_paused() is False


def test_idle_timeout_is_clamped(cfg) -> None:
    manager = manager_for(cfg)
    manager.set_idle_timeout(5)
    assert manager.idle_timeout == 60
    manager.set_idle_timeout(10**9)
    assert manager.idle_timeout == 86400


def test_drafter_downloads_into_a_subdirectory(cfg, monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, ...], Path]] = []

    async def fake_download(spec, dest, cfg_, log, on_progress, repo=None, patterns=None):
        calls.append((repo, tuple(patterns or ()), Path(dest)))
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "draft.gguf").write_bytes(b"y")

    monkeypatch.setattr("serve.manager.download_snapshot", fake_download)
    manager = ModelManager(
        cfg, Catalog(cfg.custom_models_file), FakeRuntime(), logging.getLogger("test")
    )
    spec = Catalog(cfg.custom_models_file).get("qwen3.8-27b-gguf")

    asyncio.run(manager._ensure_draft(spec))
    asyncio.run(manager._ensure_draft(spec))  # already there: no second fetch

    assert [c[0] for c in calls] == [spec.draft_repo]
    assert calls[0][1] == tuple(spec.draft_files)
    assert calls[0][2] == cfg.models_dir / spec.key / "draft"


def test_load_requires_unloading_another_model(cfg) -> None:
    manager = manager_for(cfg)
    manager.active = Catalog(cfg.custom_models_file).get("qwen3-8b-awq")
    manager.phase = "ready"
    try:
        manager._assert_can_load(Catalog(cfg.custom_models_file).get("sdxl-base"), replace=False)
    except ValueError as exc:
        assert "unload" in str(exc)
    else:
        raise AssertionError("expected a conflict")
    manager._assert_can_load(Catalog(cfg.custom_models_file).get("sdxl-base"), replace=True)
