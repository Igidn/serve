from __future__ import annotations

import json
from pathlib import Path


class Settings:
    """Small, persisted dashboard settings."""

    def __init__(self, path: Path, idle_timeout: float):
        self.path = path
        self.idle_timeout = idle_timeout
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
            value = float(data["idle_timeout"])
        except (OSError, ValueError, KeyError, TypeError):
            return
        self.idle_timeout = max(60.0, min(86400.0, value))

    def update(self, idle_timeout: float | None = None) -> None:
        if idle_timeout is not None:
            self.idle_timeout = max(60.0, min(86400.0, float(idle_timeout)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"idle_timeout": self.idle_timeout}, indent=2))
        tmp.replace(self.path)
