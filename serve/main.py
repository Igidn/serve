from __future__ import annotations

import logging
import platform
import sys

import uvicorn

from .app import create_app
from .config import Config
from .gpu import driver_version, gpus


def setup_logging(cfg: Config) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    file = logging.FileHandler(cfg.logs_dir / "app.log")
    file.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers[:] = [stream, file]
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    cfg = Config.from_env().ensure()
    setup_logging(cfg)
    log = logging.getLogger("serve")

    cards = gpus()
    log.info("serve starting: python=%s root=%s", platform.python_version(), cfg.root)
    log.info(
        "gpus=%s driver=%s",
        ", ".join(f"{g['name']} ({g['memory_total_mb']} MB)" for g in cards) or "none",
        driver_version() or "unknown",
    )
    log.info(
        "idle timeout=%ss tunnel=%s exit_on_idle=%s",
        int(cfg.idle_timeout),
        "on" if cfg.tunnel_enabled else "off",
        cfg.allow_exit,
    )

    app = create_app(cfg)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=cfg.app_port,
        log_config=None,
        access_log=False,
        timeout_keep_alive=75,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
