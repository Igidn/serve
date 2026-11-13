from __future__ import annotations

import asyncio
import logging
import os
import platform
import signal
import sys

from .catalog import Catalog
from .config import Config
from .gpu import driver_version, gpus
from .idle import IdleWatchdog
from .manager import ModelManager
from .relay_client import RelayClient
from .runtime import RuntimeManager

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(cfg: Config) -> None:
    fmt = logging.Formatter(LOG_FORMAT)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    file = logging.FileHandler(cfg.logs_dir / "app.log")
    file.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers[:] = [stream, file]
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "uvicorn.error", "uvicorn.access", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def run(cfg: Config) -> None:
    log = logging.getLogger("serve")
    catalog = Catalog(cfg.custom_models_file)
    runtime = RuntimeManager(cfg, logging.getLogger("serve.runtime"))
    manager = ModelManager(cfg, catalog, runtime, logging.getLogger("serve.engine"))
    relay = RelayClient(
        cfg,
        logging.getLogger("serve.relay"),
        on_open=manager.request_started,
        on_close=manager.request_finished,
    )
    stop = asyncio.Event()

    async def shutdown(reason: str) -> None:
        if stop.is_set():
            return
        log.warning("shutting down: %s", reason)
        try:
            await manager.shutdown()
        except Exception as exc:
            log.warning("error stopping the engine: %s", exc)
        try:
            await relay.stop()
        except Exception as exc:
            log.warning("error stopping the relay: %s", exc)
        stop.set()
        if cfg.allow_exit:
            # Let the engine flush its log, then end the Kaggle run so the GPU
            # stops being billed.
            await asyncio.sleep(1.0)
            os._exit(0)

    watchdog = IdleWatchdog(manager, shutdown, logging.getLogger("serve.idle"))

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown("signal")))
        except NotImplementedError:  # pragma: no cover - non-Unix
            pass

    log.info("serve node starting: python=%s root=%s", platform.python_version(), cfg.root)
    cards = gpus()
    log.info(
        "gpus=%s driver=%s",
        ", ".join(f"{g['name']} ({g['memory_total_mb']} MB)" for g in cards) or "none",
        driver_version() or "unknown",
    )
    log.info(
        "model=%s relay=%s idle timeout=%ss exit_on_idle=%s",
        cfg.model_key,
        cfg.relay_url,
        int(cfg.idle_timeout),
        cfg.allow_exit,
    )

    watchdog.start()
    try:
        log.info("installing llama.cpp and downloading %s", cfg.model_key)
        await manager.load_and_wait(cfg.model_key)
        log.info("model ready; connecting to the raspi relay")
        relay.start()
        await stop.wait()
    finally:
        await watchdog.stop()
        await relay.stop()
        await manager.shutdown()


def main() -> None:
    cfg = Config.from_env().ensure()
    # Keep Hugging Face metadata on the working disk instead of the home disk.
    os.environ.setdefault("HF_HOME", str(cfg.hf_home))
    setup_logging(cfg)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
