from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import httpx

from . import staging
from .catalog import Catalog, ModelSpec
from .config import Config
from .util import dir_size, exception_summary, human_bytes, now, one_line


class InsufficientSpace(RuntimeError):
    def __init__(self, need: int, free: int):
        super().__init__(
            f"this model needs {human_bytes(need)} but only {human_bytes(free)} is free. "
            "Delete a cached model, or attach the weights as a Kaggle Dataset "
            "(kernels mount /kaggle/input, which does not use the working disk)."
        )
        self.need = need
        self.free = free


def remote_size(repo: str, token: str | None, patterns: list[str] | None = None) -> int | None:
    """Byte size of the files a download will fetch, or None when unknown."""
    import fnmatch

    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).model_info(repo, files_metadata=True)
        sizes = []
        for sibling in info.siblings or []:
            size = getattr(sibling, "size", None)
            if not size:
                continue
            if patterns and not any(fnmatch.fnmatch(sibling.rfilename, p) for p in patterns):
                continue
            sizes.append(size)
        return sum(sizes) if sizes else None
    except Exception:
        return None


async def download_snapshot(
    spec: ModelSpec,
    dest: Path,
    cfg: Config,
    log,
    on_progress,
    repo: str | None = None,
    patterns: list[str] | None = None,
) -> Path:
    """Fetch the files that ``patterns`` selects from ``repo`` into ``dest``.

    Defaults to the spec's own repo and file list; the drafter passes its own
    so a second repository can land beside the target.
    """
    dest.mkdir(parents=True, exist_ok=True)
    repo = repo or spec.repo
    patterns = list(patterns if patterns is not None else spec.files) or None
    total = remote_size(repo, cfg.hf_token, patterns)
    if total:
        free = shutil.disk_usage(cfg.state_dir).free
        # Hugging Face downloads into a temporary file inside the destination
        # and renames it, so the peak is one copy of the selected files, not
        # two. Keep a small margin for logs and the destination metadata.
        if free < total * 1.03 + 512 * 1024**2:
            raise InsufficientSpace(int(total * 1.03) + 512 * 1024**2, free)
        on_progress(0, total)

    stop = asyncio.Event()

    async def poll() -> None:
        while not stop.is_set():
            on_progress(dir_size(dest), total)
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.5)
            except asyncio.TimeoutError:
                continue

    poller = asyncio.create_task(poll())
    try:
        from huggingface_hub import snapshot_download

        log.info("downloading %s → %s", repo, dest)
        await asyncio.to_thread(
            snapshot_download,
            repo_id=repo,
            local_dir=str(dest),
            token=cfg.hf_token,
            allow_patterns=patterns,
            ignore_patterns=None if patterns else ["*.gguf", "*.onnx", "*.msgpack"],
        )
    finally:
        stop.set()
        await poller

    on_progress(dir_size(dest), total)
    if not Catalog.is_checkpoint(dest, spec):
        expected = "a .gguf file" if spec.engine == "llamacpp" else "config.json"
        raise RuntimeError(
            f"{repo} was downloaded but has no {expected}; it may not be a full checkpoint"
        )
    return dest


class ModelManager:
    """Owns the single active engine and everything that mutates GPU state."""

    def __init__(self, cfg: Config, catalog: Catalog, runtime, log):
        self.cfg = cfg
        self.catalog = catalog
        self.runtime = runtime
        self.log = log

        self.phase = "idle"  # idle | loading | ready | stopping | error
        self.stage = ""
        self.error: str | None = None
        self.active: ModelSpec | None = None
        self.source: str | None = None
        self.started_at: float | None = None
        self.progress: float | None = None
        self.bytes_done = 0
        self.bytes_total: int | None = None

        self.engine = None
        self.inflight = 0
        self.last_activity = now()
        self.idle_timeout = cfg.idle_timeout
        self.idle_paused_reason: str | None = None
        self._load_task: asyncio.Task | None = None
        self._download_tasks: dict[str, asyncio.Task] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._sizes: dict[str, tuple[float, int]] = {}

    # -- activity ----------------------------------------------------------

    def touch(self) -> None:
        self.last_activity = now()

    def idle_paused(self) -> bool:
        if self.inflight > 0 or self.phase in {"loading", "stopping"}:
            return True
        return any(job["state"] == "running" for job in self._jobs.values())

    def idle_remaining(self) -> float:
        return max(0.0, self.idle_timeout - (now() - self.last_activity))

    def set_idle_timeout(self, seconds: float) -> None:
        self.idle_timeout = max(60.0, min(86400.0, float(seconds)))
        self.touch()

    # -- public actions ----------------------------------------------------

    async def load(self, key: str, replace: bool = False, actor: str = "node") -> None:
        spec = self.catalog.get(key)
        self._assert_can_load(spec, replace)
        self.log.info("load %s requested by %s", key, actor)
        self.touch()
        self._load_task = asyncio.create_task(self._load(spec))

    async def load_and_wait(self, key: str, replace: bool = False) -> None:
        """Install the runtime, download the model, and block until ready."""
        await self.load(key, replace=replace)
        task = self._load_task
        if task is not None:
            await task
        if self.phase != "ready":
            raise RuntimeError(self.error or f"{key} did not become ready")
        self.touch()

    def request_started(self) -> None:
        self.inflight += 1
        self.touch()

    def request_finished(self) -> None:
        self.inflight = max(0, self.inflight - 1)
        self.touch()

    def _assert_can_load(self, spec: ModelSpec, replace: bool) -> None:
        if self._load_task and not self._load_task.done():
            raise ValueError("another load or unload is already in progress")
        if self.active and self.active.key == spec.key and self.phase == "ready":
            return
        if self.active and self.active.key == spec.key and self.phase == "error":
            return
        if self.active and self.phase in {"ready", "error"} and not replace:
            raise ValueError(
                f"{self.active.key} is loaded; unload it first or pass replace=true"
            )

    async def unload(self, reason: str = "requested") -> None:
        self.log.info("unload requested (%s)", reason)
        self.phase = "stopping"
        self.stage = "stopping engine"
        task = self._load_task
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._teardown()
        self.touch()

    async def _teardown(self) -> None:
        if self.engine is not None:
            await self.engine.stop()
        self.engine = None
        self.active = None
        self.source = None
        self.started_at = None
        self.progress = None
        self.error = None
        self.phase = "idle"
        self.stage = ""

    async def download(self, key: str) -> None:
        spec = self.catalog.get(key)
        if self.active and self.active.key == key and self.phase == "ready":
            raise ValueError("this model is loaded and already on disk")
        job = self._jobs.get(key)
        if job and job["state"] == "running":
            return
        self._jobs[key] = {
            "key": key,
            "state": "running",
            "bytes_done": 0,
            "bytes_total": None,
            "error": None,
            "started": now(),
        }
        self.touch()

        async def run() -> None:
            job = self._jobs[key]
            try:
                await download_snapshot(
                    spec,
                    self.catalog.download_dir(spec, self.cfg.models_dir),
                    self.cfg,
                    self.log,
                    lambda done, total: job.update(
                        bytes_done=done, bytes_total=total or job.get("bytes_total")
                    ),
                )
                if spec.draft_repo:
                    await self._ensure_draft(
                        spec,
                        on_progress=lambda done, total: job.update(
                            bytes_done=done, bytes_total=total or job.get("bytes_total")
                        ),
                    )
                job["state"] = "done"
                self._sizes.pop(key, None)
            except Exception as exc:
                job["state"] = "error"
                job["error"] = one_line(str(exc), 400)
                self.log.warning("download failed for %s: %s", key, exc)
            finally:
                self._download_tasks.pop(key, None)

        self._download_tasks[key] = asyncio.create_task(run())

    async def delete(self, key: str) -> None:
        spec = self.catalog.get(key)
        if self.active and self.active.key == key:
            raise ValueError("unload the model before deleting its files")
        job = self._jobs.get(key)
        if job and job["state"] == "running":
            raise ValueError("a download is still running for this model")
        for path in (
            self.catalog.download_dir(spec, self.cfg.models_dir),
            *(Path(p) for p in spec.local_paths if Path(p).is_dir()),
        ):
            if path.is_dir() and str(path).startswith(str(self.cfg.models_dir)):
                shutil.rmtree(path, ignore_errors=True)
        self._jobs.pop(key, None)
        self._sizes.pop(key, None)
        self.log.info("deleted cached files for %s", key)

    async def shutdown(self) -> None:
        await self.unload(reason="shutdown")

    # -- loading -----------------------------------------------------------

    async def _load(self, spec: ModelSpec) -> None:
        try:
            if self.active is not None:
                await self.unload(reason="replaced")
            self.phase = "loading"
            self.error = None
            self.active = spec
            self.progress = None
            self.bytes_done = 0
            self.bytes_total = None

            self.stage = f"preparing {spec.engine}"
            await self.runtime.prepare(spec.engine)

            self.stage = "locating model weights"
            source, local = self.catalog.source_for(spec, self.cfg.models_dir)
            if local is None:
                dest = self.catalog.download_dir(spec, self.cfg.models_dir)
                self.stage = f"downloading {spec.repo}"
                await download_snapshot(
                    spec,
                    dest,
                    self.cfg,
                    self.log,
                    self._set_progress,
                )
                source, local = self.catalog.source_for(spec, self.cfg.models_dir)
            elif (
                spec.engine != "llamacpp"
                and self.cfg.stage_models
                and staging.is_network_mount(local)
            ):
                source, local = await self._stage_weights(spec, local)
            else:
                self.stage = f"using local weights at {local}"

            if spec.draft_repo:
                self.stage = f"downloading drafter {spec.draft_repo}"
                await self._ensure_draft(spec)

            self.source = source
            self.bytes_total = None
            self.progress = None

            from .catalog import build_argv
            from .engine import EngineProcess

            argv = build_argv(
                spec,
                source,
                self.cfg.engine_port,
                state_dir=self.cfg.state_dir,
                draft=self.catalog.draft_file(spec, self.cfg.models_dir),
            )
            self.stage = "starting engine"
            engine = EngineProcess(
                self.cfg, self.cfg.engine_port, self.cfg.logs_dir / "engine.log", self.log
            )
            self.engine = engine
            self.started_at = now()
            await engine.start(argv)
            self.stage = "waiting for /health"
            await engine.wait_ready(timeout=3600)
            if spec.kind == "chat":
                self.stage = "warming up"
                await self._warmup(spec)

            self.phase = "ready"
            self.stage = ""
            self.touch()
            self.log.info("model ready: %s", spec.key)
        except asyncio.CancelledError:
            self.log.info("load cancelled")
            if self.engine is not None:
                await self.engine.stop()
            self.engine = None
            self.phase = "idle"
            self.stage = ""
            self.active = None
            raise
        except Exception as exc:
            message = exception_summary(str(exc))
            self.error = message or one_line(str(exc), 800)
            self.phase = "error"
            self.stage = ""
            self.log.warning("load failed for %s: %s", spec.key, self.error)
            if self.engine is not None:
                await self.engine.stop()
            self.engine = None

    async def _warmup(self, spec: ModelSpec) -> None:
        """One tiny completion, so the first real request is not the one that
        pays for graph capture and the first full sweep over the weights.

        On the T4s that first pass costs about two minutes; absorbing it into
        the load keeps "ready" honest and makes agent harnesses look instant.
        """
        started = now()
        url = f"http://127.0.0.1:{self.cfg.engine_port}/v1/chat/completions"
        payload = {
            "model": spec.key,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
            self.log.info("warmup finished in %.1fs", now() - started)
        except Exception as exc:  # a slow first pass is not a failed load
            self.log.warning("warmup failed after %.1fs: %s", now() - started, exc)

    def _set_progress(self, done: int, total: int | None) -> None:
        self.bytes_done = done
        self.bytes_total = total
        if total:
            self.progress = min(1.0, done / total)

    async def _ensure_draft(self, spec: ModelSpec, on_progress=None) -> None:
        """Fetch the speculative drafter once, into its own subdirectory."""
        dest = self.catalog.draft_dir(spec, self.cfg.models_dir)
        if self.catalog.draft_file(spec, self.cfg.models_dir) is not None:
            return
        self.log.info("downloading drafter %s → %s", spec.draft_repo, dest)
        await download_snapshot(
            spec,
            dest,
            self.cfg,
            self.log,
            on_progress or self._set_progress,
            repo=spec.draft_repo,
            patterns=spec.draft_files,
        )

    async def _stage_weights(self, spec: ModelSpec, local: Path) -> tuple[str, Path]:
        """Copy weights off a network mount so the engine reads local disk."""
        dest = self.catalog.download_dir(spec, self.cfg.models_dir)
        files = staging.checkpoint_files(local)
        need = staging.total_bytes(files)
        if staging.ready(dest):
            self.log.info("reusing staged weights at %s", dest)
        else:
            free = shutil.disk_usage(self.cfg.state_dir).free
            if need and free < need * 1.02:
                self.log.warning(
                    "cannot stage %s: needs %s, %s free; loading from %s",
                    spec.key,
                    human_bytes(need),
                    human_bytes(free),
                    local,
                )
                self.stage = f"using local weights at {local}"
                return str(local), local
            self.stage = f"copying {human_bytes(need)} to local disk"
            self.log.info("staging %s: %s → %s (%s)", spec.key, local, dest, human_bytes(need))
            self._set_progress(0, need or None)
            started = now()
            try:
                await asyncio.to_thread(staging.stage, local, dest, self._set_progress)
            except (OSError, RuntimeError) as exc:
                # A failed copy is not worth failing the load: the mount still
                # works, it is only slower.
                self.log.warning("staging %s failed: %s", spec.key, exc)
                self.progress = None
                self.bytes_done = 0
                self.bytes_total = None
                self.stage = f"using local weights at {local}"
                return str(local), local
            self.log.info("staged %s in %.0fs", spec.key, now() - started)
        self.progress = None
        self.bytes_done = 0
        self.bytes_total = None
        self.stage = f"using staged weights at {dest}"
        return str(dest), dest

    # -- reporting ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        from .gpu import disk, gpus

        if self.phase == "ready" and (self.engine is None or not self.engine.alive):
            self.phase = "error"
            self.stage = ""
            self.error = "the engine process exited; open the logs tab for the traceback"

        models = []
        for spec in self.catalog.all():
            local = self.catalog.local_source(spec)
            cached_dir = self.catalog.download_dir(spec, self.cfg.models_dir)
            cached = self.catalog.cached(spec, self.cfg.models_dir)
            size = 0
            if cached:
                stamp, value = self._sizes.get(spec.key, (0.0, 0))
                if now() - stamp > 5:
                    value = dir_size(cached_dir)
                    self._sizes[spec.key] = (now(), value)
                size = value
            job = self._jobs.get(spec.key)
            models.append(
                {
                    **spec.to_dict(),
                    "available": local is not None or cached,
                    "local_path": str(local) if local else None,
                    "cached": cached,
                    "disk_bytes": size,
                    "job": job,
                }
            )

        active = None
        if self.active is not None:
            active = {
                "key": self.active.key,
                "label": self.active.label,
                "repo": self.active.repo,
                "kind": self.active.kind,
                "engine": self.active.engine,
                "served_name": self.active.key,
                "source": self.source,
            }

        gpu_rows = gpus()
        total_vram = sum(g["memory_total_mb"] for g in gpu_rows) / 1024

        return {
            "phase": self.phase,
            "stage": self.stage,
            "error": self.error,
            "active": active,
            "progress": self.progress,
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            "started_at": self.started_at,
            "last_activity": self.last_activity,
            "idle_remaining": self.idle_remaining(),
            "idle_timeout": self.idle_timeout,
            "idle_paused": self.idle_paused(),
            "inflight": self.inflight,
            "models": models,
            "runtime": self.runtime.all_status(),
            "gpus": gpu_rows,
            "vram_total_gb": round(total_vram, 1),
            "disk": disk(self.cfg.state_dir),
            "engine_log": str(self.cfg.logs_dir / "engine.log"),
            "endpoints": {
                "base": "/v1",
                "chat": "/v1/chat/completions",
                "images": "/v1/images/generations",
                "models": "/v1/models",
            },
        }
