from __future__ import annotations

import asyncio


class IdleWatchdog:
    """Ends the Kaggle run after a period without inference traffic.

    Status polling from the Pi's health endpoint deliberately does *not* count
    as activity, otherwise a monitoring loop would hold the GPU forever.
    A load, an unload, a download or an in-flight request pauses the timer.
    """

    def __init__(self, manager, shutdown, log, interval: float = 5.0):
        self.manager = manager
        self.shutdown = shutdown
        self.log = log
        self.interval = interval
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            if self.manager.idle_paused():
                continue
            if self.manager.idle_remaining() <= 0:
                self.log.info("idle timeout reached; shutting the kernel down")
                await self.shutdown("idle timeout")
                return

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
