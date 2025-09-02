import threading
import asyncio
from typing import Any, Awaitable, Optional

class AsyncLoopRunner:
    """Boucle asyncio persistante dans un thread dédié."""
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        def _runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.run_forever()
            loop.close()
        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

    def run(self, coro: Awaitable[Any]) -> Any:
        if not self._loop:
            raise RuntimeError("AsyncLoopRunner not started")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def stop(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
