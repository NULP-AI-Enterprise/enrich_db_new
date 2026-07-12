"""
Async helpers for Celery fork workers.

asyncio.run() closes the event loop after the coroutine finishes.
httpx (and anyio) register background cleanup tasks that then try to
finalise on the now-closed loop → RuntimeError('Event loop is closed').

run_in_worker() cancels every pending task before closing the loop so
those finalisation tasks exit cleanly instead of printing tracebacks.
"""

import asyncio
import logging
from typing import Any, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_in_worker(coro: Coroutine[Any, Any, T]) -> T:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            if pending:
                for task in pending:
                    task.cancel()
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        finally:
            asyncio.set_event_loop(None)
            loop.close()
