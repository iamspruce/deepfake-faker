import asyncio
from threading import Thread

class AsyncHandler:
    """
    Manages a dedicated thread for an asyncio event loop,
    allowing asyncio tasks to run alongside a synchronous GUI.
    """
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run_coroutine(self, coro):
        """Schedules a coroutine to be run on the asyncio event loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self):
        """Stops the asyncio event loop."""
        self._loop.call_soon_threadsafe(self._loop.stop)
