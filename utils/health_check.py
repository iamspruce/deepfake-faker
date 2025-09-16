import asyncio
import aiohttp
import logging
from PyQt6.QtCore import QObject, pyqtSignal, QThread

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HealthCheckWorker(QObject):
    status_update = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, url: str, interval: float = 1.0, timeout: float = 120.0):
        super().__init__()
        self.url = url
        self.interval = interval
        self.timeout = timeout
        self.running = True
        self.async_loop = asyncio.new_event_loop()
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self.run)

    def run(self):
        asyncio.set_event_loop(self.async_loop)
        self.async_loop.run_until_complete(self._poll_loop())

    async def _poll_loop(self):
        elapsed_time = 0.0
        backoff = 1.0
        while self.running and elapsed_time < self.timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.url, timeout=self.interval) as response:
                        if response.status == 200:
                            status_data = await response.json()
                            status_data['elapsed'] = elapsed_time
                            self.status_update.emit(status_data)
                            if status_data.get("status") in ["ready", "error"]:
                                self.running = False
                        else:
                            self.status_update.emit({
                                "status": "error",
                                "error": f"Server returned status {response.status}",
                                "elapsed": elapsed_time
                            })
                            self.running = False
                        backoff = 1.0
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self.status_update.emit({
                    "status": "retrying",
                    "error": f"Connection failed: {str(e)}",
                    "elapsed": elapsed_time
                })
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 5.0)
            elapsed_time += self.interval
        
        self.finished.emit()
        self.thread.quit()

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.quit()
        self.thread.wait()