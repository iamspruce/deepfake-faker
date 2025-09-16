import subprocess
import logging
import traceback
from PyQt6.QtCore import QObject, pyqtSignal, QThread

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProcessSignals(QObject):
    status_update = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

class ProcessWorker(QObject):
    def __init__(self, command, env=None, cwd=None):
        super().__init__()
        self.command = command
        self.env = env
        self.cwd = cwd
        self.process = None
        self.signals = ProcessSignals()

    def run(self):
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env,
                cwd=self.cwd,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            
            for line in iter(self.process.stdout.readline, ''):
                self.signals.status_update.emit(line.strip())
            
            stderr_output = self.process.stderr.read().strip()
            if self.process.wait() != 0 and stderr_output:
                self.signals.error.emit(f"Process failed: {stderr_output}")
        
        except Exception as e:
            self.signals.error.emit(f"Process error: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.signals.finished.emit()
            
    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                logging.warning(f"Process {self.command} killed after termination timeout.")

class ProcessMonitor:
    def __init__(self, command, env=None, cwd=None):
        self.thread = QThread()
        self.worker = ProcessWorker(command, env=env, cwd=cwd)
        self.worker.moveToThread(self.thread)
        
        self.signals = self.worker.signals
        
        self.thread.started.connect(self.worker.run)
        self.worker.signals.finished.connect(self.thread.quit)
        self.worker.signals.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

    def start(self):
        self.thread.start()

    def stop(self):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()