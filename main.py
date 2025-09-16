import os
import sys
import queue
import socket
import logging
import traceback
import platform
from PyQt6.QtWidgets import QApplication, QFileDialog
from PyQt6.QtCore import QTimer, QObject, pyqtSignal, QThread

# --- Local Imports ---
from gui.pyqt_ui import MainWindow, StartupDialog
from state_manager import StateManager
from utils.async_handler import AsyncHandler
from utils.virtual_devices import VirtualDeviceManager
from utils.camera_handler import CameraHandler
from utils.audio_handler import AudioHandler
from utils.gpu_utils import check_gpu
from services.media_service import MediaService
from services.settings import Settings
from services.process_monitor import ProcessMonitor
from utils.health_check import HealthCheckWorker
from services.runpod_manager import RunPodManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)

class StatsUpdateSignal(QObject):
    update = pyqtSignal(dict, dict)

class DeploymentWorker(QThread):
    status_update = pyqtSignal(str)
    deployment_complete = pyqtSignal(str, str, str, str)  # voice_url, face_url, voice_id, face_id
    error = pyqtSignal(str)

    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key

    def run(self):
        try:
            manager = RunPodManager(self.api_key)
            voice_image = "spruceemma/voice-backend:latest"
            face_image = "spruceemma/face-backend:latest"
            
            self.status_update.emit("Deploying voice backend (this can take 5-10 mins)...")
            voice_url, voice_id = manager.deploy_and_poll_endpoint("my-voice-backend", voice_image, 8080)
            
            self.status_update.emit("Deploying face backend (this can take 5-10 mins)...")
            face_url, face_id = manager.deploy_and_poll_endpoint("my-face-backend", face_image, 8081)
            
            self.deployment_complete.emit(voice_url, face_url, voice_id, face_id)
        except Exception as e:
            logging.error(f"Deployment error: {str(e)}\n{traceback.format_exc()}")
            self.error.emit(str(e))

class AppController:
    def __init__(self, initial_mode_config):
        self.app = QApplication.instance()
        self.state = StateManager()
        self.settings = Settings()
        self.async_handler = AsyncHandler()
        self.ui = MainWindow(callbacks={
            "on_connect_disconnect": self.handle_connect_disconnect,
            "on_resolution_change": self.on_resolution_change,
            "on_camera_change": self.on_camera_change,
            "on_refresh_cameras": self.scan_for_cameras,
            "on_refresh_audio": self.scan_for_audio_devices,
            "on_select_face": self.on_select_face,
            "on_enhancement_toggle": self.on_enhancement_toggle,
            "on_input_device_change": self.on_input_device_change,
            "on_output_device_change": self.on_output_device_change,
            "on_voice_change": self.on_voice_change,
            "on_space_press": lambda: self.set_push_to_talk(True),
            "on_space_release": lambda: self.set_push_to_talk(False),
            "on_closing": self.on_closing,
        })
        self.configure_initial_state(initial_mode_config)
        self.input_frame_q = queue.Queue()
        self.output_frame_q = queue.Queue()
        self.audio_input_q = queue.Queue()
        self.media_service = None
        self.virtual_devices = None
        self.voice_process_monitor = None
        self.face_process_monitor = None
        self.face_poller = None
        self.voice_poller = None
        self.deployment_worker = None
        self.stats_signal = StatsUpdateSignal()
        self.stats_signal.update.connect(self.update_stats_ui)

        # Initialize camera and audio devices
        self.scan_for_cameras()
        self.scan_for_audio_devices()

    def configure_initial_state(self, config):
        self.state.operating_mode = config.get("mode", "offline")
        if self.state.operating_mode == "cloud":
            self.state.runpod_api_key = config.get("api_key", "")
            self.deployment_worker = DeploymentWorker(self.state.runpod_api_key)
            self.deployment_worker.status_update.connect(self.ui.update_status_message)
            self.deployment_worker.error.connect(lambda msg: self.ui.update_status_message(msg, is_error=True))
            self.deployment_worker.deployment_complete.connect(self.handle_deployment_complete)
            self.deployment_worker.start()
        elif self.state.operating_mode == "custom_urls":
            self.state.face_server_endpoint = config.get("face_url", "")
            self.state.voice_server_endpoint = config.get("voice_url", "")
            self.start_media_stream()

    def is_port_in_use(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def connect_local_backends(self):
        if self.is_port_in_use(8080) or self.is_port_in_use(8081):
            self.ui.update_status_message("Ports 8080/8081 are in use. Close conflicting apps.", is_error=True)
            return
        try:
            is_windows = platform.system() == "Windows"
            venv_activate = "Scripts\\activate.bat" if is_windows else "bin/activate"
            python_cmd = "python" if is_windows else "python3"

            def start_voice():
                voice_venv_path = os.path.join("backends", "voice", venv_activate)
                voice_cwd = os.path.join(os.getcwd(), "backends", "voice")
                voice_cmd = [python_cmd, "voice_main.py"]
                if is_windows:
                    voice_cmd = [voice_venv_path, "&&", python_cmd, "voice_main.py"]
                self.voice_process_monitor = ProcessMonitor(voice_cmd, cwd=voice_cwd)
                self.voice_process_monitor.signals.status_update.connect(
                    lambda msg: self.ui.update_status_message(f"Voice backend: {msg}")
                )
                self.voice_process_monitor.signals.error.connect(
                    lambda msg: self.ui.update_status_message(f"Voice backend error: {msg}", is_error=True)
                )
                self.voice_process_monitor.start()

            def start_face():
                face_venv_path = os.path.join("backends", "face", venv_activate)
                face_cwd = os.path.join(os.getcwd(), "backends", "face")
                face_cmd = [python_cmd, "face_main.py"]
                if is_windows:
                    face_cmd = [face_venv_path, "&&", python_cmd, "face_main.py"]
                self.face_process_monitor = ProcessMonitor(face_cmd, cwd=face_cwd)
                self.face_process_monitor.signals.status_update.connect(
                    lambda msg: self.ui.update_status_message(f"Face backend: {msg}")
                )
                self.face_process_monitor.signals.error.connect(
                    lambda msg: self.ui.update_status_message(f"Face backend error: {msg}", is_error=True)
                )
                self.face_process_monitor.start()

            start_voice()
            start_face()
            self.state.face_server_endpoint = "http://localhost:8081"
            self.state.voice_server_endpoint = "http://localhost:8080"

            # Auto-restart on crash
            def handle_crash(backend, start_fn):
                logging.warning(f"{backend} backend crashed. Attempting restart...")
                self.ui.update_status_message(f"{backend} backend crashed. Retrying...", is_error=True)
                if backend == "Voice" and self.voice_process_monitor:
                    self.voice_process_monitor.stop()
                    self.voice_process_monitor = None
                elif backend == "Face" and self.face_process_monitor:
                    self.face_process_monitor.stop()
                    self.face_process_monitor = None
                start_fn()

            self.voice_process_monitor.signals.error.connect(lambda msg: handle_crash("Voice", start_voice))
            self.face_process_monitor.signals.error.connect(lambda msg: handle_crash("Face", start_face))

            self.start_media_stream()
        except Exception as e:
            logging.error(f"Failed to start local backends: {str(e)}\n{traceback.format_exc()}")
            self.ui.update_status_message(f"Failed to start local backends: {str(e)}", is_error=True)

    def handle_deployment_complete(self, voice_url, face_url, voice_id, face_id):
        self.state.voice_server_endpoint = voice_url
        self.state.face_server_endpoint = face_url
        self.state.voice_endpoint_id = voice_id
        self.state.face_endpoint_id = face_id
        self.start_media_stream()
        self.ui.update_status_message("Deployment complete. Connected to cloud backends.")

    def start_media_stream(self):
        if self.state.operating_mode in ['local_gpu', 'local_cpu', 'cloud', 'custom_urls']:
            if self.state.face_server_endpoint and self.state.voice_server_endpoint:
                self.face_poller = HealthCheckWorker(f"{self.state.face_server_endpoint}/health")
                self.face_poller.status_update.connect(
                    lambda status: self.ui.update_status_message(
                        f"Face backend: {status.get('status', 'unknown')} ({status.get('elapsed', 0):.1f}s)",
                        is_error=status.get('status') == 'error'
                    )
                )
                self.face_poller.start()
                self.voice_poller = HealthCheckWorker(f"{self.state.voice_server_endpoint}/health")
                self.voice_poller.status_update.connect(
                    lambda status: self.ui.update_status_message(
                        f"Voice backend: {status.get('status', 'unknown')} ({status.get('elapsed', 0):.1f}s)",
                        is_error=status.get('status') == 'error'
                    )
                )
                self.voice_poller.start()
            self.virtual_devices = VirtualDeviceManager(self.state, self.audio_input_q, self.state.selected_resolution)
            self.media_service = MediaService(
                self.state, self.input_frame_q, self.output_frame_q,
                self.audio_input_q, self.async_handler, self.virtual_devices, self.stats_signal
            )
            self.media_service.start()

    def handle_connect_disconnect(self):
        if self.state.is_fully_connected():
            self.disconnect_all()
            self.ui.connect_button.setText("Connect")
            self.ui.connect_button.setObjectName("connect_button")
            self.ui.setStyleSheet(self.ui.STYLESheet)
        else:
            if self.state.operating_mode in ["local_gpu", "local_cpu"]:
                self.connect_local_backends()
            elif self.state.operating_mode in ["cloud", "custom_urls"]:
                self.start_media_stream()
            self.ui.connect_button.setText("Disconnect")
            self.ui.connect_button.setObjectName("disconnect_button")
            self.ui.setStyleSheet(self.ui.STYLESheet)

    def disconnect_all(self):
        if self.media_service:
            self.media_service.stop()
            self.media_service = None
        if self.virtual_devices:
            self.virtual_devices.close()
            self.virtual_devices = None
        if self.face_poller:
            self.face_poller.stop()
            self.face_poller = None
        if self.voice_poller:
            self.voice_poller.stop()
            self.voice_poller = None
        if self.voice_process_monitor:
            self.voice_process_monitor.stop()
            self.voice_process_monitor = None
        if self.face_process_monitor:
            self.face_process_monitor.stop()
            self.face_process_monitor = None
        self.state.reset_for_disconnection()
        self.ui.update_status_message("Disconnected from all services.")

    def update_stats_ui(self, video_stats, audio_stats):
        self.ui.update_stats(video_stats, audio_stats)

    def scan_for_cameras(self):
        camera_handler = CameraHandler()
        camera_handler.start_background_scan(lambda cameras: self.ui.update_camera_list(cameras))

    def scan_for_audio_devices(self):
        audio_handler = AudioHandler()
        devices = audio_handler.list_devices()
        input_devices = devices.get('input', [])
        output_devices = devices.get('output', [])
        self.state.available_input_devices = input_devices
        self.state.available_output_devices = output_devices
        self.ui.update_audio_device_lists(input_devices, output_devices)
        if input_devices and not self.state.selected_input_device_id:
            self.state.selected_input_device_id = input_devices[0]['id']
        if output_devices and not self.state.selected_output_device_id:
            self.state.selected_output_device_id = output_devices[0]['id']

    def on_resolution_change(self, res_text):
        self.state.selected_resolution = self.state.resolutions[res_text]
        self.settings.save({"resolution": res_text})
        if self.media_service:
            self.disconnect_all()
            self.start_media_stream()
        self.ui.update_video_panel_size(self.state.selected_resolution)

    def on_camera_change(self, cam_name):
        for cam_id, name in self.state.available_cameras:
            if name == cam_name:
                self.state.selected_camera_index = cam_id
                if self.media_service:
                    self.disconnect_all()
                    self.start_media_stream()
                break

    def on_input_device_change(self, device_name):
        for dev in self.state.available_input_devices:
            if dev['name'] == device_name:
                self.state.selected_input_device_id = dev['id']
                if self.virtual_devices and self.virtual_devices.audio_stream:
                    self.virtual_devices.stop_audio_stream()
                    self.virtual_devices.start_audio_stream(
                        self.state.selected_input_device_id, self.state.selected_output_device_id
                    )
                break
    
    def on_output_device_change(self, device_name):
        for dev in self.state.available_output_devices:
            if dev['name'] == device_name:
                self.state.selected_output_device_id = dev['id']
                if self.virtual_devices and self.virtual_devices.audio_stream:
                    self.virtual_devices.stop_audio_stream()
                    self.virtual_devices.start_audio_stream(
                        self.state.selected_input_device_id, self.state.selected_output_device_id
                    )
                break

    def on_select_face(self):
        filepath, _ = QFileDialog.getOpenFileName(self.ui, "Select Target Face", "", "Images (*.png *.jpg *.jpeg)")
        if filepath:
            self.state.target_face_path = filepath
            logging.info(f"Target face selected: {filepath}")

    def on_enhancement_toggle(self, checked):
        self.state.is_enhancement_on = checked
        logging.info(f"Face enhancement {'enabled' if checked else 'disabled'}")

    def on_voice_change(self, voice_name):
        self.state.selected_voice = voice_name
        logging.info(f"AI Voice changed to: {voice_name}")

    def set_push_to_talk(self, is_active):
        self.state.is_push_to_talk_active = is_active
        self.ui.talk_button.setChecked(is_active)

    def run(self):
        self.ui.show()
        sys.exit(self.app.exec())

    def on_closing(self):
        logging.info("Application closing...")
        if self.state.operating_mode == "cloud" and self.state.runpod_api_key:
            endpoint_ids_to_terminate = [
                eid for eid in [self.state.voice_endpoint_id, self.state.face_endpoint_id] if eid
            ]
            if endpoint_ids_to_terminate:
                manager = RunPodManager(self.state.runpod_api_key)
                try:
                    manager.terminate_endpoints(endpoint_ids_to_terminate)
                except Exception as e:
                    logging.error(f"Failed to terminate endpoints: {str(e)}\n{traceback.format_exc()}")
        self.disconnect_all()
        self.async_handler.stop()
        self.settings.save({"resolution": self.ui.resolution_selector.currentText()})

if __name__ == '__main__':
    app = QApplication(sys.argv)
    has_gpu = check_gpu()
    initial_config = None
    if has_gpu:
        logging.info("Compatible GPU found. Starting in local GPU mode.")
        initial_config = {"mode": "local_gpu"}
    else:
        logging.info("No compatible GPU detected. Showing startup options.")
        startup_dialog = StartupDialog()
        if startup_dialog.exec():
            initial_config = startup_dialog.result
        else:
            initial_config = None
    if initial_config:
        controller = AppController(initial_config)
        controller.run()
    else:
        logging.info("No startup option selected. Exiting application.")
        sys.exit(0)