import os
import sys
import queue
import socket
import logging
import traceback
import platform
import requests
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt6.QtCore import QTimer, QObject, pyqtSignal, QThread

# --- Local Imports ---
from gui.pyqt_ui import ConnectionDialog, MainWindow, StartupDialog
from state_manager import StateManager
from utils.async_handler import AsyncHandler
from utils.venv_utils import get_venv_python
from utils.virtual_devices import VirtualDeviceManager
from utils.camera_handler import CameraHandler
from utils.audio_handler import AudioHandler
from utils.gpu_utils import check_gpu
from services.media_service import MediaService
from services.settings import Settings
from services.process_monitor import ProcessMonitor
from utils.health_check import HealthCheckWorker
from services.runpod_manager import RunPodManager
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)

class StatsUpdateSignal(QObject):
    update = pyqtSignal(dict, dict)

class DeploymentWorker(QThread):
    status_update = pyqtSignal(str)
    deployment_complete = pyqtSignal(str, str, str, str)
    error = pyqtSignal(str)

    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key
        self.stopped = False

    def stop(self):
        self.stopped = True

    def run(self):
        try:
            manager = RunPodManager(self.api_key)
            voice_image = "spruceemma/voice-backend:latest"
            face_image = "spruceemma/face-backend:latest"
            
            self.status_update.emit("Deploying voice backend (this can take 5-10 mins)...")
            if self.stopped:
                return
            voice_url, voice_id = manager.deploy_and_poll_endpoint("my-voice-backend", voice_image, 8080)
            
            self.status_update.emit("Deploying face backend (this can take 5-10 mins)...")
            if self.stopped:
                return
            face_url, face_id = manager.deploy_and_poll_endpoint("my-face-backend", face_image, 8081)
            
            if not self.stopped:
                self.deployment_complete.emit(voice_url, face_url, voice_id, face_id)
        except Exception as e:
            if not self.stopped:
                logging.error(f"Deployment error: {str(e)}\n{traceback.format_exc()}")
                self.error.emit(str(e))
                
class AppController:
    def __init__(self, initial_mode_config):
        self.app = QApplication.instance()
        self.state = StateManager()
        self.deployment_worker = None
        self.settings = Settings()
        self.async_handler = AsyncHandler()
        self.input_frame_q = queue.Queue()
        self.output_frame_q = queue.Queue()
        self.audio_input_q = queue.Queue()
        self.ui = MainWindow(
            callbacks={
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
                "on_speaker_toggle": self.on_speaker_toggle,
                "on_closing": self.on_closing,
            },
            input_queue=self.input_frame_q,
            output_queue=self.output_frame_q
        )
        self.configure_initial_state(initial_mode_config)
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
            # NOTE: Do NOT start worker here—delay to apply_startup_config()
        elif self.state.operating_mode == "custom_urls":
            self.state.face_server_endpoint = config.get("face_url", "")
            self.state.voice_server_endpoint = config.get("voice_url", "")
            # NOTE: Do NOT start stream here—delay to apply_startup_config()
            
    def apply_startup_config(self):
        """Start services/threads after UI is shown and events processed."""
        mode = self.state.operating_mode
        if mode == "cloud":
            self.ui.update_status_message("Starting cloud deployment...")
            self.start_deployment_worker()
        elif mode == "custom_urls":
            self.ui.update_status_message("Connecting to custom servers...")
            self.start_media_stream()
        elif mode in ["local_gpu", "local_cpu"]:
            msg = f"Starting local {mode} backends..."
            self.ui.update_status_message(msg)
            self.connect_local_backends()
        elif mode == "offline":
            self.ui.update_status_message("Running in offline mode.")

        # For connected modes, update button to "Disconnect" state
        if mode != "offline":
            self.ui.connect_button.setText("Disconnect")
            self.ui.connect_button.setObjectName("disconnect_button")
            self.ui.setStyleSheet(self.ui.styleSheet())  # Refresh for red styling

    def start_deployment_worker(self):
        if self.deployment_worker and self.deployment_worker.isRunning():
            self.deployment_worker.stop()
            self.deployment_worker.wait()
        try:
            self.deployment_worker = DeploymentWorker(self.state.runpod_api_key)
            self.deployment_worker.status_update.connect(self.ui.update_status_message)
            self.deployment_worker.error.connect(lambda msg: self.ui.update_status_message(msg, is_error=True))
            self.deployment_worker.deployment_complete.connect(self.handle_deployment_complete)
            self.deployment_worker.start()
        except (ValueError, RuntimeError) as e:
            logging.error(f"Failed to start deployment worker: {str(e)}")
            self.ui.update_status_message(f"Error: {str(e)}", is_error=True)

    def is_port_in_use(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def check_backend_health(self, url):
        try:
            response = requests.get(f"{url}/health", timeout=5)
            return response.status_code == 200 and response.json().get('status') == 'healthy'
        except Exception as e:
            logging.debug(f"Health check failed for {url}: {str(e)}")
            return False

    def terminate_port_processes(self, port):
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.laddr.port == port and conn.pid:
                    process = psutil.Process(conn.pid)
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        process.kill()
                        logging.warning(f"Process {conn.pid} on port {port} killed after timeout.")
                    logging.info(f"Terminated process {conn.pid} on port {port}.")
        except Exception as e:
            logging.error(f"Failed to terminate processes on port {port}: {str(e)}")
            return False
        return True
    
   
    def connect_local_backends(self):
        ports = {8080: "voice", 8081: "face"}
        port_status = {}
        
        # Check which ports are in use and their health
        for port, backend in ports.items():
            in_use = self.is_port_in_use(port)
            health_url = f"http://localhost:{port}"
            healthy = self.check_backend_health(health_url) if in_use else False
            port_status[port] = {
                "in_use": in_use,
                "healthy": healthy,
                "backend": backend
            }
        
        # Determine if we can use existing backends
        can_use_voice = port_status[8080]["in_use"] and port_status[8080]["healthy"]
        can_use_face = port_status[8081]["in_use"] and port_status[8081]["healthy"]
        
        if can_use_voice and can_use_face:
            logging.info("Existing backends on ports 8080/8081 are healthy. Using them.")
            self.state.face_server_endpoint = "http://localhost:8081"
            self.state.voice_server_endpoint = "http://localhost:8080"
            self.start_media_stream()
            return
        
        def start_backend(self, backend_name: str, port: int):
            backend_path = os.path.join("backends", backend_name)
            python_cmd = get_venv_python(backend_path)
            main_script = f"{backend_name}_main.py"

            process_monitor = ProcessMonitor(
                [python_cmd, main_script],
                cwd=backend_path
            )
            
            process_monitor.signals.status_update.connect(
                lambda msg: self.ui.update_status_message(f"{backend_name.capitalize()} backend: {msg}")
            )
            process_monitor.signals.error.connect(
                lambda msg: self.ui.update_status_message(f"{backend_name.capitalize()} backend error: {msg}", is_error=True)
            )
            
            process_monitor.start()
            return process_monitor

        
        # Collect ports that need action: either not in use (start new) or in use but unhealthy (terminate and start new)
        ports_to_terminate = []
        for port, status in port_status.items():
            if status["in_use"] and not status["healthy"]:
                ports_to_terminate.append(port)
        
        if ports_to_terminate:
            msg = QMessageBox(self.ui)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Ports in Use by Unhealthy Processes")
            msg.setText(
                f"The following ports are in use by unhealthy backend processes:\n{', '.join(map(str, ports_to_terminate))}\n\n"
                "To proceed, these processes must be terminated, which may affect other running applications.\n"
                "Do you want to close these processes and continue?"
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg.setDefaultButton(QMessageBox.StandardButton.No)
            
            if msg.exec() != QMessageBox.StandardButton.Yes:
                self.ui.update_status_message(f"Cannot proceed: Unhealthy processes on ports {', '.join(map(str, ports_to_terminate))}.", is_error=True)
                return
            
            # Terminate unhealthy processes
            for port in ports_to_terminate:
                if not self.terminate_port_processes(port):
                    self.ui.update_status_message(f"Failed to free port {port}. Close conflicting apps manually.", is_error=True)
                    return
        
        # Now start any missing or freshly terminated backends
        ports_to_start = [port for port, status in port_status.items() if not status["in_use"] or port in ports_to_terminate]
        
        if ports_to_start:
            # Update port_status after terminations
            for port in ports_to_start:
                port_status[port]["in_use"] = self.is_port_in_use(port)
        
        try:
         
            # Start required backends
            if 8080 in ports_to_start:
                self.voice_process_monitor = start_backend("voice", 8080)
            if 8081 in ports_to_start:
                self.face_process_monitor = start_backend("face", 8081)

            self.state.face_server_endpoint = "http://localhost:8081"
            self.state.voice_server_endpoint = "http://localhost:8080"

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

            if self.voice_process_monitor:
                self.voice_process_monitor.signals.error.connect(lambda msg: handle_crash("Voice",  start_backend("voice", 8080)))
            if self.face_process_monitor:
                self.face_process_monitor.signals.error.connect(lambda msg: handle_crash("Face", start_backend("voice", 8081)))

            self.start_media_stream()
        except Exception as e:
            logging.error(f"Failed to start local backends: {str(e)}\n{traceback.format_exc()}")
            self.ui.update_status_message(f"Failed to start local backends: {str(e)}", is_error=True)
        
    def handle_deployment_complete(self, voice_url, face_url, voice_id, face_id):
        self.state.voice_server_endpoint = voice_url
        self.state.face_server_endpoint = face_url
        self.state.voice_endpoint_id = voice_id
        self.state.face_endpoint_id = face_id
        self.ui.update_status_message("Warming up cloud workers...", is_error=False)
        self.start_media_stream()

    def start_media_stream(self):
        try:
            if self.state.operating_mode in ["local_gpu", "local_cpu", "cloud", "custom_urls"]:
                # Verify backend connectivity
                if not self.check_backend_health(self.state.voice_server_endpoint):
                    self.ui.update_status_message("Voice backend not ready.", is_error=True)
                    self.state.voice_backend_status = "error"
                    return
                if not self.check_backend_health(self.state.face_server_endpoint):
                    self.ui.update_status_message("Face backend not ready.", is_error=True)
                    self.state.face_backend_status = "error"
                    return
                
                # Fetch available voices
                try:
                    response = requests.get(f"{self.state.voice_server_endpoint}/available-models", timeout=5)
                    if response.status_code == 200:
                        voices = response.json().get("models", [])
                        self.state.available_voices = voices
                        self.ui.update_voice_list(voices)
                        if voices and not self.state.selected_voice:
                            self.state.selected_voice = voices[0]
                            self.ui.voice_selector.setCurrentText(voices[0])
                            self.on_voice_change(voices[0])
                    else:
                        logging.warning(f"Failed to fetch voices: {response.status_code} {response.text}")
                        self.ui.update_status_message("Could not fetch available voices.", is_error=True)
                except Exception as e:
                    logging.error(f"Error fetching voices: {str(e)}")
                    self.ui.update_status_message("Error fetching available voices.", is_error=True)

                # Initialize virtual devices
                if not self.virtual_devices:
                    self.virtual_devices = VirtualDeviceManager(
                        self.state, self.audio_input_q, self.state.selected_resolution
                    )
                
                if self.state.operating_mode != "offline":
                    self.virtual_devices.start_audio_stream(
                        self.state.selected_input_device_id, self.state.selected_output_device_id
                    )

                # Start health check polling
                if not self.face_poller:
                    self.face_poller = HealthCheckWorker(self.state.face_server_endpoint)
                    self.face_poller.status_update.connect(
                        lambda s: self.handle_status_update("face", s)
                    )
                    self.face_poller.start()
                
                if not self.voice_poller:
                    self.voice_poller = HealthCheckWorker(self.state.voice_server_endpoint)
                    self.voice_poller.status_update.connect(
                        lambda s: self.handle_status_update("voice", s)
                    )
                    self.voice_poller.start()

                # Start media service
                self.media_service = MediaService(
                    self.state, self.input_frame_q, self.output_frame_q,
                    self.audio_input_q, self.async_handler,
                    self.virtual_devices, self.stats_signal
                )
                self.media_service.signals.backend_failure.connect(
                    lambda msg: self.ui.update_status_message(msg, is_error=True)
                )
                self.media_service.start()
                
                if self.state.target_face_path:
                    self.on_select_face(self.state.target_face_path)

                self.ui.update_status_message("Connected to backends.", is_error=False)
                self.state.server_status = "Connected."
        except Exception as e:
            logging.error(f"Failed to start media stream: {str(e)}\n{traceback.format_exc()}")
            self.ui.update_status_message(f"Failed to start media stream: {str(e)}", is_error=True)

    def disconnect_all(self):
        try:
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
            self.ui.update_status_message("Disconnected.", is_error=False)
        except Exception as e:
            logging.error(f"Error during disconnect: {str(e)}\n{traceback.format_exc()}")
            self.ui.update_status_message(f"Error during disconnect: {str(e)}", is_error=True)
    def handle_connect_disconnect(self):
        try:
            if self.state.server_status == "Connected.":
                self.disconnect_all()
                self.ui.connect_button.setText("Connect")
                self.ui.connect_button.setObjectName("connect_button")
                self.ui.setStyleSheet(self.ui.styleSheet())  # Refresh stylesheet
            else:
                dialog = ConnectionDialog(has_gpu=check_gpu(), parent=self.ui, stop_thread_callback=self.stop_deployment_worker)
                if dialog.exec():
                    config = dialog.result
                    if config:
                        self.state.operating_mode = config.get("mode", "offline")
                        if self.state.operating_mode == "cloud":
                            self.state.runpod_api_key = config.get("api_key", "")
                            self.ui.update_status_message("Starting cloud deployment...")
                            self.start_deployment_worker()
                        elif self.state.operating_mode == "custom_urls":
                            self.state.face_server_endpoint = config.get("face_url", "")
                            self.state.voice_server_endpoint = config.get("voice_url", "")
                            self.ui.update_status_message("Connecting to custom servers...")
                            self.start_media_stream()
                        elif self.state.operating_mode == "local_gpu":
                            self.ui.update_status_message("Starting local GPU backends...")
                            self.connect_local_backends()
                        elif self.state.operating_mode == "local_cpu":
                            self.ui.update_status_message("Starting local CPU backends...")
                            self.connect_local_backends()
                        elif self.state.operating_mode == "offline":
                            self.ui.update_status_message("Running in offline mode.", is_error=False)
                        self.ui.connect_button.setText("Disconnect")
                        self.ui.connect_button.setObjectName("disconnect_button")
                        self.ui.setStyleSheet(self.ui.styleSheet())  # Refresh stylesheet
        except Exception as e:
            logging.error(f"Connect/Disconnect error: {str(e)}\n{traceback.format_exc()}")
            self.ui.update_status_message(f"Error: {str(e)}", is_error=True)

    def stop_deployment_worker(self):
        if self.deployment_worker and self.deployment_worker.isRunning():
            self.deployment_worker.stop()
            self.deployment_worker.wait()
        
    def handle_status_update(self, backend: str, status: str):
        self.state.server_status = f"{backend.capitalize()} backend: {status}"
        if status == "healthy":
            if backend == "face":
                self.state.face_backend_status = "connected"
            elif backend == "voice":
                self.state.voice_backend_status = "connected"
        else:
            if backend == "face":
                self.state.face_backend_status = "error"
            elif backend == "voice":
                self.state.voice_backend_status = "error"
            self.ui.update_status_message(f"{backend.capitalize()} backend unhealthy: {status}", is_error=True)

    def update_stats_ui(self, video_stats, audio_stats):
        self.ui.update_stats(video_stats, audio_stats)

    def scan_for_cameras(self):
        camera_handler = CameraHandler()
        camera_handler.start_background_scan(
            lambda cameras: self.ui.update_camera_list(cameras)
        )
        self.state.available_cameras = camera_handler.camera_list

    def scan_for_audio_devices(self):
        audio_handler = AudioHandler()
        audio_handler.start_background_scan(
            lambda devices: self.ui.update_audio_device_lists(devices['input'], devices['output'])
        )
        self.state.available_input_devices = audio_handler.device_list['input']
        self.state.available_output_devices = audio_handler.device_list['output']
        if self.state.available_input_devices and not self.state.selected_input_device_id:
            self.state.selected_input_device_id = self.state.available_input_devices[0]['id']
        if self.state.available_output_devices and not self.state.selected_output_device_id:
            self.state.selected_output_device_id = self.state.available_output_devices[0]['id']

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

    def on_select_face(self, filepath=None):
        if not filepath:
            filepath, _ = QFileDialog.getOpenFileName(self.ui, "Select Target Face", "", "Images (*.png *.jpg *.jpeg)")
        if filepath:
            self.state.target_face_path = filepath
            logging.info(f"Target face selected: {filepath}")
            if self.state.face_server_endpoint and self.state.face_backend_status == "connected":
                try:
                    with open(filepath, "rb") as f:
                        response = requests.post(f"{self.state.face_server_endpoint}/set-source-face", files={"file": f})
                    if response.status_code == 200:
                        logging.info("Source face set successfully on backend.")
                        self.ui.update_status_message("Source face set successfully.", is_error=False)
                    else:
                        logging.error(f"Failed to set source face: {response.status_code} {response.text}")
                        self.ui.update_status_message(f"Failed to set source face: {response.text}", is_error=True)
                except Exception as e:
                    logging.error(f"Error setting source face: {str(e)}")
                    self.ui.update_status_message(f"Error setting source face: {str(e)}", is_error=True)

    def on_enhancement_toggle(self, checked):
        self.state.is_enhancement_on = checked
        logging.info(f"Face enhancement {'enabled' if checked else 'disabled'}")
        if self.state.face_server_endpoint and self.state.face_backend_status == "connected":
            try:
                response = requests.post(
                    f"{self.state.face_server_endpoint}/update-settings",
                    json={"use_face_enhancement": checked}
                )
                if response.status_code == 200:
                    logging.info("Face enhancement settings updated on backend.")
                else:
                    logging.error(f"Failed to update face enhancement: {response.status_code} {response.text}")
            except Exception as e:
                logging.error(f"Error updating face enhancement: {str(e)}")

    def on_voice_change(self, voice_name):
        self.state.selected_voice = voice_name
        logging.info(f"AI Voice changed to: {voice_name}")
        if self.state.voice_server_endpoint and self.state.voice_backend_status == "connected":
            try:
                response = requests.post(
                    f"{self.state.voice_server_endpoint}/update-settings",
                    json={"model_name": voice_name}
                )
                if response.status_code == 200:
                    logging.info(f"Voice model set to {voice_name} on backend.")
                else:
                    logging.error(f"Failed to set voice model: {response.status_code} {response.text}")
            except Exception as e:
                logging.error(f"Error setting voice model: {str(e)}")

    def on_speaker_toggle(self, checked):
        if self.media_service:
            self.media_service.send_audio_to_speaker = checked
            logging.info(f"Speaker output {'enabled' if checked else 'disabled'}")
        else:
            self.ui.update_status_message("Cannot toggle speaker: Media service not active.", is_error=True)

    def set_push_to_talk(self, is_active):
        self.state.is_push_to_talk_active = is_active
        self.ui.talk_button.setChecked(is_active)

    def on_closing(self):
        logging.info("Application closing...")
        if self.deployment_worker and self.deployment_worker.isRunning():
            self.deployment_worker.stop()
            self.deployment_worker.wait()
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

    # --- KEY CHANGE 1 ---
    # Temporarily prevent the app from quitting when the dialog closes
    app.setQuitOnLastWindowClosed(False)

    config = None
    if has_gpu:
        logging.info("Compatible GPU found. Starting in local GPU mode.")
        config = {"mode": "local_gpu"}
    else:
        logging.info("No compatible GPU detected. Showing startup options.")
        startup_dialog = StartupDialog(stop_thread_callback=lambda: None)
        if startup_dialog.exec():
            config = startup_dialog.result
        else:
            config = None

    if config:
        controller = AppController(config)  # Sets state but does NOT start threads/services
        controller.ui.show()

        # Process any pending events (e.g., UI layout/timer init) before starting threads
        app.processEvents()

        # Apply startup config NOW (after show(), before exec())
        controller.apply_startup_config()

        # --- KEY CHANGE 2 ---
        # Restore the default behavior so the app closes when the main window is closed
        app.setQuitOnLastWindowClosed(True)

        sys.exit(app.exec())
    else:
        logging.info("No startup option selected. Exiting application.")
        sys.exit(0)