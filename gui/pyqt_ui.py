import logging
import queue
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QDialog, QFrame, QLineEdit, QCheckBox, QSpacerItem, QSizePolicy,
    QRadioButton, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QPoint, QTimer
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QPalette, QColor
import qtawesome as qta
from PIL import Image
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)

STYLESheet = """
    QWidget {
        color: #ecf0f1;
        font-family: Arial, sans-serif;
    }
    QMainWindow, QDialog {
        background-color: #2c3e50;
    }
    QFrame#sidebar {
        background-color: #34495e;
        border-right: 1px solid #2c3e50;
    }
    QLabel {
        font-size: 14px;
    }
    QLabel#title {
        font-size: 22px;
        font-weight: bold;
        padding: 10px;
        color: #ffffff;
    }
    QLabel#video_panel {
        background-color: #212121;
        border: 1px solid #34495e;
        border-radius: 5px;
        min-width: 640px;
        min-height: 480px;
    }
    QLabel#pip_view {
        border: 2px solid #3498db;
        border-radius: 5px;
        background-color: #212121;
    }
    QLabel#stats_label { font-size: 12px; color: #bdc3c7; }
    QLabel#stats_value { font-size: 14px; color: #ffffff; font-weight: bold; }
    QLabel#status_message_error { color: #e74c3c; font-weight: bold; }
    QMessageBox {
        background-color: #2c3e50;
        color: #ecf0f1;
    }
    QMessageBox QLabel {
        color: #ecf0f1;
    }
    QMessageBox QPushButton {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 8px;
        border-radius: 3px;
    }
    QMessageBox QPushButton:hover {
        background-color: #4ea8e1;
    }
    QPushButton {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px;
        border-radius: 5px;
        font-size: 14px;
    }
    QPushButton:hover {
        background-color: #4ea8e1;
    }
    QPushButton:pressed {
        background-color: #2980b9;
    }
    QPushButton:disabled {
        background-color: #555;
        color: #999;
    }
    QPushButton#disconnect_button {
        background-color: #e74c3c;
    }
    QPushButton#disconnect_button:hover {
        background-color: #f1786b;
    }
    QPushButton#talk_button:checked {
        background-color: #2ecc71;
    }
    QPushButton#talk_button:checked:hover {
        background-color: #58d68d;
    }
    QComboBox {
        padding: 5px;
        border: 1px solid #7f8c8d;
        border-radius: 3px;
        background-color: #34495e;
    }
    QLineEdit {
        padding: 5px;
        border: 1px solid #7f8c8d;
        border-radius: 3px;
        background-color: #ecf0f1;
        color: #2c3e50;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
    }
    QFrame#settings_group {
        border: 1px solid #4a627a;
        border-radius: 5px;
        padding: 5px;
        margin-top: 10px;
    }
    QProgressBar {
        border: 1px solid #7f8c8d;
        border-radius: 5px;
        text-align: center;
        color: #ecf0f1;
    }
    QProgressBar::chunk {
        background-color: #3498db;
    }
"""

class ConnectionDialog(QDialog):
    def __init__(self, has_gpu, parent=None, stop_thread_callback=None):
        super().__init__(parent)
        self.stop_thread_callback = stop_thread_callback
        self.setWindowTitle("Connection Setup")
        self.setModal(True)
        self.setStyleSheet(STYLESheet)
        self.setMinimumWidth(350)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(15)
        self.layout.addWidget(QLabel("<h2>Processing Mode</h2>"))
        self.mode_selector = QComboBox()
        modes = ["Offline"]
        if has_gpu:
            modes.insert(0, "Local GPU")
        modes.append("RunPod Cloud")
        modes.append("Custom URLs")
        self.mode_selector.addItems(modes)
        self.mode_selector.setToolTip("Select the processing mode for AI computations")
        self.layout.addWidget(self.mode_selector)
        self.api_key_label = QLabel("RunPod API Key:")
        self.runpod_entry = QLineEdit()
        self.runpod_entry.setPlaceholderText("Enter your RunPod API key")
        self.runpod_entry.setEchoMode(QLineEdit.EchoMode.Password)
        self.face_url_label = QLabel("Face Backend URL:")
        self.face_url_entry = QLineEdit()
        self.face_url_entry.setPlaceholderText("e.g., http://localhost:8081")
        self.voice_url_label = QLabel("Voice Backend URL:")
        self.voice_url_entry = QLineEdit()
        self.voice_url_entry.setPlaceholderText("e.g., http://localhost:8080")
        self.runpod_radio = QRadioButton("Use RunPod Cloud")
        self.custom_url_radio = QRadioButton("Use Custom URLs")
        self.force_local_radio = QRadioButton("Force Local CPU")
        self.layout.addWidget(self.runpod_radio)
        self.layout.addWidget(self.api_key_label)
        self.layout.addWidget(self.runpod_entry)
        self.layout.addWidget(self.custom_url_radio)
        self.layout.addWidget(self.face_url_label)
        self.layout.addWidget(self.face_url_entry)
        self.layout.addWidget(self.voice_url_label)
        self.layout.addWidget(self.voice_url_entry)
        self.layout.addWidget(self.force_local_radio)
        confirm_button = QPushButton("Confirm")
        confirm_button.clicked.connect(self.accept)
        self.layout.addWidget(confirm_button)
        self.result = None
        self.temp_result = None

        self.mode_selector.currentTextChanged.connect(self.update_ui)
        self.runpod_radio.toggled.connect(self.update_ui)
        self.custom_url_radio.toggled.connect(self.update_ui)
        self.force_local_radio.toggled.connect(self.update_ui)
        self.update_ui()

    def update_ui(self):
        is_runpod = self.mode_selector.currentText() == "RunPod Cloud" or self.runpod_radio.isChecked()
        is_custom = self.mode_selector.currentText() == "Custom URLs" or self.custom_url_radio.isChecked()
        is_force_local = self.mode_selector.currentText() == "Offline" or self.force_local_radio.isChecked()
        self.api_key_label.setVisible(is_runpod)
        self.runpod_entry.setVisible(is_runpod)
        self.face_url_label.setVisible(is_custom)
        self.face_url_entry.setVisible(is_custom)
        self.voice_url_label.setVisible(is_custom)
        self.voice_url_entry.setVisible(is_custom)
        self.runpod_radio.setVisible(self.mode_selector.currentText() != "Offline")
        self.custom_url_radio.setVisible(self.mode_selector.currentText() != "Offline")
        self.force_local_radio.setVisible(self.mode_selector.currentText() != "Offline")

    def show_confirmation(self, mode, **kwargs):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Confirm Deployment")
        if mode == "cloud":
            msg.setText(
                "You are about to deploy cloud backends on RunPod using your API key.\n\n"
                "What happens next:\n"
                "- Two serverless endpoints (voice and face) will be created.\n"
                "- This may take 5-15 minutes and will incur costs (~$0.20-$1.00/hour per endpoint, ~$0.01-0.05 per cold start).\n"
                "- The app will poll until ready and connect automatically.\n"
                "- Endpoints will be terminated when you close the app.\n\n"
                "Proceed? (You can cancel to go back.)"
            )
            msg.setInformativeText(
                "Note: Ensure sufficient credits in your RunPod account (https://runpod.io/console/billing)."
            )
            self.temp_result = {"mode": mode, "api_key": kwargs["api_key"]}
        elif mode == "custom_urls":
            msg.setText(
                "You are about to connect to your custom server URLs.\n\n"
                "What happens next:\n"
                "- The app will attempt to connect to the provided face and voice URLs.\n"
                "- Health checks will run periodically.\n"
                "- No new deployments will occur.\n\n"
                "Proceed?"
            )
            self.temp_result = {"mode": mode, "face_url": kwargs["face_url"], "voice_url": kwargs["voice_url"]}
        elif mode == "local_cpu":
            msg.setText(
                "You are about to force local CPU mode (no GPU detected).\n\n"
                "What happens next:\n"
                "- Backends will run locally on your CPU (may be slow or unstable).\n"
                "- Virtual devices and media services will start.\n\n"
                "Proceed? (Performance may be poor.)"
            )
            self.temp_result = {"mode": mode}
        
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.Yes)
        
        if msg.exec() == QMessageBox.StandardButton.Yes:
            self.result = self.temp_result
            super().accept()
        else:
            pass

    def accept(self):
        if self.runpod_radio.isChecked():
            if not self.runpod_entry.text().strip():
                self.runpod_entry.setPlaceholderText("API Key required!")
                return
            self.result = {"mode": "cloud", "api_key": self.runpod_entry.text()}
        elif self.custom_url_radio.isChecked():
            if not (self.face_url_entry.text().strip() and self.voice_url_entry.text().strip()):
                self.face_url_entry.setPlaceholderText("Face URL required!" if not self.face_url_entry.text().strip() else "")
                self.voice_url_entry.setPlaceholderText("Voice URL required!" if not self.voice_url_entry.text().strip() else "")
                return
            self.result = {
                "mode": "custom_urls",
                "face_url": self.face_url_entry.text(),
                "voice_url": self.voice_url_entry.text()
            }
        elif self.force_local_radio.isChecked():
            self.result = {"mode": "local_cpu"}
        super().accept()  # Emit accepted signal
        
    def reject(self):
        super().reject()  # Emit rejected signal
        
    def closeEvent(self, event):
        if self.stop_thread_callback:
            self.stop_thread_callback()
        super().closeEvent(event)

class StartupDialog(QDialog):
    def __init__(self, parent=None, stop_thread_callback=None):
        super().__init__(parent)
        self.stop_thread_callback = stop_thread_callback
        self.setWindowTitle("System Requirements Check")
        self.setModal(True)
        self.setStyleSheet(STYLESheet)
        self.setMinimumWidth(450)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(15)
        title = QLabel("<h2>No Compatible GPU Detected</h2>")
        title.setWordWrap(True)
        self.layout.addWidget(title)
        info_label = QLabel(
            "This application requires a powerful GPU for real-time AI processing. "
            "Please choose how you would like to proceed:"
        )
        info_label.setWordWrap(True)
        self.layout.addWidget(info_label)
        self.runpod_radio = QRadioButton("Deploy to Cloud with RunPod API Key")
        self.runpod_radio.setToolTip("Use RunPod cloud services for processing")
        self.custom_url_radio = QRadioButton("Connect to Your Own Custom Servers")
        self.custom_url_radio.setToolTip("Provide URLs for your own face and voice servers")
        self.force_local_radio = QRadioButton("Force Run on this Computer (Not Recommended)")
        self.force_local_radio.setToolTip("Attempt to run on CPU (may be slow)")
        self.runpod_entry = QLineEdit()
        self.runpod_entry.setPlaceholderText("Enter your RunPod API Key...")
        self.runpod_entry.setToolTip("Enter your RunPod API key")
        self.face_url_entry = QLineEdit()
        self.face_url_entry.setPlaceholderText("Enter Face Backend URL...")
        self.face_url_entry.setToolTip("URL for face processing server")
        self.voice_url_entry = QLineEdit()
        self.voice_url_entry.setPlaceholderText("Enter Voice Backend URL...")
        self.voice_url_entry.setToolTip("URL for voice processing server")
        self.layout.addWidget(self.runpod_radio)
        self.layout.addWidget(self.runpod_entry)
        self.layout.addWidget(self.custom_url_radio)
        self.layout.addWidget(self.face_url_entry)
        self.layout.addWidget(self.voice_url_entry)
        self.layout.addWidget(self.force_local_radio)
        button_layout = QHBoxLayout()
        self.confirm_button = QPushButton("Confirm")
        self.close_button = QPushButton("Close App")
        button_layout.addStretch(1)
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.close_button)
        self.layout.addLayout(button_layout)
        self.runpod_radio.toggled.connect(self.toggle_inputs)
        self.custom_url_radio.toggled.connect(self.toggle_inputs)
        self.force_local_radio.toggled.connect(self.toggle_inputs)
        self.confirm_button.clicked.connect(self.accept)
        self.close_button.clicked.connect(self.reject)
        self.runpod_radio.setChecked(True)
        self.toggle_inputs()
        self.result = None

    def toggle_inputs(self):
        self.runpod_entry.setVisible(self.runpod_radio.isChecked())
        self.face_url_entry.setVisible(self.custom_url_radio.isChecked())
        self.voice_url_entry.setVisible(self.custom_url_radio.isChecked())

    def accept(self):
        if self.runpod_radio.isChecked():
            if not self.runpod_entry.text().strip():
                self.runpod_entry.setPlaceholderText("API Key required!")
                return
            self.result = {"mode": "cloud", "api_key": self.runpod_entry.text()}
        elif self.custom_url_radio.isChecked():
            if not (self.face_url_entry.text().strip() and self.voice_url_entry.text().strip()):
                self.face_url_entry.setPlaceholderText("Face URL required!" if not self.face_url_entry.text().strip() else "")
                self.voice_url_entry.setPlaceholderText("Voice URL required!" if not self.voice_url_entry.text().strip() else "")
                return
            self.result = {
                "mode": "custom_urls",
                "face_url": self.face_url_entry.text(),
                "voice_url": self.voice_url_entry.text()
            }
        elif self.force_local_radio.isChecked():
            self.result = {"mode": "local_cpu"}
        super().accept()
    
    def closeEvent(self, event):
        if self.stop_thread_callback:
            self.stop_thread_callback()
        super().closeEvent(event)

class MainWindow(QMainWindow):
    def __init__(self, callbacks, input_queue=None, output_queue=None):
        super().__init__()
        self.callbacks = callbacks
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.setWindowTitle("AI Studio")
        self.setStyleSheet(STYLESheet)
        self.setMinimumSize(800, 600)
        self.installEventFilter(self)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        title_label = QLabel("AI Studio")
        title_label.setObjectName("title")
        sidebar_layout.addWidget(title_label)
        
        self.connect_button = QPushButton("Connect")
        self.connect_button.setObjectName("connect_button")
        self.connect_button.clicked.connect(self.callbacks["on_connect_disconnect"])
        sidebar_layout.addWidget(self.connect_button)
        
        settings_group = QFrame()
        settings_group.setObjectName("settings_group")
        settings_layout = QVBoxLayout(settings_group)
        
        resolution_label = QLabel("Resolution:")
        self.resolution_selector = QComboBox()
        self.resolution_selector.addItems(["480p", "720p", "1080p"])
        self.resolution_selector.currentTextChanged.connect(self.callbacks["on_resolution_change"])
        settings_layout.addWidget(resolution_label)
        settings_layout.addWidget(self.resolution_selector)
        
        camera_label = QLabel("Camera:")
        self.camera_selector = QComboBox()
        self.camera_selector.currentTextChanged.connect(self.callbacks["on_camera_change"])
        settings_layout.addWidget(camera_label)
        settings_layout.addWidget(self.camera_selector)
        
        refresh_cameras = QPushButton("Refresh Cameras")
        refresh_cameras.clicked.connect(self.callbacks["on_refresh_cameras"])
        settings_layout.addWidget(refresh_cameras)
        
        input_device_label = QLabel("Input Device:")
        self.input_device_selector = QComboBox()
        self.input_device_selector.currentTextChanged.connect(self.callbacks["on_input_device_change"])
        settings_layout.addWidget(input_device_label)
        settings_layout.addWidget(self.input_device_selector)
        
        output_device_label = QLabel("Output Device:")
        self.output_device_selector = QComboBox()
        self.output_device_selector.currentTextChanged.connect(self.callbacks["on_output_device_change"])
        settings_layout.addWidget(output_device_label)
        settings_layout.addWidget(self.output_device_selector)
        
        refresh_audio = QPushButton("Refresh Audio")
        refresh_audio.clicked.connect(self.callbacks["on_refresh_audio"])
        settings_layout.addWidget(refresh_audio)
        
        face_select_button = QPushButton("Select Target Face")
        face_select_button.clicked.connect(self.callbacks["on_select_face"])
        settings_layout.addWidget(face_select_button)
        
        enhancement_cb = QCheckBox("Enable Face Enhancement")
        enhancement_cb.stateChanged.connect(self.callbacks["on_enhancement_toggle"])
        settings_layout.addWidget(enhancement_cb)
        
        voice_label = QLabel("AI Voice:")
        self.voice_selector = QComboBox()
        self.voice_selector.currentTextChanged.connect(self.callbacks["on_voice_change"])
        settings_layout.addWidget(voice_label)
        settings_layout.addWidget(self.voice_selector)
        
        self.talk_button = QPushButton("Push to Talk")
        self.talk_button.setCheckable(True)
        self.talk_button.setObjectName("talk_button")
        settings_layout.addWidget(self.talk_button)
        
        speaker_cb = QCheckBox("Enable Speaker Output")
        speaker_cb.stateChanged.connect(self.callbacks.get("on_speaker_toggle", lambda x: None))
        settings_layout.addWidget(speaker_cb)
        
        sidebar_layout.addWidget(settings_group)
        sidebar_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        main_layout.addWidget(sidebar)
        
        content_layout = QVBoxLayout()
        self.video_panel = QLabel()
        self.video_panel.setObjectName("video_panel")
        self.video_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.video_panel)
        
        self.pip_view = QLabel()
        self.pip_view.setObjectName("pip_view")
        self.pip_view.setFixedSize(160, 120)
        self.pip_view.setParent(self.video_panel)
        
        stats_layout = QHBoxLayout()
        video_stats_group = QFrame()
        video_stats_group.setObjectName("settings_group")
        video_stats_vlayout = QVBoxLayout(video_stats_group)
        self.video_stats_labels = {
            "sent_fps": QLabel("Sent FPS: 0.0"),
            "recv_fps": QLabel("Recv FPS: 0.0"),
            "rtt_ms": QLabel("RTT: 0.0 ms")
        }
        for label in self.video_stats_labels.values():
            label.setObjectName("stats_value")
            video_stats_vlayout.addWidget(label)
        stats_layout.addWidget(video_stats_group)
        
        audio_stats_group = QFrame()
        audio_stats_group.setObjectName("settings_group")
        audio_stats_vlayout = QVBoxLayout(audio_stats_group)
        self.audio_stats_labels = {
            "sent_ps": QLabel("Sent PS: 0.0"),
            "recv_ps": QLabel("Recv PS: 0.0"),
            "rtt_ms": QLabel("RTT: 0.0 ms")
        }
        for label in self.audio_stats_labels.values():
            label.setObjectName("stats_value")
            audio_stats_vlayout.addWidget(label)
        stats_layout.addWidget(audio_stats_group)
        
        content_layout.addLayout(stats_layout)
        self.status_message = QLabel("Ready to connect.")
        self.status_message.setObjectName("status_message")
        content_layout.addWidget(self.status_message)
        
        self.status_progress = QProgressBar()
        self.status_progress.setVisible(False)
        self.status_progress.setMaximum(0)
        content_layout.addWidget(self.status_progress)
        
        content_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        main_layout.addLayout(content_layout)
        
        self._reposition_pip()
        
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_frames)
        self.frame_timer.start(33)  # ~30 FPS

    def _reposition_pip(self):
        if not hasattr(self, 'pip_view'):
            return
        margin = 10
        parent_size = self.video_panel.size()
        pip_size = self.pip_view.size()
        x = parent_size.width() - pip_size.width() - margin
        y = margin
        self.pip_view.move(x, y)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_pip()

    def eventFilter(self, obj, event):
        if event.type() == event.Type.KeyPress and event.key() == Qt.Key.Key_Space:
            if not self.talk_button.isChecked():
                self.talk_button.setChecked(True)
                self.callbacks["on_space_press"]()
            return True
        elif event.type() == event.Type.KeyRelease and event.key() == Qt.Key.Key_Space:
            if self.talk_button.isChecked():
                self.talk_button.setChecked(False)
                self.callbacks["on_space_release"]()
            return True
        return super().eventFilter(obj, event)

    def update_camera_list(self, cameras):
        self.camera_selector.clear()
        self.camera_selector.addItems([name for _, name in cameras])

    def update_voice_list(self, voices):
        self.voice_selector.clear()
        self.voice_selector.addItems(voices)

    def update_audio_device_lists(self, input_devices, output_devices):
        self.input_device_selector.clear()
        self.input_device_selector.addItems([dev['name'] for dev in input_devices])
        self.output_device_selector.clear()
        self.output_device_selector.addItems([dev['name'] for dev in output_devices])

    def update_stats(self, video_stats, audio_stats):
        self.video_stats_labels["sent_fps"].setText(f"Sent FPS: {video_stats['sent_fps']:.1f}")
        self.video_stats_labels["recv_fps"].setText(f"Recv FPS: {video_stats['recv_fps']:.1f}")
        self.video_stats_labels["rtt_ms"].setText(f"RTT: {video_stats['rtt_ms']:.1f} ms")
        self.audio_stats_labels["sent_ps"].setText(f"Sent PS: {audio_stats['sent_ps']:.1f}")
        self.audio_stats_labels["recv_ps"].setText(f"Recv PS: {audio_stats['recv_ps']:.1f}")
        self.audio_stats_labels["rtt_ms"].setText(f"RTT: {audio_stats['rtt_ms']:.1f} ms")

    def update_status_message(self, message, is_error=False):
        self.status_message.setText(message)
        self.status_message.setObjectName("status_message_error" if is_error else "")
        self.status_message.setStyleSheet(self.styleSheet())
        is_loading = any(keyword in message.lower() for keyword in ["waiting", "deploying", "warming up"])
        self.status_progress.setVisible(is_error or is_loading)
        if is_error:
            if "no available gpus" in message.lower():
                message += "\nCheck RunPod credits or try a different region/time (https://runpod.io/console)."
            elif "api v2 is unavailable" in message.lower():
                message += "\nCheck RunPod API docs for updates (https://docs.runpod.io)."
            elif "webrtc" in message.lower():
                message += "\nCheck network or firewall settings."
            QMessageBox.warning(self, "Error", message)

    def update_video_panel_size(self, resolution):
        width, height = resolution
        self.video_panel.setFixedSize(width, height)
        self.adjustSize()

    def update_frames(self):
        try:
            if self.input_queue and not self.input_queue.empty():
                raw_frame = self.input_queue.get_nowait()
                if isinstance(raw_frame, np.ndarray):
                    raw_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = raw_rgb.shape
                    qimage = QImage(raw_rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimage).scaled(
                        self.pip_view.size(), Qt.AspectRatioMode.KeepAspectRatio
                    )
                    self.pip_view.setPixmap(pixmap)
            
            if self.output_queue and not self.output_queue.empty():
                processed_image = self.output_queue.get_nowait()
                if isinstance(processed_image, Image.Image):
                    processed_rgb = np.array(processed_image)
                    h, w, ch = processed_rgb.shape
                    qimage = QImage(processed_rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimage).scaled(
                        self.video_panel.size(), Qt.AspectRatioMode.KeepAspectRatio
                    )
                    self.video_panel.setPixmap(pixmap)
        except queue.Empty:
            pass
        except Exception as e:
            logging.error(f"Error updating frames: {str(e)}")
            
    def closeEvent(self, event):
        self.callbacks.get("on_closing", lambda: None)()
        super().closeEvent(event)