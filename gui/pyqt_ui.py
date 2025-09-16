from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QDialog, QFrame, QLineEdit, QCheckBox, QSpacerItem, QSizePolicy,
    QRadioButton, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QPalette, QColor
import qtawesome as qta

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
    }
    QProgressBar::chunk {
        background-color: #3498db;
    }
"""

class ConnectionDialog(QDialog):
    def __init__(self, has_gpu, parent=None):
        super().__init__(parent)
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
        self.mode_selector.addItems(modes)
        self.mode_selector.setToolTip("Select the processing mode for AI computations")
        self.layout.addWidget(self.mode_selector)
        self.api_key_label = QLabel("RunPod API Key:")
        self.api_key_entry = QLineEdit()
        self.api_key_entry.setToolTip("Enter your RunPod API key for cloud deployment")
        self.layout.addWidget(self.api_key_label)
        self.layout.addWidget(self.api_key_entry)

class StartupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
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

class MainWindow(QMainWindow):
    def __init__(self, callbacks):
        super().__init__()
        self.setWindowTitle("DeepFaker")
        self.setStyleSheet(STYLESheet)
        self.callbacks = callbacks
        self.setGeometry(100, 100, 1200, 720) # Adjusted initial size for a better view

        # Main layout: Sidebar on the left, main content on the right
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # --- Sidebar (Left Panel) ---
        self._setup_sidebar()

        # --- Main Content (Right Panel) ---
        # This area will contain the video panel and the stats panel below it
        main_content_widget = QWidget()
        main_content_layout = QVBoxLayout(main_content_widget)
        main_content_layout.setContentsMargins(10, 10, 10, 10)
        main_content_layout.setSpacing(10)

        # Video Panel (the main view)
        self.video_panel = QLabel()
        self.video_panel.setObjectName("video_panel")
        self.video_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # We set a plain black background to see the frame clearly
        self.video_panel.setPixmap(QPixmap(640, 480))
        self.video_panel.pixmap().fill(QColor("black"))

        # Add video panel to the main content area
        main_content_layout.addWidget(self.video_panel, 1) # The '1' makes it expand

        # Stats Panel (below the video)
        self._setup_stats_panel()
        main_content_layout.addWidget(self.stats_frame) # Add the stats frame here

        # Add the entire main content area to the main window layout
        self.main_layout.addWidget(main_content_widget, 1)

        # --- Picture-in-Picture (PiP) View ---
        # This is a child of the main video panel so it can be overlaid
        self.pip_view = QLabel(self.video_panel)
        self.pip_view.setObjectName("pip_view")
        self.pip_view.setFixedSize(200, 150)
        self.pip_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Set a plain black background to see the PiP frame
        self.pip_view.setPixmap(QPixmap(200, 150))
        self.pip_view.pixmap().fill(QColor("black"))
        self.pip_view.show() # Make it visible

        # Install event filter for spacebar key presses
        self.installEventFilter(self)

    def _setup_sidebar(self):
        """Creates and configures all widgets for the sidebar."""
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(300)
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.sidebar_layout.setContentsMargins(10, 0, 10, 10)
        self.sidebar_layout.setSpacing(10)

        self.title_label = QLabel("DeepFaker")
        self.title_label.setObjectName("title")
        self.sidebar_layout.addWidget(self.title_label)

        self.status_message = QLabel("Ready to connect.")
        self.status_message.setWordWrap(True)
        self.status_progress = QProgressBar()
        self.status_progress.setRange(0, 0)
        self.status_progress.setVisible(False)
        self.sidebar_layout.addWidget(self.status_message)
        self.sidebar_layout.addWidget(self.status_progress)

        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.callbacks["on_connect_disconnect"])
        self.sidebar_layout.addWidget(self.connect_button)

        # Create a frame for better grouping of settings
        settings_frame = QFrame()
        settings_frame.setObjectName("settings_group")
        settings_layout = QVBoxLayout(settings_frame)

        self.resolution_selector = QComboBox()
        self.resolution_selector.addItems(["480p", "720p", "1080p"])
        self.resolution_selector.currentTextChanged.connect(self.callbacks["on_resolution_change"])
        settings_layout.addWidget(QLabel("Resolution:"))
        settings_layout.addWidget(self.resolution_selector)

        self.camera_selector = QComboBox()
        self.camera_selector.currentTextChanged.connect(self.callbacks["on_camera_change"])
        settings_layout.addWidget(QLabel("Camera:"))
        settings_layout.addWidget(self.camera_selector)
        
        self.refresh_cameras_button = QPushButton("Refresh Cameras")
        self.refresh_cameras_button.clicked.connect(self.callbacks["on_refresh_cameras"])
        settings_layout.addWidget(self.refresh_cameras_button)

        self.input_device_selector = QComboBox()
        self.input_device_selector.currentTextChanged.connect(self.callbacks["on_input_device_change"])
        settings_layout.addWidget(QLabel("Input Device:"))
        settings_layout.addWidget(self.input_device_selector)

        self.output_device_selector = QComboBox()
        self.output_device_selector.currentTextChanged.connect(self.callbacks["on_output_device_change"])
        settings_layout.addWidget(QLabel("Output Device:"))
        settings_layout.addWidget(self.output_device_selector)
        
        self.refresh_audio_button = QPushButton("Refresh Audio Devices")
        self.refresh_audio_button.clicked.connect(self.callbacks["on_refresh_audio"])
        settings_layout.addWidget(self.refresh_audio_button)

        self.sidebar_layout.addWidget(settings_frame)

        self.face_button = QPushButton("Select Target Face")
        self.face_button.clicked.connect(self.callbacks["on_select_face"])
        self.sidebar_layout.addWidget(self.face_button)

        self.enhancement_check = QCheckBox("Enable Face Enhancement")
        self.enhancement_check.stateChanged.connect(self.callbacks["on_enhancement_toggle"])
        self.sidebar_layout.addWidget(self.enhancement_check)
        
        self.voice_selector = QComboBox()
        self.voice_selector.currentTextChanged.connect(self.callbacks["on_voice_change"])
        self.sidebar_layout.addWidget(QLabel("Voice:"))
        self.sidebar_layout.addWidget(self.voice_selector)

        self.talk_button = QPushButton("Talk (Hold Space)")
        self.talk_button.setCheckable(True)
        self.talk_button.setObjectName("talk_button")
        self.sidebar_layout.addWidget(self.talk_button)
        
        self.sidebar_layout.addStretch(1) # Pushes everything up
        
        self.main_layout.addWidget(self.sidebar)

    def _setup_stats_panel(self):
        """Creates the horizontal stats panel for the main content area."""
        self.stats_frame = QFrame()
        # Use a QHBoxLayout to place stats groups side-by-side
        stats_layout = QHBoxLayout(self.stats_frame)
        stats_layout.setContentsMargins(0,0,0,0)

        # Video Stats Group
        video_stats_group = QFrame()
        video_stats_group.setObjectName("settings_group")
        video_stats_vlayout = QVBoxLayout(video_stats_group)
        video_stats_vlayout.addWidget(QLabel("<b>Video Stats</b>"))
        self.video_stats_labels = {
            "sent_fps": QLabel("Sent FPS: 0.0"), "recv_fps": QLabel("Recv FPS: 0.0"),
            "rtt_ms": QLabel("RTT: 0.0 ms")
        }
        for label in self.video_stats_labels.values():
            label.setObjectName("stats_value")
            video_stats_vlayout.addWidget(label)
        stats_layout.addWidget(video_stats_group)

        # Audio Stats Group
        audio_stats_group = QFrame()
        audio_stats_group.setObjectName("settings_group")
        audio_stats_vlayout = QVBoxLayout(audio_stats_group)
        audio_stats_vlayout.addWidget(QLabel("<b>Audio Stats</b>"))
        self.audio_stats_labels = {
            "sent_ps": QLabel("Sent PS: 0.0"), "recv_ps": QLabel("Recv PS: 0.0"),
            "rtt_ms": QLabel("RTT: 0.0 ms")
        }
        for label in self.audio_stats_labels.values():
            label.setObjectName("stats_value")
            audio_stats_vlayout.addWidget(label)
        stats_layout.addWidget(audio_stats_group)

    def _reposition_pip(self):
        """Positions the PiP view at the top-right of the video panel."""
        if not hasattr(self, 'pip_view'):
            return
        margin = 10
        parent_size = self.video_panel.size()
        pip_size = self.pip_view.size()
        x = parent_size.width() - pip_size.width() - margin
        y = margin
        self.pip_view.move(x, y)

    def resizeEvent(self, event):
        """Overrides the resize event to keep the PiP in the corner."""
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

    # --- Public Methods (unchanged) ---
    def update_camera_list(self, cameras):
        self.camera_selector.clear()
        self.camera_selector.addItems([name for _, name in cameras])

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
        self.status_message.setStyleSheet(self.styleSheet()) # Re-apply stylesheet
        is_loading = "waiting" in message.lower() or "deploying" in message.lower()
        self.status_progress.setVisible(is_error or is_loading)
        if is_error and ("backend error" in message.lower() or "failed to start" in message.lower()):
            QMessageBox.warning(self, "Backend Error", message)

    def update_video_panel_size(self, resolution):
        # This function might not be needed anymore as the panel auto-sizes
        # but can be kept if you need to force a specific aspect ratio logic
        width, height = resolution
        self.video_panel.setFixedSize(width, height)
        self.adjustSize()