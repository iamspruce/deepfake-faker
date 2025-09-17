# state_manager.py (Adapted for Dual-Backend Architecture)

class StateManager:
    def __init__(self):
        # MODIFIED: operating_mode can be 'local', 'cloud', or 'offline'
        self.operating_mode = 'offline'
        self.target_face_path = ""
        self.selected_voice = ""
        
        self.runpod_api_key = None
        
        # MODIFIED: More granular status management
        self.server_status = "Ready to connect."
        self.voice_backend_status = 'disconnected' # e.g., 'disconnected', 'connecting', 'downloading', 'connected', 'error'
        self.face_backend_status = 'disconnected'  # e.g., 'disconnected', 'connecting', 'downloading', 'connected', 'error'
        
        self.face_server_endpoint = None
        self.voice_server_endpoint = None
        
        # --- NEW: Store Endpoint IDs for cleanup ---
        self.face_endpoint_id = None
        self.voice_endpoint_id = None

        # --- Static App Configuration ---
        self.resolutions = {"480p": (640, 480), "720p": (1280, 720), "1080p": (1920, 1080)}
        self.available_cameras = []
        self.available_voices = [] # This would be populated from the voice backend
        self.available_input_devices = []
        self.available_output_devices = []
        
        # --- Dynamic State ---
        self.selected_resolution = self.resolutions["480p"]
        self.selected_camera_index = 0
        self.selected_input_device_id = None
        self.selected_output_device_id = None
        self.is_enhancement_on = False
        self.is_push_to_talk_active = False
        
        # --- Connection & Stats ---
        self.video_stats = {"sent_fps": 0.0, "recv_fps": 0.0, "rtt_ms": 0.0}
        self.audio_stats = {"sent_ps": 0.0, "recv_ps": 0.0, "rtt_ms": 0.0}
        

    # NEW: Helper methods to check overall connection state
    def is_fully_connected(self):
        return self.voice_backend_status == 'connected' and self.face_backend_status == 'connected'
        
    def is_fully_disconnected(self):
        return self.voice_backend_status in ['disconnected', 'error'] and self.face_backend_status in ['disconnected', 'error']

    def reset_for_disconnection(self):
        self.server_status = "Disconnected."
        self.voice_backend_status = 'disconnected'
        self.face_backend_status = 'disconnected'
        self.video_stats = {"sent_fps": 0.0, "recv_fps": 0.0, "rtt_ms": 0.0}
        self.audio_stats = {"sent_ps": 0.0, "recv_ps": 0.0, "rtt_ms": 0.0}
        self.face_server_endpoint = None
        self.voice_server_endpoint = None
        self.face_endpoint_id = None
        self.voice_endpoint_id = None
        
        
        
        
        

