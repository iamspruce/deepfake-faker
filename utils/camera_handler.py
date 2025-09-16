import cv2
import platform
import threading

class CameraHandler:
    """
    A dedicated class to handle camera detection using the robust logic
    from your successful video_handler.py project.
    """
    def __init__(self):
        self.camera_list = []
        self.scan_thread = None

    def get_camera_list_sync(self):
        """
        Performs a synchronous scan using the proven method:
        1. Limit the scan range on macOS.
        2. Read a test frame to confirm the camera is working.
        """
        print("Performing robust camera scan...")
        cameras = []
        # --- KEY INSIGHT 1: Limit the scan on macOS ---
        max_scan_index = 3 if platform.system() == 'Darwin' else 8
        
        for i in range(max_scan_index):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # --- KEY INSIGHT 2: Read a test frame ---
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cameras.append({'id': i, 'name': f'Camera {i}'})
                cap.release()
            except Exception as e:
                print(f"Error checking camera {i}: {str(e)}")
        
        # If after all that, no cameras were found, add a default.
        if not cameras:
            cameras.append({'id': 0, 'name': 'Camera 0'})
            
        print(f"Robust scan found {len(cameras)} camera(s).")
        return cameras

    def start_background_scan(self, callback):
        """Starts the camera scan on a background thread."""
        def scanner_worker():
            self.camera_list = self.get_camera_list_sync()
            camera_tuples = [(cam['id'], cam['name']) for cam in self.camera_list]
            callback(camera_tuples)

        self.scan_thread = threading.Thread(target=scanner_worker, daemon=True)
        self.scan_thread.start()