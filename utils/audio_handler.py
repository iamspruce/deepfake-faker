import sounddevice as sd
import threading
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)

class AudioHandler:
    """
    A dedicated class to handle audio device detection in a non-blocking way.
    """
    def __init__(self):
        self.device_list = {'input': [], 'output': []}

    def get_audio_devices_sync(self):
        """Performs a synchronous scan for all audio devices."""
        logging.info("Performing audio device scan...")
        inputs = []
        outputs = []
        try:
            for i, device in enumerate(sd.query_devices()):
                # On macOS, filter out some low-quality default devices
                if "Apple" in device["name"] and "Sound" in device["name"]:
                    continue
                if device['max_input_channels'] > 0:
                    inputs.append({'id': i, 'name': device['name']})
                if device['max_output_channels'] > 0:
                    outputs.append({'id': i, 'name': device['name']})
        except Exception as e:
            logging.error(f"Could not query audio devices: {str(e)}")
            return {'input': [], 'output': []}

        logging.info(f"Scan found {len(inputs)} input(s) and {len(outputs)} output(s).")
        return {'input': inputs, 'output': outputs}

    def list_devices(self):
        """Synchronous method for initial device population."""
        self.device_list = self.get_audio_devices_sync()
        return self.device_list

    def start_background_scan(self, callback):
        """Starts the device scan on a background thread."""
        def scanner_worker():
            devices = self.get_audio_devices_sync()
            self.device_list = devices
            callback(devices)

        threading.Thread(target=scanner_worker, daemon=True).start()

    def start(self, input_device_id, output_device_id):
        """Placeholder for starting audio stream (to be implemented)."""
        logging.info(f"Starting audio with input ID {input_device_id}, output ID {output_device_id}")
        # Implement audio streaming logic here if needed
        pass
