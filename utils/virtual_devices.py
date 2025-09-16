import numpy as np
import pyvirtualcam
import sounddevice as sd
import platform
import queue
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VirtualDeviceManager:
    def __init__(self, state, audio_input_queue: queue.Queue, resolution=(1280, 720)):
        self.state = state
        self.width, self.height = resolution
        self.audio_input_queue = audio_input_queue
        self.raw_cam = None
        self.processed_cam = None
        self.audio_stream = None
        
        try:
            backend = 'obs' if platform.system() == "Darwin" else None
            self.raw_cam = pyvirtualcam.Camera(
                width=self.width, height=self.height, fps=30,
                device="AI Studio Cam (Raw)", backend=backend, print_fps=False
            )
            self.processed_cam = pyvirtualcam.Camera(
                width=self.width, height=self.height, fps=30,
                device="AI Studio Cam (Processed)", backend=backend, print_fps=False
            )
            logging.info(f"Virtual cameras created: '{self.raw_cam.device}' and '{self.processed_cam.device}'")
        except Exception as e:
            logging.warning(f"Could not create virtual cameras: {str(e)}")
            logging.warning("On macOS, this requires OBS Studio. Video will not be sent to external applications.")
            self.raw_cam = self.processed_cam = None

    def start_audio_stream(self, input_device_id, output_device_id):
        if self.audio_stream and self.audio_stream.active:
            logging.info("Audio stream is already active.")
            return

        def audio_callback(indata, outdata, frames, time, status):
            if status.input_overflow:
                return
            if status:
                logging.warning(f"Audio Stream Warning: {status}")
            
            if self.state.is_push_to_talk_active:
                outdata[:] = indata
                if self.state.operating_mode != 'offline' and self.audio_input_queue:
                    try:
                        self.audio_input_queue.put_nowait(indata.copy())
                    except queue.Full:
                        logging.warning("Audio queue full: Dropping packet to prevent blocking.")
            else:
                outdata.fill(0)

        try:
            logging.info(f"Attempting to start audio stream with Input ID: {input_device_id} and Output ID: {output_device_id}")
            self.audio_stream = sd.Stream(
                samplerate=16000,
                latency='low',
                channels=1,
                device=(input_device_id, output_device_id),
                dtype='float32',
                callback=audio_callback
            )
            self.audio_stream.start()
            logging.info("Duplex audio stream started successfully.")
        except Exception as e:
            logging.error(f"Could not start duplex audio stream: {str(e)}")
            self.audio_stream = None

    def stop_audio_stream(self):
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
            logging.info("Duplex audio stream stopped.")
    
    def send_raw_frame(self, frame: np.ndarray):
        if self.raw_cam:
            try:
                self.raw_cam.send(frame)
            except Exception as e:
                logging.warning(f"Failed to send raw frame: {str(e)}")
                
    def send_processed_frame(self, frame: np.ndarray):
        if self.processed_cam:
            try:
                self.processed_cam.send(frame)
            except Exception as e:
                logging.warning(f"Failed to send processed frame: {str(e)}")
                
    def close(self):
        if self.raw_cam:
            self.raw_cam.close()
        if self.processed_cam:
            self.processed_cam.close()
        self.stop_audio_stream()
        logging.info("Virtual devices closed.")