import threading
import time
import cv2
import queue
from PIL import Image
import numpy as np
import platform
import logging

from utils.webrtc_handler import WebRTCHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MediaService:
    def __init__(self, state, in_q, out_q, audio_in_q, async_handler, virtual_devices, stats_signal):
        self.state = state
        self.input_queue = in_q
        self.output_queue = out_q
        self.audio_input_queue = audio_in_q
        self.async_handler = async_handler
        self.virtual_devices = virtual_devices
        self.stats_signal = stats_signal
        
        self.running = False
        self.cap = None
        
        self.video_capture_thread = None
        self.audio_sender_thread = None
        self.video_receiver_thread = None
        self.audio_receiver_thread = None
        
        self.face_webrtc: WebRTCHandler = None
        self.voice_webrtc: WebRTCHandler = None

        self._sent_frames = 0
        self._recv_frames = 0
        self._sent_audio_packets = 0
        self._recv_audio_packets = 0
        self._last_stats_time = time.time()
        self.video_fps_history = []
        self.audio_ps_history = []

        if self.state.operating_mode not in ['offline_passthrough', 'offline']:
            self.face_webrtc = WebRTCHandler(
                endpoint_url=self.state.face_server_endpoint,
                async_handler=self.async_handler,
                on_video_track=self._start_receiving_video,
                on_audio_track=None,
                add_video_track=True,
                add_audio_track=False
            )
            self.voice_webrtc = WebRTCHandler(
                endpoint_url=self.state.voice_server_endpoint,
                async_handler=self.async_handler,
                on_video_track=None,
                on_audio_track=self._start_receiving_audio,
                add_video_track=False,
                add_audio_track=True
            )

    def start(self):
        self.running = True
        
        if self.face_webrtc:
            self.face_webrtc.connect()
        if self.voice_webrtc:
            self.voice_webrtc.connect()
        
        self.video_capture_thread = threading.Thread(target=self._video_capture_loop, daemon=True)
        self.video_capture_thread.start()

        if self.voice_webrtc:
            self.audio_sender_thread = threading.Thread(target=self._audio_sender_loop, daemon=True)
            self.audio_sender_thread.start()

    def stop(self):
        self.running = False
        if self.face_webrtc:
            self.face_webrtc.close()
        if self.voice_webrtc:
            self.voice_webrtc.close()
        time.sleep(0.1)
        if self.video_capture_thread and self.video_capture_thread.is_alive():
            self.video_capture_thread.join()
        if self.audio_sender_thread and self.audio_sender_thread.is_alive():
            self.audio_sender_thread.join()
        if self.video_receiver_thread and self.video_receiver_thread.is_alive():
            self.video_receiver_thread.join()
        if self.audio_receiver_thread and self.audio_receiver_thread.is_alive():
            self.audio_receiver_thread.join()
        if self.cap:
            self.cap.release()

    def _open_camera(self, camera_id):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened() and platform.system() == 'Darwin':
            cap.release()
            cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            logging.error(f"Could not open camera {camera_id} for video stream.")
            return None
        return cap

    def _video_capture_loop(self):
        logging.info("Starting video capture loop...")
        w, h = self.state.selected_resolution
        
        read_failures = 0
        max_retries = 3
        while self.running:
            if not self.cap or not self.cap.isOpened():
                self.cap = self._open_camera(self.state.selected_camera_index)
                if not self.cap:
                    read_failures += 1
                    if read_failures >= max_retries:
                        logging.error(f"Failed to open camera after {max_retries} attempts.")
                        break
                    time.sleep(1)
                    continue
                read_failures = 0

            ret, frame_bgr = self.cap.read()
            if not ret or frame_bgr is None:
                read_failures += 1
                if read_failures >= max_retries:
                    logging.error(f"Failed to read frame after {max_retries} attempts.")
                    break
                time.sleep(0.1)
                continue
            
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if self.virtual_devices:
                self.virtual_devices.send_raw_frame(frame_rgb)
            
            if self.state.operating_mode in ['offline_passthrough', 'offline']:
                try:
                    self.output_queue.put_nowait(Image.fromarray(frame_rgb))
                    if self.virtual_devices:
                        self.virtual_devices.send_processed_frame(frame_rgb)
                    self._sent_frames += 1
                    self._recv_frames += 1
                except queue.Full:
                    logging.warning("Output queue full, dropping video frame.")
            elif self.face_webrtc and self.face_webrtc.is_connected():
                self.face_webrtc.send_video_frame(frame_bgr)
                self._sent_frames += 1
            
            self._update_stats()
            time.sleep(1/30)
        
        logging.info("Video capture loop finished.")
        if self.cap:
            self.cap.release()

    def _audio_sender_loop(self):
        while self.running:
            try:
                audio_frame = self.audio_input_queue.get_nowait()
                if self.voice_webrtc and self.voice_webrtc.is_connected():
                    self.voice_webrtc.send_audio_frame(audio_frame)
                    self._sent_audio_packets += 1
            except queue.Empty:
                time.sleep(0.01)
                continue
            except Exception as e:
                logging.error(f"Error in audio sender loop: {e}")

    def _start_receiving_video(self, track):
        self.video_receiver_thread = threading.Thread(target=self._video_receiver_loop, args=(track,), daemon=True)
        self.video_receiver_thread.start()

    def _video_receiver_loop(self, track):
        future = self.async_handler.run_coroutine(self._video_receiver_loop_async(track))
        try:
            future.result()
        except Exception as e:
            logging.error(f"Error in video receiver loop: {e}")

    async def _video_receiver_loop_async(self, track):
        while self.running:
            try:
                frame = await track.recv()
                processed_rgb = frame.to_ndarray(format="rgb24")
                try:
                    self.output_queue.put_nowait(Image.fromarray(processed_rgb))
                    if self.virtual_devices:
                        self.virtual_devices.send_processed_frame(processed_rgb)
                    self._recv_frames += 1
                except queue.Full:
                    logging.warning("Output queue full, dropping received video frame.")
            except Exception as e:
                logging.error(f"Error receiving video frame: {e}")
                break

    def _start_receiving_audio(self, track):
        self.audio_receiver_thread = threading.Thread(target=self._audio_receiver_loop, args=(track,), daemon=True)
        self.audio_receiver_thread.start()

    def _audio_receiver_loop(self, track):
        future = self.async_handler.run_coroutine(self._audio_receiver_loop_async(track))
        try:
            future.result()
        except Exception as e:
            logging.error(f"Error in audio receiver loop: {e}")
        
    async def _audio_receiver_loop_async(self, track):
        while self.running:
            try:
                audio_frame = await track.recv()
                self._recv_audio_packets += 1
            except Exception as e:
                logging.error(f"Error receiving audio frame: {e}")
                break

    def _update_stats(self):
        now = time.time()
        delta = now - self._last_stats_time
        if delta >= 1.0:
            sent_fps = self._sent_frames / delta
            recv_fps = self._recv_frames / delta
            sent_ps = self._sent_audio_packets / delta
            recv_ps = self._recv_audio_packets / delta
            
            self.video_fps_history.append(sent_fps)
            self.audio_ps_history.append(sent_ps)
            if len(self.video_fps_history) > 5:
                self.video_fps_history.pop(0)
            if len(self.audio_ps_history) > 5:
                self.audio_ps_history.pop(0)
            
            self.state.video_stats['sent_fps'] = sum(self.video_fps_history) / len(self.video_fps_history) if self.video_fps_history else 0.0
            self.state.video_stats['recv_fps'] = recv_fps
            self.state.video_stats['rtt_ms'] = self.face_webrtc.get_rtt_ms() if self.face_webrtc else -1.0
            self.state.audio_stats['sent_ps'] = sum(self.audio_ps_history) / len(self.audio_ps_history) if self.audio_ps_history else 0.0
            self.state.audio_stats['recv_ps'] = recv_ps
            self.state.audio_stats['rtt_ms'] = self.voice_webrtc.get_rtt_ms() if self.voice_webrtc else -1.0
            
            self.stats_signal.update.emit(self.state.video_stats, self.state.audio_stats)
            self._sent_frames = 0
            self._recv_frames = 0
            self._sent_audio_packets = 0
            self._recv_audio_packets = 0
            self._last_stats_time = now