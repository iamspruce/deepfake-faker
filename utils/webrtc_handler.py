import asyncio
import threading
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
import numpy as np
import cv2
import random
import string
from av import VideoFrame, AudioFrame
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomVideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()

    async def recv(self):
        frame = await self.queue.get()
        return frame

    async def send_frame(self, frame: np.ndarray):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(rgb_frame, format="rgb24")
            await self.queue.put(video_frame)
        except Exception as e:
            logging.error(f"Error sending video frame: {str(e)}")

class CustomAudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self.SAMPLE_RATE = 16000
        self.SAMPLES_PER_FRAME = 320

    async def recv(self):
        frame = await self.queue.get()
        return frame

    async def send_frame(self, frame: np.ndarray):
        try:
            audio_samples = frame.reshape(-1, 1)
            audio_frame = AudioFrame.from_ndarray(audio_samples, format='s16', layout='mono')
            audio_frame.sample_rate = self.SAMPLE_RATE
            audio_frame.pts = self.SAMPLES_PER_FRAME
            await self.queue.put(audio_frame)
        except Exception as e:
            logging.error(f"Error sending audio frame: {str(e)}")

class WebRTCHandler:
    def __init__(self, endpoint_url: str, async_handler, 
                 on_video_track: callable, on_audio_track: callable,
                 add_video_track: bool = False, add_audio_track: bool = False):
        
        self.endpoint_url = endpoint_url
        self.async_handler = async_handler
        self.pc = RTCPeerConnection()
        self.add_video_track = add_video_track
        self.add_audio_track = add_audio_track
        self.video_sender_track = None
        self.audio_sender_track = None
        self._on_video_track_callback = on_video_track
        self._on_audio_track_callback = on_audio_track
        self.reconnect_timer = None
        
        def monitor_connection():
            if self.pc.connectionState in ["failed", "disconnected"]:
                logging.warning(f"WebRTC disconnected from {self.endpoint_url}. Reconnecting...")
                self.close()
                self.connect()
            self.reconnect_timer = threading.Timer(5.0, monitor_connection)
            self.reconnect_timer.start()
        
        monitor_connection()

        @self.pc.on("track")
        def on_track(track):
            logging.info(f"Track {track.kind} received from {self.endpoint_url}")
            if track.kind == "video" and self._on_video_track_callback:
                self._on_video_track_callback(track)
            elif track.kind == "audio" and self._on_audio_track_callback:
                self._on_audio_track_callback(track)

    def connect(self):
        logging.info(f"Attempting to connect to aiortc endpoint: {self.endpoint_url}")
        self.async_handler.run_coroutine(self._connect_async())

    async def _connect_async(self):
        if self.add_video_track:
            self.video_sender_track = CustomVideoStreamTrack()
            self.pc.addTrack(self.video_sender_track)
        
        if self.add_audio_track:
            self.audio_sender_track = CustomAudioStreamTrack()
            self.pc.addTrack(self.audio_sender_track)

        try:
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            webrtc_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=7))
            body = { "sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type, "webrtc_id": webrtc_id }
            
            async with aiohttp.ClientSession() as session:
                post_url = f"{self.endpoint_url}/webrtc/offer"
                async with session.post(post_url, json=body) as resp:
                    if resp.status != 200:
                        logging.error(f"Signaling error for {post_url}: {resp.status} {await resp.text()}")
                        return
                    data = await resp.json()
                    answer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                    await self.pc.setRemoteDescription(answer)
                    logging.info(f"Successfully connected to {self.endpoint_url}")
        except Exception as e:
            logging.error(f"Failed to connect to WebRTC endpoint {self.endpoint_url}: {str(e)}")

    def send_video_frame(self, frame: np.ndarray):
        if self.is_connected() and self.video_sender_track:
            self.async_handler.run_coroutine(self.video_sender_track.send_frame(frame))
    
    def send_audio_frame(self, frame: np.ndarray):
        if self.is_connected() and self.audio_sender_track:
            self.async_handler.run_coroutine(self.audio_sender_track.send_frame(frame))

    def is_connected(self) -> bool:
        return self.pc.connectionState == "connected"

    def get_rtt_ms(self) -> float:
        if self.pc and self.pc.sctp and self.pc.sctp.transport and hasattr(self.pc.sctp.transport, 'rtt'):
            if self.pc.sctp.transport.rtt is not None:
                return self.pc.sctp.transport.rtt * 1000
        return -1.0

    async def _close_async(self):
        if self.pc.connectionState != "closed":
            await self.pc.close()

    def close(self):
        if self.pc:
            logging.info(f"Closing WebRTC connection to {self.endpoint_url}")
            self.async_handler.run_coroutine(self._close_async())