# rvc_wrapper.py

import numpy as np
import torch
import librosa
from src.rvc import Config, load_hubert, get_vc

class RVCWrapper:
    def __init__(self, model_path, hubert_path, device="cuda:0", is_half=False):
        self.model_path = model_path
        self.hubert_path = hubert_path
        self.device = device
        self.is_half = is_half
        
        # Load the configuration
        self.config = Config(self.device, self.is_half)
        
        # Initialize model variables to None
        self.hubert_model = None
        self.cpt = None
        self.version = None
        self.net_g = None
        self.tgt_sr = None
        self.vc = None

        # --- Default Tunable Parameters ---
        self.pitch = 0
        self.f0_method = 'rmvpe'
        self.index_rate = 0.5
        self.filter_radius = 3
        self.rms_mix_rate = 0.25
        self.protect = 0.33
        self.crepe_hop_length = 160
        self.index_path = "" # Optional: Add path to an index file if you have one

    def load_model(self):
        """
        Loads the Hubert and RVC models into memory.
        This is a slow operation and should be done only once.
        """
        print("[RVC Wrapper] Loading Hubert model...")
        self.hubert_model = load_hubert(self.device, self.is_half, self.hubert_path)
        
        print("[RVC Wrapper] Loading RVC model...")
        self.cpt, self.version, self.net_g, self.tgt_sr, self.vc = get_vc(
            self.device, self.is_half, self.config, self.model_path
        )
        print("[RVC Wrapper] All models loaded successfully.")

    def process_chunk(self, audio_chunk_float32):
        """
        Processes a small chunk of audio for real-time conversion.
        Assumes input is a float32 NumPy array at 48000 Hz.
        """
        if self.hubert_model is None or self.net_g is None:
            raise RuntimeError("Models are not loaded. Call load_model() first.")

        # Resample input audio from 48kHz to 16kHz for the model
        audio_16k = librosa.resample(
            y=audio_chunk_float32,
            orig_sr=48000,
            target_sr=16000
        )

        times = [0, 0, 0] # Placeholder for performance tracking
        if_f0 = self.cpt.get('f0', 1)

        # Perform inference using the VC pipeline
        converted_audio = self.vc.pipeline(
            self.hubert_model,
            self.net_g,
            0, # Speaker ID
            audio_16k,
            "realtime_chunk", # Dummy input path name
            times,
            self.pitch,
            self.f0_method,
            self.index_path,
            self.index_rate,
            if_f0,
            self.filter_radius,
            self.tgt_sr,
            0, # Resample SR (0 means no resampling inside the pipeline)
            self.rms_mix_rate,
            self.version,
            self.protect,
            self.crepe_hop_length,
        )

        # ------------------- THE FIX IS HERE -------------------
        # Convert the integer output from the model to float32 for resampling
        converted_audio_float = converted_audio.astype(np.float32) / 32767.0

        # --- ADD THIS DEBUG LINE ---
        print(f"DEBUG: Data type before resampling is {converted_audio_float.dtype}")

        # Resample the output back to 48kHz for the client
        output_audio_48k = librosa.resample(
            y=converted_audio_float, # Ensure we are using the float version
            orig_sr=self.tgt_sr,
            target_sr=48000
        )
        # --------------------------------------------------------

        return output_audio_48k

