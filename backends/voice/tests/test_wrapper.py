# backends/voice/tests/test_wrapper.py

import numpy as np
import soundfile as sf
import os
import sys
import torch
import librosa
import traceback
from tqdm import tqdm
from pathlib import Path

# --- CORRECTED PATH SETUP ---
# Get the root directory of the voice backend ('voice/')
voice_backend_root = Path(__file__).resolve().parents[1]

# Add the voice backend's root and its 'src' directory to the Python path
if str(voice_backend_root) not in sys.path:
    sys.path.append(str(voice_backend_root))
rvc_src_dir = voice_backend_root / "src"
if str(rvc_src_dir) not in sys.path:
    sys.path.append(str(rvc_src_dir))

try:
    from rvc_wrapper import RVCWrapper
    # Import the model download functions from your new script
    from src.download_models import download_pretrained_models, download_voice_model
except ImportError as e:
    print("\n--- IMPORT ERROR ---")
    print(f"Could not import required modules: {e}")
    print("Please ensure rvc_wrapper.py and src/download_models.py exist.\n")
    sys.exit(1)

print("--- RVC Wrapper Test Script (with Integrated Model Downloader) ---")

# --- Configuration ---
MODEL_NAME = "Klee"
INPUT_AUDIO_NAME = "test_input.wav"

# --- ASSET PATHS ---
tests_dir = Path(__file__).resolve().parent
INPUT_AUDIO_PATH = tests_dir / INPUT_AUDIO_NAME
OUTPUT_AUDIO_PATH = tests_dir / "test_output_voice.wav"

# Model paths are relative to the voice backend's root
MODELS_DIR = voice_backend_root / "rvc_models"
HUBERT_MODEL_PATH = MODELS_DIR / "hubert_base.pt"
MODEL_FOLDER_PATH = MODELS_DIR / MODEL_NAME


if __name__ == "__main__":
    try:
        # --- Step 1: Download and Prepare Models ---
        print("\n[Step 1] Checking for and downloading necessary models...")
        # Ensure the base directory for all models exists
        MODELS_DIR.mkdir(exist_ok=True)
        # Download essential pretrained files (like hubert_base.pt)
        download_pretrained_models(str(MODELS_DIR))
        # Download the specific voice model for the test
        download_voice_model(MODEL_NAME, str(MODELS_DIR))
        print("  -> Models are ready.")

        # --- Step 2: Find the .pth model file ---
        # This logic remains the same, as the downloader creates the expected structure
        print(f"\n[Step 2] Locating .pth file for model '{MODEL_NAME}'...")
        pth_file = next(f for f in os.listdir(MODEL_FOLDER_PATH) if f.endswith(".pth"))
        MODEL_PATH = MODEL_FOLDER_PATH / pth_file
        print(f"  -> Found model file: {pth_file}")

        # --- Step 3: Initialize the Wrapper ---
        print("\n[Step 3] Initializing RVCWrapper...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        is_half = False
        
        rvc_wrapper = RVCWrapper(str(MODEL_PATH), str(HUBERT_MODEL_PATH), device, is_half)
        print(f"  -> Wrapper created for model: {MODEL_PATH.name}")

        # --- Step 4: Load the Models ---
        print("\n[Step 4] Loading models into memory...")
        rvc_wrapper.load_model()
        print("  -> Models loaded successfully.")

        # --- Step 5: Load and Prepare Input Audio ---
        print(f"\n[Step 5] Loading input audio: {INPUT_AUDIO_NAME}")
        if not INPUT_AUDIO_PATH.exists():
            raise FileNotFoundError(f"Input audio not found at {INPUT_AUDIO_PATH}")
            
        input_audio, sr = sf.read(INPUT_AUDIO_PATH)
        
        # Resample to the target sample rate used by the model
        target_sr = 48000
        if sr != target_sr:
            print(f"  -> Resampling audio from {sr}Hz to {target_sr}Hz...")
            input_audio = librosa.resample(y=input_audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        
        input_audio = input_audio.astype(np.float32)
        print(f"  -> Audio is ready for processing (duration: {len(input_audio)/target_sr:.2f}s).")
        
        # --- Step 6: Process Audio in Chunks ---
        print("\n[Step 6] Processing audio in chunks...")
        chunk_size = 48000  # Process 1 second at a time
        output_chunks = []

        for i in tqdm(range(0, len(input_audio), chunk_size), desc="Converting"):
            chunk = input_audio[i : i + chunk_size]
            converted_chunk = rvc_wrapper.process_chunk(chunk)
            output_chunks.append(converted_chunk)
            
        final_audio = np.concatenate(output_chunks)
        print("  -> Audio processing complete.")
        
        # --- Step 7: Save the Output ---
        print(f"\n[Step 7] Saving converted audio to: {OUTPUT_AUDIO_PATH.name}")
        sf.write(OUTPUT_AUDIO_PATH, final_audio, target_sr)
        
        print("\n🎉 Test complete! 🎉")
        print("If the output file sounds correct, your wrapper and downloader are working properly.")

    except Exception as e:
        print(f"\n--- An error occurred during the test ---")
        traceback.print_exc()