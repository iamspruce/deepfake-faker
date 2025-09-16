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
# Get the root directory of the voice backend ('voice/') by going up one level from this script's parent ('tests/')
voice_backend_root = Path(__file__).resolve().parents[1]

# Add the voice backend's root to the Python path to find rvc_wrapper, etc.
if str(voice_backend_root) not in sys.path:
    sys.path.append(str(voice_backend_root))

# Add the RVC 'src' directory to the path for its internal imports
rvc_src_dir = voice_backend_root / "src"
if str(rvc_src_dir) not in sys.path:
    sys.path.append(str(rvc_src_dir))

try:
    from rvc_wrapper import RVCWrapper
except ImportError as e:
    print("\n--- IMPORT ERROR ---")
    print(f"Could not import RVCWrapper: {e}")
    print("Please ensure your directory structure is correct:")
    print("  - backends/voice/")
    print("    - rvc_wrapper.py")
    print("    - src/ (containing the RVC code)")
    print("    - tests/test_wrapper.py (this file)\n")
    sys.exit(1)

print("--- RVC Wrapper Test Script (Modular Structure) ---")

# --- Configuration ---
MODEL_NAME = "Klee"
INPUT_AUDIO_NAME = "test_input.wav"

# --- CORRECTED ASSET PATHS ---
# Test files are located relative to this script
tests_dir = Path(__file__).resolve().parent
INPUT_AUDIO_PATH = tests_dir / INPUT_AUDIO_NAME
OUTPUT_AUDIO_PATH = tests_dir / "test_output_voice.wav"

# Model paths are relative to the voice backend's root
MODELS_DIR = voice_backend_root / "rvc_models"
HUBERT_MODEL_PATH = MODELS_DIR / "hubert_base.pt"
MODEL_FOLDER_PATH = MODELS_DIR / MODEL_NAME

# Find the .pth file automatically
try:
    pth_file = next(f for f in os.listdir(MODEL_FOLDER_PATH) if f.endswith(".pth"))
    MODEL_PATH = MODEL_FOLDER_PATH / pth_file
except (StopIteration, FileNotFoundError):
    print(f"Error: Could not find a .pth model file in '{MODEL_FOLDER_PATH}'")
    sys.exit(1)


if __name__ == "__main__":
    try:
        # --- Step 1: Initialize the Wrapper ---
        print("\n[Step 1] Initializing RVCWrapper...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        is_half = False
        
        rvc_wrapper = RVCWrapper(str(MODEL_PATH), str(HUBERT_MODEL_PATH), device, is_half)
        print(f"  -> Wrapper created for model: {MODEL_PATH.name}")

        # --- Step 2: Load the Models ---
        print("\n[Step 2] Loading models into memory...")
        rvc_wrapper.load_model()
        print("  -> Models loaded successfully.")

        # --- Step 3: Load and Prepare Input Audio ---
        print(f"\n[Step 3] Loading input audio: {INPUT_AUDIO_NAME}")
        if not INPUT_AUDIO_PATH.exists():
            raise FileNotFoundError(f"Input audio not found at {INPUT_AUDIO_PATH}")
            
        input_audio, sr = sf.read(INPUT_AUDIO_PATH)
        
        if sr != 48000:
            print(f"  -> Resampling audio from {sr}Hz to 48000Hz...")
            input_audio = librosa.resample(y=input_audio.astype(np.float32), orig_sr=sr, target_sr=48000)
        
        input_audio = input_audio.astype(np.float32)
        print(f"  -> Audio is ready for processing (duration: {len(input_audio)/48000:.2f}s).")
        
        # --- Step 4: Process Audio in Chunks ---
        print("\n[Step 4] Processing audio in chunks...")
        chunk_size = 48000
        output_chunks = []

        for i in tqdm(range(0, len(input_audio), chunk_size), desc="Converting"):
            chunk = input_audio[i : i + chunk_size]
            converted_chunk = rvc_wrapper.process_chunk(chunk)
            output_chunks.append(converted_chunk)
            
        final_audio = np.concatenate(output_chunks)
        print("  -> Audio processing complete.")
        
        # --- Step 5: Save the Output ---
        print(f"\n[Step 5] Saving converted audio to: {OUTPUT_AUDIO_PATH.name}")
        sf.write(OUTPUT_AUDIO_PATH, final_audio, 48000)
        
        print("\n🎉 Test complete! 🎉")
        print("If the output file sounds correct, your wrapper is working properly.")

    except Exception as e:
        print(f"\n--- An error occurred during the test ---")
        traceback.print_exc()