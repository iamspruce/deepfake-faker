# face/model_downloader_face.py

import os
import requests
from pathlib import Path

# --- Configuration ---
MODELS_DIR = Path(__file__).resolve().parent / "models"


MODEL_MANIFEST = {
   "inswapper_128_fp16": {
        "url": "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128_fp16.onnx",
        "description": "The main face swapping model (High Quality).",
    },
    "GFPGANv1.4.pth": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "description": "Face enhancement model.",
    },
}

def download_file(url, destination_path):
    """Downloads a file with a progress bar."""
    print(f"Downloading {destination_path.name}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(destination_path, 'wb') as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                if total_size > 0:
                    downloaded += len(chunk)
                    progress = (downloaded / total_size) * 100
                    print(f"\r  -> Progress: {progress:.2f}%", end="")
            print()

def download_models():
    """Checks for and downloads all required face models."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for model_name, model_info in MODEL_MANIFEST.items():
        destination = MODELS_DIR / model_name
        if not os.path.exists(destination):
            print(f"Model '{model_name}' not found.")
            download_file(model_info["url"], destination)
        else:
            print(f"Model '{model_name}' already exists.")
            
  