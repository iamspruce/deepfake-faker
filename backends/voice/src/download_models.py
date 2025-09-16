# model_downloader.py

import os
import requests
import zipfile
import shutil

# --- Configuration ---
# A manifest of models your UI can offer for download
MODEL_MANIFEST = {
    "Klee": {
        "url": "https://huggingface.co/qweshkka/Klee/resolve/main/Klee.zip",
        "description": "Klee from Genshin Impact",
        "credit": "qweshsmashjuicefruity",
    },
    # You can add other models here that you want to make downloadable
    # "AnotherModel": {
    #     "url": "https://example.com/AnotherModel.zip",
    #     "description": "A different character voice.",
    #     "credit": "SomeCreator",
    # },
}

# Link for downloading essential pretrained models
PRETRAINED_DOWNLOAD_LINK = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/'

def download_file(url, destination_path):
    """Downloads a file, showing progress."""
    print(f"Downloading {os.path.basename(destination_path)}...")
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
            print() # Newline after download finishes

def download_and_extract_zip(url, extract_folder):
    """Downloads and extracts a zip file."""
    temp_zip_path = os.path.join(extract_folder, "temp_model.zip")
    download_file(url, temp_zip_path)

    print(f"Extracting {os.path.basename(extract_folder)}...")
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    os.remove(temp_zip_path)
    print(f"Extraction complete for {os.path.basename(extract_folder)}.")

def download_pretrained_models(pretrain_dir):
    """Checks for and downloads hubert_base.pt and rmvpe.pt."""
    os.makedirs(pretrain_dir, exist_ok=True)
    pretrained_models = ['hubert_base.pt', 'rmvpe.pt']

    for model_name in pretrained_models:
        destination = os.path.join(pretrain_dir, model_name)
        if not os.path.exists(destination):
            print(f"Essential file '{model_name}' not found.")
            download_file(f'{PRETRAINED_DOWNLOAD_LINK}{model_name}', destination)
        else:
            print(f"Essential file '{model_name}' already exists.")

def download_voice_model(model_name, models_dir):
    """Downloads a specific voice model from the manifest."""
    if model_name not in MODEL_MANIFEST:
        raise ValueError(f"Model '{model_name}' not found in the manifest.")

    model_info = MODEL_MANIFEST[model_name]
    model_folder_path = os.path.join(models_dir, model_name)

    if os.path.exists(model_folder_path):
        print(f"Model '{model_name}' directory already exists. Skipping download.")
        return model_folder_path

    print(f"Model '{model_name}' not found, starting download...")
    os.makedirs(model_folder_path, exist_ok=True)

    try:
        download_and_extract_zip(model_info["url"], model_folder_path)
        return model_folder_path
    except Exception as e:
        if os.path.exists(model_folder_path):
            shutil.rmtree(model_folder_path)
        raise RuntimeError(f"Failed to download and extract '{model_name}': {e}")