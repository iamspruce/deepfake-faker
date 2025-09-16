# backends/face/src/core/enhancer.py (Modified to be skippable)

import os
import torch
import logging

from .typing import Frame
from . import config
from .utilities import conditional_download

# --- MODIFICATION: Wrap the problematic import ---
try:
    import gfpgan
    GFPGAN_AVAILABLE = True
except ImportError:
    print("\n--- WARNING ---")
    print("Could not import 'gfpgan'. The face enhancement feature will be disabled.")
    print("This is likely due to a torch/torchvision version conflict.")
    print("---------------\n")
    GFPGAN_AVAILABLE = False

# Module-level variable to hold the loaded model
FACE_ENHANCER = None
logger = logging.getLogger(__name__)

def load_face_enhancer() -> None:
    """
    Downloads and loads the GFPGAN model into memory.
    Skips loading if the library is not available.
    """
    global FACE_ENHANCER
    # --- MODIFICATION: Check if the library was imported successfully ---
    if FACE_ENHANCER is not None or not GFPGAN_AVAILABLE:
        return

    model_path = os.path.join(config.MODELS_DIR, config.ENHANCER_MODEL_NAME)
    model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
    
    print("Checking for face restoration model (GFPGAN)...")
    if not os.path.exists(model_path):
        conditional_download(config.MODELS_DIR, [model_url])
    
    print("Loading face restoration model (GFPGAN)...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=device)
        print(f"GFPGAN model loaded successfully on device: {device}")
    except Exception as e:
        logger.error(f"Could not load GFPGAN model: {e}")
        FACE_ENHANCER = None

def enhance_face(target_frame: Frame) -> Frame:
    """
    Enhances a face in the target frame. Returns the original if the model isn't loaded.
    """
    if FACE_ENHANCER is None:
        return target_frame
    
    try:
        _, _, restored_img = FACE_ENHANCER.enhance(
            target_frame, has_aligned=False, only_center_face=True, paste_back=True
        )
        if restored_img is not None:
            return restored_img
    except Exception as e:
        logger.error(f"Failed to enhance face: {e}")

    return target_frame