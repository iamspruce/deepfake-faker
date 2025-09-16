# face/src/core/config.py
import os
import torch
import platform
import logging
from pathlib import Path

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Correctly resolves the 'models' directory, assuming this file is in 'src/core/'
MODELS_DIR = str(Path(__file__).resolve().parents[1] / "models")

# --- Hardware-Specific Configuration Profiles ---

# For NVIDIA GPUs (CUDA)
NVIDIA_GPU_CONFIG = {
    "DEVICE": "cuda",
    "SWAPPER_BACKEND": "onnx",
    "EXECUTION_PROVIDERS": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "SWAPPER_MODEL_NAME": "inswapper_128_fp16.onnx"
}

# For Apple Silicon (M-series Macs)
APPLE_SILICON_CONFIG = {
    "DEVICE": "mps",
    "SWAPPER_BACKEND": "coreml",
    "EXECUTION_PROVIDERS": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "SWAPPER_MODEL_NAME": "inswapper_128.onnx"
}

# ✅ NEW: For Intel-based Macs with Core ML support
INTEL_MAC_CONFIG = {
    "DEVICE": "cpu",  # PyTorch will still use the CPU
    "SWAPPER_BACKEND": "onnx", # The backend is ONNX, accelerated by Core ML
    "EXECUTION_PROVIDERS": ["CPUExecutionProvider"],
    "SWAPPER_MODEL_NAME": "inswapper_128_fp16.onnx" # Use the full-precision ONNX model
}

# For generic CPU
CPU_CONFIG = {
    "DEVICE": "cpu",
    "SWAPPER_BACKEND": "onnx",
    "EXECUTION_PROVIDERS": ["CPUExecutionProvider"],
    "SWAPPER_MODEL_NAME": "inswapper_128.onnx"
}

# --- Automatic Hardware Detection and Configuration Loading ---

config = CPU_CONFIG # Default to CPU
logger.info("Detecting optimal hardware configuration...")

try:
    import onnxruntime
    available_providers = onnxruntime.get_available_providers()

    # ✅ CORRECTED: Multi-stage check with proper priority
    # Priority 1: NVIDIA GPU
    if torch.cuda.is_available() and "CUDAExecutionProvider" in available_providers:
        logger.info("✅ NVIDIA GPU (CUDA) detected. Using GPU configuration.")
        config = NVIDIA_GPU_CONFIG
    # Priority 2: Apple Silicon (M-series Mac)
    elif torch.backends.mps.is_available():
        logger.info("✅ Apple Silicon (MPS/CoreML) detected. Using Apple Silicon configuration.")
        config = APPLE_SILICON_CONFIG
    # Priority 3: Intel Mac (if CoreML provider is available)
    elif platform.system() == "Darwin" and "CoreMLExecutionProvider" in available_providers:
        logger.info("✅ Intel Mac with CoreML support detected. Using optimized ONNX configuration.")
        config = INTEL_MAC_CONFIG
    else:
        logger.info("⚠️ No compatible GPU or accelerator detected. Falling back to CPU configuration.")
        config = CPU_CONFIG

except ImportError:
    logger.warning("onnxruntime not found. Configuration will be basic CPU.")


# Unpack the selected configuration into global variables
DEVICE = config["DEVICE"]
SWAPPER_BACKEND = config["SWAPPER_BACKEND"]
EXECUTION_PROVIDERS = config["EXECUTION_PROVIDERS"]
SWAPPER_MODEL_NAME = config["SWAPPER_MODEL_NAME"]

logger.info(f"  - PyTorch Device: {DEVICE}")
logger.info(f"  - Swapper Backend: {SWAPPER_BACKEND}")
logger.info(f"  - ONNX Providers: {EXECUTION_PROVIDERS}")
logger.info(f"  - Swapper Model: {SWAPPER_MODEL_NAME}")


# --- User-Tunable Settings ---


mouth_mask_down_size = 0.02
mouth_mask_size = 1.0
mouth_mask_feather_ratio = 8
mask_down_size = 1.0
mask_size=1.0
execution_threads = 4
use_mouth_mask = True
mask_feather_ratio = 8
mask_padding = 0.05
video_encoder = 'libx264'
video_quality = 20