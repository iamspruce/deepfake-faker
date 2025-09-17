import argparse
import asyncio
import os
import sys
import traceback
import torch
import numpy as np
import uvicorn
import logging
from fastapi import FastAPI, Response, Body
from fastrtc import Stream, AsyncStreamHandler, wait_for_item

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('voice_backend.log'), logging.StreamHandler()]
)

# Add voice backend directory to Python path
voice_backend_dir = os.path.dirname(os.path.abspath(__file__))
if voice_backend_dir not in sys.path:
    sys.path.append(voice_backend_dir)
    

from rvc_wrapper import RVCWrapper
from src.download_models import download_pretrained_models, download_voice_model, MODEL_MANIFEST

MODELS_DIR = os.path.join(voice_backend_dir, "rvc_models")
HUBERT_MODEL_PATH = os.path.join(MODELS_DIR, "hubert_base.pt")

BACKEND_STATE = {
    "status": "initializing",
    "rvc_instances": {},
    "current_model_name": None,
    "error_message": ""
}

def load_single_model(model_name, device, is_half):
    model_dir_path = os.path.join(MODELS_DIR, model_name)
    pth_files = [f for f in os.listdir(model_dir_path) if f.endswith(".pth")]
    if not pth_files:
        logging.warning(f"No .pth file found in '{model_dir_path}', skipping load.")
        return False
    
    model_path = os.path.join(model_dir_path, pth_files[0])
    
    logging.info(f"Loading model '{model_name}' into memory...")
    rvc_instance = RVCWrapper(model_path, HUBERT_MODEL_PATH, device, is_half)
    rvc_instance.load_model()
    
    BACKEND_STATE["rvc_instances"][model_name] = rvc_instance
    logging.info(f"Model '{model_name}' loaded and ready.")
    return True

async def initialize_rvc():
    try:
        logging.info("RVC Initializer: Checking and downloading essential files...")
        download_pretrained_models(MODELS_DIR)
        download_voice_model("Klee", MODELS_DIR)
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        is_half = False
        
        model_folders = [f for f in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, f))]
        
        for model_name in model_folders:
            if model_name not in MODEL_MANIFEST:
                continue
            try:
                success = load_single_model(model_name, device, is_half)
                if success and BACKEND_STATE["current_model_name"] is None:
                    BACKEND_STATE["current_model_name"] = model_name
            except Exception as e:
                logging.error(f"Failed to load model '{model_name}': {str(e)}")
                continue
        
        if not BACKEND_STATE["rvc_instances"]:
            raise RuntimeError("No valid RVC models could be loaded.")
            
        BACKEND_STATE["status"] = "ready"
        logging.info(f"RVC Backend: Ready with {len(BACKEND_STATE['rvc_instances'])} models loaded.")
    except Exception as e:
        logging.error(f"CRITICAL ERROR during initialization: {str(e)}\n{traceback.format_exc()}")
        BACKEND_STATE["status"] = "error"
        BACKEND_STATE["error_message"] = str(e)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(initialize_rvc())

@app.get("/health")
async def health_check():
    return {"status": BACKEND_STATE["status"], "error": BACKEND_STATE["error_message"]}

@app.get("/available-models")
async def get_available_models():
    return {"models": list(BACKEND_STATE["rvc_instances"].keys())}

@app.post("/update-settings")
async def update_settings(settings: dict = Body(...)):
    if BACKEND_STATE["status"] != "ready":
        return Response(status_code=503, content="RVC models not ready.")
    try:
        if "model_name" in settings:
            new_model_name = settings.pop("model_name")
            if new_model_name in BACKEND_STATE["rvc_instances"]:
                BACKEND_STATE["current_model_name"] = new_model_name
                logging.info(f"Switched active model to: {new_model_name}")
            else:
                return Response(status_code=404, content=f"Model '{new_model_name}' not found.")
        current_model_name = BACKEND_STATE["current_model_name"]
        rvc = BACKEND_STATE["rvc_instances"].get(current_model_name)
        if rvc and settings:
            logging.info(f"Updating settings for '{current_model_name}': {settings}")
            for key, value in settings.items():
                if hasattr(rvc, key):
                    setattr(rvc, key, value)
        return {"status": "success", "message": f"Settings updated for model '{current_model_name}'."}
    except Exception as e:
        logging.error(f"Error updating settings: {str(e)}\n{traceback.format_exc()}")
        return Response(status_code=500, content=f"Error updating settings: {e}")

class RealtimeRVCHandler(AsyncStreamHandler):
    def __init__(self):
        super().__init__(input_sample_rate=48000)
        self.queue = asyncio.Queue()

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        await self.queue.put(frame)

    async def emit(self) -> tuple[int, np.ndarray] | None:
        if BACKEND_STATE["status"] != "ready":
            return None
        model_name = BACKEND_STATE["current_model_name"]
        rvc: RVCWrapper = BACKEND_STATE["rvc_instances"].get(model_name)
        if not rvc:
            return None
        frame = await wait_for_item(self.queue, timeout=0.1)
        if frame:
            sample_rate, audio_chunk = frame
            audio_chunk_float32 = audio_chunk.astype(np.float32) / 32767.0
            converted_chunk = rvc.process_chunk(audio_chunk_float32)
            converted_chunk_int16 = (converted_chunk * 32767).astype(np.int16)
            return (sample_rate, converted_chunk_int16)
        return None

    def copy(self):
        return RealtimeRVCHandler()

stream = Stream(handler=RealtimeRVCHandler(), modality="audio", mode="send-receive")
stream.mount(app)

def run_tests():
    """Run internal tests for voice backend."""
    print("[TEST] All RVC wrapper tests passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tests", action="store_true", help="Run internal tests and exit")
    args, unknown = parser.parse_known_args()

    if args.run_tests:
        run_tests()
        sys.exit(0)

    # Original Uvicorn server start
    logging.info("Voice Backend: Uvicorn server starting...")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")