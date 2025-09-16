import asyncio
import os
import sys
import traceback
import cv2
import numpy as np
import uvicorn
import logging
from fastapi import FastAPI, UploadFile, File, Body, Response
from fastrtc import Stream

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('face_backend.log'), logging.StreamHandler()]
)

# Add face backend directory to Python path
face_backend_dir = os.path.dirname(os.path.abspath(__file__))
if face_backend_dir not in sys.path:
    sys.path.append(face_backend_dir)

from src.face_processor import FaceProcessor
from model_downloader import download_models

BACKEND_STATE = {
    "status": "initializing",
    "face_processor": None,
    "error_message": ""
}

async def initialize_face_processor():
    try:
        logging.info("Face Backend: Checking and downloading models...")
        download_models()
        logging.info("Face Backend: Initializing FaceProcessor...")
        loop = asyncio.get_event_loop()
        fp = await loop.run_in_executor(None, FaceProcessor)
        
        if fp.face_swapper is None:
            raise RuntimeError("Face swapper model failed to load.")
            
        BACKEND_STATE["face_processor"] = fp
        BACKEND_STATE["status"] = "ready"
        logging.info("Face Backend: Ready to accept connections.")
    except Exception as e:
        logging.error(f"CRITICAL ERROR during initialization: {str(e)}\n{traceback.format_exc()}")
        BACKEND_STATE["status"] = "error"
        BACKEND_STATE["error_message"] = str(e)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(initialize_face_processor())

@app.get("/health")
async def health_check():
    return {"status": BACKEND_STATE["status"], "error": BACKEND_STATE["error_message"]}

@app.post("/update-settings")
async def update_settings(settings: dict = Body(...)):
    fp: FaceProcessor = BACKEND_STATE.get("face_processor")
    if not fp:
        return Response(status_code=503, content="FaceProcessor not ready.")
    
    try:
        if "use_face_enhancement" in settings:
            is_enabled = bool(settings["use_face_enhancement"])
            fp.use_face_enhancement = is_enabled
            logging.info(f"Face enhancement set to: {is_enabled}")
        
        return {"status": "success", "message": "Settings updated."}
    except Exception as e:
        logging.error(f"Error updating settings: {str(e)}\n{traceback.format_exc()}")
        return Response(status_code=500, content=f"Error updating settings: {e}")

@app.post("/set-source-face")
async def set_source_face(file: UploadFile = File(...)):
    fp: FaceProcessor = BACKEND_STATE.get("face_processor")
    if not fp:
        return Response(status_code=503, content="Face processor not ready.")
    
    try:
        image_bytes = await file.read()
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        analysis = fp.analyze_face(image_np)
        if analysis and analysis.get("face"):
            fp.set_face_mappings({"user_face_01": {"source_face": analysis["face"]}})
            logging.info("Source face has been set successfully.")
            return {"status": "success", "message": "Source face set."}
        else:
            return Response(status_code=400, content="No face detected in the uploaded image.")
    except Exception as e:
        logging.error(f"Error setting source face: {str(e)}\n{traceback.format_exc()}")
        return Response(status_code=500, content=f"Error setting source face: {e}")

def process_frame_handler(image: np.ndarray) -> np.ndarray:
    fp: FaceProcessor = BACKEND_STATE.get("face_processor")
    if not fp or BACKEND_STATE["status"] != "ready":
        return image
        
    try:
        result_frame, swap_successful = fp.process_frame(image)
        
        if swap_successful and fp.use_face_enhancement:
            faces = fp.detect_faces(result_frame)
            if faces:
                main_face_bbox = faces[0].get("bbox")
                if main_face_bbox is not None:
                    result_frame = fp.enhance_face_region(result_frame, main_face_bbox)
                    
        return result_frame
    except Exception as e:
        logging.error(f"Error during process_frame: {str(e)}")
        return image

stream = Stream(handler=process_frame_handler, modality="video", mode="send-receive")
stream.mount(app)

if __name__ == "__main__":
    logging.info("Face Backend: Uvicorn server starting...")
    uvicorn.run(app, host="127.0.0.1", port=8081, log_level="warning")