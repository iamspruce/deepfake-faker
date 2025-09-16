# face/analyzer.py

from typing import Any
import os
import insightface
import numpy as np
import logging # --- NEW ---
from . import config
from .typing import Frame

FACE_ANALYSER = None
logger = logging.getLogger(__name__) # --- NEW ---

def get_face_analyser() -> Any:
    """Initializes and returns the FaceAnalysis model."""
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        # --- MODIFIED: Added a log message ---
        logger.info(f"Initializing FaceAnalyser with providers: {config.EXECUTION_PROVIDERS}")
        FACE_ANALYSER = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=config.MODELS_DIR,
            providers=config.EXECUTION_PROVIDERS,
        )
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(320, 320))
    return FACE_ANALYSER

def get_one_face(frame: Frame) -> Any:
    """Gets the most prominent face from a frame."""
    face_analyser = get_face_analyser()
    faces = face_analyser.get(frame)
    try:
        return min(faces, key=lambda x: x.bbox[0])
    except (ValueError, IndexError):
        return None