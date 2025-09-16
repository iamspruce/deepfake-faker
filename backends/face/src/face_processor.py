# src/core/face_processor.py (New Orchestrator Version)

import numpy as np
import logging

from . import swapper, enhancer, config
from .analyzer import get_face_analyser, get_one_face
from .typing import Frame, Face

logger = logging.getLogger(__name__)

class FaceProcessor:
    def __init__(self):
        logger.info("Initializing FaceProcessor Orchestrator...")
        self.source_face: Face = None
        self.use_face_enhancement: bool = False

        # Pre-download and load all necessary models on initialization
        swapper.pre_check()
        enhancer.load_face_enhancer()
        get_face_analyser() # Initializes the buffalo_l model
        swapper.get_face_swapper()   # Initializes the inswapper model
        
        self.face_swapper = swapper.get_face_swapper()
        
        logger.info("FaceProcessor Orchestrator is ready.")

    def set_face_mappings(self, mappings):
        """Sets the source face from the provided mapping."""
        if "user_face_01" in mappings and hasattr(mappings["user_face_01"]["source_face"], "embedding"):
            self.source_face = mappings["user_face_01"]["source_face"]
            logger.info("Source face has been updated in the processor.")
        else:
            logger.warning("set_face_mappings received invalid data or no face embedding.")

    def analyze_face(self, image: np.ndarray):
        """Analyzes an image and returns the detected face data."""
        face = get_one_face(image)
        if face:
            # The 'face' object from analyzer is what the swapper needs
            return {"face": face} 
        return None

    def process_frame(self, frame: Frame) -> (Frame, bool):
        """The main processing pipeline for each frame."""
        swap_successful = False
        result_frame = frame
        
        # We process every face found in the frame
        target_face = get_one_face(frame)

        if target_face and self.source_face:
            result_frame = swapper.swap_face(
                    self.source_face,
                    target_face,
                    result_frame # Use the result of the previous swap as the input for the next
                )
            swap_successful = True
        
        # Apply enhancement to the final frame if enabled and a swap happened
        if swap_successful and self.use_face_enhancement:
            result_frame = enhancer.enhance_face(result_frame)
            
        return result_frame, swap_successful