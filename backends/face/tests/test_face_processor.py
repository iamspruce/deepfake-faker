# backends/face/tests/test_face_processor.py

import os
import sys
import cv2
import traceback
from pathlib import Path

# --- Path Setup ---
# Add the backend's root directory ('face/') to the Python path
face_backend_root = str(Path(__file__).resolve().parents[1])
if face_backend_root not in sys.path:
    sys.path.append(face_backend_root)

try:
    # Import the model downloader and the core processor components
    from model_downloader import download_models
    from src import analyzer
    from src.face_processor import FaceProcessor
except ImportError as e:
    print(f"\n--- IMPORT ERROR ---: {e}")
    print("Could not import necessary modules. Please check your file paths.")
    traceback.print_exc()
    sys.exit(1)


def run_all_tests():
    """
    Runs a series of tests to showcase different features of the face processor.
    """
    print("--- Starting FaceProcessor Feature Showcase ---")

    # --- Configuration ---
    tests_dir = Path(__file__).resolve().parent
    SOURCE_IMAGE_PATH = str(tests_dir / "source_face.jpeg")
    TARGET_IMAGE_PATH = str(tests_dir / "target_image.JPG")

    # --- Step 1: Download Models ---
    print("\n[Step 1] Checking for and downloading required models...")
    try:
        download_models()
        print("  -> Models are downloaded and ready.")
    except Exception as e:
        print(f"\n--- ERROR DURING MODEL DOWNLOAD ---"); traceback.print_exc(); return
    
    # --- Step 2: Initialize FaceProcessor ---
    print("\n[Step 2] Initializing FaceProcessor...")
    try:
        face_processor = FaceProcessor()
        print("  -> FaceProcessor initialized successfully.")
    except Exception as e:
        print(f"\n--- ERROR DURING INITIALIZATION ---"); traceback.print_exc(); return

    # --- Step 3: Load Source and Target Images/Faces ---
    print("\n[Step 3] Loading and analyzing source/target images...")
    try:
        source_img = cv2.imread(SOURCE_IMAGE_PATH)
        target_img = cv2.imread(TARGET_IMAGE_PATH)
        if source_img is None or target_img is None:
            raise FileNotFoundError("Could not read source or target image.")

        source_face = analyzer.get_one_face(source_img)
        if not source_face:
            raise ValueError("No face detected in the source image.")
            
        face_processor.source_face = source_face # Directly set the source face
        print("  -> Source and target images loaded successfully.")
    except Exception as e:
        print(f"\n--- ERROR LOADING IMAGES ---"); traceback.print_exc(); return

    # --- Run Individual Tests ---
    
    # Test 1: Simple Face Swap
    test_simple_swap(face_processor, target_img.copy(), str(tests_dir / "output_1_simple_swap.jpg"))
    
    # Test 2: Swap with Face Enhancement
    test_swap_with_enhancement(face_processor, target_img.copy(), str(tests_dir / "output_2_swap_enhanced.jpg"))
    
    print("\n🎉 All tests finished successfully! 🎉")
    print(f"Check the output files in your '{tests_dir.name}' folder.")


def test_simple_swap(processor, target_img, output_path):
    """Tests the basic face swapping functionality."""
    print("\n--- Testing: Simple Face Swap ---")
    processor.use_face_enhancement = False
    result_frame, success = processor.process_frame(target_img)
    if success:
        cv2.imwrite(output_path, result_frame)
        print(f"  -> Success! Result saved to {Path(output_path).name}")
    else:
        print("  -> Swap failed for this test.")


def test_swap_with_enhancement(processor, target_img, output_path):
    """Tests swapping combined with GFPGAN face enhancement."""
    print("\n--- Testing: Swap with Face Enhancement ---")
    processor.use_face_enhancement = True
    result_frame, success = processor.process_frame(target_img)
    if success:
        cv2.imwrite(output_path, result_frame)
        print(f"  -> Success! Result saved to {Path(output_path).name}")
    else:
        print("  -> Swap failed for this test.")
    processor.use_face_enhancement = False # Reset for other tests


if __name__ == "__main__":
    run_all_tests()