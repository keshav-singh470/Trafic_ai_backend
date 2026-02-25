import sys
import os
import cv2
import numpy as np

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.anpr_service import ANPRService

def test_eager_mode():
    print("\n--- Testing Eager Reporting Mode ---")
    service = ANPRService(anpr_model="models/anpr_plat.pt")
    service.reset()
    
    # Simulate a high-confidence strict reading
    # PaddleOCR usually returns (text, conf)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Mocking the OCR result to be 'strict'
    # We'll need to patch process_plate or mock its behavior
    # Instead, let's just test if the logic in run_detection handles it.
    
    # We can't easily mock the AI model here without deep patching, 
    # but we can test the helper functions that drive the logic.
    
    plate_strict = "KA03MV0927"
    print(f"Testing strict plate: {plate_strict}")
    is_strict = service.validate_indian_plate(plate_strict, mode='strict')
    print(f"Is strict: {is_strict}")
    
    # Test position aware correction
    bad_plate = "KA03MV092Z"
    corrected = service.position_aware_correct(bad_plate)
    print(f"Corrected {bad_plate} -> {corrected}")

def test_ocr_fallback_logic():
    print("\n--- Testing OCR Fallback Logic ---")
    service = ANPRService(anpr_model="models/anpr_plat.pt")
    
    # Create a dummy noisy image
    noisy_img = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(noisy_img, "DL1LY2046", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
    
    # Verify multiple preprocessing modes
    for mode in ['standard', 'high_contrast', 'inverse']:
        processed = service.preprocess_for_ocr(noisy_img, mode=mode)
        print(f"Mode '{mode}': output shape {processed.shape if processed is not None else 'FAILED'}")

if __name__ == "__main__":
    test_eager_mode()
    test_ocr_fallback_logic()
