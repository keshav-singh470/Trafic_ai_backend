import sys
import os
import cv2
import numpy as np

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.anpr_service import ANPRService
from services.ocr_service import OCRService

def test_ocr_service():
    print("\n--- Testing OCR Service (PaddleOCR) ---")
    ocr = OCRService()
    
    # Create a dummy image with some text if possible, or just check initialization
    dummy_img = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(dummy_img, "KA03MV0927", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    text, conf = ocr.get_text(dummy_img)
    print(f"Results for dummy plate: text='{text}', conf={conf:.2f}")
    
def test_preprocessing():
    print("\n--- Testing Preprocessing Logic ---")
    service = ANPRService(anpr_model="models/anpr_plat.pt")
    
    # Test image (small)
    small_img = np.zeros((30, 100, 3), dtype=np.uint8)
    processed = service.preprocess_for_ocr(small_img)
    print(f"Small image (30x100): processed={'OK' if processed is not None else 'SKIPPED (Correct)'}")
    
    # Test image (normal)
    normal_img = np.zeros((50, 150, 3), dtype=np.uint8)
    cv2.putText(normal_img, "DL1LY2046", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    processed = service.preprocess_for_ocr(normal_img)
    if processed is not None:
        print(f"Normal image (50x150): processed shape {processed.shape} (3x upscale expected)")
        # cv2.imwrite("outputs/test_preprocessed.jpg", processed)
    else:
        print(f"Normal image (50x150): processed=FAILED")

def test_aggregation():
    print("\n--- Testing Multi-frame Aggregation Logic ---")
    service = ANPRService(anpr_model="models/anpr_plat.pt")
    
    readings = [
        {"text": "KA03MV0927", "conf": 0.9},
        {"text": "KA03MV0927", "conf": 0.85},
        {"text": "KA03MV0927", "conf": 0.88},
        {"text": "KA03MV0922", "conf": 0.4},
        {"text": "KA03MV0927", "conf": 0.92},
    ]
    
    best_text, best_conf = service._best_vote(readings)
    print(f"Aggregation result: text='{best_text}', conf={best_conf:.2f} (Expected: KA03MV0927)")
    
    # Tie breaker test
    readings_tie = [
        {"text": "DL1LY2046", "conf": 0.8},
        {"text": "DL1LY2046", "conf": 0.9},
        {"text": "DL1LY2046XX", "conf": 0.7}, # Longer but lower conf
        {"text": "DL1LY2046XX", "conf": 0.75},
    ]
    best_text_tie, best_conf_tie = service._best_vote(readings_tie)
    print(f"Tie-breaker result: text='{best_text_tie}', conf={best_conf_tie:.2f}")

if __name__ == "__main__":
    try:
        test_ocr_service()
    except Exception as e:
        print(f"OCR Service Test Error: {e}")
        
    try:
        test_preprocessing()
    except Exception as e:
        print(f"Preprocessing Test Error: {e}")
        
    try:
        test_aggregation()
    except Exception as e:
        print(f"Aggregation Test Error: {e}")
