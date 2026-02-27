import cv2
import sys
import os
from ultralytics import YOLO

# Add parent directory to path to import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.vehicle_type_service import VehicleTypeService

def test_inference():
    model_path = "models/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        return

    print("--- Initializing Service ---")
    service = VehicleTypeService(model_path=model_path)
    
    # Use a sample image or a frame from test.mp4 if it exists
    video_path = "test.mp4"
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print("--- Running Detection on video frame ---")
            detections = service.detect_vehicle_type(frame)
            print(f"Found {len(detections)} detections:")
            for d in detections:
                print(f"  - {d['type']}: {d['confidence']:.2f} at {d['bbox']}")
            
            best = service.get_best_detection(frame)
            print(f"Best detection: {best['type']} ({best['confidence']:.2f})")
        else:
            print("❌ Failed to read frame from test.mp4")
    else:
        print("ℹ️ test.mp4 not found, skipping video frame test.")

    print("\n✅ Verification script finished.")

if __name__ == "__main__":
    test_inference()
