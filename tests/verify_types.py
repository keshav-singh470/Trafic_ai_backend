import cv2
import numpy as np
import os
from services.helmet_service import HelmetService
from services.overload_service import OverloadService
from services.wrong_side_service import WrongSideService

def test_services():
    # Create dummy frame
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Instantiate services
    print("Initializing services...")
    try:
        helmet_svc = HelmetService()
        overload_svc = OverloadService()
        wrong_side_svc = WrongSideService()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    # Run detection on dummy frame
    # (Won't detect anything, but ensures no crash on basic processing)
    print("Running detections...")
    try:
        h_res = helmet_svc.run_detection(frame)
        o_res = overload_svc.run_detection(frame)
        w_res = wrong_side_svc.run_detection(frame)
        
        print("Helmet result type:", type(h_res), h_res)
        print("Overload result type:", type(o_res), o_res)
        print("WrongSide result type:", type(w_res), w_res)
        
        print("Verification passed!")
    except Exception as e:
        print(f"Detection failed: {e}")

if __name__ == "__main__":
    test_services()
