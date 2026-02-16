import sys
import os
import cv2

# Add parent directory to path to allow importing services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set env var for protobuf just in case
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from services.overload_service import OverloadService

# Pick a video that exists
VIDEO_PATH = r"uploads/0908b2fd-8dd1-405d-91bf-339e4506fe2c_Hulimavu_Circle_FIX_2_Helmet.mp4"

try:
    print("Initializing OverloadService...")
    # Using explicit path that matches what API would send (relative to ANPR-System root)
    service = OverloadService(model_path="models/helmet_triple_model.pt") 
    print("Service initialized.")

    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found: {VIDEO_PATH}")
        # Try to find any video
        files = os.listdir("uploads")
        mp4s = [f for f in files if f.endswith(".mp4")]
        if mp4s:
            VIDEO_PATH = os.path.join("uploads", mp4s[0])
            print(f"Using alternative video: {VIDEO_PATH}")
        else:
            print("No videos found in uploads/")
            sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Failed to open video.")
        sys.exit(1)

    print(f"Processing video frames from {VIDEO_PATH}...")
    frames_to_process = 30
    for i in range(frames_to_process):
        ret, frame = cap.read()
        if not ret: break
        
        violations = service.run_detection(frame)
        print(f"Frame {i}: {len(violations)} violations")
        for v in violations:
            print(f"  - {v}")

    cap.release()
    print("Test completed successfully.")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
