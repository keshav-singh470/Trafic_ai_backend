import cv2
from ultralytics import YOLO
import torch
import os

MODELS = [
    r"D:\Python\Smart-Traffic-Analytics\ANPR\runs\detect\anpr_training\step3_run2\weights\best.pt",
    r"D:\Python\Smart-Traffic-Analytics\ANPR\runs\detect\anpr_training\step3_run\weights\best.pt",
    r"D:\Python\Smart-Traffic-Analytics\ANPR\yolo26n.pt",
    r"D:\Python\Smart-Traffic-Analytics\ANPR-System\best.pt"
]
VIDEO_PATH = r"d:\Python\Smart-Traffic-Analytics\test-video\Navranag_Circle_FIX_2_Wrong_Route_2.mp4"

def diagnose():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = []
    for _ in range(10): # Take 10 frames spread out
        for _ in range(30): cap.read() # skip 30
        ret, frame = cap.read()
        if ret: frames.append(frame)
    cap.release()

    for mpath in MODELS:
        if not os.path.exists(mpath): 
            print(f"Skipping {mpath} (not found)")
            continue
        print(f"\n--- Testing Model: {mpath} ---")
        model = YOLO(mpath)
        total_dets = 0
        for i, frame in enumerate(frames):
            results = model(frame, conf=0.1, verbose=False)[0]
            if len(results.boxes) > 0:
                print(f"Frame {i}: Detected {len(results.boxes)} plates.")
                total_dets += len(results.boxes)
        print(f"Total detections across 10 frames: {total_dets}")

if __name__ == "__main__":
    diagnose()
