import cv2
from ultralytics import YOLO
import torch
import os

MODEL_PATH = r"D:\Python\Smart-Traffic-Analytics\ANPR\runs\detect\anpr_training\step3_run2\weights\best.pt"
VIDEO_PATH = r"d:\Python\Smart-Traffic-Analytics\test-video\Navranag_Circle_FIX_2_Wrong_Route_2.mp4"

def visual_diagnose():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        # Run at very low conf to see if it finds ANY plates
        res = model(frame, conf=0.1, verbose=False)[0]
        if len(res.boxes) > 0:
            print(f"Frame {frame_idx}: Detected {len(res.boxes)} plates.")
            for i, box in enumerate(res.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].item()
                plate = frame[y1:y2, x1:x2]
                if plate.size > 0:
                    cv2.imwrite(f"debug_plate_{frame_idx}_{i}_{conf:.2f}.jpg", plate)
            if frame_idx > 500: break # stop after some detections

    cap.release()

if __name__ == "__main__":
    visual_diagnose()
