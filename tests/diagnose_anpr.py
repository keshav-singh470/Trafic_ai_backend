import cv2
from ultralytics import YOLO
import torch
import os

def diagnose_anpr():
    model_path = r"D:\Python\Smart-Traffic-Analytics\ANPR\runs\detect\anpr_training\step3_run2\weights\best.pt"
    video_path = r"d:\Python\Smart-Traffic-Analytics\test-video\Navranag_Circle_FIX_2_Wrong_Route_2.mp4"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
        
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read video")
        return
        
    # Run detection on full frame first to see if it detects anything
    results = model(frame)[0]
    print(f"Detected {len(results.boxes)} objects in full frame.")
    for box in results.boxes:
        print(f"Object: {box.cls} | Conf: {box.conf.item():.2f} | XYXY: {box.xyxy[0].tolist()}")
        
    cap.release()

if __name__ == "__main__":
    diagnose_anpr()
