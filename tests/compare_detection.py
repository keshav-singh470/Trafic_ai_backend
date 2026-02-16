import cv2
from ultralytics import YOLO
import os

MODEL_PATH = r"D:\Python\Smart-Traffic-Analytics\ANPR\runs\detect\anpr_training\step3_run2\weights\best.pt"
VIDEO_PATH = r"d:\Python\Smart-Traffic-Analytics\test-video\Navranag_Circle_FIX_2_Wrong_Route_2.mp4"

def compare():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Base model for vehicle detection
    base = YOLO("yolo11n.pt") 
    
    for _ in range(100): cap.read() # Skip to a frame with cars
    ret, frame = cap.read()
    if not ret: return

    print("--- Full Frame Detection ---")
    res_full = model(frame, conf=0.1, verbose=False)[0]
    print(f"Detected {len(res_full.boxes)} plates in full frame.")

    print("\n--- Crop Based Detection ---")
    res_base = base(frame, verbose=False)[0]
    V_CLASSES = [2, 3, 5, 7]
    for box, cls in zip(res_base.boxes.xyxy, res_base.boxes.cls):
        if int(cls) in V_CLASSES:
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            roi = frame[y1:y2, x1:x2]
            res_crop = model(roi, conf=0.1, verbose=False)[0]
            if len(res_crop.boxes) > 0:
                print(f"Detected plate in vehicle at {x1,y1}")

    cap.release()

if __name__ == "__main__":
    compare()
