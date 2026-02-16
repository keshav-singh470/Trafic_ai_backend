from ultralytics import YOLO
import os

model_path = "../external_helmet_repo/best (2).pt"
try:
    model = YOLO(model_path)
    print("Model loaded successfully.")
    for k, v in model.names.items():
        print(f"Class {k}: {v}")
except Exception as e:
    print(f"Failed to load model: {e}")
