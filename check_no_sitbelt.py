from ultralytics import YOLO
import os

model_path = "no_sitbelt.pt"
if not os.path.exists(model_path):
    print(f"Error: {model_path} does not exist.")
else:
    try:
        model = YOLO(model_path)
        print("Model loaded successfully.")
        for k, v in model.names.items():
            print(f"Class {k}: {v}")
    except Exception as e:
        print(f"Failed to load model: {e}")
