from ultralytics import YOLO
import os

model_path = "models/seatbelt_specialized.pt"
if not os.path.exists(model_path):
    print(f"Error: {model_path} does not exist.")
else:
    try:
        model = YOLO(model_path)
        print("Model loaded successfully.")
        print("Classes:", model.names)
    except Exception as e:
        print(f"Failed to load model: {e}")
