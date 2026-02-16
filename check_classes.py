from ultralytics import YOLO
import os

model_path = "models/helmet_triple_model.pt"
try:
    model = YOLO(model_path)
    print("Model loaded successfully.")
    for k, v in model.names.items():
        print(f"Class {k}: {v}")
except Exception as e:
    print(f"Failed to load model: {e}")

try:
    size = os.path.getsize("models/seatbelt_specialized.pt")
    print(f"seatbelt_specialized.pt size: {size} bytes")
except Exception as e:
    print(f"Error checking seatbelt_specialized.pt size: {e}")
