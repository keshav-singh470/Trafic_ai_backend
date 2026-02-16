from ultralytics import YOLO

model_path = "models/helmet_triple_model.pt"
try:
    model = YOLO(model_path)
    print("Model loaded successfully.")
    print("Classes:", model.names)
except Exception as e:
    print(f"Failed to load model: {e}")
