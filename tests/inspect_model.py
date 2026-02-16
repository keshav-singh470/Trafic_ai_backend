from ultralytics import YOLO

model_path = r"d:\Python\Smart-Traffic-Analytics\best (3).pt"
try:
    model = YOLO(model_path)
    print("Model Classes:", model.names)
except Exception as e:
    print(f"Error loading model: {e}")
