from ultralytics import YOLO
import json
import os

model_path = r'models/best.pt'
if os.path.exists(model_path):
    model = YOLO(model_path)
    print(json.dumps(model.names, indent=4))
else:
    print(f"File not found: {model_path}")
