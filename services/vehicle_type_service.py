import torch
from ultralytics import YOLO
import numpy as np
from typing import List, Dict

class VehicleTypeService:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = "models/best.pt"):
        if VehicleTypeService._initialized:
            return
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing VehicleTypeService on {self.device} with model {model_path}...")
        
        # Load the model ONCE
        self.model = YOLO(model_path).to(self.device)
        
        # Class names from the trained YOLOv8 model (car, bus, truck, motorcycle, scooter, autorickshaw)
        # Assuming the model classes align with the user's list. 
        # YOLOv8 stores class names in model.names
        self.class_names = self.model.names
        
        VehicleTypeService._initialized = True

    def detect_vehicle_type(self, image_or_frame) -> List[Dict]:
        """
        Detect vehicle types in an image or frame.
        Returns a list of detections: [{"type": str, "confidence": float, "bbox": [x1, y1, x2, y2]}]
        """
        if image_or_frame is None or image_or_frame.size == 0:
            return []

        # Run inference with 0.25 confidence threshold
        results = self.model(image_or_frame, conf=0.25, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = self.class_names.get(cls_id, "unknown")
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            detections.append({
                "type": label,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
            
        return detections

    def get_best_detection(self, image_or_frame) -> Dict:
        """
        Helper to return the single highest confidence detection for a crop.
        Useful when processing a cropped vehicle image.
        """
        detections = self.detect_vehicle_type(image_or_frame)
        if not detections:
            return {"type": "unknown", "confidence": 0.0, "bbox": [0, 0, 0, 0]}
        
        # Sort by confidence descending
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return detections[0]
