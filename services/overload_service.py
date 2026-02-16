from .base import BaseTrafficService
import cv2
import numpy as np

class OverloadService(BaseTrafficService):
    def __init__(self, model_path="models/helmet_triple_model.pt"):
        # Use user-provided model that detects riders/helmets
        super().__init__(model_path=model_path)
        print(f"OverloadService initialized with {model_path}")
        # Classes: 0: with_helmet, 1: without_helmet, 3: motorcyclist

    def run_detection(self, frame):
        detections, tracked_dets = self.process_frame(frame)
        violations = []

        if tracked_dets is None:
            return violations

        boxes = tracked_dets.xyxy
        class_ids = tracked_dets.class_id
        tracker_ids = tracked_dets.tracker_id
        
        if tracker_ids is None:
            return violations
        
        bikes = []
        riders = []

        # Separate bikes and riders (heads/helmets)
        for box, cls_id, tid in zip(boxes, class_ids, tracker_ids):
            if cls_id == 3: # motorcyclist (The bike itself + rider usually, or just bike)
                bikes.append({"box": box, "id": tid, "rider_count": 0})
            elif cls_id in [0, 1]: # with_helmet or without_helmet (represents a person on bike)
                riders.append({"box": box, "id": tid})

        # Association: Count riders per bike based on proximity/overlap
        for bike in bikes:
            bx1, by1, bx2, by2 = bike["box"]
            bike_area = (bx2 - bx1) * (by2 - by1)
            
            for rider in riders:
                rx1, ry1, rx2, ry2 = rider["box"]
                
                # Check if rider is roughly above or within bike box
                # Simple logic: Rider center is within bike horizontal range 
                # and Rider bottom is close to Bike top/center
                
                rcx = (rx1 + rx2) / 2
                rcy = (ry1 + ry2) / 2
                
                # Expanded bike box for association (sometimes bike detection is small)
                if bx1 - 20 < rcx < bx2 + 20 and by1 - 100 < rcy < by2 + 50:
                     bike["rider_count"] += 1

        # Check for Triple Riding (> 2 riders)
        for bike in bikes:
             if bike["rider_count"] >= 3:
                 violations.append({
                    "id": int(bike["id"]),
                    "type": "TRIPLE RIDING",
                    "box": bike["box"].astype(int).tolist(),
                    "confidence": 0.8 # Placeholder or derived
                })
        
        return violations
