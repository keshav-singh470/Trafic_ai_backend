from .base import BaseTrafficService
import cv2
import numpy as np

class HelmetService(BaseTrafficService):
    def __init__(self, model_path="models/helmet_triple_model.pt"):
        # Use user-provided trained model
        super().__init__(model_path=model_path)
        print(f"HelmetService initialized with {model_path}")
        # Model classes: {0: 'with_helmet', 1: 'without_helmet', ...}

    def run_detection(self, frame):
        # Base service handles inference and tracking
        detections, tracked_dets = self.process_frame(frame)
        violations = []

        if tracked_dets is None:
            return violations

        # supervision Detections object
        boxes = tracked_dets.xyxy
        class_ids = tracked_dets.class_id
        tracker_ids = tracked_dets.tracker_id
        confidences = tracked_dets.confidence
        
        if tracker_ids is None:
            return violations

        bikes = []
        no_helmets = []

        # Separate detections
        for box, cls_id, track_id, conf in zip(boxes, class_ids, tracker_ids, confidences):
            if cls_id == 3: # Motorcyclist
                bikes.append(box)
            elif cls_id == 1: # Without Helmet
                no_helmets.append({
                    "id": int(track_id),
                    "box": box,
                    "confidence": float(conf) if conf is not None else 0.0
                })

        # Association: Only flag No Helmet if on a bike
        for person in no_helmets:
            px1, py1, px2, py2 = person["box"]
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2
            
            is_riding = False
            for bbox in bikes:
                bx1, by1, bx2, by2 = bbox
                
                # Check if person is roughly above or detection overlaps bike
                # Logic: Person center X within expanded bike width AND 
                # Person center Y is reasonable relative to bike (not too far up/down)
                if (bx1 - 30 < pcx < bx2 + 30) and (by1 - 100 < pcy < by2 + 50):
                    is_riding = True
                    break
            
            if is_riding:
                violations.append({
                    "id": person["id"],
                    "type": "NO HELMET",
                    "box": person["box"].astype(int).tolist(),
                    "confidence": person["confidence"]
                })
        
        return violations
