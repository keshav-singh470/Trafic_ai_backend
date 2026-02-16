from .base import BaseTrafficService
import cv2
import numpy as np
from ultralytics import YOLO
import os

class SeatbeltService(BaseTrafficService):
    def __init__(self, model_path="no_sitbelt.pt", base_model="models/yolo11n.pt"):
        # Initialize base service with the base YOLO model for vehicle/person detection
        super().__init__(model_path=base_model)
        
        self.seatbelt_model = None
        self.model_path = model_path
        
        # specific seatbelt model validation
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            try:
                print(f"Loading specialized seatbelt model from {model_path}...")
                self.seatbelt_model = YOLO(model_path).to(self.device)
            except Exception as e:
                print(f"Failed to load seatbelt model: {e}")
        else:
            print(f"Warning: Seatbelt model {model_path} not found or empty. Service will run in passive mode.")

    def run_detection(self, frame):
        # 1. Detect Vehicles (Cars) and Persons using base model
        # We use the base process_frame which uses conf=0.3. 
        # For seatbelt (interior view), we might need lower conf for persons.
        detections, tracked_dets = self.process_frame(frame)
        violations = []

        if tracked_dets is None:
            return violations

        # Parse base detections
        boxes = tracked_dets.xyxy
        class_ids = tracked_dets.class_id
        tracker_ids = tracked_dets.tracker_id
        
        # Identify Cars (2, 5, 7 - Car, Bus, Truck) and Persons (0)
        vehicle_boxes = []
        person_indices = []
        
        for i, cls_id in enumerate(class_ids):
            if cls_id in [2, 5, 7]: # Vehicles
                vehicle_boxes.append(boxes[i])
            elif cls_id == 0: # Person
                # Filter low-confidence persons (often empty seats/headrests)
                conf = detections.confidence[i]
                if conf < 0.5:
                    print(f"DEBUG: Ignored low-confidence person (conf={conf:.2f})")
                    continue
                person_indices.append(i)

        # 2. Run Seatbelt Model (Detects "Seatbelt" presence)
        seatbelt_boxes = []
        if self.seatbelt_model:
            results = self.seatbelt_model(frame, conf=0.1, verbose=False)[0] # Very low conf to start
            for box in results.boxes:
                # We assume Class 0 is "Seatbelt"
                if int(box.cls[0]) == 0:
                     seatbelt_boxes.append(box.xyxy[0].cpu().numpy())
            
            # DEBUG: Print detections
            if len(seatbelt_boxes) > 0:
                print(f"DEBUG: Found {len(seatbelt_boxes)} Seatbelts in frame.")
            else:
                print("DEBUG: No Seatbelts found in frame.")

        # Logic for Interior Views (Dashcams):
        # If we see Persons but NO Vehicles bounding the frame, we might be INSIDE the vehicle.
        # In this case, we should assume ALL persons are passengers/drivers.
        assume_all_in_vehicle = (len(vehicle_boxes) == 0)

        # 3. Check Persons
        for idx in person_indices:
            p_box = boxes[idx]
            p_id = tracker_ids[idx]
            
            # Check if Person is inside ANY Vehicle
            in_vehicle = assume_all_in_vehicle
            if not in_vehicle:
                for v_box in vehicle_boxes:
                    # Check if person center is inside vehicle box
                    p_center = ((p_box[0] + p_box[2])/2, (p_box[1] + p_box[3])/2)
                    if (v_box[0] < p_center[0] < v_box[2]) and (v_box[1] < p_center[1] < v_box[3]):
                        in_vehicle = True
                        break
            
            if not in_vehicle:
                continue # Ignore pedestrians if we have identified cars
            
            # Check if Person has a Seatbelt (overlap)
            has_seatbelt = False
            for s_box in seatbelt_boxes:
                # Calculate Intersection
                ix1 = max(p_box[0], s_box[0])
                iy1 = max(p_box[1], s_box[1])
                ix2 = min(p_box[2], s_box[2])
                iy2 = min(p_box[3], s_box[3])
                
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                
                if iw * ih > 0: # Any overlap is considered "Has Seatbelt"
                    has_seatbelt = True
                    break
            
            if not has_seatbelt:
                # VIOLATION
                violations.append({
                    "id": int(p_id),
                    "type": "NO SEATBELT",
                    "box": p_box.astype(int).tolist(),
                    "confidence": 1.0 # Logic-based confidence
                })

        return violations
