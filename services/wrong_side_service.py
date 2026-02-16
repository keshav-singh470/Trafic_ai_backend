import cv2
import numpy as np
from collections import deque
from shapely.geometry import Polygon, Point
from .base import BaseTrafficService

class WrongSideService(BaseTrafficService):
    def __init__(self, base_model="yolo11n.pt", allowed_direction="entering", threshold=100, zones=None):
        super().__init__(model_path=base_model)
        self.tracks = {}
        self.allowed_direction = allowed_direction
        self.threshold = threshold
        # Zones: List of dictionaries {"polygon": Polygon([(x,y),...]), "forbidden_classes": [0, 1]}
        self.zones = zones if zones else []

    def run_detection(self, frame):
        _, tracked = self.process_frame(frame)
        violations = []
        
        # If no tracked objects, return empty
        if tracked is None or not hasattr(tracked, 'tracker_id') or tracked.tracker_id is None:
             return violations

        # supervision Detections object has numpy arrays directly
        boxes = tracked.xyxy
        ids = tracked.tracker_id if tracked.tracker_id is not None else []
        classes = tracked.class_id if tracked.class_id is not None else []

        for box, tid, cls in zip(boxes, ids, classes):
            if tid not in self.tracks:
                self.tracks[tid] = {
                    "pos_hist": deque(maxlen=30),
                    "wrong_side": False,
                    "reported": False,
                    "lane_violation": False
                }
            
            tr = self.tracks[tid]
            
            # 1. ZONE/LANE LOGIC (Static Position)
            centroid = Point((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            for zone in self.zones:
                poly = zone.get("polygon")
                forbidden = zone.get("forbidden_classes", [])
                
                if poly and poly.contains(centroid):
                    if cls in forbidden:
                        tr["lane_violation"] = True
                        if not tr["reported"]:
                            violations.append({
                                "id": int(tid),
                                "type": "WRONG LANE", # Distinct from WRONG SIDE
                                "box": box.astype(int).tolist(),
                                "zone_info": "Restricted Lane"
                            })
                            tr["reported"] = True # Report once

            # 2. WRONG SIDE LOGIC (Movement Vector)
            y_center = (box[1] + box[3]) / 2
            tr["pos_hist"].append(y_center)

            if len(tr["pos_hist"]) >= 15:
                start_y = tr["pos_hist"][0]
                end_y = tr["pos_hist"][-1]
                movement = end_y - start_y

                if abs(movement) > self.threshold:
                    detected_dir = "entering" if movement > 0 else "exiting"
                    if detected_dir != self.allowed_direction:
                        tr["wrong_side"] = True
                        if not tr["reported"]:
                            violations.append({
                                "id": int(tid),
                                "type": "WRONG SIDE",
                                "box": box.astype(int).tolist()
                            })
                            tr["reported"] = True
        
        return violations
