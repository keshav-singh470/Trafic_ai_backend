from .base import BaseTrafficService
import cv2
import numpy as np
from collections import deque

class StalledService(BaseTrafficService):
    def __init__(self, model_path="yolo11n.pt", frame_threshold=60, move_threshold=10):
        # frame_threshold: Number of frames a vehicle must be stationary to be considered stalled
        # move_threshold: Maximum pixel movement allowed to still be considered "stationary"
        super().__init__(model_path=model_path)
        self.frame_threshold = frame_threshold
        self.move_threshold = move_threshold
        self.tracks = {} # {track_id: {"history": deque, "stalled": bool}}

    def run_detection(self, frame):
        detections, tracked_dets = self.process_frame(frame)
        violations = []

        if tracked_dets is None or tracked_dets.tracker_id is None:
            return violations

        # supervision Detections object
        boxes = tracked_dets.xyxy
        class_ids = tracked_dets.class_id
        tracker_ids = tracked_dets.tracker_id
        
        # Target classes: Car(2), Motorcycle(3), Bus(5), Truck(7)
        target_classes = [2, 3, 5, 7]

        for box, cls_id, track_id in zip(boxes, class_ids, tracker_ids):
            if cls_id not in target_classes:
                continue

            tid = int(track_id)
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

            if tid not in self.tracks:
                self.tracks[tid] = {
                    "history": deque(maxlen=self.frame_threshold + 10),
                    "stalled": False,
                    "reported": False
                }
            
            track_data = self.tracks[tid]
            track_data["history"].append(center)

            # Check for stalled condition
            if len(track_data["history"]) >= self.frame_threshold:
                # Get position from 'frame_threshold' frames ago
                past_pos = track_data["history"][0] # Oldest in deque (maxlen is close to threshold)
                current_pos = track_data["history"][-1]
                
                # Calculate movement distance
                dist = np.linalg.norm(np.array(current_pos) - np.array(past_pos))
                
                if dist < self.move_threshold:
                    track_data["stalled"] = True
                    if not track_data["reported"]:
                         violations.append({
                            "id": tid,
                            "type": "STALLED VEHICLE",
                            "box": box.astype(int).tolist(),
                            "confidence": 1.0
                        })
                         track_data["reported"] = True # Report once
                else:
                    # Vehicle is moving, reset stalled status if it was stalled (optional, or just keep reported)
                    # For now, if it moves again, we don't clear 'stalled' but we stop reporting it.
                    track_data["stalled"] = False

        return violations
