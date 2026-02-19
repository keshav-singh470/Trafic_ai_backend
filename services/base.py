import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
import time

class BaseTrafficService:
    # --- Configuration ---
    STABILITY_FRAMES = 3
    DUPLICATE_PLATE_TIME_WINDOW_SECONDS = 60

    def __init__(self, model_path="yolo11n.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Base Service on {self.device} with model {model_path}...")
        self.model = YOLO(model_path).to(self.device)
        self.tracker = sv.ByteTrack()
        
        # --- De-duplication and Stability State ---
        self.alerted_track_ids = set()
        self.recent_plates = {}    # plate -> timestamp
        self.track_stability = {}  # track_id -> count

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB-xA) * max(0, yB-yA)
        areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

        return inter / (areaA + areaB - inter + 1e-6)

    def process_frame(self, frame):
        """Performs detection and tracking on a single frame."""
        results = self.model(frame, conf=0.3, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        tracked_detections = self.tracker.update_with_detections(detections)
        return detections, tracked_detections

    def reset(self):
        """Resets the state of the service (e.g., tracker, alerts)."""
        print(f"Resetting {self.__class__.__name__} state...")
        self.tracker = sv.ByteTrack()
        self.alerted_track_ids.clear()
        self.recent_plates.clear()
        self.track_stability.clear()

    def update_stability(self, current_ids):
        """Update stability counters for currently tracked IDs."""
        new_stability = {}
        for tid in current_ids:
            new_stability[tid] = self.track_stability.get(tid, 0) + 1
        self.track_stability = new_stability

    def should_alert(self, track_id, plate_text=None):
        """
        Combined logic for alert triggering.
        Returns True ONLY if all criteria are met.
        """
        # 1. Track ID de-duplication
        if track_id in self.alerted_track_ids:
            print(f"   [SKIP] Track {track_id} already alerted.")
            return False

        # 2. Stability check
        stability = self.track_stability.get(track_id, 0)
        if stability < self.STABILITY_FRAMES:
            print(f"   [SKIP] Track {track_id} stability insufficient: {stability}/{self.STABILITY_FRAMES}")
            return False

        # 3. Plate de-duplication (time window)
        if plate_text and plate_text != "N/A":
            last_time = self.recent_plates.get(plate_text, 0)
            elapsed = time.time() - last_time
            if elapsed < self.DUPLICATE_PLATE_TIME_WINDOW_SECONDS:
                print(f"   [SKIP] Plate {plate_text} seen {elapsed:.1f}s ago (window: {self.DUPLICATE_PLATE_TIME_WINDOW_SECONDS}s).")
                return False

        return True

    def mark_alerted(self, track_id, plate_text=None):
        """Record that an alert has been sent for this track/plate."""
        self.alerted_track_ids.add(track_id)
        if plate_text and plate_text != "N/A":
            self.recent_plates[plate_text] = time.time()
