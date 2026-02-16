import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv

class BaseTrafficService:
    def __init__(self, model_path="yolo11n.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Base Service on {self.device} with model {model_path}...")
        self.model = YOLO(model_path).to(self.device)
        self.tracker = sv.ByteTrack()

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
        
        # We only track specific classes if needed, or track everything
        # For base, we'll return all detections and updated tracker results
        tracked_detections = self.tracker.update_with_detections(detections)
        return detections, tracked_detections
