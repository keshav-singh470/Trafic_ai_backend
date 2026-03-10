"""
services/vehicle_type_service.py
==================================
Fixed vehicle type classifier for Indian traffic.

ROOT CAUSE OF BUG (was in api.py):
  - best.pt (Roboflow) returns STRING class names like "car", "bike"
  - But api.py was doing VEHICLE_TYPE_MAP.get(cls_id) with INTEGER cls_id
  - Result: COCO class 5 (bus) was matching everything even for cars/bikes
  
FIXES IN THIS FILE:
  1. Correct string-based class lookup from best.pt
  2. CLAHE night-vision preprocessing (auto-detect low light)
  3. Confidence threshold + shape-based fallback
  4. Indian vehicle categories: Car, Bike, Bus, Truck, Auto-Rickshaw, Scooter
"""

import cv2
import numpy as np
import torch

MIN_CONF_THRESHOLD = 0.30

LABEL_TO_INDIAN = {
    # ── EXACT best.pt class names (model has exactly these 6 classes) ──
    # {0: 'autorickshaw', 1: 'bus', 2: 'car', 3: 'motorcycle', 4: 'scooter', 5: 'truck'}
    "autorickshaw":  "Auto-Rickshaw",
    "bus":           "Bus",
    "car":           "Car",
    "motorcycle":    "Bike",
    "scooter":       "Scooter",
    "truck":         "Truck",

    # ── Common variations for safety ──
    "auto":               "Auto-Rickshaw",
    "auto rickshaw":      "Auto-Rickshaw",
    "auto-rickshaw":      "Auto-Rickshaw",
    "three-wheeler":      "Auto-Rickshaw",
    "bike":               "Bike",
    "motorbike":          "Bike",
    "bicycle":            "Bicycle",
    "sedan":              "Car",
    "suv":                "Car",
    "van":                "Car",
    "lorry":              "Truck",
    "tempo":              "Truck",
    "pickup":             "Truck",
}

from ultralytics import YOLO


# ── Night vision helpers ──────────────────────────────────────────────────────

def is_low_light(frame: np.ndarray, threshold: int = 55) -> bool:
    """Returns True if frame is dark (night / low light CCTV)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < threshold


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """
    CLAHE enhancement — makes night frames much clearer for YOLO.
    Works on LAB color space (only L channel enhanced).
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ── Shape-based fallback ──────────────────────────────────────────────────────

def classify_by_shape(bbox_w: int, bbox_h: int) -> str:
    """
    Fallback classification using bounding box aspect ratio.
    Used ONLY when model confidence < MIN_CONF_THRESHOLD.

    CCTV top-down typical ratios:
      Bike/Scooter : tall, narrow  → ratio < 0.75
      Auto-Rickshaw: squarish      → ratio 0.75–1.3
      Car          : wide rect     → ratio 1.3–2.4
      Bus/Truck    : very wide     → ratio > 2.4
    """
    if bbox_w == 0 or bbox_h == 0:
        return "Vehicle"
    ratio = bbox_w / max(bbox_h, 1)

    if ratio < 0.75:
        return "Bike"
    elif ratio < 1.3:
        return "Auto-Rickshaw"
    elif ratio < 2.4:
        return "Car"
    else:
        return "Bus"


# ── Main Service Class ────────────────────────────────────────────────────────

class VehicleTypeService:
    def __init__(self, model_path: str = "models/best.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VehicleTypeService] Loading {model_path} on {self.device}...")
        self.model = YOLO(model_path).to(self.device)
        # Store raw class names from model for debugging
        self.class_names = self.model.names  # {0: 'car', 1: 'bike', ...}
        print(f"[VehicleTypeService] Classes: {self.class_names}")

    # ── Core: map raw label → Indian category ────────────────────────────────
    def _map_label(self, raw_label: str) -> str:
        """Maps best.pt raw string label to Indian traffic category."""
        key = raw_label.lower().strip()

        # Direct match
        if key in LABEL_TO_INDIAN:
            return LABEL_TO_INDIAN[key]

        # Partial match (e.g. "motorcycle_2w" → "Bike")
        for known_key, value in LABEL_TO_INDIAN.items():
            if known_key in key or key in known_key:
                return value

        # Return title-cased original if nothing matched
        return raw_label.title()

    # ── Single image: get best detection ─────────────────────────────────────
    def get_best_detection(self, image: np.ndarray) -> dict:
        """
        Run best.pt on a single vehicle crop.
        Returns {"type": str, "confidence": float, "raw_label": str, "method": str}
        """
        if image is None or image.size == 0:
            return {"type": "Vehicle", "confidence": 0.0, "raw_label": "", "method": "empty_image"}

        # Auto night enhancement
        night = is_low_light(image)
        proc = apply_clahe(image) if night else image

        # Ensure minimum size for YOLO
        h, w = proc.shape[:2]
        if h < 64 or w < 64:
            proc = cv2.resize(proc, (max(64, w), max(64, h)))

        try:
            with torch.no_grad():
                results = self.model(proc, verbose=False, conf=MIN_CONF_THRESHOLD)
        except Exception as e:
            print(f"[VehicleTypeService] Inference error: {e}")
            return {"type": "Vehicle", "confidence": 0.0, "raw_label": "", "method": "inference_error"}

        best_conf = 0.0
        best_type = None
        best_raw  = ""

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                # ✅ KEY FIX: Use STRING class name, not integer ID
                raw_label = self.model.names.get(cls_id, "unknown")
                mapped    = self._map_label(raw_label)

                if conf > best_conf:
                    best_conf = conf
                    best_type = mapped
                    best_raw  = raw_label

        # Low confidence → shape fallback
        if best_conf < MIN_CONF_THRESHOLD or best_type is None:
            fallback = classify_by_shape(w, h)
            return {
                "type":       fallback,
                "confidence": best_conf,
                "raw_label":  best_raw,
                "method":     "shape_fallback",
                "night":      night,
            }

        return {
            "type":       best_type,
            "confidence": round(best_conf, 3),
            "raw_label":  best_raw,
            "method":     "model",
            "night":      night,
        }

    # ── Full frame: detect all vehicles ──────────────────────────────────────
    def detect_vehicle_type(self, frame: np.ndarray) -> list:
        """
        Run detection on full frame.
        Returns list of dicts with type, confidence, bbox, method.
        Used by /api/detect-vehicle-type test endpoint.
        """
        if frame is None or frame.size == 0:
            return []

        night = is_low_light(frame)
        proc  = apply_clahe(frame) if night else frame

        try:
            with torch.no_grad():
                results = self.model(proc, verbose=False, conf=MIN_CONF_THRESHOLD)
        except Exception as e:
            print(f"[VehicleTypeService] Frame inference error: {e}")
            return []

        detections = []
        for r in results:
            for box in r.boxes:
                conf    = float(box.conf[0])
                cls_id  = int(box.cls[0])
                # ✅ KEY FIX: String-based label lookup
                raw_label = self.model.names.get(cls_id, "unknown")
                mapped    = self._map_label(raw_label)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bw = x2 - x1
                bh = y2 - y1

                method = "model"
                if conf < MIN_CONF_THRESHOLD:
                    mapped = classify_by_shape(bw, bh)
                    method = "shape_fallback"

                detections.append({
                    "type":       mapped,
                    "raw_label":  raw_label,
                    "confidence": round(conf, 3),
                    "bbox":       [x1, y1, x2, y2],
                    "method":     method,
                    "night":      night,
                })

        return detections