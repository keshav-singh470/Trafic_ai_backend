import cv2
import re
import os
import threading
from datetime import datetime
from ultralytics import YOLO
import easyocr  # ✅ Change this
from difflib import SequenceMatcher
from .base import BaseTrafficService

# 🔒 Global lock
OCR_LOCK = threading.Lock()

class ANPRService(BaseTrafficService):
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, base_model="yolo11n.pt", anpr_model="anpr_plat.pt"):
        if ANPRService._initialized:
            return
            
        super().__init__(model_path=base_model)

        print("Loading ANPR YOLO model...")
        self.anpr_model = YOLO(anpr_model).to(self.device)

        print("Loading EasyOCR model (ONCE)...")
        # ✅ EasyOCR - much more stable on Windows
        self.ocr = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR loaded.")

        self.seen_plates = set()
        self.blacklist = self._load_blacklist()
        
        ANPRService._initialized = True

    def _load_blacklist(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(base_dir)

        paths = [
            os.path.join(root_dir, "blacklist.txt"),
            os.path.join(base_dir, "blacklist.txt"),
            r"C:\Users\keshav singh\Desktop\Project\anpr-system-main\anpr-system-main\blacklist.txt"
        ]

        loaded = set()
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding='utf-8-sig') as f:
                        for line in f:
                            clean = re.sub(r'[^A-Z0-9]', '', line.upper())
                            if clean:
                                loaded.add(clean)
                    self.log_debug(f"Loaded blacklist from {path}: {len(loaded)} entries")
                except Exception as e:
                    self.log_debug(f"Error loading {path}: {e}")

        final_list = list(loaded)
        self.log_debug(f"Final Blacklist ({len(final_list)}): {sorted(final_list)}")
        return final_list

    def clean_plate_text(self, text):
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def validate_indian_plate(self, text):
        std_pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{4}$'
        bh_pattern = r'^[0-9]{2}BH[0-9]{1,4}[A-Z]{1,2}[0-9]{1,4}$'
        if len(text) < 8 or len(text) > 12:
            return False
        return bool(re.match(std_pattern, text) or re.match(bh_pattern, text))

    def is_similar(self, text, threshold=0.8):
        for seen_text in self.seen_plates:
            if SequenceMatcher(None, text, seen_text).ratio() > threshold:
                return True, seen_text
        return False, None

    def extract_indian_plate(self, text):
        std_pattern = r'[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}'
        bh_pattern = r'[0-9]{2}BH[0-9]{1,4}[A-Z]{1,2}[0-9]{1,4}'

        std_match = re.search(std_pattern, text)
        if std_match:
            return std_match.group(0)

        bh_match = re.search(bh_pattern, text)
        if bh_match:
            return bh_match.group(0)

        return None

    def process_plate(self, plate_img):
        """✅ Updated for EasyOCR"""
        if plate_img is None or plate_img.size == 0:
            return ""

        try:
            # Preprocessing
            plate_processed = cv2.resize(plate_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(plate_processed, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # ✅ EasyOCR works with grayscale or BGR
            with OCR_LOCK:
                results = self.ocr.readtext(enhanced, detail=0)  # detail=0 returns only text
                
            if results:
                detected_text = " ".join(results)
                return detected_text

        except Exception as e:
            self.log_debug(f"OCR error: {e}")
            import traceback
            self.log_debug(traceback.format_exc())

        return ""

    def log_debug(self, message):
        try:
            with open("anpr_debug.log", "a", encoding="utf-8") as f:
                f.write(f"{datetime.now()} - {message}\n")
        except:
            pass

    def run_detection(self, frame, frame_count):
        results = self.anpr_model(frame, verbose=False)[0]
        detected_plates = []

        self.log_debug(f"Frame {frame_count}: Detected {len(results.boxes)} potential plates.")

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])

            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            plate_img = frame[y1:y2, x1:x2]

            if plate_img is None or plate_img.size == 0:
                continue

            raw_text = self.process_plate(plate_img)
            clean_text = self.clean_plate_text(raw_text)

            self.log_debug(f"Frame {frame_count}: Box [{x1},{y1},{x2},{y2}] Conf {conf:.2f} Raw '{raw_text}' Clean '{clean_text}'")

            extracted = self.extract_indian_plate(clean_text)
            if extracted:
                clean_text = extracted

            if not clean_text:
                continue

            is_new = False
            if clean_text in self.seen_plates:
                is_new = False
            else:
                similar, original = self.is_similar(clean_text)
                if similar:
                    is_new = False
                    clean_text = original
                else:
                    is_new = True

            is_valid = self.validate_indian_plate(clean_text)

            if is_valid:
                if is_new:
                    self.seen_plates.add(clean_text)

                alert = clean_text in self.blacklist

                detected_plates.append({
                    "frame": int(frame_count),
                    "text": str(clean_text),
                    "conf": float(conf),
                    "valid": "Yes",
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "is_new": is_new,
                    "alert": alert,
                    "alert_type": "SECURITY ALERT" if alert else None
                })

        return detected_plates