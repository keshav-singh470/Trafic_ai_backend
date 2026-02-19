import cv2
import re
import os
import threading
from datetime import datetime
from collections import Counter
from ultralytics import YOLO
import easyocr
from difflib import SequenceMatcher
from .base import BaseTrafficService

OCR_LOCK = threading.Lock()

# ─────────────────────────────────────────────
#  HOW THIS WORKS (read before editing):
#
#  Problem: Same plate reads differently every frame
#  e.g. DL1LY2046 → LJ120461, DU11Y2046, JL011204
#
#  Solution:
#  1. SPATIAL DEDUP  – track plate by screen position (bbox),
#     not by text. If a new detection overlaps >40% with an
#     already-seen region → same vehicle → skip.
#  2. VOTING BUFFER  – collect N readings per spatial zone,
#     pick the most common valid plate text.
#  3. OCR CORRECTION – fix common I↔1, O↔0 errors by position.
#  4. CONFIRMED LIST – once a plate is confirmed+reported,
#     any future detection in the same zone is skipped for
#     COOLDOWN_FRAMES frames.
# ─────────────────────────────────────────────

COOLDOWN_FRAMES  = 600   # ~20s at 30fps – ignore same zone after reporting
MIN_VOTES_NEEDED = 3     # how many OCR reads before we accept a plate
SPATIAL_IOU_THRESH = 0.35  # overlap ratio to consider "same vehicle"


class ANPRService(BaseTrafficService):
    _instance     = None
    _initialized  = False

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

        print("Loading EasyOCR...")
        self.ocr = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR loaded.")

        self.blacklist = self._load_blacklist()

        # ── Spatial zone tracker ──────────────────────────────
        # Each entry: {
        #   "box"          : [x1,y1,x2,y2],   ← representative bbox
        #   "readings"     : [text, text, ...], ← OCR votes collected
        #   "reported"     : bool,
        #   "reported_frame": int,
        #   "best_plate"   : str or None        ← confirmed plate text
        # }
        self.zones = []

        ANPRService._initialized = True

    # ── Blacklist ─────────────────────────────────────────────
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
                    self.log_debug(f"Loaded blacklist: {len(loaded)} entries from {path}")
                except Exception as e:
                    self.log_debug(f"Blacklist error: {e}")
        return list(loaded)

    # ── Text helpers ──────────────────────────────────────────
    def clean_plate_text(self, text):
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def fix_common_ocr_errors(self, text):
        """
        Indian plate format: XX 00 XX 0000
        pos 0-1 → letters, pos 2-3 → digits,
        middle 0-3 chars → letters, last 4 → digits
        """
        if len(text) < 8:
            return text

        L = {'0':'O','1':'I','5':'S','8':'B','6':'G','2':'Z'}
        D = {'O':'0','I':'1','L':'1','S':'5','B':'8','G':'6','Z':'2','Q':'0'}

        t = list(text)
        # First 2 must be letters
        for i in [0, 1]:
            if i < len(t) and t[i] in L:
                t[i] = L[t[i]]
        # Pos 2-3 must be digits
        for i in [2, 3]:
            if i < len(t) and t[i] in D:
                t[i] = D[t[i]]
        # Last 4 must be digits
        for i in range(max(0, len(t)-4), len(t)):
            if t[i] in D:
                t[i] = D[t[i]]
        return ''.join(t)

    def validate_indian_plate(self, text):
        if len(text) < 8 or len(text) > 12:
            return False
        std = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{4}$'
        bh  = r'^[0-9]{2}BH[0-9]{1,4}[A-Z]{1,2}[0-9]{1,4}$'
        return bool(re.match(std, text) or re.match(bh, text))

    def extract_indian_plate(self, text):
        std = r'[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}'
        bh  = r'[0-9]{2}BH[0-9]{1,4}[A-Z]{1,2}[0-9]{1,4}'
        m = re.search(std, text)
        if m: return m.group(0)
        m = re.search(bh, text)
        if m: return m.group(0)
        return None

    # ── Spatial helpers ───────────────────────────────────────
    def _iou(self, a, b):
        """Intersection over Union for two [x1,y1,x2,y2] boxes."""
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        aA = (a[2]-a[0]) * (a[3]-a[1])
        bA = (b[2]-b[0]) * (b[3]-b[1])
        denom = aA + bA - inter
        return inter / (denom + 1e-6)

    def _find_zone(self, box):
        """Return index of matching zone, or -1 if new location."""
        for i, z in enumerate(self.zones):
            if self._iou(box, z["box"]) >= SPATIAL_IOU_THRESH:
                return i
        return -1

    def _best_vote(self, readings):
        """
        From a list of OCR strings, return the most-common valid plate.
        Falls back to fuzzy grouping if no single text wins.
        """
        valid = [r for r in readings if self.validate_indian_plate(r)]
        if valid:
            winner, count = Counter(valid).most_common(1)[0]
            if count >= 1:
                return winner

        # Fuzzy group all readings
        groups = {}
        for r in readings:
            placed = False
            for key in groups:
                if SequenceMatcher(None, r, key).ratio() > 0.72:
                    groups[key].append(r)
                    placed = True
                    break
            if not placed:
                groups[r] = [r]

        if not groups:
            return None

        best_group = max(groups.values(), key=len)
        candidate, _ = Counter(best_group).most_common(1)[0]
        return candidate if self.validate_indian_plate(candidate) else None

    # ── OCR ───────────────────────────────────────────────────
    def process_plate(self, plate_img):
        if plate_img is None or plate_img.size == 0:
            return ""
        try:
            resized   = cv2.resize(plate_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            gray      = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced  = clahe.apply(gray)
            denoised  = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

            with OCR_LOCK:
                results = self.ocr.readtext(
                    denoised, detail=0,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )
            return " ".join(results) if results else ""
        except Exception as e:
            self.log_debug(f"OCR error: {e}")
            return ""

    # ── Main detection ────────────────────────────────────────
    def run_detection(self, frame, frame_count):
        results = self.anpr_model(frame, verbose=False)[0]
        to_report = []

        h, w = frame.shape[:2]

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.35:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            det_box = [x1, y1, x2, y2]

            # ── Find or create spatial zone ───────────────
            zone_idx = self._find_zone(det_box)

            if zone_idx == -1:
                # Brand new location – create zone
                self.zones.append({
                    "box"           : det_box,
                    "readings"      : [],
                    "reported"      : False,
                    "reported_frame": -1,
                    "best_plate"    : None
                })
                zone_idx = len(self.zones) - 1

            zone = self.zones[zone_idx]
            # Update zone's representative box to latest detection
            zone["box"] = det_box

            # ── Already reported? Check cooldown ──────────
            if zone["reported"]:
                frames_since = frame_count - zone["reported_frame"]
                if frames_since < COOLDOWN_FRAMES:
                    # Same vehicle still on screen – draw but DON'T re-report
                    self.log_debug(
                        f"Frame {frame_count}: zone {zone_idx} cooldown "
                        f"({frames_since}/{COOLDOWN_FRAMES}) plate={zone['best_plate']}"
                    )
                    continue
                else:
                    # Vehicle left and came back – reset zone
                    self.log_debug(f"Frame {frame_count}: zone {zone_idx} RESET (vehicle reappeared)")
                    zone["readings"]       = []
                    zone["reported"]       = False
                    zone["reported_frame"] = -1
                    zone["best_plate"]     = None

            # ── Collect OCR reading ───────────────────────
            plate_img  = frame[y1:y2, x1:x2]
            raw_text   = self.process_plate(plate_img)
            clean_text = self.clean_plate_text(raw_text)
            clean_text = self.fix_common_ocr_errors(clean_text)
            extracted  = self.extract_indian_plate(clean_text)
            if extracted:
                clean_text = extracted

            if clean_text:
                zone["readings"].append(clean_text)

            self.log_debug(
                f"Frame {frame_count}: zone {zone_idx} reading='{clean_text}' "
                f"total_votes={len(zone['readings'])}"
            )

            # ── Not enough votes yet ──────────────────────
            if len(zone["readings"]) < MIN_VOTES_NEEDED:
                continue

            # ── Vote for best plate ───────────────────────
            best = self._best_vote(zone["readings"])
            if not best:
                continue

            # ── Mark reported ─────────────────────────────
            zone["reported"]       = True
            zone["reported_frame"] = frame_count
            zone["best_plate"]     = best

            alert = best in self.blacklist
            self.log_debug(f"Frame {frame_count}: CONFIRMED plate={best} alert={alert}")

            to_report.append({
                "frame"     : int(frame_count),
                "text"      : best,
                "conf"      : float(conf),
                "valid"     : "Yes",
                "box"       : det_box,
                "is_new"    : True,
                "alert"     : alert,
                "alert_type": "SECURITY ALERT" if alert else None
            })

        return to_report

    def reset(self):
        """Resets the state of the service (tracker and zones)."""
        print("Resetting ANPRService state (tracker and zones)...")
        super().reset()
        self.zones = []

    def log_debug(self, message):
        try:
            with open("anpr_debug.log", "a", encoding="utf-8") as f:
                f.write(f"{datetime.now()} - {message}\n")
        except:
            pass