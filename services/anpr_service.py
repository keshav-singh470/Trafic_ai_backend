import cv2
import re
import os
import threading
from datetime import datetime
from collections import Counter
from ultralytics import YOLO
import easyocr
import numpy as np
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
MIN_VOTES_NEEDED = 10    # Focus on accuracy: collect many frames for stable result
SPATIAL_IOU_THRESH = 0.35  # overlap ratio to consider "same vehicle"
TIGHT_PAD_PX = 8         # Aggressive tight padding to remove background
DEBUG_CROP_DIR = "outputs/debug_crops"

INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CT", "MH", "DL", "GA", "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LD", "MP", "MN", "ML", "MZ", "NL", "OD", "PY", "PB", "RJ", "SK", "TN", "TG", "TR", "UP", "UK", "WB", "TS", "BH"
}


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

    def position_aware_correct(self, text):
        """
        Corrects confusing characters based on Indian plate structure:
        XX 00 XX 0000 or XX 00 X 0000 or 22 BH 1234 AA
        Character Confusion Map: O↔0, I↔1, B↔8, S↔5, Z↔2, M↔H, G↔6, A↔4
        """
        if not text: return ""
        
        # Mapping dictionaries
        to_L = {'0':'O','1':'I','5':'S','8':'B','6':'G','2':'Z', '4':'A', 'D':'0', 'Q':'0'} 
        to_D = {'O':'0','I':'1','L':'1','S':'5','B':'8','G':'6','Z':'2','A':'4', 'H': 'H', 'M': 'M'}
        
        t = list(text)
        length = len(t)

        # 1. State Code (First 2 chars): ALWAYS LETTERS
        for i in range(min(2, length)):
            if t[i] in to_L: t[i] = to_L[t[i]]

        if length >= 4:
            # 2. District/Serial (Next 1-2 chars): ALWAYS DIGITS
            # Usually XX 00 or XX 0
            # We'll just map indices 2 and 3 as digits if they exist
            for i in range(2, min(4, length)):
                if t[i] in to_D: t[i] = to_D[t[i]]
            
            # 3. Last 4 characters (Identifier): ALWAYS DIGITS
            for i in range(max(4, length-4), length):
                if t[i] in to_D: t[i] = to_D[t[i]]
                
            # 4. Middle characters (Category): ALWAYS LETTERS
            # For XX 00 XX 0000, indices 4 and 5 are letters
            for i in range(4, length-4):
                if t[i] in to_L: t[i] = to_L[t[i]]
        
        # 5. Handle M <-> H swap (Special request)
        # Note: Position aware doesn't swap M/H because both are letters.
        # But for robustness, we can try to validate if it's a known state or common mistake.
        # However, simple mapping isn't enough as both are valid.
        
        return "".join(t)

    def validate_indian_plate(self, text, mode='strict'):
        """
        Two-Level Validation:
        'strict' -> Indian Regex + State Code
        'soft'   -> Alphanumeric + Length (7-11)
        """
        if not text: return False
        
        # Normalize: Remove spaces, upper case
        text = text.replace(" ", "").upper()
        
        # ── SOFT VALIDATION ──────────────────────────────────────────
        # Just needs to be alphanumeric and reasonable length
        if mode == 'soft':
            return len(text) in range(7, 12) and text.isalnum()
            
        # ── STRICT VALIDATION ────────────────────────────────────────
        # Must match Indian plate format precisely.
        # Regex: ^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$
        state_code = text[:2]
        if state_code not in INDIAN_STATE_CODES:
            # Check for BH format: 22BH1234AA
            if not (len(text) == 10 and text[2:4] == "BH"):
                return False

        pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'
        bh_pattern = r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$'
        
        return bool(re.match(pattern, text) or re.match(bh_pattern, text))

    def extract_indian_plate(self, text):
        # We look for 9 or 10 character patterns
        std_9  = r'[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}'
        std_10 = r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}'
        bh_10  = r'[0-9]{2}BH[0-9]{4}[A-Z]{1,2}'
        
        for p in [std_10, std_9, bh_10]:
            m = re.search(p, text)
            if m: return m.group(0)
        return None

    def calculate_score(self, original, corrected, conf):
        """
        Scores a candidate with a strong pattern preference.
        """
        score = 0
        if not corrected: return -100
        
        # 1. HARD RULE: Must be exactly 9 or 10 chars
        length = len(corrected)
        if length not in [9, 10]:
            return -50 # Heavy penalty for wrong length
        
        # 2. HARD RULE: Regex Match
        is_valid = self.validate_indian_plate(corrected)
        if is_valid: 
            score += 200 # Major boost for valid format
        else:
            return -100 # Reject if not regex valid per requested hard rule

        # 3. PREFERRED PATTERN: Starts with 2 letters, ends with 4 digits
        if re.match(r'^[A-Z]{2}.*[0-9]{4}$', corrected):
            score += 50
        
        # 4. Confidence factor
        score += int(conf * 10)
        
        # 5. Translation penalty
        diffs = sum(1 for a, b in zip(original, corrected) if a != b)
        score -= (diffs * 5)
        
        return score

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
        Tiered Voting Logic:
        1. Attempt to find a "Strictly Valid" majority.
        2. Fallback to "Softly Valid" if no strict results exist.
        3. Returns None if neither level finds a stable candidate.
        """
        if not readings: return None
        
        # Normalize all readings for consistency
        readings = [r.replace(" ", "").upper() for r in readings if r]
        if not readings: return None

        # Level 1: Strict Validation (The "Secure" winner)
        stricts = [r for r in readings if self.validate_indian_plate(r, mode='strict')]
        if stricts:
            winner, count = Counter(stricts).most_common(1)[0]
            # Stability check for strict: at least 30% of total readings or 2+ instances
            if count >= 2 or (len(stricts) > 5 and count >= len(stricts) * 0.3):
                self.log_debug(f"VOTE: Strict Winner '{winner}' ({count}/{len(stricts)} strict votes)")
                return winner

        # Level 2: Soft Fallback (The "Recall" winner)
        softs = [r for r in readings if self.validate_indian_plate(r, mode='soft')]
        if softs:
            winner, count = Counter(softs).most_common(1)[0]
            # Stability check for soft: require more consistency (repetition check)
            # Must have at least 3 votes to trust a soft match
            if count >= 3:
                self.log_debug(f"VOTE: Soft Fallback Winner '{winner}' ({count}/{len(softs)} soft votes)")
                return winner
            else:
                self.log_debug(f"VOTE: Soft candidate '{winner}' ({count} votes) rejected as unstable.")

        # If we have reached MIN_VOTES_NEEDED but no winner emerged
        if len(readings) >= MIN_VOTES_NEEDED:
            self.log_debug(f"VOTE: No stable candidate found after {len(readings)} frames.")
            return "Unreadable"
            
        return None

    # ── OCR ───────────────────────────────────────────────────
    def preprocess_for_ocr(self, img, mode='standard'):
        """
        DETERMINISTIC SINGLE-PASS PREPROCESSING:
        Grayscale -> Resize (3x) -> CLAHE -> Light Sharpen
        """
        if img is None or img.size == 0: return None
        
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Resize (3x for OCR clarity)
        h, w = gray.shape[:2]
        resized = cv2.resize(gray, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
        
        # 3. CLAHE (Better than global histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        
        # 4. Light Sharpening (Laplacian-like)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        processed = cv2.filter2D(enhanced, -1, kernel)
            
        return processed

    def _ocr_pass(self, processed_img):
        """Helper for a single OCR pass with confidence score."""
        with OCR_LOCK:
            results = self.ocr.readtext(
                processed_img, 
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
        if not results: return "", 0.0
        
        full_text = "".join([r[1] for r in results]).replace(" ", "")
        avg_conf = sum([r[2] for r in results]) / len(results)
        return full_text, avg_conf

    def process_plate(self, plate_img, modes=None):
        """
        DETERMINISTIC SINGLE OCR CALL:
        Takes tight crop, applies robust enhancement, runs OCR, corrects confusions.
        """
        if plate_img is None or plate_img.size == 0:
            return ""

        # 1. Enhancement
        processed = self.preprocess_for_ocr(plate_img)
        if processed is None: return ""

        # 2. RUN OCR (Single determininstic call)
        raw_text, conf = self._ocr_pass(processed)
        if not raw_text: return ""

        # 3. Position-Aware Correlation Fixes (O↔0, I↔1, etc.)
        corrected = self.position_aware_correct(raw_text)
        
        # 4. Extract based on Regex
        extracted = self.extract_indian_plate(corrected)
        final_text = extracted if extracted else corrected
        
        # 5. M <-> H Dynamic Check (User request)
        # If text has H but fails regex, try M
        if not self.validate_indian_plate(final_text):
            if "H" in final_text:
                test_m = final_text.replace("H", "M")
                if self.validate_indian_plate(test_m):
                    final_text = test_m
            elif "M" in final_text:
                test_h = final_text.replace("M", "H")
                if self.validate_indian_plate(test_h):
                    final_text = test_h

        # 6. Scoring for Confidence Reporting (only report if it might be valid)
        score = self.calculate_score(raw_text, final_text, conf)
        
        # We only return valid results to the voting buffer
        if score >= 200 or self.validate_indian_plate(final_text):
            self.log_debug(f"OCR: '{raw_text}' -> '{final_text}' (Score: {score})")
            return final_text
            
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

            # ── 1. Relaxed Quality Filter & Tight Crop ────────────────────────────
            # Confidence Threshold 0.25 (Lowered for better recall)
            if float(box.conf[0]) < 0.25: continue

            # TIGHT PADDING (8px)
            px1, py1 = max(0, x1 - TIGHT_PAD_PX), max(0, y1 - TIGHT_PAD_PX)
            px2, py2 = min(w, x2 + TIGHT_PAD_PX), min(h, y2 + TIGHT_PAD_PX)
            
            plate_img = frame[py1:py2, px1:px2]
            
            # LOOSENED QUALITY CHECK (Relaxed thresholds)
            ch, cw = plate_img.shape[:2]
            if cw < 50 or ch < 15: # Relaxed from 70x20
                # self.log_debug(f"Frame {frame_count} skipped: Tiny crop ({cw}x{ch})")
                continue
            
            gray_crop = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
            if variance < 40: # Relaxed from 80
                self.log_debug(f"Frame {frame_count} skipped: Too Blurry (Var: {variance:.1f})")
                continue
            
            avg_brightness = np.mean(gray_crop)
            if avg_brightness < 25 or avg_brightness > 245: # Expanded from 35-225
                self.log_debug(f"Frame {frame_count} skipped: Bad exposure ({avg_brightness:.1f})")
                continue

            # Save debug crop
            if not os.path.exists(DEBUG_CROP_DIR): os.makedirs(DEBUG_CROP_DIR)
            crop_fn = f"{DEBUG_CROP_DIR}/z{zone_idx}_f{frame_count}.jpg"
            cv2.imwrite(crop_fn, plate_img)

            # ── 2. Deterministic OCR ────────────────────────────────────
            plate_text = self.process_plate(plate_img)
            
            if plate_text:
                # Log what we found for debugging recall/precision
                is_strict = self.validate_indian_plate(plate_text, mode='strict')
                is_soft = self.validate_indian_plate(plate_text, mode='soft')
                self.log_debug(f"Frame {frame_count}: OCR result='{plate_text}' (Strict: {is_strict}, Soft: {is_soft})")
                
                zone["readings"].append(plate_text)
                if len(zone["readings"]) > 20:
                    zone["readings"] = zone["readings"][-20:]

            # ── 3. Tiered Voting Check ──────────────────────────────────
            if len(zone["readings"]) < MIN_VOTES_NEEDED:
                continue

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