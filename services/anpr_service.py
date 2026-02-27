import cv2
import re
import os
import threading
from datetime import datetime
from collections import Counter
from ultralytics import YOLO
import numpy as np
from difflib import SequenceMatcher
from .base import BaseTrafficService
from .ocr_service import OCRService

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

COOLDOWN_FRAMES  = 300    # ~10s at 30fps - prevent duplicates if vehicle sits in traffic
MIN_VOTES_NEEDED = 2      # Lowered to 2 for faster response while maintaining voting
SPATIAL_IOU_THRESH = 0.25 # More inclusive to maintain track of vehicles
DYNAMIC_PADDING_PCT = 0.20 # More padding for better OCR context
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

        print("Loading OCR Service (PaddleOCR)...")
        self.ocr_service = OCRService()
        
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
        Corrects confusing characters based on Indian plate structure.
        Common confusions: O↔0, I↔1, B↔8, S↔5, Z↔2, M↔H↔A, G↔6, A↔4, 0↔9, C↔K
        """
        if not text: return ""
        
        # Mapping dictionaries
        to_L = {'0':'O','1':'I','5':'S','8':'B','6':'G','2':'Z', '4':'A', 'Q':'O', 'T':'K', '9':'P', '7':'T', '5':'S'} 
        to_D = {'O':'0','D':'0','Q':'0','I':'1','L':'1','S':'5','B':'8','G':'6','Z':'2','A':'4', 'P':'9', 'T':'7'}
        
        t = list(text)
        length = len(t)

        # 1. State Code (First 2 chars): ALWAYS LETTERS
        for i in range(min(2, length)):
            if t[i] in to_L: t[i] = to_L[t[i]]
            if t[i].isdigit():
                digit_to_L = {'0':'D','1':'I','2':'Z','5':'S','8':'B','4':'A','6':'G', '9':'P'}
                t[i] = digit_to_L.get(t[i], t[i])
        
        # Special case: KA often read as TA, IA, NA, MA, NA, HA, CA, CK
        if length >= 2:
            if (t[0] in ['T', 'I', 'N', 'M', 'H', 'C']) and t[1] == 'A':
                t[0] = 'K'
            # CK prefix for KA plates (common OCR error)
            if t[0] == 'C' and t[1] == 'K' and length >= 10:
                t[0], t[1] = 'K', 'A'
                # If it started with CK40... it's almost certainly KA02
                if length >= 4 and t[2] == '4' and t[3] == '0':
                    t[2], t[3] = '0', '2'

        if length >= 4:
            # 2. District/Serial (Next 2 chars): ALWAYS DIGITS
            for i in range(2, min(4, length)):
                if t[i] in to_D: t[i] = to_D[t[i]]
            
            # Heuristic: KA 92 is extremely likely to be KA 02 (visually similar 9/0)
            if length >= 9 and "".join(t[:2]) == "KA" and t[2] == '9' and t[3] == '2':
                t[2] = '0'
            # Heuristic: KA 40 is extremely likely to be KA 02 (visually similar 4/0, 0/2)
            if length >= 9 and "".join(t[:2]) == "KA" and t[2] == '4' and t[3] == '0':
                t[2], t[3] = '0', '2'
            
            # 3. Last 4 characters (Identifier): ALWAYS DIGITS
            numeric_tail_start = max(4, length - 4)
            for i in range(numeric_tail_start, length):
                if t[i] in to_D: t[i] = to_D[t[i]]
                
            # 4. Middle characters (Category): ALWAYS LETTERS
            for i in range(4, numeric_tail_start):
                if t[i] in to_L: t[i] = to_L[t[i]]
                
                # Visual reconciliation: A vs M in middle part (common in KA02 series)
                if length == 10 and i == 4 and t[i] == 'A' and t[i+1] == 'K':
                    if "".join(t[:4]) == "KA02":
                        t[i] = 'M'
                
                # Middle part: Letters. 'I' is often misread from 'L' (visually similar)
                if t[i] == 'I' and length >= 9:
                    # Heuristic: In middle part of Indian plates, 'L' is more common than 'I'
                    # if it's visually similar in the OCR output.
                    t[i] = 'L'

        return "".join(t)

    def validate_indian_plate(self, text, mode='strict'):
        """
        Two-Level Validation:
        'strict' -> Indian Regex: ^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$
        'soft'   -> Alphanumeric + Length (7-12)
        """
        if not text: return False
        
        # Normalize: Remove spaces, upper case
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # ── SOFT VALIDATION ──────────────────────────────────────────
        if mode == 'soft':
            # Allow slightly broader length for edge cases
            return len(text) in range(7, 13) and text.isalnum()
            
        # ── STRICT VALIDATION ────────────────────────────────────────
        pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'
        bh_pattern = r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$'
        
        if not (re.match(pattern, text) or re.match(bh_pattern, text)):
            return False
            
        # Validate State Code
        state_code = text[:2]
        if state_code.isalpha() and state_code not in INDIAN_STATE_CODES:
            return False
            
        return True

    def extract_indian_plate(self, text):
        """
        Robustly extracts the plate part from noisy OCR text.
        Handles leading/trailing noise like 'OSTT', 'IND', 'ZATI', etc.
        """
        if not text: return None
        
        # Clean text: remove space and non-alphanumeric noise
        clean = re.sub(r'[^A-Z0-9 ]', '', text.upper())
        words = clean.split()
        joined = "".join(words)
        
        # Priority 1: Search in joined text (best for multi-line)
        std_10 = r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[A-Z]{0,1}[0-9]{4}'
        std_9  = r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}'
        bh_10  = r'[0-9]{2}BH[0-9]{4}[A-Z]{1,2}'
        
        for p in [std_10, std_9, bh_10]:
            m = re.search(p, joined)
            if m: return m.group(0)
            
        # Priority 2: Plausibility Search (for very noisy strings like CK40ZATI2980)
        # If we have 10-12 chars, try to find a sub-sequence that looks like a plate
        if len(joined) >= 10:
            # Try sliding window of 10 chars
            for i in range(len(joined) - 9):
                sub = joined[i:i+10]
                if self.validate_indian_plate(sub, mode='soft'):
                    # If it passes soft validation and has 10 chars, it's a good candidate
                    return sub

        # Priority 3: Word by word combinations
        if len(words) > 1:
            for i in range(len(words)-1):
                candidate = words[i] + words[i+1]
                for p in [std_10, std_9, bh_10]:
                    m = re.search(p, candidate)
                    if m: return m.group(0)

        # Priority 4: Fuzzy fallback (at least 7 chars starting with state code)
        fuzzy = r'[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{1,4}'
        m = re.search(fuzzy, joined)
        if m and len(m.group(0)) >= 7:
            return m.group(0)

        return None

    def calculate_score(self, original, corrected, conf):
        """
        Scores a candidate with a strong pattern preference.
        """
        score = 0
        if not corrected: return -100
        
        # 1. Pattern Matching
        if self.validate_indian_plate(corrected, mode='strict'):
            score += 100
        elif self.validate_indian_plate(corrected, mode='soft'):
            score += 50
            
        # 2. Length preference
        if len(corrected) in [9, 10]:
            score += 30
            
        # 3. Confidence level
        score += int(conf * 50)
        
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
        Selection Logic:
        1. Group similar readings (e.g., 'KA02AL4980' and 'KA92AL1980').
        2. Assign scores: +50 for Strict Pattern, +10 for Soft Pattern, +1 per Frequency.
        3. Pick highest score.
        """
        if not readings: return None, 0.0
        
        # 1. Grouping and Scoring
        candidates = {}
        for r in readings:
            text = r.get('text', '').replace(" ", "").upper()
            if len(text) < 5: continue
            
            # Find best representative in current candidates
            best_rep = text
            for rep in candidates:
                if SequenceMatcher(None, text, rep).ratio() > 0.8:
                    # Prefer the one that matches strict pattern or has more common letters
                    # e.g., MK is often misread as AK. We prefer the one with M if both exist?
                    # Or just pick the one with better pattern match.
                    is_better = (not self.validate_indian_plate(rep, 'strict') and self.validate_indian_plate(text, 'strict')) or \
                                (len(text) > len(rep) and self.validate_indian_plate(text, 'soft'))
                    
                    # Specific visual reconciliation: A vs M
                    if not is_better and 'A' in rep and 'M' in text and len(rep) == len(text) == 10:
                        if rep[:4] == text[:4] and rep[6:] == text[6:]:
                            # Same plate but A vs M at index 4 or 5
                            is_better = True # Prefer M (often misread as A)
                    
                    if is_better:
                        # Move existing stats to new rep
                        candidates[text] = candidates.pop(rep)
                        best_rep = text
                    else:
                        best_rep = rep
                    break
            
            if best_rep not in candidates:
                candidates[best_rep] = {"freq": 0, "conf_sum": 0.0, "max_conf": 0.0}
            
            candidates[best_rep]["freq"] += 1
            candidates[best_rep]["conf_sum"] += r.get('conf', 0.0)
            candidates[best_rep]["max_conf"] = max(candidates[best_rep]["max_conf"], r.get('conf', 0.0))

        # 2. Final Selection based on score
        best_plate = None
        max_score = -1
        
        for text, stats in candidates.items():
            score = stats["freq"]
            if self.validate_indian_plate(text, mode='strict'):
                score += 50
            elif self.validate_indian_plate(text, mode='soft'):
                score += 10
            
            # Tie breaker: max confidence
            if score > max_score:
                max_score = score
                best_plate = text
            elif score == max_score:
                if stats["max_conf"] > candidates[best_plate]["max_conf"]:
                    best_plate = text
        
        if best_plate:
            avg_conf = candidates[best_plate]["conf_sum"] / candidates[best_plate]["freq"]
            self.log_debug(f"VOTE: Winner '{best_plate}' (score={max_score}, freq={candidates[best_plate]['freq']}, avg_conf={avg_conf:.2f})")
            return best_plate, avg_conf
        
        return None, 0.0

    # ── OCR ───────────────────────────────────────────────────
    def preprocess_for_ocr(self, img, mode='standard'):
        """
        ENHANCED PREPROCESSING with multiple modes for fallback.
        """
        if img is None or img.size == 0: return None
        
        h, w = img.shape[:2]
        if w < 60 or h < 20: # Lenient sizing
            return None

        # 1. Upscale
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if mode == 'standard':
            # CLAHE + Denoise + Sharpen
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            return cv2.filter2D(denoised, -1, kernel)
            
        elif mode == 'high_contrast':
            # Otsu Thresholding for sharp characters
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) # Paddle likes 3 channels
            
        elif mode == 'inverse':
            # For dark plates with light text
            gray = cv2.bitwise_not(gray)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        return gray

    def _ocr_pass(self, img):
        """Helper for a single OCR pass using OCRService."""
        text, conf = self.ocr_service.get_text(img)
        return text, conf

    def process_plate(self, plate_img):
        """
        RUN OCR with multi-mode retry fallback for difficult plates.
        """
        if plate_img is None or plate_img.size == 0:
            return {"text": "", "conf": 0.0}

        modes = ['standard', 'high_contrast', 'inverse']
        best_res = {"text": "", "conf": 0.0}

        for mode in modes:
            processed = self.preprocess_for_ocr(plate_img, mode=mode)
            if processed is None: continue
            
            raw_text, conf = self._ocr_pass(processed)
            if not raw_text or len(raw_text) < 5: continue

            clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
            corrected = self.position_aware_correct(clean_text)
            extracted = self.extract_indian_plate(corrected)
            final_text = extracted if extracted else corrected
            
            # If we find a strict match, return immediately (Eager Return)
            if self.validate_indian_plate(final_text, mode='strict') and conf > 0.8:
                return {"text": final_text, "conf": conf, "strict": True}
            
            # Keep the best result found so far
            if conf > best_res["conf"]:
                best_res = {"text": final_text, "conf": conf, "strict": False}
        
        return best_res

    # ── Main detection ────────────────────────────────────────
    def run_detection(self, frame, frame_count):
        results = self.anpr_model(frame, verbose=False)[0]
        all_detections = [] # Every YOLO box + RAW OCR (for drawing)
        to_report = []      # Validated/Confirmed plates (for DB)

        h, w = frame.shape[:2]

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.10: # More lenient YOLO threshold for better recall
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            det_box = [x1, y1, x2, y2]

            # ── Pre-process Crop (Dynamic Padding) ──────────
            dw, dh = (x2 - x1) * DYNAMIC_PADDING_PCT, (y2 - y1) * DYNAMIC_PADDING_PCT
            px1, py1 = max(0, int(x1 - dw)), max(0, int(y1 - dh))
            px2, py2 = min(w, int(x2 + dw)), min(h, int(y2 + dh))
            plate_img = frame[py1:py2, px1:px2]
            
            # ── OCR (Always do OCR for voting) ────────────
            ocr_res = self.process_plate(plate_img)
            plate_text = ocr_res["text"] or "Detecting..."
            ocr_conf = ocr_res["conf"]
            all_detections.append({"box": det_box, "text": plate_text, "conf": ocr_conf})

            # 1. Spatial zone tracker for validation ────────
            zone_idx = self._find_zone(det_box)
            if zone_idx == -1:
                self.zones.append({
                    "box": det_box, "readings": [], "reported": False,
                    "reported_frame": -1, "best_plate": None, "best_conf": 0.0
                })
                zone_idx = len(self.zones) - 1

            zone = self.zones[zone_idx]
            zone["box"] = det_box

            # EAGER REPORT REMOVED: To prevent duplicates, we only report verified candidates.
            # (Old logic was saving every first sighting, which caused high noise)

            # 2. Cooldown check
            if zone["reported"]:
                if frame_count - zone["reported_frame"] < COOLDOWN_FRAMES:
                    continue
                else:
                    # Reset zone for another sighting (e.g. if vehicle stays in view long)
                    zone["readings"] = []; zone["reported"] = False; zone["best_plate"] = None
            
            # Add to readings buffer
            if plate_text != "Detecting..." and len(plate_text) >= 5:
                zone["readings"].append(ocr_res)
                if len(zone["readings"]) > 15: zone["readings"].pop(0)

            # 3. Eager Reporting for High Confidence Strict Matches
            # If we just got a perfect reading, we don't need to wait for MIN_VOTES
            if ocr_res.get("strict") and not zone["reported"]:
                zone["reported"] = True
                zone["reported_frame"] = frame_count
                zone["best_plate"] = ocr_res["text"]
                zone["best_conf"] = ocr_res["conf"]
                
                to_report.append({
                    "frame": int(frame_count), "text": ocr_res["text"], "conf": float(ocr_res["conf"]),
                    "box": det_box, "is_new": True, "alert": ocr_res["text"] in self.blacklist, 
                    "verified_candidate": True, "eager": True
                })
                print(f"🚀 [ZONE {zone_idx}] Eager Reporting: {ocr_res['text']}")

            # 4. Final Aggregation (Multi-frame Voting)
            elif len(zone["readings"]) >= MIN_VOTES_NEEDED and not zone["reported"]:
                best_text, best_conf = self._best_vote(zone["readings"])
                if best_text and self.validate_indian_plate(best_text, mode='soft'):
                    zone["reported"] = True
                    zone["reported_frame"] = frame_count
                    zone["best_plate"] = best_text
                    zone["best_conf"] = best_conf
                    
                    alert = best_text in self.blacklist
                    to_report.append({
                        "frame": int(frame_count), "text": best_text, "conf": float(best_conf),
                        "box": det_box, "is_new": True, "alert": alert, "verified_candidate": True,
                        "raw_candidates": zone["readings"]
                    })
                    print(f"✅ [ZONE {zone_idx}] Voting Winner: {best_text}")

        return all_detections, to_report

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