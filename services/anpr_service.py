import cv2
import re
import os
import threading
from datetime import datetime
from collections import Counter
from ultralytics import YOLO
import numpy as np
import torch
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
DYNAMIC_PADDING_PCT = 0.50 # 50% padding ensures FULL plate is always captured even at frame edges
DEBUG_CROP_DIR = "outputs/debug_crops"
SHARPNESS_THRESHOLD = 100  # Laplacian variance threshold for sharp plate crops
UNREAD_FRAME_LIMIT = 5    # Report UNREAD after this many frames without a valid plate read (lowered for faster response)

INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "CT", "DD", "DL", "DN",
    "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TG", "TR", "TS", "UK", "UP", "WB", "BH"
}

# Map invalid OCR state codes to closest valid state
INVALID_STATE_MAP = {
    'IN': 'TN', 'KN': 'KA', 'IK': 'JK',
}

BANNED_WORDS = {
    "HINDUJA", "GROUP", "GOODS", "CARRIER", "HSRP", "INDIA", 
    "STOP", "SIGNAL", "POLICE", "AMBULANCE", "ARMY", "NAVY"
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
        #   "box"              : [x1,y1,x2,y2],   ← representative bbox
        #   "readings"         : [text, text, ...], ← OCR votes collected
        #   "reported"         : bool,
        #   "reported_frame"   : int,
        #   "best_plate"       : str or None        ← confirmed plate text
        #   "frames_without_read": int              ← track blurry / no-read frames
        #   "total_frames"     : int                ← total frames this zone has existed
        #   "unread_reported"  : bool               ← whether UNREAD entry was already sent
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

    # ── Sharpness check ──────────────────────────────────────
    @staticmethod
    def calculate_sharpness(image):
        """
        Calculate sharpness of an image region using Laplacian variance.
        Higher value = sharper image. Threshold: 100.
        """
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    # ── Text helpers ──────────────────────────────────────────
    def clean_plate_text(self, text):
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def position_aware_correct(self, text):
        """
        Corrects confusing characters based on Indian plate structure.
        Indian format: [LL][DD][L{1,2}][DDDD]
        Positions 1,2,5,6 = LETTERS | Positions 3,4,7,8,9,10 = DIGITS
        
        KEY FIXES for Indian ANPR:
        - M vs H confusion (very common in Indian fonts)
        - N vs H confusion
        - Strict series-letter recovery
        """
        if not text: return ""
        
        raw_text = text  # Save original for logging
        
        # Correction maps
        digit_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '7': 'T'}
        letter_to_digit = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7'}
        
        # INDIAN FONT: H and M look nearly identical in bold fonts.
        # N and M also get confused. These only apply in LETTER positions.
        # We'll try M first in series positions when H/N appears.
        SERIES_LETTER_FIX = {'H': 'M', 'N': 'M'}
        
        t = list(text)
        length = len(t)

        # 1. Positions 1,2 (idx 0,1) — State Code: MUST be LETTERS
        for i in range(min(2, length)):
            if t[i].isdigit() and t[i] in digit_to_letter:
                t[i] = digit_to_letter[t[i]]

        if length >= 4:
            # 2. Positions 3,4 (idx 2,3) — District: MUST be DIGITS
            for i in range(2, 4):
                if t[i].isalpha() and t[i] in letter_to_digit:
                    t[i] = letter_to_digit[t[i]]

            # 3. Last 4 chars — Number: MUST be DIGITS
            numeric_tail_start = max(4, length - 4)
            for i in range(numeric_tail_start, length):
                if t[i].isalpha() and t[i] in letter_to_digit:
                    # Specific check for Indian font 'A' vs '4' or 'L' vs '1'
                    if t[i] == 'A' and i >= length - 4:
                        t[i] = '4'
                    elif t[i] == 'L' and i >= length - 4:
                        t[i] = '1'
                    else:
                        t[i] = letter_to_digit[t[i]]

            # 4. Middle chars (idx 4 to tail) — Series: MUST be LETTERS
            for i in range(4, numeric_tail_start):
                if t[i].isdigit() and t[i] in digit_to_letter:
                    t[i] = digit_to_letter[t[i]]
                # FIX: M/H/N confusion in series positions
                # e.g., KA55H4L → KA55M4L (H misread as M)
                # Apply M swap only if total length looks like a full plate (9-10 chars)
                elif t[i].isalpha() and t[i] in SERIES_LETTER_FIX and length >= 9:
                    candidate_t = list(t)
                    candidate_t[i] = SERIES_LETTER_FIX[t[i]]
                    candidate_text = ''.join(candidate_t)
                    if self.validate_indian_plate(candidate_text, mode='strict'):
                        print(f"[M/H FIX] Swapped '{t[i]}' → '{SERIES_LETTER_FIX[t[i]]}' at pos {i}: {''.join(t)} → {candidate_text}")
                        t[i] = SERIES_LETTER_FIX[t[i]]

        # 5. State code validation — map invalid to closest valid
        if length >= 2:
            state = "".join(t[:2])
            if state not in INDIAN_STATE_CODES and state in INVALID_STATE_MAP:
                corrected_state = INVALID_STATE_MAP[state]
                t[0], t[1] = corrected_state[0], corrected_state[1]
                print(f"🔧 State code corrected: {state} → {corrected_state}")

        # 6. O→K, Z→L state code recovery for common OCR misreads
        if length >= 2:
            state = "".join(t[:2])
            if state not in INDIAN_STATE_CODES:
                ocr_state_swap = {'O': 'K', 'Z': 'L', '0': 'K', '2': 'L'}
                new_t = list(t)
                changed = False
                for i in range(min(2, len(new_t))):
                    if new_t[i] in ocr_state_swap:
                        new_t[i] = ocr_state_swap[new_t[i]]
                        changed = True
                new_state = "".join(new_t[:2])
                if changed and new_state in INDIAN_STATE_CODES:
                    t = new_t

        # 7. Final Polish
        corrected = "".join(t)
        
        if length >= 9:
            # '1' where a letter should be (e.g. KA02L1...)
            if length == 10 and corrected[5].isdigit() and corrected[5] == '1':
                corrected = corrected[:5] + 'I' + corrected[6:]
            elif length == 10 and corrected[4].isdigit() and corrected[4] == '1':
                corrected = corrected[:4] + 'I' + corrected[5:]

        if corrected != raw_text:
            print(f"[OCR FIX] raw={raw_text} corrected={corrected}")
        
        return corrected

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


    def is_full_plate(self, plate_img, text):
        """
        Verifies if the plate is likely 'full' and not a partial view.
        Uses aspect ratio, character count, and banned words filter.
        """
        if not text or plate_img is None or plate_img.size == 0:
            return False
            
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # 1. BANNED WORDS FILTER (False Positive Prevention)
        for word in BANNED_WORDS:
            if word in clean_text or word in text.upper():
                print(f"🚫 [FILTER] Blocked false positive word: {word}")
                return False

        h, w = plate_img.shape[:2]
        if w == 0 or h == 0: return False
        
        aspect_ratio = w / h
        # 2. ASPECT RATIO: Relaxed to 1.5 to allow partial/angled Indian plates
        # (Old value 2.1 was rejecting many valid plates with ratio 1.5–2.0)
        if aspect_ratio < 1.5 or aspect_ratio > 7.0: 
            return False
            
        # 3. CHARACTER COUNT: Full Indian plate MUST have 8-10 characters
        # (e.g., KA02LA7982 is 10, DL1C1234 is 8)
        if len(clean_text) < 7:
            return False
            
        # 4. PATTERN VESTIGE: Should generally look like an Indian plate (LL at start)
        if len(clean_text) >= 2:
            state = clean_text[:2]
            # If start is digits, it might be a BH plate or error, but LL is standard
            if state.isdigit() and "BH" not in clean_text:
                pass # Allow BH fallback 
            elif not state[0].isalpha(): # If first char is digit and not BH, likely not a plate
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
        if len(joined) >= 10:
            for i in range(len(joined) - 9):
                sub = joined[i:i+10]
                if self.validate_indian_plate(sub, mode='soft'):
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
                    is_better = (not self.validate_indian_plate(rep, 'strict') and self.validate_indian_plate(text, 'strict')) or \
                                (len(text) > len(rep) and self.validate_indian_plate(text, 'soft'))
                    
                    # Specific visual reconciliation: A vs M
                    if not is_better and 'A' in rep and 'M' in text and len(rep) == len(text) == 10:
                        if rep[:4] == text[:4] and rep[6:] == text[6:]:
                            is_better = True
                    
                    if is_better:
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

        # EXTRA: If two candidates differ only in H/M at series position, prefer the strictly valid one
        keys = list(candidates.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                if len(a) == len(b) and len(a) >= 9:
                    diffs = [(idx, a[idx], b[idx]) for idx in range(len(a)) if a[idx] != b[idx]]
                    if len(diffs) == 1:
                        idx_d, ca, cb = diffs[0]
                        if 4 <= idx_d <= 5 and {ca, cb} <= {'H', 'M', 'N'}:
                            # Merge into the strictly valid one
                            if self.validate_indian_plate(a, 'strict') and not self.validate_indian_plate(b, 'strict'):
                                candidates[a]["freq"] += candidates[b]["freq"]
                                candidates[a]["conf_sum"] += candidates[b]["conf_sum"]
                                candidates[a]["max_conf"] = max(candidates[a]["max_conf"], candidates[b]["max_conf"])
                                del candidates[b]
                                keys[j] = a  # Prevent re-processing
                                print(f"[M/H MERGE] Merged '{b}' into '{a}'")
                            elif self.validate_indian_plate(b, 'strict') and not self.validate_indian_plate(a, 'strict'):
                                candidates[b]["freq"] += candidates[a]["freq"]
                                candidates[b]["conf_sum"] += candidates[a]["conf_sum"]
                                candidates[b]["max_conf"] = max(candidates[b]["max_conf"], candidates[a]["max_conf"])
                                del candidates[a]
                                keys[i] = b
                                print(f"[M/H MERGE] Merged '{a}' into '{b}'")
                                break

        # 2. Final Selection based on score
        best_plate = None
        max_score = -1
        
        for text, stats in candidates.items():
            score = stats["freq"]
            if self.validate_indian_plate(text, mode='strict'):
                score += 50
            elif self.validate_indian_plate(text, mode='soft'):
                score += 10
            
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
            # CLAHE + Light Denoise (reduced from h=10)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            denoised = cv2.fastNlMeansDenoising(gray, None, h=5)
            # Subtle sharpening only if needed
            return denoised
            
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
            
            # ── Full Plate Verification ──
            if not self.is_full_plate(plate_img, final_text):
                continue

            # If we find a strict match, return immediately (Eager Return)
            if self.validate_indian_plate(final_text, mode='strict') and conf > 0.8:
                return {"text": final_text, "conf": conf, "strict": True}
            
            # Keep the best result found so far
            if conf > best_res["conf"]:
                best_res = {"text": final_text, "conf": conf, "strict": False}
        
        return best_res

    # ── Main detection ────────────────────────────────────────
    def run_detection(self, frame, frame_count):
        """
        Processing order:
        1. YOLO detects plates at 640px resolution
        2. For each plate region → sharpness check
        3. If sharp → OCR → position_aware_correct()
        4. Spatial zone tracking + voting
        5. Returns (all_detections, to_report)
        
        Vehicle type (best.pt) and bounding box drawing are handled in api.py.
        """
        h_frame, w_frame = frame.shape[:2]

        # ── Run YOLO plate detection at 640px for speed ──────
        with torch.no_grad():
            results = self.anpr_model(frame, imgsz=640, conf=0.35, verbose=False)[0]

        all_detections = []  # Every YOLO box + RAW OCR (for drawing)
        to_report = []       # Validated/Confirmed plates (for DB)

        for box in results.boxes:
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame, x2), min(h_frame, y2)
            det_box = [x1, y1, x2, y2]

            # ── Sharpness check (Problem 1) ──────────────────
            plate_region = frame[y1:y2, x1:x2]
            sharpness = self.calculate_sharpness(plate_region)

            # ── Pre-process Crop (Dynamic Padding) ──────────
            dw, dh = (x2 - x1) * DYNAMIC_PADDING_PCT, (y2 - y1) * DYNAMIC_PADDING_PCT
            px1, py1 = max(0, int(x1 - dw)), max(0, int(y1 - dh))
            px2, py2 = min(w_frame, int(x2 + dw)), min(h_frame, int(y2 + dh))
            plate_img = frame[py1:py2, px1:px2]

            # 1. Spatial zone tracker ────────────────────────
            zone_idx = self._find_zone(det_box)
            if zone_idx == -1:
                self.zones.append({
                    "box": det_box, "readings": [], "reported": False,
                    "reported_frame": -1, "best_plate": None, "best_conf": 0.0,
                    "frames_without_read": 0, "total_frames": 0,
                    "unread_reported": False
                })
                zone_idx = len(self.zones) - 1

            zone = self.zones[zone_idx]
            zone["box"] = det_box
            zone["total_frames"] = zone.get("total_frames", 0) + 1

            # 2. Cooldown check
            if zone["reported"]:
                if frame_count - zone["reported_frame"] < COOLDOWN_FRAMES:
                    # Still add to all_detections for visualization
                    all_detections.append({
                        "box": det_box, "text": zone.get("best_plate", ""),
                        "conf": zone.get("best_conf", 0.0), "confirmed": True
                    })
                    continue
                else:
                    zone["readings"] = []; zone["reported"] = False
                    zone["best_plate"] = None; zone["unread_reported"] = False
                    zone["frames_without_read"] = 0

            # ── If blurry, skip OCR but keep tracking (Problem 1) ──
            if sharpness < SHARPNESS_THRESHOLD:
                zone["frames_without_read"] = zone.get("frames_without_read", 0) + 1
                all_detections.append({
                    "box": det_box, "text": "Blurry...", "conf": 0.0,
                    "confirmed": False
                })
                # Check if we should report UNREAD (Problem 4)
                if (zone.get("frames_without_read", 0) >= UNREAD_FRAME_LIMIT
                        and not zone.get("unread_reported", False)
                        and not zone["reported"]):
                    zone["unread_reported"] = True
                    to_report.append({
                        "frame": int(frame_count), "text": "UNREAD", "conf": 0.0,
                        "box": det_box, "is_new": True, "alert": False,
                        "verified_candidate": True, "unread": True
                    })
                    print(f"📋 [ZONE {zone_idx}] UNREAD after {zone['frames_without_read']} blurry frames")
                continue

            # ── OCR (sharp frame) ─────────────────────────────
            ocr_res = self.process_plate(plate_img)
            plate_text = ocr_res["text"] or ""
            ocr_conf = ocr_res["conf"]

            display_text = plate_text if plate_text else "Detecting..."
            all_detections.append({
                "box": det_box, "text": display_text, "conf": ocr_conf,
                "confirmed": False
            })

            # Add to readings buffer — only if we got a meaningful valid read
            if plate_text and len(plate_text) >= 5:
                zone["readings"].append(ocr_res)
                zone["frames_without_read"] = 0  # Reset since we got a valid read
                if len(zone["readings"]) > 15: zone["readings"].pop(0)
            else:
                # FIX: Increment NOT just for blurry, but also when OCR ran but returned
                # nothing useful (short text, failed aspect ratio, etc.)
                # This ensures UNREAD is triggered for sharp-but-unreadable plates too.
                zone["frames_without_read"] = zone.get("frames_without_read", 0) + 1

            # 3. Eager Reporting for High Confidence Strict Matches
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
                # Mark as confirmed for visualization
                all_detections[-1]["confirmed"] = True
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
                    all_detections[-1]["confirmed"] = True
                    print(f"✅ [ZONE {zone_idx}] Voting Winner: {best_text}")

            # 5. UNREAD fallback (Problem 4) — no read after many frames
            if (zone.get("frames_without_read", 0) >= UNREAD_FRAME_LIMIT
                    and not zone.get("unread_reported", False)
                    and not zone["reported"]):
                zone["unread_reported"] = True
                to_report.append({
                    "frame": int(frame_count), "text": "UNREAD", "conf": 0.0,
                    "box": det_box, "is_new": True, "alert": False,
                    "verified_candidate": True, "unread": True
                })
                print(f"📋 [ZONE {zone_idx}] UNREAD after {zone['frames_without_read']} frames without plate read")

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