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

COOLDOWN_FRAMES  = 90    # ~3s at 30fps - Reduced to allow re-detecting successive vehicles in high traffic
MIN_VOTES_NEEDED = 8     # Improved from 4 to 8 to ensure OCR stabilizes before reporting
SPATIAL_IOU_THRESH = 0.3 # Slightly more relaxed overlap
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
        to_L = {'0':'O','1':'I','L':'I','5':'S','8':'B','6':'G','2':'Z', '4':'A', 'Q':'O', 'T':'K'} 
        to_D = {'O':'0','D':'0','Q':'0','I':'1','L':'1','S':'5','B':'8','G':'6','Z':'2','A':'4'}
        
        t = list(text)
        length = len(t)

        # 1. State Code (First 2 chars): ALWAYS LETTERS
        for i in range(min(2, length)):
            if t[i] in to_L: t[i] = to_L[t[i]]
            # If digit found in state code, try to map to letter
            if t[i].isdigit():
                digit_to_L = {'0':'D','1':'I','2':'Z','5':'S','8':'B','4':'A','6':'G'}
                t[i] = digit_to_L.get(t[i], t[i])
        
        # Heuristic: KA often read as TA or IA
        if length >= 2 and (t[0] in ['T', 'I']) and t[1] == 'A':
            t[0] = 'K'

        if length >= 4:
            # 2. District/Serial (Next 2 chars): ALWAYS DIGITS
            for i in range(2, min(4, length)):
                if t[i] in to_D: t[i] = to_D[t[i]]
            
            # 3. Last 4 characters (Identifier): ALWAYS DIGITS
            for i in range(max(4, length-4), length):
                if t[i] in to_D: t[i] = to_D[t[i]]
                
            # 4. Middle characters (Category): ALWAYS LETTERS
            for i in range(4, max(4, length-4)):
                if t[i] in to_L: t[i] = to_L[t[i]]
        
        return "".join(t)

    def validate_indian_plate(self, text, mode='strict'):
        """
        Two-Level Validation:
        'strict' -> Indian Regex: ^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$
        'soft'   -> Alphanumeric + Length (7-11)
        """
        if not text: return False
        
        # Normalize: Remove spaces, upper case
        text = text.replace(" ", "").upper()
        
        # ── SOFT VALIDATION ──────────────────────────────────────────
        if mode == 'soft':
            return len(text) in range(7, 12) and text.isalnum()
            
        # ── STRICT VALIDATION ────────────────────────────────────────
        # Regex: ^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$ (Support up to 3 letters in middle for safety)
        pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$'
        bh_pattern = r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$'
        
        if not (re.match(pattern, text) or re.match(bh_pattern, text)):
            return False
            
        # Validate State Code
        state_code = text[:2]
        if state_code.isalpha() and state_code not in INDIAN_STATE_CODES:
            return False
            
        return True

    def extract_indian_plate(self, text):
        # Increased robustness: ignore small noise at edges, look for 9 or 10 chars
        # Pattern 1: KA02MK6346 (Standard 10)
        # Pattern 2: DL1LY2046 (Standard 9 - but often read with extra digit or missed space)
        
        # Clean text from common OCR noise prefix/suffix (like dots or small symbols)
        text = re.sub(r'^[^A-Z0-9]+|[^A-Z0-9]+$', '', text)

        std_10 = r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}'
        std_9  = r'[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}'
        std_8  = r'[A-Z]{2}[0-9]{2}[A-Z]{0,1}[0-9]{4}' # Fallback for 8 chars if we are desperate
        bh_10  = r'[0-9]{2}BH[0-9]{4}[A-Z]{1,2}'
        
        for p in [std_10, std_9, bh_10, std_8]:
            m = re.search(p, text)
            if m: 
                res = m.group(0)
                if len(res) >= 8: return res
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
        1. Keep all >= 0.2 conf.
        2. Assign scores: +5 for Pattern match, +1 per Frequency.
        3. Pick highest score.
        """
        if not readings: return None, 0.0
        
        valid_candidates = []
        for r in readings:
            if not r or not r.get('text'): continue
            text = r['text'].replace(" ", "").upper()
            if len(text) >= 4 and r.get('conf', 0.0) >= 0.2:
                valid_candidates.append(r)

        if not valid_candidates: 
            return None, 0.0

        # Group and calculate scores
        scores = {}
        for r in valid_candidates:
            text = r['text']
            if text not in scores:
                scores[text] = {"freq": 0, "conf_sum": 0.0, "bonus": 0}
            
            scores[text]["freq"] += 1
            scores[text]["conf_sum"] += r['conf']
            
            # Pattern Bonus
            if self.validate_indian_plate(text, mode='strict'):
                scores[text]["bonus"] = 20 

        # Group similar texts to avoid splitting votes (e.g. KA03 vs KA031)
        # We'll use the longest/most patterned one in each fuzzy group
        final_scores = {}
        for text, stats in scores.items():
            # Find if there's a highly similar text already in final_scores
            found = False
            for target in final_scores.keys():
                similarity = SequenceMatcher(None, text, target).ratio()
                if similarity > 0.8: # Very similar
                    # Merge into the "better" one (pattern bonus or length)
                    found = True
                    is_better = stats["bonus"] > final_scores[target]["bonus"] or \
                                (stats["bonus"] == final_scores[target]["bonus"] and len(text) > len(target))
                    
                    final_scores[target]["freq"] += stats["freq"]
                    final_scores[target]["conf_sum"] += stats["conf_sum"]
                    if is_better:
                        # Swap key to the better text representation
                        old_stats = final_scores.pop(target)
                        final_scores[text] = old_stats
                        final_scores[text]["bonus"] = stats["bonus"]
                    break
            if not found:
                final_scores[text] = stats

        # Pick best based on (bonus + freq) then avg_conf
        best_plate = None
        best_total_score = -1
        best_avg_conf = -1.0
        
        for text, stats in final_scores.items():
            total_score = stats["bonus"] + stats["freq"]
            avg_conf = stats["conf_sum"] / stats["freq"]
            
            if total_score > best_total_score:
                best_total_score = total_score
                best_plate = text
                best_avg_conf = avg_conf
            elif total_score == best_total_score:
                if avg_conf > best_avg_conf:
                    best_avg_conf = avg_conf
                    best_plate = text
                    
        self.log_debug(f"VOTE: Winner '{best_plate}' (score={best_total_score}, freq={scores[best_plate]['freq']}, avg_conf={best_avg_conf:.2f})")
        return best_plate, best_avg_conf
        return best_plate, best_avg_conf

    # ── OCR ───────────────────────────────────────────────────
    def preprocess_for_ocr(self, img):
        """
        ENHANCED PREPROCESSING:
        Grayscale -> Resize (2x) -> Denoise -> Sharpen -> CLAHE -> Threshold
        """
        if img is None or img.size == 0: return None
        
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Resize (2x as requested)
        h, w = gray.shape[:2]
        resized = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        
        # 3. Denoise
        denoised = cv2.fastNlMeansDenoising(resized, h=10)
        
        # 4. CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 5. Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # 6. Thresholding (Adaptive)
        processed = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
            
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

    def process_plate(self, plate_img):
        """
        RUN OCR ON ORIGINAL AND ENHANCED CROPS:
        Returns: { 'text': str, 'conf': float }
        """
        if plate_img is None or plate_img.size == 0:
            return {"text": "", "conf": 0.0}

        # 1. OCR on Original
        text_orig, conf_orig = self._ocr_pass(plate_img)
        
        # 2. OCR on Enhanced
        processed = self.preprocess_for_ocr(plate_img)
        text_enh, conf_enh = ("", 0.0)
        if processed is not None:
            text_enh, conf_enh = self._ocr_pass(processed)

        # 3. Pick best result
        if conf_enh > conf_orig:
            raw_text, conf = text_enh, conf_enh
        else:
            raw_text, conf = text_orig, conf_orig

        if not raw_text or len(raw_text) < 6: 
            return {"text": "", "conf": 0.0}

        # 4. Position-Aware Correlation Fixes
        corrected = self.position_aware_correct(raw_text)
        
        # 5. Extract based on Regex (Softened)
        extracted = self.extract_indian_plate(corrected)
        final_text = extracted if extracted else corrected
        
        # If still very short, it's likely noise, but user asked for 20%+, so we allow it.
        if len(final_text) < 4:
            return {"text": "", "conf": 0.0}
        
        # 6. M <-> H Dynamic Check
        if not self.validate_indian_plate(final_text):
            if "H" in final_text:
                test_m = final_text.replace("H", "M")
                if self.validate_indian_plate(test_m):
                    final_text = test_m
            elif "M" in final_text:
                test_h = final_text.replace("M", "H")
                if self.validate_indian_plate(test_h):
                    final_text = test_h

        return {"text": final_text, "conf": conf}

    # ── Main detection ────────────────────────────────────────
    def run_detection(self, frame, frame_count):
        results = self.anpr_model(frame, verbose=False)[0]
        all_detections = [] # Every YOLO box + RAW OCR (for drawing)
        to_report = []      # Validated/Confirmed plates (for DB)

        h, w = frame.shape[:2]

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.12: # Lowered from 0.15 for maximum recall on small/fast plates
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            det_box = [x1, y1, x2, y2]

            # ── Pre-process Crop ────────────────────────
            px1, py1 = max(0, x1 - TIGHT_PAD_PX), max(0, y1 - TIGHT_PAD_PX)
            px2, py2 = min(w, x2 + TIGHT_PAD_PX), min(h, y2 + TIGHT_PAD_PX)
            plate_img = frame[py1:py2, px1:px2]
            
            # Save debug crop
            if not os.path.exists(DEBUG_CROP_DIR): os.makedirs(DEBUG_CROP_DIR)
            crop_fn = f"{DEBUG_CROP_DIR}/z_f{frame_count}_{x1}.jpg"
            cv2.imwrite(crop_fn, plate_img)

            # ── OCR (Always do OCR for drawing) ──────────
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

            # 2. Cooldown check for "Verified" reporting
            if zone["reported"]:
                if frame_count - zone["reported_frame"] < COOLDOWN_FRAMES:
                    continue
                else:
                    zone["readings"] = []; zone["reported"] = False; zone["best_plate"] = None; zone["best_conf"] = 0.0
                    zone["first_frame_saved"] = False

            # Add to voting buffer
            if plate_text != "Detecting..." and len(plate_text) >= 5:
                zone["readings"].append(ocr_res)
                if len(zone["readings"]) > 20: zone["readings"] = zone["readings"][-20:]

            # 3. Validation Report (High confidence / Stability)
            if len(zone["readings"]) >= MIN_VOTES_NEEDED:
                best_text, best_conf = self._best_vote(zone["readings"])
                if best_text:
                    # Only report if it's better or we haven't reported a "stable" one yet
                    # Enforce Indian plate length (9-10) for 'Verified' status to satisfy user request
                    is_valid_length = len(best_text) >= 9
                    
                    if (not zone["reported"] or best_conf > zone["best_conf"]) and is_valid_length:
                        zone["reported"] = True
                        zone["reported_frame"] = frame_count
                        zone["best_plate"] = best_text
                        zone["best_conf"] = best_conf
                        
                        alert = best_text in self.blacklist
                        to_report.append({
                            "frame": int(frame_count), "text": best_text, "conf": float(best_conf),
                            "box": det_box, "is_new": True, "alert": alert, "verified_candidate": True
                        })

        return all_detections, to_report

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