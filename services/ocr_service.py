import os
import cv2
import numpy as np
import logging
import threading
import time
import traceback

# ── Force Stable Paddle Environment ──────────────────────────────────────────
os.environ["FLAGS_enable_onednn"]   = "0"
os.environ["FLAGS_use_onednn"]      = "0"
os.environ["FLAGS_enable_pir_api"]  = "0"
os.environ["FLAGS_use_pir_api"]     = "0"
os.environ["FLAGS_use_mkldnn"]      = "0"

from paddleocr import PaddleOCR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OCRService")

class OCRService:
    """
    Standardized OCR helper for ANPR using PaddleOCR.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(OCRService, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, lang='en'):
        if self._initialized:
            return
        
        logger.info("Initializing PaddleOCR engine...")
        try:
            # Initialize PaddleOCR with angle classifier enabled for rotated plates
            self.ocr = PaddleOCR(lang=lang, use_angle_cls=True)
            self._initialized = True
            logger.info("PaddleOCR engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            self.ocr = None

    def get_text(self, plate_crop):
        """
        Accepts a plate crop (BGR image or bytes) and returns (text, confidence) candidates.
        """
        if self.ocr is None:
            logger.error("OCR engine not initialized.")
            return "", 0.0

        try:
            # 1. Convert bytes to BGR image if necessary
            if isinstance(plate_crop, bytes):
                nparr = np.frombuffer(plate_crop, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = plate_crop

            # 2. Strict Validation
            if img is None:
                return "", 0.0
            
            # Size check
            h, w = img.shape[:2]
            if h < 10 or w < 10:
                logger.debug(f"Image too small for OCR: {w}x{h}")
                return "", 0.0

            # Channel / Color Space validation
            if len(img.shape) == 2:
                # Grayscale to BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                # BGRA to BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Final check to ensure it's a valid 3-channel image for PaddleOCR
            if not (len(img.shape) == 3 and img.shape[2] == 3):
                logger.warning(f"Unexpected image shape for OCR: {img.shape}")
                return "", 0.0

            # 3. Run PaddleOCR (Inside Try/Except already but we'll be extra careful)
            start_time = time.time()
            result = self.ocr.ocr(img)
            elapsed = time.time() - start_time

            # 4. Safe Parsing
            if not result or not isinstance(result, list) or len(result) == 0 or result[0] is None:
                return "", 0.0

            full_text = ""
            confidences = []

            # PaddleOCR result is a list of results (one per image), we pass one image
            if result[0]:
                # Sort blocks by Y-coordinate to handle multi-line plates (top-to-bottom)
                # Each 'line' is [[ [x1,y1], [x2,y1], [x2,y2], [x1,y2] ], (text, conf)]
                sorted_lines = sorted(result[0], key=lambda x: x[0][0][1]) 

                for line in sorted_lines:
                    try:
                        if not line or len(line) < 2:
                            continue
                        
                        text_data = line[1]
                        if isinstance(text_data, (tuple, list)) and len(text_data) >= 2:
                            text, conf = text_data[0], text_data[1]
                            # Add space if we already have text to keep multi-line parts distinct
                            if full_text:
                                full_text += " "
                            full_text += str(text)
                            confidences.append(float(conf))
                    except (IndexError, TypeError, ValueError) as line_err:
                        logger.warning(f"Skipping malformed OCR line: {line_err}")
                        continue

            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            full_text = full_text.strip().upper()
            
            return full_text, avg_conf

        except Exception as e:
            logger.error(f"CRITICAL: OCR Pipeline Error: {e}")
            logger.error(traceback.format_exc())
            return "", 0.0
