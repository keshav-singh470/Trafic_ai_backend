import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime, timezone
import time
import logging

# Load environment variables FIRST
load_dotenv()

# ── OpenMP / Paddle / PaddleOCR flags ─────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"]                    = "TRUE"
os.environ["OMP_NUM_THREADS"]                          = "1"
os.environ["FLAGS_enable_onednn"]                      = "0"
os.environ["FLAGS_use_onednn"]                         = "0"
os.environ["FLAGS_enable_pir_api"]                     = "0"
os.environ["FLAGS_use_pir_api"]                        = "0"
os.environ["FLAGS_use_mkldnn"]                         = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]   = "python"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"]    = "True"
os.environ["GLOG_minloglevel"]                         = "2"

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Body
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import cv2
import json
import numpy as np
import threading
import torch
import re
import contextlib
from typing import Dict, List
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor

from services.anpr_service import ANPRService
from services.helmet_service import HelmetService
from services.overload_service import OverloadService
from services.wrong_side_service import WrongSideService
from services.stalled_service import StalledService
from services.seatbelt_service import SeatbeltService
from services.vehicle_type_service import VehicleTypeService
from services.s3_service import (
    upload_video as s3_upload_video,
    upload_image as s3_upload_image,
    upload_file_to_s3,
    get_presigned_url,
    generate_presigned_urls_for_report,
)
from services.telegram_service import send_local_violation, create_combined_image
from services.pdf_service import generate_violation_pdf
from shapely.geometry import Polygon

# COCO class IDs (from yolov8n.pt base model) + best.pt string labels
VEHICLE_TYPE_MAP = {
    # ── yolov8n.pt COCO integer class IDs (ByteTrack base fallback only) ──
    0:  "Person",
    1:  "Bicycle",
    2:  "Car",
    3:  "Bike",
    5:  "Bus",
    6:  "Truck",
    7:  "Truck",
    8:  "Truck",

    # ── best.pt EXACT string labels (6 classes) ──
    "autorickshaw":  "Auto-Rickshaw",
    "bus":           "Bus",
    "car":           "Car",
    "motorcycle":    "Bike",
    "scooter":       "Scooter",
    "truck":         "Truck",

    # ── common variations ──
    "auto":          "Auto-Rickshaw",
    "bike":          "Bike",
    "van":           "Car",
    "lorry":         "Truck",
    "tempo":         "Truck",
    "minibus":       "Bus",
    "ambulance":     "Car",
}
VEHICLE_COLORS = ["red", "blue", "white", "black", "silver", "unknown"]

def get_majority_vtype(labels: List[str]):
    """Returns the most frequent vehicle type label and its relative frequency."""
    if not labels:
        return "unknown", 0.0
    from collections import Counter
    counts = Counter(labels)
    # Filter out 'unknown' if other valid labels exist
    if len(counts) > 1 and ("unknown" in counts or "Unknown" in counts):
        if "unknown" in counts: counts.pop("unknown")
        if "Unknown" in counts: counts.pop("Unknown")
        if not counts: return "unknown", 0.0
            
    most_common = counts.most_common(1)[0]
    return most_common[0], most_common[1] / len(labels)

def finalize_vtype_detection(base_v_type, voted_type, voted_conf, vehicle_bbox, frame_w, frame_h):
    """Applies sanity checks and guards to determine the final vehicle type label."""
    TWO_WHEELER_LABELS = {"Scooter", "Bike"}
    HEAVY_BASE_LABELS  = {"Car", "Bus", "Truck"}
    
    frame_area = frame_w * frame_h
    vehicle_area = 0
    if vehicle_bbox:
        vehicle_area = (vehicle_bbox[2]-vehicle_bbox[0]) * (vehicle_bbox[3]-vehicle_bbox[1])
        
    is_large_vehicle = (vehicle_area > frame_area * 0.12) if frame_area > 0 else False
    
    final_v_type = voted_type
    
    # 1. Size Sanity Check: Large objects aren't two-wheelers
    if voted_type in TWO_WHEELER_LABELS and is_large_vehicle:
        final_v_type = base_v_type if base_v_type not in ["unknown", "Unknown", None, ""] else "Vehicle"
        
    # 2. Confidence Guard: If voting is weak, trust the base YOLO model (COCO class)
    BEST_PT_CONF_THRESHOLD = 0.55
    if voted_conf < BEST_PT_CONF_THRESHOLD:
        if base_v_type and base_v_type not in ["unknown", "Unknown"]:
            final_v_type = base_v_type
        elif final_v_type == "unknown":
            final_v_type = "Vehicle"
    
    # 3. Heavy Protection: Protect Bus/Truck/Car labels from flip-flopping to two-wheelers
    if (base_v_type in HEAVY_BASE_LABELS and final_v_type in TWO_WHEELER_LABELS and voted_conf < 0.70):
        final_v_type = base_v_type
        
    # 4. Final Fallback
    if final_v_type in ["Vehicle", "Unknown", "unknown"] and base_v_type not in ["unknown", "Unknown", None, ""]:
        final_v_type = base_v_type
        
    return final_v_type

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create indexes for optimized search
    if db is not None:
        try:
            db.detections.create_index([("plate_text", 1)])
            db.detections.create_index([("vehicle_type", 1)])
            db.detections.create_index([("vehicle_color", 1)])
            db.detections.create_index([("violation_type", 1)])
            db.detections.create_index([("created_at", -1)])
            db.detections.create_index([("job_id", 1)])
            print("✅ MongoDB Indexes updated.")
        except Exception as e:
            print(f"Index error: {e}")
    yield
    # Shutdown logic (if any) could go here

app = FastAPI(title="3rd AI APP - Traffic AI System", lifespan=lifespan)

# Groq Client (Removed as per request)
GROQ_CLIENT = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8000",
        "https://trafic-ai-frontend.vercel.app",
        "https://ai-trafic.servepics.com",
        "http://ai-trafic.servepics.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

# ── Environment Configuration ──────────────────────────────────────────────────
REQUIRED_ENV_VARS = [
    "MONGO_URI",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_S3_BUCKET",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID"
]

missing_vars = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing_vars:
    print(f"❌ CRITICAL ERROR: Missing environment variables: {', '.join(missing_vars)}")
else:
    print("✅ All required environment variables are set.")

# ── MongoDB ────────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI")
db = None

if not MONGO_URI:
    print("❌ CRITICAL ERROR: MONGO_URI not found in environment variables.")
else:
    try:
        # Atlas connection needs specific options sometimes, but generally works with default
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, connectTimeoutMS=10000)
        client.admin.command('ping')
        db = client.get_database() # Uses the DB name from the URI (ai_traffic_db)
        print(f"✅ MongoDB connected successfully to database: {db.name}")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to connect to MongoDB. Error: {e}")

# ── Load AI Models once at startup ────────────────────────────────────────────
print("Loading models once at startup...")

ANPR_SERVICE = ANPRService(
    base_model="models/yolov8n.pt",
    anpr_model="models/anpr_plat.pt"
)
HELMET_SERVICE    = HelmetService(model_path="models/helmet_triple_model.pt")
OVERLOAD_SERVICE  = OverloadService(model_path="models/helmet_triple_model.pt")
WRONGSIDE_SERVICE = WrongSideService(
    base_model="models/yolov8n.pt",
    zones=[{
        "polygon": Polygon([(960, 0), (1920, 0), (1920, 1080), (960, 1080)]),
        "forbidden_classes": [2, 3, 5, 7]
    }]
)
STALLED_SERVICE  = StalledService(model_path="models/yolov8n.pt")
SEATBELT_SERVICE = SeatbeltService(model_path="no_sitbelt.pt")
VEHICLE_TYPE_SERVICE = VehicleTypeService(model_path="models/best.pt")
print(f"[BEST.PT CLASSES] {VEHICLE_TYPE_SERVICE.class_names}")

# Thread pool for vehicle type detection (non-blocking)
vtype_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vtype")

print("✅ All models loaded.")


# ── Directory helpers ─────────────────────────────────────────────────────────

def check_dirs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "assets"), exist_ok=True)

check_dirs()
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

jobs: Dict[str, Dict] = {}

# ── FIX 2: Global Plate Cooldown Tracker ──────────────────────────────────────
# Prevents duplicate processing of the same plate within 60 seconds
plate_cooldown_tracker: Dict[str, float] = {}
PLATE_COOLDOWN_SECONDS = 60


# ── MongoDB helpers ───────────────────────────────────────────────────────────

def create_job_in_db(job_id: str, case_type: str, s3_video_key: str):
    """
    Insert a new job document.
    Stores the S3 key (not a URL) — presigned URL generated on demand.
    """
    if db is None:
        return
    try:
        db.jobs.insert_one({
            "job_id":        job_id,
            "video_url":     s3_video_key,   # Store S3 key as requested (can be converted to URL)
            "case_type":     case_type,
            "status":        "pending",
            "violation_count": 0,
            "created_at":    datetime.now(timezone.utc),
        })
    except Exception as e:
        print(f"ERROR creating job in MongoDB: {e}")


def update_job_in_db(job_id: str, status: str, violation_count: int, error_message: str = None):
    if db is None:
        return
    try:
        update_data = {
            "status":          status,
            "violation_count": violation_count,
            "completed_at":    datetime.now(timezone.utc),
        }
        if error_message:
            update_data["error_message"] = error_message
            
        db.jobs.update_one(
            {"job_id": job_id},
            {"$set": update_data}
        )
    except Exception as e:
        print(f"ERROR updating job in MongoDB: {e}")


def save_detection(job_id: str, frame: int, vehicle_id: str, 
                   plate: str, confidence: float = 0.0, status: str = "low_confidence",
                   violation_type: str = "N/A", vehicle_type: str = "unknown",
                   vehicle_color: str = "unknown", vehicle_type_confidence: float = 0.0,
                   s3_vehicle_key: str = None, 
                   s3_plate_key: str = None, fps: float = 30.0, raw_ocr: str = None,
                   visual_signature: List[float] = None,
                   source_id: str = "video_upload", job_status: str = "processing",
                   processing_time: float = 0.0):
    """
    Phase 2: Save to 'vehicles' collection with Plate-based De-duplication.
    Strict Schema: job_id, plate_number, vehicle_type, color, timestamp, frame_number, violations, images.
    """
    if db is None: return
    import uuid
    try:
        total_seconds = int(frame / fps)
        time_str = f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"

        # Clean plate
        plate_number = plate.strip().upper() if plate and plate != "N/A" else "N/A"
        
        from difflib import SequenceMatcher
        
        # Primary Key: vehicle_id (Tracking ID) within a job
        # If vehicle_id is N/A, fallback to fuzzy plate matching
        filter_query = {"job_id": job_id}
        existing = None
        
        if vehicle_id and vehicle_id != "N/A":
            filter_query["vehicle_id"] = vehicle_id
            existing = db.vehicles.find_one(filter_query)
        
        if not existing and plate_number != "N/A":
            # Hybrid Deduplication: Check for plate text AND visual similarity
            similar_plates = db.vehicles.find({"job_id": job_id, "plate_number": {"$ne": "N/A"}})
            for sp in similar_plates:
                # 1. Text Similarity (Fuzzy OCR matching)
                text_sim = SequenceMatcher(None, plate_number, sp.get("plate_number", "")).ratio()
                
                # 2. Visual Similarity (Re-ID signature comparison)
                visual_sim = 0.0
                if visual_signature and sp.get("visual_signature"):
                    visual_sim = compare_signatures(visual_signature, sp["visual_signature"])
                
                # Deduplication Rule: 
                # Match if text is very similar (>80%) OR text is moderately similar (>65%) AND visual is high (>75%)
                if text_sim > 0.8 or (text_sim > 0.65 and visual_sim > 0.75):
                    existing = sp
                    filter_query = {"_id": sp["_id"]}
                    print(f"🔄 Hybrid Dedup Match: {plate_number} matches {sp['plate_number']} (Text: {text_sim:.2f}, Visual: {visual_sim:.2f})")
                    break
            
            if not existing:
                filter_query = {"job_id": job_id, "plate_number": plate_number}
                existing = db.vehicles.find_one(filter_query)
        elif not vehicle_id or vehicle_id == "N/A":
            # If no vehicle ID and no plate, it's likely a generic violation or low-confidence noise
            filter_query["vehicle_id"] = "N/A"
            filter_query["plate_number"] = "N/A"
            existing = db.vehicles.find_one(filter_query)
        if existing:
            # If we already have a high-confidence plate, don't overwrite it with a lower-confidence eager one
            if existing.get("status") == "verified" and status != "verified":
                if raw_ocr:
                    db.vehicles.update_one(filter_query, {"$push": {"raw_ocr_candidates": raw_ocr}})
                return
            
            # Tie-break: if new confidence is lower, don't overwrite the plate text/images
            if existing.get("confidence", 0.0) > confidence:
                if raw_ocr:
                    db.vehicles.update_one(filter_query, {"$push": {"raw_ocr_candidates": raw_ocr}})
                return

        update_data = {
            "$set": {
                "vehicle_id":        vehicle_id,
                "plate_number":      plate_number,
                "vehicle_type":      vehicle_type,
                "vehicle_type_confidence": float(vehicle_type_confidence),
                "color":             vehicle_color,
                "timestamp":         time_str,
                "frame_number":      int(frame),
                "confidence":        float(confidence),
                "plate_confidence":   float(confidence),
                "vehicle_confidence": float(vehicle_type_confidence),
                "source_id":         source_id,
                "job_status":        job_status,
                "frames_processed":  int(frame),
                "processing_time_seconds": float(processing_time),
                "status":            status,
                "updated_at":        datetime.now(timezone.utc),
            },
            "$addToSet": {
                "violations":        violation_type.upper() if violation_type else "ANPR"
            },
            "$setOnInsert": {
                "job_id":            job_id,
                "detection_id":      str(uuid.uuid4()),
                "created_at":        datetime.now(timezone.utc),
                "violation_type":    violation_type.upper() if violation_type else "ANPR",
            }
        }

        if s3_plate_key:   
            update_data["$set"]["plate_image_url"] = s3_plate_key
            update_data["$set"]["plate_crop_image"] = s3_plate_key
            update_data["$set"]["_s3_plate_key"] = s3_plate_key
        if s3_vehicle_key: 
            update_data["$set"]["vehicle_image_url"] = s3_vehicle_key
            update_data["$set"]["vehicle_image"] = s3_vehicle_key
            update_data["$set"]["_s3_vehicle_key"] = s3_vehicle_key
        if visual_signature: update_data["$set"]["visual_signature"] = visual_signature
        if raw_ocr:        update_data.setdefault("$push", {})["raw_ocr_candidates"] = raw_ocr

        db.vehicles.update_one(filter_query, update_data, upsert=True)
        print(f"✅ Vehicle record synced: {plate_number} | {status}")
        
    except Exception as e:
        print(f"ERROR saving to 'vehicles': {e}")


def calculate_visual_signature(img):
    """
    Creates a visual fingerprint of a vehicle using color histograms.
    Matches are based on color distribution to distinguish between different cars 
    even if the tracker flips IDs.
    """
    if img is None or img.size == 0:
        return None
    
    # 1. Resize for consistency
    img = cv2.resize(img, (128, 128))
    
    # 2. Convert to HSV (more robust to lighting than BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 3. Calculate Histogram for H and S channels (ignoring V for brightness robustness)
    # H: 18 bins, S: 10 bins
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 10], [0, 180, 0, 256])
    
    # 4. Normalize
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    return hist.flatten().tolist()

def compare_signatures(sig1, sig2):
    """Returns correlation between two signatures (0 to 1)."""
    if sig1 is None or sig2 is None: return 0.0
    s1 = np.array(sig1, dtype=np.float32).reshape((18, 10))
    s2 = np.array(sig2, dtype=np.float32).reshape((18, 10))
    return cv2.compareHist(s1, s2, cv2.HISTCMP_CORREL)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea  = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea  = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


# ── Video Processing Task ─────────────────────────────────────────────────────

def process_video_task(job_id: str, input_path: str, output_path: str,
                       case_type: str, s3_video_key: str = None):
    print(f"[JOB {job_id}] Starting background task | type={case_type}")
    
    # 1. Update job status to processing
    if job_id in jobs:
        jobs[job_id]["status"] = "processing"
    update_job_in_db(job_id, "processing", 0)
    
    cap = None
    out = None
    violation_count = 0
    # Track sent alerts to ensure one alert per unique plate per job (URGENT)
    sent_alerts = set()
    
    # NEW: Per-job processed plates set for duplicate filtering
    job_processed_plates = set()
    job_start_time = time.time()
    
    # NEW: Vehicle Type Majority Voting Store
    tid_vtype_history: Dict[int, List[str]] = {} # tid -> list of labels
    pending_vtype_reports: List[Dict] = []       # reports waiting for more votes
    
    try:
        # Select service
        if case_type in ["anpr", "security", "blacklist"]:
            service = ANPR_SERVICE
        elif case_type == "helmet":
            service = HELMET_SERVICE
        elif case_type in ["overload", "triple"]:
            service = OVERLOAD_SERVICE
        elif case_type in ["wrong_side", "wrong_lane"]:
            service = WRONGSIDE_SERVICE
        elif case_type == "stalled":
            service = STALLED_SERVICE
        elif case_type == "seatbelt":
            service = SEATBELT_SERVICE
        else:
            raise Exception(f"Unknown case_type: {case_type}")

        # Reset service state for this unique job
        if hasattr(service, "reset"):
            service.reset()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {input_path}")
            
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sent_alerts = {} 
        # NEW: Zero-Miss Tracking Metadata
        # { tid: { "first_frame": int, "last_frame": int, "reported": bool, "vtype": str, "vconf": float, "box": [] } }
        tracked_meta = {}

        # NEW: Vehicle Type Majority Voting Store
        tid_vtype_history: Dict[int, List[str]] = {} # tid -> list of labels
        pending_vtype_reports: List[Dict] = []       # reports waiting for more votes

        def finalize_and_report_vehicle(p):
            """Helper to process a buffered report with majority voting and finalize it."""
            try:
                # 1. Get majority vote
                tid_inner = p.get("tid")
                votes = tid_vtype_history.get(tid_inner, [])
                voted_type, voted_conf = get_majority_vtype(votes)
                
                # 2. Finalize with sanity checks
                final_v_type_inner = finalize_vtype_detection(
                    p["base_v_type"], voted_type, voted_conf, 
                    p["vehicle_full_box"], w, h
                )

                # Use majority vote for vehicle type (Fix 2)
                vtype_history = tid_vtype_history.get(tid_inner, [])
                if vtype_history:
                    type_counts = {}
                    for t in vtype_history:
                        if t and t.lower() not in ("unknown", "none", ""):
                            type_counts[t] = type_counts.get(t, 0) + 1
                    if type_counts:
                        final_v_type_inner = max(type_counts, key=type_counts.get)

                plate_val_log = p.get("plate", "N/A")
                print(f"[VTYPE FINAL] plate={plate_val_log} | history={vtype_history} | final={final_v_type_inner}")
                
                # 3. Save to DB
                save_detection(
                    job_id=p["job_id"], frame=p["frame"], vehicle_id=str(p["tid"]),
                    plate=p["plate"], confidence=p["confidence"], status=p["status"],
                    violation_type=p["violation_type"], vehicle_type=final_v_type_inner,
                    vehicle_color=p["vehicle_color"], vehicle_type_confidence=voted_conf,
                    s3_vehicle_key=p["full_s3_key"], s3_plate_key=p["crop_s3_key"],
                    fps=p["fps"], visual_signature=p["v_sig"],
                    source_id=p["source_id"], job_status=p.get("job_status", "processing"),
                    processing_time=time.time() - job_start_time
                )
                
                # 4. Add to job stats for report table
                if job_id in jobs:
                    jobs[job_id]["report"].append({
                        "Frame": p["frame"], "VehicleID": str(p["tid"]),
                        "Type": final_v_type_inner, "Plate": p["plate"],
                        "CropImgUrl": get_presigned_url(p["crop_s3_key"]),
                        "FullImgUrl": get_presigned_url(p["full_s3_key"]),
                        "vehicle_image": get_presigned_url(p["full_s3_key"]),
                        "plate_image": get_presigned_url(p["crop_s3_key"]),
                        "_s3_vehicle_key": p["full_s3_key"],
                        "_s3_plate_key": p["crop_s3_key"]
                    })
                
                # 5. Telegram Alert
                plate_val = p.get("plate_cleanup", "N/A")
                is_v_plate = len(plate_val) >= 7 and plate_val != "UNREAD"
                alert_k = f"tid_{p['tid']}" if p.get('tid') else f"plate_{plate_val}"
                
                cur_t = time.time()
                can_snd = True
                if alert_k in sent_alerts:
                    if (cur_t - sent_alerts[alert_k]) < 120: can_snd = False
                
                if is_v_plate and can_snd:
                     sent_alerts[alert_k] = cur_t
                     timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                     combined_p = os.path.join(OUTPUT_DIR, f"alert_{job_id}_{p['frame']}.jpg")
                     if create_combined_image(p["full_local"], p["crop_local"], combined_p):
                         send_local_violation(
                             combined_path=combined_p, plate_text=p["plate"],
                             job_id=job_id, case_type=case_type,
                             violation_count=len(jobs[job_id]["report"]),
                             status="Verified" if p["confidence"] >= 0.2 else "Low Confidence",
                             timestamp=timestamp_now, vehicle_type=final_v_type_inner,
                             confidence=voted_conf
                         )
                return True
            except Exception as ex:
                print(f"⚠️ Failed to finalize report for TID {p.get('tid')}: {ex}")
                return False

        # VideoWriter Setup
        codecs = [('mp4v', 'mp4'), ('XVID', 'avi')]
        final_output_path = output_path
        for codec, ext in codecs:
            try:
                base = os.path.splitext(output_path)[0]
                temp_path = f"{base}.{ext}"
                fourcc = cv2.VideoWriter_fourcc(*codec)
                temp_out = cv2.VideoWriter(temp_path, fourcc, int(fps), (w, h))
                if temp_out.isOpened():
                    out = temp_out
                    output_path = temp_path
                    break
            except Exception:
                continue

        if not out:
            raise Exception("Failed to initialize VideoWriter")

        frame_count = 0
        reported_violations = []
        assets_dir = os.path.join(OUTPUT_DIR, "assets")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if case_type in ["anpr", "security", "blacklist"]:
                # ── Step 1-2: YOLO base detection + ByteTrack for Vehicle ID ──
                with torch.no_grad():
                    _, tracked = service.process_frame(frame)

                # ── NEW: Per-frame Vehicle Type Voting Collection ──
                if tracked is not None and hasattr(tracked, 'tracker_id'):
                    for i in range(len(tracked.tracker_id)):
                        tid = int(tracked.tracker_id[i])
                        vbox = tracked.xyxy[i]
                        
                        # Initialize history if new
                        if tid not in tid_vtype_history:
                            tid_vtype_history[tid] = []
                        
                        # Collect up to 30 votes per vehicle (10-20 frames recommended)
                        if len(tid_vtype_history[tid]) < 30:
                            vx1, vy1, vx2, vy2 = map(int, vbox)
                            vx1, vy1 = max(0, vx1), max(0, vy1)
                            vx2, vy2 = min(w, vx2), min(h, vy2)
                            
                            if vx2 - vx1 > 40 and vy2 - vy1 > 40: # Minimum size for quality
                                v_crop = frame[vy1:vy2, vx1:vx2]
                                
                                def _vote_task(tid_inner, crop_img):
                                    res = VEHICLE_TYPE_SERVICE.get_best_detection(crop_img)
                                    # get_best_detection already returns mapped Indian category ("Car", "Bike", etc.)
                                    # Do NOT remap through VEHICLE_TYPE_MAP — that caused wrong types (double-mapping bug)
                                    vehicle_type = res.get("type", "unknown") if res else "unknown"
                                    if not vehicle_type or vehicle_type.lower() in ("", "none", "unknown"):
                                        vehicle_type = "unknown"
                                    tid_vtype_history[tid_inner].append(vehicle_type)
                                
                                # Run classification in thread pool
                                vtype_executor.submit(_vote_task, tid, v_crop.copy())

                # ── Step 1 (cont): YOLO plate detection (runs at 640px inside anpr_service) ──
                all_detections, to_report = service.run_detection(frame, frame_count)

                # ── NEW: Process Pending Reports ──
                remaining_pending = []
                curr_t_ids = []
                if tracked is not None and hasattr(tracked, 'tracker_id'):
                    curr_t_ids = tracked.tracker_id.tolist()

                for pend in pending_vtype_reports:
                    tid_p = pend["tid"]
                    vts_p = tid_vtype_history.get(tid_p, [])
                    # Finalize if we have 15 votes OR if the vehicle is no longer in frame
                    if len(vts_p) >= 15 or tid_p not in curr_t_ids:
                        finalize_and_report_vehicle(pend)
                        # Sync global violation count
                        if job_id in jobs:
                            violation_count = len(jobs[job_id]["report"])
                    else:
                        remaining_pending.append(pend)
                pending_vtype_reports = remaining_pending

                # ── Step 7 (partial): Draw DIM GREY box on all unconfirmed detections ──
                #    Only confirmed reads get bright green boxes (drawn below in to_report loop)
                for det in all_detections:
                    b = det["box"]
                    if not det.get("confirmed", False):
                        # Dim grey box for unconfirmed / blurry / detecting
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (100, 100, 100), 1)

                for res in to_report:
                    # ── Step 2: Find matching Vehicle ID from ByteTrack ──
                    pcx = (res["box"][0] + res["box"][2]) / 2
                    pcy = (res["box"][1] + res["box"][3]) / 2
                    vehicle_tid = None
                    v_type = "unknown"
                    v_color = "unknown"
                    vehicle_full_box = None  # Full vehicle bbox from ByteTrack

                    if tracked is not None and hasattr(tracked, 'tracker_id') and len(tracked.tracker_id) > 0:
                        min_dist = float('inf')
                        for i in range(len(tracked.tracker_id)):
                            tid = tracked.tracker_id[i]
                            vbox = tracked.xyxy[i]

                            vcx = (vbox[0] + vbox[2]) / 2
                            vcy = (vbox[1] + vbox[3]) / 2
                            dist = ((pcx - vcx)**2 + (pcy - vcy)**2)**0.5

                            if dist < min_dist:
                                min_dist = dist
                                vehicle_tid = tid
                                cls_id = int(tracked.class_id[i])
                                # Fallback base type
                                v_type = VEHICLE_TYPE_MAP.get(cls_id, "unknown")
                                vehicle_full_box = [int(vbox[0]), int(vbox[1]), int(vbox[2]), int(vbox[3])]

                                # ── Improved Vehicle Heuristic ──
                                vw_box = vbox[2] - vbox[0]
                                vh_box = vbox[3] - vbox[1]
                                aspect = vh_box / vw_box if vw_box > 0 else 0
                                vehicle_area_ratio = (vw_box * vh_box) / (w * h) if w * h > 0 else 0

                                if cls_id == 5:  # COCO "bus" — small one is likely Auto-Rickshaw
                                    if vehicle_area_ratio < 0.12 and 0.7 < aspect < 1.4:
                                        v_type = "Auto-Rickshaw"
                                elif cls_id == 7:  # COCO "truck" — small one might be tempo/auto
                                    if vehicle_area_ratio < 0.08 and aspect > 0.6:
                                        v_type = "Auto-Rickshaw"
                                elif cls_id == 3:  # COCO "motorcycle" — wide one is scooter
                                    if vw_box > vh_box * 1.5:
                                        v_type = "Scooter"

                        # Update Tracking Metadata for Zero-Miss
                        if vehicle_tid is not None:
                            if vehicle_tid not in tracked_meta:
                                tracked_meta[vehicle_tid] = {
                                    "first_frame": frame_count, "last_frame": frame_count,
                                    "reported": False, "vtype": v_type, "vconf": 0.0, "box": vehicle_full_box
                                }
                            else:
                                tracked_meta[vehicle_tid]["last_frame"] = frame_count
                                tracked_meta[vehicle_tid]["box"] = vehicle_full_box

                    print(f"🔍 Plate detected: {res['text']} (conf={res['conf']:.2f})")

                    # ── Step 5: Cooldown check — skip if duplicate within 60s ──
                    plate_cleanup = res["text"].strip().upper()
                    current_time = time.time()
                    is_cooldown = False

                    # Skip cooldown for UNREAD entries (they should always be saved)
                    if plate_cleanup != "UNREAD":
                        if plate_cleanup in plate_cooldown_tracker:
                            if (current_time - plate_cooldown_tracker[plate_cleanup]) < PLATE_COOLDOWN_SECONDS:
                                print(f"🚫 [COOLDOWN] Skipping {plate_cleanup} (seen {current_time - plate_cooldown_tracker[plate_cleanup]:.0f}s ago)")
                                is_cooldown = True

                        if not is_cooldown:
                            for tp, tt in list(plate_cooldown_tracker.items()):
                                if (current_time - tt) >= PLATE_COOLDOWN_SECONDS:
                                    del plate_cooldown_tracker[tp]
                                    continue
                                if SequenceMatcher(None, plate_cleanup, tp).ratio() > 0.8:
                                    print(f"🚫 [COOLDOWN] Fuzzy skip: {plate_cleanup} ≈ {tp}")
                                    is_cooldown = True
                                    break

                    if is_cooldown:
                        continue

                    # Record this plate in cooldown tracker (except UNREAD)
                    if plate_cleanup != "UNREAD":
                        plate_cooldown_tracker[plate_cleanup] = current_time

                    # ── Step 4: best.pt on FULL vehicle bbox (in thread) ──
                    # Extract full vehicle image from tracked bbox (min 80×80)
                    print(f"[DEBUG VTYPE] plate={res.get('plate','?')} | vehicle_full_box={vehicle_full_box} | tid={res.get('track_id','?')}")
                    # Vehicle type comes from Block 1 voting only — no reclassification here
                    pass

                    assets_id = uuid.uuid4().hex[:8]
                    full_name  = f"anpr_{assets_id}.jpg"
                    crop_name  = f"anpr_crop_{assets_id}.jpg"
                    full_local = os.path.join(assets_dir, full_name)
                    crop_local = os.path.join(assets_dir, crop_name)
                    full_name  = f"anpr_{assets_id}.jpg"
                    crop_name  = f"anpr_crop_{assets_id}.jpg"
                    full_local = os.path.join(assets_dir, full_name)
                    crop_local = os.path.join(assets_dir, crop_name)

                    b = res["box"]

                    # Visual signature from vehicle region for Re-ID
                    v_reid_box = [max(0, b[0]-50), max(0, b[1]-50), min(w, b[2]+50), min(h, b[3]+50)]
                    v_img = frame[v_reid_box[1]:v_reid_box[3], v_reid_box[0]:v_reid_box[2]]
                    v_sig = calculate_visual_signature(v_img)

                    # ── Step 7: Draw GREEN box ONLY on successfully read vehicles ──
                    highlighted_frame = frame.copy()
                    if not res.get("unread", False):
                        # Bright green box with plate text + Vehicle ID
                        box_color = (0, 255, 0)  # Green
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), box_color, 3)
                        label = f"{res['text']}"
                        if vehicle_tid is not None:
                            label += f" [V{vehicle_tid}]"
                        cv2.putText(frame, label, (b[0], b[1]-10), 0, 0.6, box_color, 2)
                        # Also on highlighted frame for evidence image
                        cv2.rectangle(highlighted_frame, (b[0], b[1]), (b[2], b[3]), box_color, 3)
                        cv2.putText(highlighted_frame, label, (b[0], b[1]-10), 0, 0.6, box_color, 2)
                    else:
                        # UNREAD: dim yellow box
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 200, 255), 2)
                        cv2.rectangle(highlighted_frame, (b[0], b[1]), (b[2], b[3]), (0, 200, 255), 2)

                    cv2.imwrite(full_local, highlighted_frame)

                    # ── Phase 1: High Quality Plate Crop (25% padding to prevent half crops) ──
                    bw, bh_crop = b[2]-b[0], b[3]-b[1]
                    # FIX: Use 50% padding on each side - ensures FULL plate is always captured
                    # even if the YOLO box clips the edge of the plate
                    pad_w = int(bw * 0.50)
                    pad_h = int(bh_crop * 0.50)
                    x1_pad = max(0, b[0] - pad_w)
                    y1_pad = max(0, b[1] - pad_h)
                    x2_pad = min(w, b[2] + pad_w)
                    y2_pad = min(h, b[3] + pad_h)

                    crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                    if crop.size > 0:
                        ch, cw = crop.shape[:2]
                        # Upscale if too small for readability
                        if cw < 300 or ch < 80:
                            new_w = max(300, cw)
                            new_h = int(new_w * (ch / cw)) if cw > 0 else 80
                            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                        cv2.imwrite(crop_local, crop)
                    else:
                        cv2.imwrite(crop_local, frame[b[1]:b[3], b[0]:b[2]])

                    # Upload to S3
                    full_s3_key = s3_upload_image(full_local, f"evidence/{full_name}")
                    crop_s3_key = s3_upload_image(crop_local, f"evidence/{crop_name}")

                    # ── BUFFER REPORT FOR MAJORITY VOTING ──
                    if vehicle_tid is not None:
                        report_pend = {
                            "job_id": job_id, "frame": frame_count, "tid": vehicle_tid,
                            "plate": res["text"], "confidence": res["conf"],
                            "status": "unread" if res.get("unread") else ("verified" if res["conf"] >= 0.2 else "low_confidence"),
                            "violation_type": "ANPR", "vehicle_color": v_color, "v_sig": v_sig, "fps": fps,
                            "full_local": full_local, "crop_local": crop_local,
                            "full_s3_key": full_s3_key, "crop_s3_key": crop_s3_key,
                            "plate_cleanup": plate_cleanup, "base_v_type": v_type,
                            "vehicle_full_box": vehicle_full_box, "source_id": "video_upload"
                        }
                        
                        v_votes_loop = tid_vtype_history.get(vehicle_tid, [])
                        if len(v_votes_loop) >= 15:
                            # Finalize immediately if we already have 15+ votes
                            finalize_and_report_vehicle(report_pend)
                            if job_id in jobs: violation_count = len(jobs[job_id]["report"])
                        else:
                            pending_vtype_reports.append(report_pend)
                            print(f"⏳ [BUFFERED] Plate {res['text']} (TID {vehicle_tid}, votes {len(v_votes_loop)}/15)")
                        
                        # Mark as reported for Zero-Miss coverage
                        if vehicle_tid in tracked_meta:
                            tracked_meta[vehicle_tid]["reported"] = True
                    else:
                        # Fallback for untracked
                        save_detection(
                            job_id=job_id, frame=frame_count, vehicle_id="N/A",
                            plate=res["text"], confidence=res["conf"], 
                            status="unread" if res.get("unread") else "verified",
                            violation_type="ANPR", vehicle_type=v_type,
                            vehicle_color=v_color, vehicle_type_confidence=0.0,
                            s3_vehicle_key=full_s3_key, s3_plate_key=crop_s3_key,
                            fps=fps, visual_signature=v_sig,
                            source_id="video_upload", job_status="processing"
                        )
                        violation_count += 1


            # ── Other violation types ──────────────────────────────────────
            else:
                _, tracked = service.process_frame(frame)
                violations = service.run_detection(frame)

                for v in violations:
                    if not service.should_alert(v["id"], "N/A"):
                        continue
                    
                    is_duplicate = False
                    for rv in reported_violations:
                        if rv["id"] == v["id"] and rv["type"] == v["type"]:
                            is_duplicate = True
                            break
                    if is_duplicate: continue

                    reported_violations.append({
                        "id": v["id"], "box": v["box"],
                        "type": v["type"], "frame": frame_count,
                    })
                    violation_count += 1

                    assets_id = uuid.uuid4().hex[:8]
                    full_name  = f"violation_{assets_id}.jpg"
                    crop_name  = f"violation_crop_{assets_id}.jpg"
                    full_local = os.path.join(assets_dir, full_name)
                    crop_local = os.path.join(assets_dir, crop_name)

                    # Highlight the violation
                    highlighted_frame = frame.copy()
                    b = v["box"]
                    cv2.rectangle(highlighted_frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
                    cv2.putText(highlighted_frame, v["type"], (b[0], b[1]-10), 0, 0.7, (0, 0, 255), 2)
                    cv2.imwrite(full_local, highlighted_frame)

                    crop = frame[max(0, b[1]):min(h, b[3]), max(0, b[0]):min(w, b[2])]
                    if crop.size > 0: cv2.imwrite(crop_local, crop)

                    full_s3_key = s3_upload_image(full_local, f"evidence/{full_name}")
                    crop_s3_key = s3_upload_image(crop_local, f"evidence/{crop_name}")

                    current_processing_time = time.time() - job_start_time
                    # Save to MongoDB
                    save_detection(
                        job_id=job_id,
                        frame=frame_count,
                        vehicle_id=str(v["id"]),
                        plate="VIOLATION",
                        confidence=1.0, 
                        status="verified",
                        violation_type=v["type"],
                        vehicle_type="unknown",
                        vehicle_color="unknown",
                        vehicle_type_confidence=0.0,
                        s3_vehicle_key=full_s3_key,
                        s3_plate_key=crop_s3_key,
                        fps=fps,
                        source_id="video_upload",
                        job_status="processing",
                        processing_time=current_processing_time
                    )

                    if job_id in jobs:
                        jobs[job_id]["report"].append({
                            "Frame": frame_count,
                            "VehicleID": str(v["id"]),
                            "Type": v["type"],
                            "Plate": "VIOLATION",
                            "CropImgUrl": get_presigned_url(crop_s3_key),
                            "FullImgUrl": get_presigned_url(full_s3_key),
                            "vehicle_image": get_presigned_url(full_s3_key),
                            "plate_image": get_presigned_url(crop_s3_key),
                            "_s3_vehicle_key": full_s3_key,
                            "_s3_plate_key": crop_s3_key
                        })

                    # Telegram alert
                    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    combined_path = os.path.join(OUTPUT_DIR, f"alert_{job_id}_{frame_count}.jpg")
                    if create_combined_image(full_local, crop_local, combined_path):
                        send_local_violation(
                            combined_path=combined_path,
                            plate_text="VIOLATION",
                            job_id=job_id,
                            case_type=case_type,
                            violation_count=violation_count,
                            status=v["type"],
                            timestamp=timestamp_now
                        )
                    service.mark_alerted(v["id"], "N/A")

            out.write(frame)

        # ── Step 9-B: Final Flush of Pending Reports ──
        if pending_vtype_reports:
            print(f"🏁 [JOB {job_id}] Video ended. Flushing {len(pending_vtype_reports)} remaining reports...")
            for p_flush in pending_vtype_reports:
                finalize_and_report_vehicle(p_flush)
            pending_vtype_reports.clear()
            if job_id in jobs: violation_count = len(jobs[job_id]["report"])

        # ── Step 9: Post-Processing Zero-Miss Coverage ──
        # Any vehicle tracked for > 15 frames that was NEVER reported is forced as UNREAD
        for tid, meta in tracked_meta.items():
            if not meta["reported"] and (meta["last_frame"] - meta["first_frame"]) > 15:
                print(f"📦 [ZERO-MISS] Forcing report for TrackID {tid} (No plate detected)")
                # Force report so it shows in the table even without a plate
                save_detection(
                    job_id=job_id, frame=meta["last_frame"], vehicle_id=str(tid),
                    plate="UNREAD", confidence=0.0, status="unread",
                    violation_type="ANPR", vehicle_type=meta["vtype"].title(),
                    vehicle_type_confidence=0.0,
                    source_id="zero_miss_fallback"
                )

        # Finalize job on success
        current_processing_time = time.time() - job_start_time
        update_job_in_db(job_id, "completed", violation_count)
        # Update metadata for all detections of this job
        if db is not None:
            db.vehicles.update_many(
                {"job_id": job_id},
                {"$set": {
                    "job_status": "completed",
                    "processing_time_seconds": current_processing_time
                }}
            )
        if job_id in jobs:
            jobs[job_id]["status"] = "completed"
        print(f"✅ Job completed successfully: {job_id}")

    except Exception as e:
        import traceback
        err_msg = str(e)
        print(f"❌ Job failed: {job_id} | Error: {err_msg}\n{traceback.format_exc()}")
        update_job_in_db(job_id, "failed", violation_count, error_message=err_msg)
        if db is not None:
            db.vehicles.update_many(
                {"job_id": job_id},
                {"$set": {"job_status": "failed", "error_message": err_msg}}
            )
        if job_id in jobs:
            jobs[job_id]["status"] = "error"
    finally:
        if cap: cap.release()
        if out: out.release()

# ── NEW API Endpoints ─────────────────────────────────────────────────────────

@app.get("/traffic-analytics")
def get_traffic_analytics(job_id: str = None):
    """
    Returns aggregated traffic and violation statistics.
    If job_id is provided, filters by that specific video.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    
    query = {}
    if job_id:
        query["job_id"] = job_id
        
    try:
        # 1. Total Vehicles (Count documents in 'vehicles' collection)
        total_vehicles = db.vehicles.count_documents(query)
        
        # 2. Violation Counts
        helmet_query = query.copy()
        helmet_query["violations"] = "HELMET"
        total_helmet = db.vehicles.count_documents(helmet_query)
        
        seatbelt_query = query.copy()
        seatbelt_query["violations"] = "SEATBELT"
        total_seatbelt = db.vehicles.count_documents(seatbelt_query)
        
        # 3. Vehicle Type Counts (all common Indian vehicle types)
        vtypes = ["car", "bike", "truck", "scooter", "bus", "auto-rickshaw"]
        type_counts = {}
        for vt in vtypes:
            type_query = query.copy()
            type_query["vehicle_type"] = {"$regex": f"^{re.escape(vt)}$", "$options": "i"}
            type_counts[vt] = db.vehicles.count_documents(type_query)

        # 4. More violation types
        wrong_side_query = query.copy()
        wrong_side_query["violations"] = "WRONG_SIDE"
        total_wrong_side = db.vehicles.count_documents(wrong_side_query)

        triple_query = query.copy()
        triple_query["violations"] = {"$regex": "TRIPLE", "$options": "i"}
        total_triple = db.vehicles.count_documents(triple_query)

        return {
            "total_vehicles": total_vehicles,
            "violations": {
                "helmet": total_helmet,
                "seatbelt": total_seatbelt,
                "wrong_side": total_wrong_side,
                "triple_riding": total_triple,
            },
            "vehicle_types": type_counts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")


@app.get("/search-plate")
def search_plate_endpoint(number: str):
    """
    Returns all violations and metadata for a specific plate number.
    """
    if not number:
        raise HTTPException(status_code=400, detail="Plate number required")
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
        
    try:
        query = {"plate_number": number.strip().upper()}
        results = list(db.vehicles.find(query).sort("created_at", -1))
        
        formatted_results = []
        for r in results:
            formatted_results.append({
                "plate_number":   r.get("plate_number"),
                "vehicle_type":   r.get("vehicle_type"),
                "violation_type": r.get("violation_type", "ANPR"),
                "violations":     r.get("violations", []),
                "timestamp":      r.get("timestamp"),
                "created_at":     r.get("created_at"),
                "vehicle_image":  get_presigned_url(r.get("vehicle_image_url")),
                "plate_crop_image": get_presigned_url(r.get("plate_image_url")),
                "confidence":     r.get("confidence", 0.0),
                "source_id":      r.get("source_id", "unknown")
            })
            
        return formatted_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

    finally:
        if cap: cap.release()
        if out: out.release()
        print(f"Resources released for job: {job_id}")


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/{case_type}")
async def start_job(case_type: str, background_tasks: BackgroundTasks,
                    file: UploadFile = File(...)):
    check_dirs()
    job_id     = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = os.path.join(OUTPUT_DIR, f"output_{job_id}_{file.filename}")

    # Upload original video to S3 — store KEY
    s3_video_key = None
    try:
        s3_video_key = s3_upload_video(input_path, f"{job_id}_{file.filename}")
    except Exception as e:
        print(f"WARNING: S3 video upload failed: {e}")

    # Create job in MongoDB with S3 key
    create_job_in_db(job_id, case_type, s3_video_key or "")

    jobs[job_id] = {
        "job_id":        job_id,
        "status":        "pending",
        "case_type":     case_type,
        "report":        [],
        "s3_video_key":  s3_video_key,
    }

    background_tasks.add_task(
        process_video_task, job_id, input_path, output_path, case_type, s3_video_key
    )
    return {"job_id": job_id}


@app.post("/upload")
async def legacy_upload(background_tasks: BackgroundTasks,
                        file: UploadFile = File(...),
                        case_type: str = Form("anpr")):
    return await start_job(case_type, background_tasks, file)


@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id in jobs:
        return jobs[job_id]
    
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    
    # Fallback to DB
    job_db = db.jobs.find_one({"job_id": job_id})
    if job_db:
        del job_db["_id"]
        return job_db
        
    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/report/{job_id}")
def get_report(job_id: str, verified_only: bool = True):
    """
    Returns the violation report with correct keys for frontend and fresh presigned URLs.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection not available")

    query = {"job_id": job_id}
    if verified_only:
        # Include both verified AND unread (vehicles detected but plate not read)
        query["status"] = {"$in": ["verified", "unread"]}
    
    raw_results = list(db.vehicles.find(query).sort("frame_number", 1))
    
    # Map MongoDB fields to what the frontend expects
    report_data = []
    for r in raw_results:
        plate_val = r.get("plate_number") or "N/A"
        # Show UNKNOWN for unread plates (clearer than N/A)
        if plate_val == "N/A" and r.get("status") == "unread":
            plate_val = "UNKNOWN"
        
        report_data.append({
            "Frame":            r.get("frame_number", 0),
            "Timestamp":        r.get("timestamp", "00:00:00"),
            "VehicleID":        r.get("track_id") or r.get("vehicle_id") or str(r.get("_id", ""))[:6] or "N/A",
            "Type":             r.get("vehicle_type") or r.get("class") or "unknown",
            "TypeConfidence":   r.get("vehicle_type_confidence", 0.0),
            "Plate":            plate_val,
            "Status":           r.get("status", "low_confidence"),
            "Confidence":       r.get("confidence", 0.0),
            "Violation":        r.get("violations")[0] if r.get("violations") else "ANPR",
            "_s3_vehicle_key":  r.get("vehicle_image_url"),
            "_s3_plate_key":    r.get("plate_image_url"),
        })
    
    # Enrich with presigned URLs
    enriched = generate_presigned_urls_for_report(report_data)
    return enriched

@app.post("/api/detect-vehicle-type")
async def api_detect_vehicle_type(file: UploadFile = File(...)):
    """
    Test endpoint for vehicle type detection.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")
        
    detections = VEHICLE_TYPE_SERVICE.detect_vehicle_type(frame)
    return {"detections": detections}

@app.get("/ping")
def ping():
    return {"status": "ok", "message": "API is alive"}

@app.get("/rag-data/all")
@app.get("/api/rag-data/all")
def get_all_rag_data(limit: int = 500):
    """
    Export ALL detection data across ALL jobs for RAG ingestion.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection not available")

    detections = list(db.vehicles.find({}).sort("created_at", -1).limit(limit))
    for det in detections:
        det["id"] = str(det["_id"])
        del det["_id"]
    return {
        "description": "Consolidated data for RAG chatbot",
        "total_returned": len(detections),
        "data": detections
    }

@app.get("/rag-data/{job_id}")
@app.get("/api/rag-data/{job_id}")
def get_rag_data(job_id: str):
    """
    Export all extracted data for RAG ingestion.
    Includes both verified and low-confidence detections.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    detections = list(db.vehicles.find({"job_id": job_id}).sort("frame_number", 1))
    for det in detections:
        det["id"] = str(det["_id"])
        del det["_id"]
    return {
        "job_id": job_id,
        "total_detections": len(detections),
        "data": detections
    }

@app.get("/api/report/{job_id}/pdf")
def get_pdf_report(job_id: str):
    """
    Generate and return a PDF report of all detections.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    detections = list(db.vehicles.find({"job_id": job_id}).sort("frame_number", 1))
    if not detections:
        raise HTTPException(status_code=404, detail="No data found for this job")
    
    pdf_filename = f"report_{job_id}.pdf"
    pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
    
    try:
        generate_violation_pdf(job_id, detections, pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")
        
    # Build a simple URL or use a file response? 
    # For now, let's just return the URL to the file in /outputs
    return {
        "pdf_url": f"/outputs/{pdf_filename}",
        "job_id": job_id
    }



@app.get("/presigned")
def get_presigned_endpoint(key: str, expiry: int = 3600):
    """
    Generate a fresh presigned URL for any S3 key.
    Usage: GET /presigned?key=evidence/abc.jpg&expiry=3600
    """
    if not key:
        raise HTTPException(status_code=400, detail="key parameter is required")
    url = get_presigned_url(key, expires_in=expiry)
    if not url:
        raise HTTPException(status_code=500, detail="Failed to generate presigned URL")
    return {"key": key, "url": url, "expires_in_seconds": expiry}


@app.get("/report/{job_id}/refresh")
def refresh_report_urls(job_id: str):
    """
    Force-refresh all presigned URLs in a report.
    Call this if URLs have expired (after 1 hour).
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    raw_report = jobs[job_id].get("report", [])
    # Use the robust helper from s3_service
    refreshed  = generate_presigned_urls_for_report(raw_report)
    return refreshed


@app.get("/api/search/plate")
def search_by_plate(plate: str, include_low_confidence: bool = False):
    if not plate: raise HTTPException(400, "Plate required")
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    query = {"plate_number": plate.strip().upper()}
    if not include_low_confidence:
        query["status"] = "verified"
    
    results = list(db.vehicles.find(query).sort("created_at", -1).limit(50))
    for r in results:
        r["id"] = str(r["_id"])
        del r["_id"]
        if r.get("vehicle_image_url"):
            r["vehicle_image_url"] = get_presigned_url(r["vehicle_image_url"])
        if r.get("plate_image_url"):
            r["plate_image_url"] = get_presigned_url(r["plate_image_url"])
    return results

@app.get("/api/search/vehicles")
def search_vehicles(color: str = None, type: str = None, video_id: str = None, 
                   page: int = 1, limit: int = 20,
                   include_low_confidence: bool = False):
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    query = {}
    if color: query["color"] = color.lower()
    if type: query["vehicle_type"] = type.lower()
    if video_id: query["job_id"] = video_id
    if not include_low_confidence:
        query["status"] = "verified"

    skip = (page - 1) * limit
    results = list(db.vehicles.find(query).sort("created_at", -1).skip(skip).limit(limit))
    for r in results:
        r["id"] = str(r["_id"])
        del r["_id"]
        if r.get("vehicle_image_url"):
            r["vehicle_image_url"] = get_presigned_url(r["vehicle_image_url"])
        if r.get("plate_image_url"):
            r["plate_image_url"] = get_presigned_url(r["plate_image_url"])
    return {
        "results": results,
        "page": page,
        "limit": limit,
        "total": db.vehicles.count_documents(query)
    }

@app.get("/api/violations")
def get_violations(type: str = None, page: int = 1, limit: int = 20,
                   include_low_confidence: bool = False, video_id: str = None):
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    query = {}
    if type: query["violations"] = type.upper()
    if video_id: query["job_id"] = video_id  # Filter by specific job/video
    if not include_low_confidence:
        query["status"] = "verified"
        
    skip = (page - 1) * limit
    results = list(db.vehicles.find(query).sort("created_at", -1).skip(skip).limit(limit))
    for r in results:
        r["id"] = str(r["_id"])
        del r["_id"]
        if r.get("vehicle_image_url"):
            r["vehicle_image_url"] = get_presigned_url(r["vehicle_image_url"])
        if r.get("plate_image_url"):
            r["plate_image_url"] = get_presigned_url(r["plate_image_url"])
    return {
        "results": results,
        "page": page,
        "limit": limit,
        "total": db.vehicles.count_documents(query)
    }

@app.get("/api/search/time")
def search_by_time(from_time: str = None, to_time: str = None, fuzzy: str = None, include_low_confidence: bool = False):
    # fuzzy examples: "today", "last night", "yesterday"
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    query = {}
    now = datetime.now(timezone.utc)
    
    if not include_low_confidence:
        query["status"] = "verified"

    if fuzzy:
        fuzzy = fuzzy.lower()
        if "today" in fuzzy:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            query["created_at"] = {"$gte": start}
        elif "yesterday" in fuzzy:
            end = now.replace(hour=0, minute=0, second=0, microsecond=0)
            from datetime import timedelta
            start = end - timedelta(days=1)
            query["created_at"] = {"$gte": start, "$lt": end}
        elif "last night" in fuzzy:
            end = now.replace(hour=6, minute=0, second=0, microsecond=0)
            from datetime import timedelta
            start = (end - timedelta(days=1)).replace(hour=20, minute=0, second=0, microsecond=0)
            query["created_at"] = {"$gte": start, "$lt": end}
            
    elif from_time or to_time:
        time_query = {}
        if from_time: time_query["$gte"] = datetime.fromisoformat(from_time)
        if to_time: time_query["$lte"] = datetime.fromisoformat(to_time)
        query["created_at"] = time_query

    results = list(db.vehicles.find(query).sort("created_at", -1).limit(50))
    for r in results:
        r["id"] = str(r["_id"])
        del r["_id"]
        if r.get("vehicle_image_url"):
            r["vehicle_image_url"] = get_presigned_url(r["vehicle_image_url"])
        if r.get("plate_image_url"):
            r["plate_image_url"] = get_presigned_url(r["plate_image_url"])
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
