import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime, timezone

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
import re
import contextlib
from typing import Dict, List
from difflib import SequenceMatcher

from services.anpr_service import ANPRService
from services.helmet_service import HelmetService
from services.overload_service import OverloadService
from services.wrong_side_service import WrongSideService
from services.stalled_service import StalledService
from services.seatbelt_service import SeatbeltService
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

# Fixed COCO mapping: 0=pedestrian, 1=cycle, 2=car, 3=bike, 5=bus, 7=truck (yolov8/11 standard)
VEHICLE_TYPE_MAP = {
    0: "pedestrian",
    1: "cycle",
    2: "car",
    3: "bike",
    5: "bus",
    7: "truck",
    "auto": "auto" # for custom models
}
VEHICLE_COLORS = ["red", "blue", "white", "black", "silver", "unknown"]

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
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client["ai_traffic_db"]
        print("✅ MongoDB connected successfully")
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

print("✅ All models loaded.")


# ── Directory helpers ─────────────────────────────────────────────────────────

def check_dirs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "assets"), exist_ok=True)

check_dirs()
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

jobs: Dict[str, Dict] = {}


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
                   vehicle_color: str = "unknown", s3_vehicle_key: str = None, 
                   s3_plate_key: str = None, fps: float = 30.0, raw_ocr: str = None,
                   visual_signature: List[float] = None):
    """
    Phase 2: Save to 'vehicles' collection with Plate-based De-duplication.
    Strict Schema: job_id, plate_number, vehicle_type, color, timestamp, frame_number, violations, images.
    """
    if db is None: return
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
                "color":             vehicle_color,
                "timestamp":         time_str,
                "frame_number":      int(frame),
                "confidence":        float(confidence),
                "status":            status,
                "updated_at":        datetime.now(timezone.utc),
            },
            "$addToSet": {
                "violations":        violation_type.upper() if violation_type else "ANPR"
            },
            "$setOnInsert": {
                "job_id":            job_id,
                "created_at":        datetime.now(timezone.utc),
            }
        }

        if s3_plate_key:   update_data["$set"]["plate_image_url"]   = s3_plate_key
        if s3_vehicle_key: update_data["$set"]["vehicle_image_url"] = s3_vehicle_key
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
                # Get tracked objects for vehicle matching first
                _, tracked = service.process_frame(frame)
                
                all_detections, to_report = service.run_detection(frame, frame_count)
                
                # Draw raw detections for debugging/output video
                for det in all_detections:
                    b = det["box"]
                    text = f"{det['text']} ({det['conf']:.2f})"
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
                    cv2.putText(frame, text, (b[0], b[1]-10), 0, 0.5, (255, 0, 0), 2)

                for res in to_report:
                    # Find matching vehicle ID for mapping
                    pcx = (res["box"][0] + res["box"][2]) / 2
                    pcy = (res["box"][1] + res["box"][3]) / 2
                    vehicle_tid = None
                    v_type = "unknown"
                    v_color = "unknown"
                    
                    if tracked is not None and hasattr(tracked, 'tracker_id'):
                        for i in range(len(tracked.tracker_id)):
                            tid = tracked.tracker_id[i]
                            vbox = tracked.xyxy[i]
                            if vbox[0] <= pcx <= vbox[2] and vbox[1] <= pcy <= vbox[3]:
                                vehicle_tid = tid
                                cls_id = int(tracked.class_id[i])
                                v_type = VEHICLE_TYPE_MAP.get(cls_id, "unknown")
                                break

                    print(f"🔍 Plate detected: {res['text']} (conf={res['conf']:.2f})")
                    
                    assets_id = uuid.uuid4().hex[:8]
                    full_name  = f"anpr_{assets_id}.jpg"
                    crop_name  = f"anpr_crop_{assets_id}.jpg"
                    full_local = os.path.join(assets_dir, full_name)
                    crop_local = os.path.join(assets_dir, crop_name)

                    # Save Full Vehicle Image
                    highlighted_frame = frame.copy()
                    b = res["box"]
                    # Calculate visual signature from vehicle region if possible, otherwise plate
                    # In ANPR mode, we use a larger harvest around the plate for visual Re-ID
                    v_reid_box = [max(0, b[0]-50), max(0, b[1]-50), min(w, b[2]+50), min(h, b[3]+50)]
                    v_img = frame[v_reid_box[1]:v_reid_box[3], v_reid_box[0]:v_reid_box[2]]
                    v_sig = calculate_visual_signature(v_img)

                    cv2.rectangle(highlighted_frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
                    cv2.imwrite(full_local, highlighted_frame)
                    
                    # ── Phase 1: High Quality Plate Crop (Intelligent Padding + Sharpening) ──
                    bw, bh_crop = b[2]-b[0], b[3]-b[1]
                    # Increase padding for better visual context but keep it tight enough
                    pad_w, pad_h = int(bw * 0.15), int(bh_crop * 0.15)
                    x1_pad = max(0, b[0] - pad_w)
                    y1_pad = max(0, b[1] - pad_h)
                    x2_pad = min(w, b[2] + pad_w)
                    y2_pad = min(h, b[3] + pad_h)
                    
                    crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                    if crop.size > 0:
                        # Sharpening for visual clarity
                        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                        crop = cv2.filter2D(crop, -1, kernel)
                        
                        ch, cw = crop.shape[:2]
                        if cw < 300 or ch < 100:
                            # Upscale for better visibility in report
                            new_w = max(300, cw)
                            new_h = int(new_w * (ch/cw))
                            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(crop_local, crop)
                    else:
                        cv2.imwrite(crop_local, frame[b[1]:b[3], b[0]:b[2]])

                    # Upload to S3
                    full_s3_key = s3_upload_image(full_local, f"evidence/{full_name}")
                    crop_s3_key = s3_upload_image(crop_local, f"evidence/{crop_name}")

                    # Phase 1: Relaxed Gate 0.85
                    # Phase 1: Relaxed Gate 0.2 (URGENT)
                    status = "verified" if res["conf"] >= 0.2 else "low_confidence"
                    
                    plate_cleanup = res["text"].strip().upper()
                    
                    # ── SAVING LOGIC ──
                    if res.get("verified_candidate"):
                        # Save result
                        save_detection(
                            job_id=job_id,
                            frame=frame_count,
                            vehicle_id=str(vehicle_tid) if vehicle_tid is not None else "N/A",
                            plate=res["text"],
                            confidence=res["conf"],
                            status=status,
                            violation_type="ANPR",
                            vehicle_type=v_type,
                            vehicle_color=v_color,
                            s3_vehicle_key=full_s3_key,
                            s3_plate_key=crop_s3_key,
                            fps=fps,
                            visual_signature=v_sig
                        )
                        
                        violation_count += 1
                        
                        # ── TELEGRAM ALERT LOGIC (Unique per Tracking ID or Plate - UNREPEATED) ──
                        # Fuzzy check for existing alerts to prevent duplicates on ID switches
                        plate_cleanup = res["text"].strip().upper()
                        is_duplicate_alert = False
                        
                        # 1. Direct ID/Exact Plate check
                        alert_id_primary = str(vehicle_tid) if vehicle_tid is not None else f"plate_{plate_cleanup}"
                        if alert_id_primary in sent_alerts:
                            is_duplicate_alert = True
                        
                        # 2. Fuzzy Plate check against already sent alerts
                        if not is_duplicate_alert:
                            for sa in sent_alerts:
                                if sa.startswith("plate_"):
                                    sa_plate = sa.replace("plate_", "")
                                    if SequenceMatcher(None, plate_cleanup, sa_plate).ratio() > 0.8:
                                        print(f"🚫 [SKIP] Fuzzy Duplicate Alert: {plate_cleanup} matches {sa_plate}")
                                        is_duplicate_alert = True
                                        break
                        
                        is_valid_plate = len(plate_cleanup) >= 8 # More lenient length for recall
                        
                        if not is_duplicate_alert and is_valid_plate:
                            sent_alerts.add(alert_id_primary)
                            sent_alerts.add(f"plate_{plate_cleanup}") # Track both for stability
                            
                            timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            combined_path = os.path.join(OUTPUT_DIR, f"alert_{job_id}_{frame_count}.jpg")
                            
                            if create_combined_image(full_local, crop_local, combined_path):
                                send_local_violation(
                                    combined_path=combined_path,
                                    plate_text=res["text"],
                                    job_id=job_id,
                                    case_type=case_type,
                                    violation_count=violation_count,
                                    status="Verified" if res["conf"] >= 0.2 else "Low Confidence",
                                    timestamp=timestamp_now
                                )
                                print(f"✈️ Telegram alert sent for unique detection: {res['text']} (conf={res['conf']:.2f})")

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
                        s3_vehicle_key=full_s3_key,
                        s3_plate_key=crop_s3_key,
                        fps=fps
                    )

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

        # Finalize job on success
        update_job_in_db(job_id, "completed", violation_count)
        if job_id in jobs:
            jobs[job_id]["status"] = "completed"
        print(f"✅ Job completed successfully: {job_id}")

    except Exception as e:
        import traceback
        err_msg = str(e)
        print(f"❌ Job failed: {job_id} | Error: {err_msg}\n{traceback.format_exc()}")
        update_job_in_db(job_id, "failed", violation_count, error_message=err_msg)
        if job_id in jobs:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = err_msg

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
    query = {"job_id": job_id}
    if verified_only:
        query["status"] = "verified"
    
    raw_results = list(db.vehicles.find(query).sort("frame_number", 1))
    
    # Map MongoDB fields to what the frontend expects
    report_data = []
    for r in raw_results:
        report_data.append({
            "Frame":            r.get("frame_number", 0),
            "Timestamp":        r.get("timestamp", "00:00:00"),
            "VehicleID":        r.get("vehicle_id", "N/A"),
            "Type":             r.get("vehicle_type", "unknown"),
            "Plate":            r.get("plate_number") or "N/A",
            "Status":           r.get("status", "low_confidence"),
            "Confidence":       r.get("confidence", 0.0),
            "Violation":        r.get("violations")[0] if r.get("violations") else "ANPR",
            "_s3_vehicle_key":  r.get("vehicle_image_url"),
            "_s3_plate_key":    r.get("plate_image_url"),
        })
    
    # Enrich with presigned URLs
    enriched = generate_presigned_urls_for_report(report_data)
    return enriched

@app.get("/ping")
def ping():
    return {"status": "ok", "message": "API is alive"}

@app.get("/rag-data/all")
@app.get("/api/rag-data/all")
def get_all_rag_data(limit: int = 500):
    """
    Export ALL detection data across ALL jobs for RAG ingestion.
    """
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
def search_vehicles(color: str = None, type: str = None, video_id: str = None, include_low_confidence: bool = False):
    query = {}
    if color: query["color"] = color.lower()
    if type: query["vehicle_type"] = type.lower()
    if video_id: query["job_id"] = video_id
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

@app.get("/api/violations")
def get_violations(type: str = None, include_low_confidence: bool = False):
    query = {}
    if type: query["violations"] = type.upper()
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

@app.get("/api/search/time")
def search_by_time(from_time: str = None, to_time: str = None, fuzzy: str = None, include_low_confidence: bool = False):
    # fuzzy examples: "today", "last night", "yesterday"
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
