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
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]   = "python"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"]    = "True"
os.environ["GLOG_minloglevel"]                         = "2"

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import cv2
from typing import Dict, List

from services.anpr_service import ANPRService
from services.helmet_service import HelmetService
from services.overload_service import OverloadService
from services.wrong_side_service import WrongSideService
from services.stalled_service import StalledService
from services.seatbelt_service import SeatbeltService
from services.s3_service import (
    upload_video as s3_upload_video,
    upload_image as s3_upload_image,
    get_presigned_url,                   # Updated name
    generate_presigned_urls_for_report,
)
from services.telegram_service import send_local_violation, create_combined_image
from shapely.geometry import Polygon

app = FastAPI(title="Smart Traffic AI - Modular API")

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

# ── MongoDB ────────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI")
db = None

if not MONGO_URI:
    print("CRITICAL ERROR: MONGO_URI not found in environment variables.")
else:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client["ai_traffic_db"]
        print("✅ MongoDB connected successfully")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to connect to MongoDB. Error: {e}")

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
            "case_type":     case_type,
            "s3_video_key":  s3_video_key,   # ← S3 key stored here
            "status":        "pending",
            "violation_count": 0,
            "created_at":    datetime.now(timezone.utc),
        })
    except Exception as e:
        print(f"ERROR creating job in MongoDB: {e}")


def update_job_in_db(job_id: str, status: str, violation_count: int):
    if db is None:
        return
    try:
        db.jobs.update_one(
            {"job_id": job_id},
            {"$set": {
                "status":          status,
                "violation_count": violation_count,
                "completed_at":    datetime.now(timezone.utc),
            }}
        )
    except Exception as e:
        print(f"ERROR updating job in MongoDB: {e}")


def save_violation(job_id: str, frame: int, vehicle_id: str, cls: str, plate: str,
                   s3_vehicle_key: str = None, s3_plate_key: str = None,
                   local_evidence_url: str = None):
    """
    Save violation to MongoDB.
    Stores S3 keys (not URLs) for vehicle_image and plate_image.
    local_evidence_url is the fallback /outputs/... path.
    """
    if db is None:
        return
    try:
        db.violations.insert_one({
            "job_id":          job_id,
            "frame":           frame,
            "vehicle_id":      vehicle_id,
            "class":           cls,
            "plate":           plate,
            "s3_vehicle_key":  s3_vehicle_key,   # ← S3 key
            "s3_plate_key":    s3_plate_key,      # ← S3 key
            "local_evidence":  local_evidence_url,
            "created_at":      datetime.now(timezone.utc),
        })
    except Exception as e:
        print(f"ERROR saving violation to MongoDB: {e}")


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
    try:
        print(f">>> Job started: {job_id} | type={case_type}")
        jobs[job_id]["status"] = "processing"

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
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        codecs = [('mp4v', 'mp4'), ('XVID', 'avi'), ('MJPG', 'avi'), ('DIVX', 'avi')]
        out = None
        final_output_path = output_path

        for codec, ext in codecs:
            try:
                current_ext = final_output_path.split('.')[-1]
                if ext != current_ext:
                    base = os.path.splitext(output_path)[0]
                    final_output_path = f"{base}.{ext}"
                fourcc   = cv2.VideoWriter_fourcc(*codec)
                temp_out = cv2.VideoWriter(final_output_path, fourcc, fps, (w, h))
                if temp_out.isOpened():
                    print(f"VideoWriter: codec={codec} → {final_output_path}")
                    out         = temp_out
                    output_path = final_output_path
                    break
            except Exception as e:
                print(f"Codec {codec} failed: {e}")

        if not out or not out.isOpened():
            raise Exception("Failed to initialize VideoWriter with any codec")

        frame_count         = 0
        reported_violations = []
        assets_dir          = os.path.join(OUTPUT_DIR, "assets")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # ── ANPR / Security / Blacklist ────────────────────────────────
            if case_type in ["anpr", "security", "blacklist"]:
                _, tracked = service.process_frame(frame)
                
                # Update stability early in the loop
                if hasattr(service, "update_stability"):
                    service.update_stability(tracked.tracker_id)
                
                results    = service.run_detection(frame, frame_count)

                for res in results:
                    is_alert = res.get("alert", False)
                    if case_type in ["security", "blacklist"] and not is_alert:
                        continue

                    pcx   = (res["box"][0] + res["box"][2]) / 2
                    pcy   = (res["box"][1] + res["box"][3]) / 2
                    owner = "N/A"
                    vehicle_tid = None
                    for tid, vbox in zip(tracked.tracker_id, tracked.xyxy):
                        if vbox[0] <= pcx <= vbox[2] and vbox[1] <= pcy <= vbox[3]:
                            owner = f"V-{tid}"
                            vehicle_tid = tid
                            break

                    if res.get("is_new", True):
                        # Apply de-duplication and stability rules BEFORE saving and alerting
                        if vehicle_tid is not None:
                            if not service.should_alert(vehicle_tid, res["text"]):
                                continue
                        else:
                            # Skip if no track_id associated (requirement for track-based de-dup)
                            print(f"   [SKIP] Plate {res['text']} detected but no associated track ID found.")
                            continue
                        msg_type = "SECURITY ALERT" if is_alert else "ANPR"

                        if is_alert:
                            b = res["box"]
                            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                            cv2.putText(frame, f"{res['text']} [ALERT]",
                                        (b[0], b[1]-10), 0, 0.6, (0, 0, 255), 2)

                        check_dirs()
                        assets_id = uuid.uuid4().hex[:8]
                        full_name  = f"anpr_{assets_id}.jpg"
                        crop_name  = f"anpr_crop_{assets_id}.jpg"
                        full_local = os.path.join(assets_dir, full_name)
                        crop_local = os.path.join(assets_dir, crop_name)

                        # Use a clean copy of the frame for the alert background
                        # to ensure no other vehicle boxes or text are drawn on it.
                        clean_frame = frame.copy()
                        cv2.imwrite(full_local, clean_frame)

                        b = res["box"]
                        x1, y1, x2, y2 = b
                        crop = clean_frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if crop.size > 0:
                            cv2.imwrite(crop_local, crop)

                        # Upload to S3 — get KEYS back
                        full_s3_key = s3_upload_image(full_local, f"evidence/{full_name}")
                        crop_s3_key = s3_upload_image(crop_local, f"evidence/{crop_name}")

                        # Local fallback URLs
                        full_local_url = f"/outputs/assets/{full_name}"
                        crop_local_url = f"/outputs/assets/{crop_name}"

                        # Generate presigned URLs for current report display
                        full_presigned = get_presigned_url(full_s3_key) if full_s3_key else full_local_url
                        crop_presigned = get_presigned_url(crop_s3_key) if crop_s3_key else crop_local_url

                        formatted = {
                            "Frame":            int(frame_count),
                            "VehicleID":        owner,
                            "Type":             msg_type,
                            "Plate":            res["text"],
                            "extracted_number": res["text"],
                            # Presigned URLs for frontend display
                            "vehicle_image":    full_presigned,
                            "plate_image":      crop_presigned,
                            "FullImgUrl":       full_presigned,
                            "CropImgUrl":       crop_presigned,
                            # S3 keys stored for re-generating URLs later
                            "_s3_vehicle_key":  full_s3_key,
                            "_s3_plate_key":    crop_s3_key,
                        }
                        jobs[job_id]["report"].append(formatted)

                        # ── Inline Telegram Alert (Synchronized) ──
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        violation_number = len(jobs[job_id]["report"])
                        combined_path = os.path.join(OUTPUT_DIR, f"alert_{job_id}_{violation_number}.jpg")
                        
                        print(f"🎨 Composing PIP alert for Job {job_id} [ANPR]...")
                        if create_combined_image(full_local, crop_local, combined_path):
                            send_local_violation(
                                combined_path=combined_path,
                                plate_text=res["text"],
                                job_id=job_id,
                                case_type=case_type,
                                violation_count=len(jobs[job_id]["report"]),
                                status="Processing",
                                timestamp=timestamp
                            )
                            # Mark as alerted to prevent duplicates
                            service.mark_alerted(vehicle_tid, res["text"])

                        # MongoDB — store S3 keys
                        save_violation(
                            job_id=job_id,
                            frame=int(frame_count),
                            vehicle_id=str(owner),
                            cls=str(msg_type),
                            plate=str(res["text"]),
                            s3_vehicle_key=full_s3_key,
                            s3_plate_key=crop_s3_key,
                            local_evidence_url=full_local_url,
                        )

                    if not res.get("alert", False):
                        b     = res["box"]
                        color = (0, 255, 0) if res.get("is_new", True) else (0, 255, 255)
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
                        cv2.putText(frame, res["text"], (b[0], b[1]-10), 0, 0.6, color, 2)

            # ── Other violation types ──────────────────────────────────────
            else:
                _, tracked = service.process_frame(frame)
                if hasattr(service, "update_stability"):
                    service.update_stability(tracked.tracker_id)

                violations = service.run_detection(frame)

                for v in violations:
                    # De-duplication check for other violations
                    # (Note: many services return track_id in v["id"])
                    if not service.should_alert(v["id"], "N/A"):
                        continue
                    is_duplicate = False
                    for rv in reported_violations:
                        if rv["id"] == v["id"] and rv["type"] == v["type"]:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        for rv in reported_violations:
                            if rv["type"] == v["type"]:
                                iou = calculate_iou(v["box"], rv["box"])
                                if iou > 0.5 and frame_count - rv["frame"] < 1000:
                                    is_duplicate = True
                                    break
                    if is_duplicate:
                        continue

                    reported_violations.append({
                        "id": v["id"], "box": v["box"],
                        "type": v["type"], "frame": frame_count,
                    })

                    b = v["box"]
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cv2.putText(frame, v["type"], (b[0], b[1]-10), 0, 0.6, (0, 0, 255), 2)

                    formatted = {
                        "Frame":            int(frame_count),
                        "VehicleID":        f"V-{v['id']}",
                        "Type":             v["type"],
                        "Plate":            "VIOLATION",
                        "extracted_number": "N/A",
                        "vehicle_image":    None,
                        "plate_image":      None,
                        "_s3_vehicle_key":  None,
                        "_s3_plate_key":    None,
                    }

                    if v["type"] in ["NO HELMET", "WRONG SIDE", "WRONG LANE",
                                     "TRIPLE RIDING", "STALLED VEHICLE", "NO SEATBELT"]:
                        check_dirs()
                        full_name  = f"full_{uuid.uuid4().hex[:8]}.jpg"
                        crop_name  = f"crop_{uuid.uuid4().hex[:8]}.jpg"
                        full_local = os.path.join(assets_dir, full_name)
                        crop_local = os.path.join(assets_dir, crop_name)

                        cv2.imwrite(full_local, frame)
                        x1, y1, x2, y2 = b
                        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if crop.size > 0:
                            cv2.imwrite(crop_local, crop)

                        full_s3_key = s3_upload_image(full_local, f"evidence/{full_name}")
                        crop_s3_key = s3_upload_image(crop_local, f"evidence/{crop_name}")

                        full_local_url = f"/outputs/assets/{full_name}"
                        crop_local_url = f"/outputs/assets/{crop_name}"

                        full_presigned = get_presigned_url(full_s3_key) if full_s3_key else full_local_url
                        crop_presigned = get_presigned_url(crop_s3_key) if crop_s3_key else crop_local_url

                        formatted["vehicle_image"]   = full_presigned
                        formatted["plate_image"]     = crop_presigned
                        formatted["FullImgUrl"]      = full_presigned
                        formatted["CropImgUrl"]      = crop_presigned
                        formatted["_s3_vehicle_key"] = full_s3_key
                        formatted["_s3_plate_key"]   = crop_s3_key

                        save_violation(
                            job_id=job_id,
                            frame=int(frame_count),
                            vehicle_id=str(formatted["VehicleID"]),
                            cls=str(formatted["Type"]),
                            plate=str(formatted["Plate"]),
                            s3_vehicle_key=full_s3_key,
                            s3_plate_key=crop_s3_key,
                            local_evidence_url=full_local_url,
                        )
                        # Mark as alerted to prevent duplicates for this track
                        service.mark_alerted(v["id"], "N/A")

                    jobs[job_id]["report"].append(formatted)

            out.write(frame)

        cap.release()
        out.release()

        violation_count = len(jobs[job_id]["report"])
        jobs[job_id]["status"]    = "completed"
        jobs[job_id]["video_url"] = f"/outputs/{os.path.basename(output_path)}"

        # Generate presigned URL for processed output video (if uploaded to S3)
        output_s3_key = s3_upload_video(output_path, f"output_{os.path.basename(output_path)}")
        if output_s3_key:
            jobs[job_id]["s3_output_key"]    = output_s3_key
            jobs[job_id]["s3_output_presigned"] = get_presigned_url(output_s3_key)

        # Update MongoDB job
        update_job_in_db(job_id, "completed", violation_count)


    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Job {job_id} failed: {error_msg}")
        with open("debug_log.txt", "w") as f:
            f.write(error_msg)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
        update_job_in_db(job_id, "error", 0)

    print(f">>> Job finished: {job_id} | Violations: {len(jobs[job_id]['report'])}")


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
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/report/{job_id}")
def get_report(job_id: str):
    """
    Returns the violation report with FRESH presigned URLs.
    S3 keys stored in report are converted to presigned URLs on every call.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    raw_report = jobs[job_id].get("report", [])
    # Regenerate presigned URLs fresh on every call (they expire after 1 hour)
    enriched = generate_presigned_urls_for_report(raw_report)
    return enriched


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
