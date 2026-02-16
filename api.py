import os
# Fix OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Disable OneDNN and PIR features to fix PaddlePaddle compatibility issues on some CPUs
os.environ["FLAGS_enable_onednn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

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
from shapely.geometry import Polygon

app = FastAPI(title="Smart Traffic AI - Modular API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8000/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

def check_dirs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "assets"), exist_ok=True)

check_dirs()

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

jobs: Dict[str, Dict] = {}

def get_service(case_type: str):
    if case_type in ["anpr", "security", "blacklist"]:
        return ANPRService(
            base_model="models/yolov8n.pt", 
            anpr_model="models/anpr_plat.pt"
        )
    elif case_type == "helmet":
        # Use user-provided trained model (best (3).pt)
        return HelmetService(model_path="models/helmet_triple_model.pt")
    elif case_type == "overload" or case_type == "triple":
        # Use same user-provided model for triple riding
        return OverloadService(model_path="models/helmet_triple_model.pt")
    elif case_type == "wrong_side" or case_type == "wrong_lane":
        # Default Zone: Assume right half of a 1920x1080 frame is for oncoming traffic only
        # So if we detect "car" (class 2) or "truck" (class 7) there, moving wrong way, it handles.
        # But for Lane Logic: "If ANY vehicle is in this zone, it's wrong lane"
        zones = [{
             "polygon": Polygon([(960, 0), (1920, 0), (1920, 1080), (960, 1080)]),
             "forbidden_classes": [2, 3, 5, 7] # Cars, Motorcycles, Buses, Trucks
        }]
        return WrongSideService(base_model="models/yolov8n.pt", zones=zones)
    elif case_type == "stalled":
        return StalledService(model_path="models/yolov8n.pt")
    elif case_type == "seatbelt":
        return SeatbeltService(model_path="no_sitbelt.pt")
    else:
        return None

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def process_video_task(job_id: str, input_path: str, output_path: str, case_type: str):
    try:
        print(">>> Job started:", job_id)
        jobs[job_id]["status"] = "processing"
        service = get_service(case_type)
        if not service: raise Exception("Invalid service type")

        cap = cv2.VideoCapture(input_path)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Try multiple codecs for robustness
        codecs = [('avc1', 'mp4'), ('mp4v', 'mp4'), ('XVID', 'avi'), ('MJPG', 'avi'), ('DIVX', 'avi')]
        out = None
        final_output_path = output_path
        
        for codec, ext in codecs:
            try:
                # Check if we need to change extension
                current_ext = final_output_path.split('.')[-1]
                if ext != current_ext:
                    # Replace extension
                    base = os.path.splitext(output_path)[0]
                    final_output_path = f"{base}.{ext}"

                fourcc = cv2.VideoWriter_fourcc(*codec)
                temp_out = cv2.VideoWriter(final_output_path, fourcc, fps, (w, h))
                if temp_out.isOpened():
                    print(f"Initialized VideoWriter with codec {codec} -> {final_output_path}")
                    out = temp_out
                    # Update global jobs dict with correctness path? 
                    # Actually we need to update output_path used later
                    output_path = final_output_path
                    break
            except Exception as e:
                print(f"Codec {codec} failed: {e}")
                continue

        if not out or not out.isOpened():
            raise Exception("Failed to initialize VideoWriter with any codec (avc1, mp4v, XVID, MJPG, DIVX)")

        frame_count = 0
        reported_violations = [] # List of {id, box, type, frame}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            if case_type in ["anpr", "security", "blacklist"]:
                # For ANPR, we also want to track vehicles to associate plates
                _, tracked = service.process_frame(frame)
                results = service.run_detection(frame, frame_count)
                for res in results:
                    is_alert = res.get("alert", False)
                    # Filter: In security/blacklist mode, ONLY process hits that are on the blacklist
                    if case_type in ["security", "blacklist"] and not is_alert:
                        continue

                    # Attempt association with tracked vehicles
                    pcx, pcy = (res["box"][0] + res["box"][2]) / 2, (res["box"][1] + res["box"][3]) / 2
                    owner = "N/A"
                    for tid, vbox in zip(tracked.tracker_id, tracked.xyxy):
                        if vbox[0] <= pcx <= vbox[2] and vbox[1] <= pcy <= vbox[3]:
                            owner = f"V-{tid}"
                            break
                    
                    if res.get("is_new", True):
                        is_alert = res.get("alert", False)
                        msg_type = "SECURITY ALERT" if is_alert else "ANPR"

                        # If Alert, draw on frame BEFORE saving so evidence shows the red box
                        if is_alert:
                            b = res["box"]
                            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                            cv2.putText(frame, f"{res['text']} [ALERT]", (b[0], b[1]-10), 0, 0.6, (0, 0, 255), 2)

                        # Save ANPR Images (Full + Crop)
                        check_dirs()
                        assets_dir = os.path.join(OUTPUT_DIR, "assets")
                        
                        full_name = f"anpr_{uuid.uuid4().hex[:8]}.jpg"
                        crop_name = f"anpr_crop_{uuid.uuid4().hex[:8]}.jpg"
                        
                        # Save Full
                        cv2.imwrite(os.path.join(assets_dir, full_name), frame)
                        
                        # Save Crop
                        b = res["box"]
                        x1, y1, x2, y2 = b
                        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if crop.size > 0:
                            cv2.imwrite(os.path.join(assets_dir, crop_name), crop)

                        formatted = {
                            "Frame": int(frame_count),
                            "VehicleID": owner,
                            "Type": msg_type,
                            "Plate": res["text"],
                            "FullImgUrl": f"/outputs/assets/{full_name}",
                            "CropImgUrl": f"/outputs/assets/{crop_name}"
                        }
                        jobs[job_id]["report"].append(formatted)
                    
                    # Draw for output video (if not already drawn as alert)
                    if not res.get("alert", False):
                        b = res["box"]
                        
                        # Visualization Logic
                        if res.get("is_new", True):
                            color = (0, 255, 0) # Green for New
                            label = res["text"] 
                        else:
                            color = (0, 255, 255) # Yellow for Seen/Duplicate
                            label = res["text"]

                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
                        cv2.putText(frame, label, (b[0], b[1]-10), 0, 0.6, color, 2)
            else:
                violations = service.run_detection(frame)
                for v in violations:
                    # DEDUPLICATION LOGIC
                    is_duplicate = False
                    
                    # 1. ID Check (if tracker is stable)
                    for rv in reported_violations:
                        if rv["id"] == v["id"] and rv["type"] == v["type"]:
                            # Already reported this ID for this violation type
                            # potentially update timestamp?
                            is_duplicate = True
                            break
                    
                    # 2. Spatial Check (if tracker switched ID but it's same object)
                    if not is_duplicate:
                        for rv in reported_violations:
                            if rv["type"] == v["type"]:
                                # Check IoU
                                iou = calculate_iou(v["box"], rv["box"])
                                if iou > 0.5: # 50% overlap means likely same object
                                    # Also check time? If it's been a long time (e.g. 500 frames), maybe re-report?
                                    # For now, strict deduplication -> strict is better to avoid spam
                                    if frame_count - rv["frame"] < 1000: # Within ~30-60 seconds
                                        is_duplicate = True
                                        break
                    
                    if is_duplicate:
                        continue
                    
                    # New violation
                    reported_violations.append({
                        "id": v["id"],
                        "box": v["box"],
                        "type": v["type"],
                        "frame": frame_count
                    })

                    b = v["box"]
                    # Draw first so images have the detection box
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cv2.putText(frame, v["type"], (b[0], b[1]-10), 0, 0.6, (0, 0, 255), 2)

                    formatted = {
                        "Frame": int(frame_count),
                        "VehicleID": f"V-{v['id']}",
                        "Type": v["type"],
                        "Plate": "VIOLATION"
                    }

                    if v["type"] in ["NO HELMET", "WRONG SIDE", "WRONG LANE", "TRIPLE RIDING", "STALLED VEHICLE", "NO SEATBELT"]:
                        check_dirs()
                        assets_dir = os.path.join(OUTPUT_DIR, "assets")
                        
                        full_name = f"full_{uuid.uuid4().hex[:8]}.jpg"
                        crop_name = f"crop_{uuid.uuid4().hex[:8]}.jpg"
                        
                        # Save Full
                        cv2.imwrite(os.path.join(assets_dir, full_name), frame)
                        
                        # Save Crop
                        x1, y1, x2, y2 = b
                        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if crop.size > 0:
                            cv2.imwrite(os.path.join(assets_dir, crop_name), crop)
                            
                        formatted["FullImgUrl"] = f"/outputs/assets/{full_name}"
                        formatted["CropImgUrl"] = f"/outputs/assets/{crop_name}"

                    jobs[job_id]["report"].append(formatted)

            out.write(frame)

        cap.release()
        out.release()
        jobs[job_id]["status"] = "completed"
        # Return full URL so frontend on different port/host can find it
        jobs[job_id]["video_url"] = f"/outputs/{os.path.basename(output_path)}"
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Job failed: {error_msg}")
        with open("debug_log.txt", "w") as f:
            f.write(error_msg)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

    print(">>> Job finished:", job_id, "Report size:", len(jobs[job_id]["report"]))

@app.post("/api/{case_type}")
async def start_job(case_type: str, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    check_dirs()
    job_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    output_path = os.path.join(OUTPUT_DIR, f"output_{job_id}_{file.filename}")
    jobs[job_id] = {"job_id": job_id, "status": "pending", "case_type": case_type, "report": []}
    background_tasks.add_task(process_video_task, job_id, input_path, output_path, case_type)
    return {"job_id": job_id}

@app.post("/upload")
async def legacy_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...), case_type: str = Form("anpr")):
    return await start_job(case_type, background_tasks, file)

@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs: raise HTTPException(status_code=404)
    return jobs[job_id]

@app.get("/report/{job_id}")
def get_report(job_id: str):
    if job_id not in jobs: raise HTTPException(status_code=404)
    return jobs[job_id].get("report", [])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
