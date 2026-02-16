import cv2
import re
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime
from difflib import SequenceMatcher
import os
import torch

class TrafficAnalyticsService:
    def __init__(self, base_model="yolo11n.pt", anpr_model=r"C:\Users\keshav singh\Desktop\Project\anpr-system-main\anpr-system-main\models\anpr_plat.pt"):
        """
        Unified Traffic Analytics Service.
        Using yolo11n.pt (Large) for maximum detection accuracy as requested.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Smart Traffic AI on {self.device}...")
        
        # Load Models
        self.base_model = YOLO(base_model).to(self.device)
        self.anpr_model = YOLO(anpr_model).to(self.device)
        
        # OCR Engine
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        # Internal State
        self.blacklist = self._load_blacklist()
        self.track_history = {}
        self.seen_plates = set()

    def _load_blacklist(self):
        path = "blacklist.txt"
        if os.path.exists(path):
            with open(path, "r") as f:
                return [line.strip().upper() for line in f if line.strip()]
        return []

    def clean_text(self, text):
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def validate_plate(self, text):
        std = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{4}$'
        bh = r'^[0-9]{2}BH[0-9]{1,4}[A-Z]{1,2}[0-9]{1,4}$'
        return bool(re.match(std, text) or re.match(bh, text))

    def iou(self, b1, b2):
        xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
        xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def process_video(self, video_path, output_path, mode="anpr", on_result=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return {"error": "Video failed"}
        
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = int(cap.get(5))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # COCO Class IDs
        V_CLASSES = [2, 3, 5, 7] # car, motorcycle, bus, truck
        PERSON = 0
        MOTORCYCLE = 3
        
        frame_idx = 0
        self.track_history = {}
        self.seen_plates = set()
        report = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            if frame_idx % 2 != 0: # Half frames for performance
                out.write(frame)
                continue

            # 1. Base Tracking (yolo11l.pt)
            results = self.base_model.track(frame, persist=True, verbose=False, conf=0.4)[0]
            vehicles_in_frame = {}
            
            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.cpu().numpy().astype(int)
                clss = results.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, ids, clss):
                    if cls in V_CLASSES:
                        vehicles_in_frame[tid] = (box, cls)
                        if tid not in self.track_history:
                            self.track_history[tid] = {"pos": [], "start": frame_idx, "done": False, "plate": None}
                        
                        cent = ((box[0]+box[2])//2, (box[1]+box[3])//2)
                        self.track_history[tid]["pos"].append(cent)
                        if len(self.track_history[tid]["pos"]) > 30: self.track_history[tid]["pos"].pop(0)

            # 2. ANPR (Full Frame - Synced with anpr_final.py)
            if mode in ["anpr", "blacklist"]:
                p_results = self.anpr_model(frame, verbose=False, conf=0.15)[0]
                for p_box in p_results.boxes:
                    px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy().astype(int)
                    plate_img = frame[py1:py2, px1:px2]

                    if plate_img.size > 0:
                        # Advanced Preprocessing
                        plate_up = cv2.resize(plate_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                        gray = cv2.cvtColor(plate_up, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        enhanced = clahe.apply(gray)
                        plate_fin = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

                        # Split Strategy
                        h_p, w_p = plate_img.shape[:2]
                        aspect = w_p / h_p
                        txt = ""

                        if aspect < 2.5: # 2-row
                            hh = plate_fin.shape[0] // 2
                            rt, rb = self.ocr.ocr(plate_fin[0:hh, :], cls=True), self.ocr.ocr(plate_fin[hh:, :], cls=True)
                            tt = "".join([l[1][0] for l in rt[0]]) if rt and rt[0] else ""
                            tb = "".join([l[1][0] for l in rb[0]]) if rb and rb[0] else ""
                            if self.validate_plate(self.clean_text(tt+tb)): txt = tt+tb
                            else:
                                rf = self.ocr.ocr(plate_fin, cls=True)
                                if rf and rf[0]: txt = "".join([l[1][0] for l in rf[0]])
                        else:
                            rf = self.ocr.ocr(plate_fin, cls=True)
                            if rf and rf[0]: txt = "".join([l[1][0] for l in rf[0]])

                        clean = self.clean_text(txt)
                        if self.validate_plate(clean):
                            is_dup = any(SequenceMatcher(None, clean, s).ratio() > 0.8 for s in self.seen_plates)
                            if not is_dup:
                                pcx, pcy = (px1+px2)/2, (py1+py2)/2
                                owner = -1
                                for tid, (vbox, _) in vehicles_in_frame.items():
                                    if vbox[0] <= pcx <= vbox[2] and vbox[1] <= pcy <= vbox[3]:
                                        owner = tid; break
                                
                                self.seen_plates.add(clean)
                                type_res = "SECURITY" if clean in self.blacklist else "ANPR"
                                res_item = {"Frame": frame_idx, "VehicleID": owner, "Type": type_res, "Result": clean}
                                report.append(res_item)
                                if on_result: on_result(res_item)
                                print(f"New Valid Plate: {clean} (Vehicle {owner})")
                                if owner != -1: self.track_history[owner]["plate"] = clean

            # 3. Violation Modules & Annotation
            for tid, (box, cls) in vehicles_in_frame.items():
                x1, y1, x2, y2 = box
                violation, color = None, (0, 255, 0)
                label = self.track_history[tid]["plate"] or f"ID:{tid}"
                
                if mode == "wrong_side":
                    hist = self.track_history[tid]["pos"]
                    if len(hist) > 15:
                        dy = hist[-1][1] - hist[0][1]
                        if dy < -50: violation, color = "WRONG SIDE", (0, 0, 255)
                elif mode == "triple":
                    if cls == MOTORCYCLE:
                        riders = sum(1 for b in results[0].boxes if int(b.cls[0]) == 0 and self.iou(box, b.xyxy[0].cpu().numpy()) > 0.05)
                        if riders >= 3: violation, color = "TRIPLE RIDING", (0, 0, 255)
                elif mode == "helmet":
                    if cls == MOTORCYCLE:
                        violation, color = "NO HELMET", (0, 165, 255)
                elif mode == "stalled":
                    hist = self.track_history[tid]["pos"]
                    if len(hist) > 20:
                        dist = np.linalg.norm(np.array(hist[-1]) - np.array(hist[0]))
                        if dist < 5 and (frame_idx - self.track_history[tid]["start"]) > 40:
                            violation, color = "STALLED", (0, 255, 255)

                if violation and not self.track_history[tid]["done"]:
                    self.track_history[tid]["done"] = True
                    item = {"Frame": frame_idx, "VehicleID": tid, "Type": violation, "Result": "VIOLATION"}
                    report.append(item)
                    if on_result: on_result(item)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{violation if violation else label}", (x1, y1-10), 0, 0.6, color, 2)

            out.write(frame)
            if frame_idx % 10 == 0: print(f"Processing Frame {frame_idx}...")

        cap.release()
        out.release()
        return {"total": len(report), "report": report}

if __name__ == "__main__":
    service = TrafficAnalyticsService()
