import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Match api.py flags
os.environ["FLAGS_enable_onednn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

# Force CPU to rule out CUDA conflicts for now
os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    print("Importing paddleocr...")
    from paddleocr import PaddleOCR
    # print("Importing torch...")
    # import torch
    # print("Importing ultralytics...")
    # from ultralytics import YOLO

    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=False, lang='en', enable_mkldnn=False, device='cpu', ocr_version='PP-OCRv3')
    
    # print("Initializing YOLO...")
    # model = YOLO("models/yolo11n.pt")
    
    print("Success!")
except Exception:
    import traceback
    traceback.print_exc()
