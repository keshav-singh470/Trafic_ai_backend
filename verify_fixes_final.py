import cv2
import numpy as np
import sys
import os

# Add the project directory to sys.path
sys.path.append(r'c:\Users\keshav singh\Desktop\Project\anpr-system-old\Trafic_ai_backend')

from services.anpr_service import ANPRService
from services.vehicle_type_service import VehicleTypeService

def test_full_plate_logic():
    print("--- Testing Full Plate Logic ---")
    service = ANPRService()
    
    # Mock a partial plate (Small aspect ratio)
    partial_img = np.zeros((100, 100, 3), dtype=np.uint8)
    partial_text = "KA05MK12"
    result = service.is_full_plate(partial_img, partial_text)
    print(f"Partial Plate (100x100, {partial_text}): Expected=False, Got={result}")
    
    # Mock a short text plate
    short_text_img = np.zeros((100, 400, 3), dtype=np.uint8)
    short_text = "KA05"
    result = service.is_full_plate(short_text_img, short_text)
    print(f"Short Text Plate (100x400, {short_text}): Expected=False, Got={result}")
    
    # Mock a full plate
    full_img = np.zeros((100, 400, 3), dtype=np.uint8)
    full_text = "KA05MK1234"
    result = service.is_full_plate(full_img, full_text)
    print(f"Full Plate (100x400, {full_text}): Expected=True, Got={result}")

def test_vehicle_classification():
    print("\n--- Testing Vehicle Classification Mapping ---")
    # This just checks if the service loads and has the correct names
    service = VehicleTypeService()
    print(f"Model Class Names: {service.class_names}")
    
    # Verify mapping in api.py logic (mocked here)
    VEHICLE_TYPE_MAP = {
        0: "Auto-Rickshaw",
        1: "Bus",
        2: "Car",
        3: "Bike",
        4: "Scooter",
        5: "Truck"
    }
    
    for cls_id, expected_name in VEHICLE_TYPE_MAP.items():
        name = service.class_names.get(cls_id, "unknown")
        # Note: model might use 'motorcycle' for 'bike' or 'autorickshaw' for 'auto-rickshaw'
        # We just want to see if the ID exists and is reasonable
        print(f"ID {cls_id}: Model says '{name}', Map says '{expected_name}'")

if __name__ == "__main__":
    try:
        test_full_plate_logic()
        test_vehicle_classification()
        print("\n✅ Verification data gathered.")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
