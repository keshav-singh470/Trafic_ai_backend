import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.anpr_service import ANPRService

def test_extraction_and_classification():
    service = ANPRService(base_model="yolo11n.pt", anpr_model="yolo11n.pt")
    
    test_cases = [
        ("CK40ZATI2980", "KA02AL4980"), # CK40 -> KA02, ZATI is tough
        ("KA02AI4980", "KA02AL4980"),   # I -> L (Middle part)
        ("KA02AK6346", "KA02MK6346"),   # AK -> MK (M misread as A)
    ]
    
    print("\nEXTRACT_TESTS")
    for raw, expected in test_cases:
        corrected = service.position_aware_correct(raw.replace(" ", ""))
        extracted = service.extract_indian_plate(corrected)
        final = extracted if extracted else corrected
        
        print(f"RAW: {raw} -> FINAL: {final} | EXPECT: {expected} | {'OK' if final == expected else 'FAIL'}")

    # Heuristic Check
    w, h = 1920, 1080
    vbox = [100, 100, 400, 500] 
    vw, vh = vbox[2]-vbox[0], vbox[3]-vbox[1]
    aspect = vh / vw if vw > 0 else 0
    is_auto = (vw * vh < (w * h * 0.15)) and aspect > 0.8
    print(f"\nBus(Small, Tall) -> Auto-Rickshaw? {is_auto}")

if __name__ == "__main__":
    test_extraction_and_classification()
