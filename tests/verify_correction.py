import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.anpr_service import ANPRService

def test_corrections():
    service = ANPRService(base_model="yolo11n.pt", anpr_model="yolo11n.pt")
    
    test_cases = [
        ("KA02A L4980", "KA02AL4980"),
        ("KA92A I1980", "KA02AL4980"),
        ("OSTTKA03MV0927", "KA03MV0927"),
        ("KA02AK6346", "KA02MK6346"), 
    ]
    
    print("\nVAL_START")
    for raw, expected in test_cases:
        corrected = service.position_aware_correct(raw.replace(" ", ""))
        extracted = service.extract_indian_plate(corrected)
        final = extracted if extracted else corrected
        
        # Simulating voting winner preference for M vs A if both exist
        # In this unit test, we only test single-shot correction
        # But for 'KA02AK' -> 'KA02MK' if we specifically want to test single shot:
        # We need to see if position_aware_correct did anything.
        
        print(f"{raw} -> {final} | {'OK' if final == expected else 'FAIL'}")
    print("VAL_END")

if __name__ == "__main__":
    test_corrections()
