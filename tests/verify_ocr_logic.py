import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.anpr_service import ANPRService

def test_production_logic():
    class TestableANPRService(ANPRService):
        def __init__(self):
            # Skip super().__init__ which loads models
            self.blacklist = []
            self.zones = []
            print("Initialized Mock ANPRService for production logic testing.")

    service = TestableANPRService()
    
    # 1. Tiered Validation (Strict vs Soft)
    print("\n--- Testing Tiered Validation ---")
    validation_cases = [
        ("KA03MV0927", True, True),   # Strict: OK, Soft: OK
        ("MH12AB1234", True, True),   # Strict: OK, Soft: OK
        ("KA05NG5139", True, True),   # User example: OK
        ("KA03 O 927", False, True),  # Strict: FAIL (spaces), Soft: OK (after norm)
        ("XX12AB1234", False, True),  # Strict: FAIL (Invalid state XX), Soft: OK
        ("ABC 12345", False, True),   # Strict: FAIL, Soft: OK (Length 8)
        ("TOO_SHORT", False, False),  # Strict: FAIL, Soft: FAIL (Length < 7)
    ]
    
    for plate, expected_strict, expected_soft in validation_cases:
        s_res = service.validate_indian_plate(plate, mode='strict')
        w_res = service.validate_indian_plate(plate, mode='soft')
        print(f"Plate: {plate:10} | Strict: {s_res:5} | Soft: {w_res:5}")

    # 4. Correction Logic
    print("\n--- Testing Correction Logic ---")
    correction_cases = [
        ("TA05H05139", "KA05H05139"), # T -> K heuristic
        ("KAU5NG5139", "KA05NG5139"), # U -> 0 digit fix
        ("B3JT34", "B3JT34"),         # Short/Partial
        ("I31302", "I31302"),         # Short/Partial
    ]
    
    for raw, expected in correction_cases:
        corrected = service.position_aware_correct(raw)
        print(f"Raw: {raw:12} | Corrected: {corrected:12} | Match: {corrected == expected}")

if __name__ == "__main__":
    test_production_logic()

if __name__ == "__main__":
    test_production_logic()
