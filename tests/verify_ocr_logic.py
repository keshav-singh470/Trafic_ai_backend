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
        ("KA03 O 927", False, True),  # Strict: FAIL (spaces), Soft: OK (after norm)
        ("ABC 12345", False, True),   # Strict: FAIL, Soft: OK (Length 8)
        ("A1B2C3D4", False, True),    # Strict: FAIL, Soft: OK (Length 8)
        ("TOO_SHORT", False, False),  # Strict: FAIL, Soft: FAIL (Length < 7)
    ]
    
    for plate, expected_strict, expected_soft in validation_cases:
        s_res = service.validate_indian_plate(plate, mode='strict')
        w_res = service.validate_indian_plate(plate, mode='soft')
        print(f"Plate: {plate:10} | Strict: {s_res:5} | Soft: {w_res:5}")

    # 2. Tiered Voting (Strict Priority)
    print("\n--- Testing Tiered Voting (Strict Priority) ---")
    # Even if soft-valid outvotes strict-valid, strict should win (security)
    # UNLESS strict is below the 2-vote stability floor.
    readings = ["KA03MV0927"] * 2 + ["SOFTPLATE1"] * 10
    winner = service._best_vote(readings)
    print(f"Strict (2 votes) vs Soft (10 votes). Winner: {winner} | Match: {winner == 'KA03MV0927'}")

    # 3. Tiered Voting (Soft Fallback with Repetition Check)
    print("\n--- Testing Soft Fallback (Repetition Check) ---")
    # Soft needs at least 3 votes to be trusted
    readings_2 = ["SOFTPLATE1"] * 2
    winner_2 = service._best_vote(readings_2)
    print(f"Soft (2 votes). Winner: {winner_2} | Match: {winner_2 is None}") # Should be None/Rejected
    
    readings_3 = ["SOFTPLATE1"] * 3
    winner_3 = service._best_vote(readings_3)
    print(f"Soft (3 votes). Winner: {winner_3} | Match: {winner_3 == 'SOFTPLATE1'}")

if __name__ == "__main__":
    test_production_logic()

if __name__ == "__main__":
    test_production_logic()
