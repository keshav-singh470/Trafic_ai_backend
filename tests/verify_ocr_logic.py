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
    
    # 1. State Code & Regex Validation
    print("\n--- Testing State Code & Regex Validation ---")
    validation_cases = [
        ("KA03MV0927", True),   # Valid State (Karnataka)
        ("DL01AA1234", True),   # Valid State (Delhi)
        ("22BH1234AA", True),   # Valid BH
        ("XX03MV0927", False),  # Invalid State (XX)
        ("KA03MV092", False),   # Invalid length
        ("MH12AB1234", True),   # Valid State (Maharashtra)
    ]
    
    for plate, expected in validation_cases:
        res = service.validate_indian_plate(plate)
        print(f"Plate: {plate:12} | Valid: {res:6} | Match: {res == expected}")

    # 2. Voting Logic (Stable)
    print("\n--- Testing Multi-frame Voting (Stable) ---")
    # Stable readings for one vehicle
    readings_stable = ["KA03MV0927"] * 5 + ["KA03HV0927"] * 2 + ["KA03MV0927"] * 3 # 8 vs 2
    winner = service._best_vote(readings_stable)
    print(f"Readings: {len(readings_stable)} samples | Winner: {winner} | Match: {winner == 'KA03MV0927'}")

    # 3. Voting Logic (Unstable -> Unreadable)
    print("\n--- Testing Multi-frame Voting (Unstable -> Unreadable) ---")
    readings_unstable = [
        "KA03MV0927", "DL01AA1234", "MH12AB1234", "TN38BZ1234", 
        "HR26DQ1234", "UP16CK1234", "KA05NN5555"
    ] * 3 # Many different valid plates, no consensus
    winner_unstable = service._best_vote(readings_unstable)
    print(f"Readings: {len(readings_unstable)} samples | Winner: {winner_unstable} | Match: {winner_unstable == 'Unreadable'}")

    # 4. Position-Aware Correction (1 ↔ I)
    print("\n--- Testing 1 ↔ I Correction ---")
    correction_cases = [
        ("DL0IAA1234", "DL01AA1234"), # I -> 1 in digit area
        ("DL01AAI234", "DL01AA1234"), # I -> 1 in digit area
    ]
    for raw, expected in correction_cases:
        corrected = service.position_aware_correct(raw)
        print(f"Raw: {raw:12} | Corrected: {corrected:12} | Match: {corrected == expected}")

if __name__ == "__main__":
    test_production_logic()
