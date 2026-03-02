"""
Standalone unit tests for ANPR fixes:
- Problem 1: Sharpness calculation (Laplacian variance)
- Problem 5: OCR state code correction (O->K, Z->L)

These tests do NOT import ANPRService (which requires YOLO/PaddleOCR).
Instead, they copy the exact logic from anpr_service.py for isolated testing.
"""
import cv2
import numpy as np
import re


# ── Copy of INDIAN_STATE_CODES from anpr_service.py ──
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "CT", "DD", "DL", "DN",
    "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TG", "TR", "TS", "UK", "UP", "WB", "BH"
}

INVALID_STATE_MAP = {
    'IN': 'TN', 'KN': 'KA', 'IK': 'JK',
}


def calculate_sharpness(image):
    """Copy of ANPRService.calculate_sharpness for testing."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def position_aware_correct(text):
    """Copy of ANPRService.position_aware_correct for testing."""
    if not text: return ""
    
    raw_text = text
    digit_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '7': 'T'}
    letter_to_digit = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7'}
    
    t = list(text)
    length = len(t)

    for i in range(min(2, length)):
        if t[i].isdigit() and t[i] in digit_to_letter:
            t[i] = digit_to_letter[t[i]]

    if length >= 4:
        for i in range(2, 4):
            if t[i].isalpha() and t[i] in letter_to_digit:
                t[i] = letter_to_digit[t[i]]

        numeric_tail_start = max(4, length - 4)
        for i in range(numeric_tail_start, length):
            if t[i].isalpha() and t[i] in letter_to_digit:
                t[i] = letter_to_digit[t[i]]

        for i in range(4, numeric_tail_start):
            if t[i].isdigit() and t[i] in digit_to_letter:
                t[i] = digit_to_letter[t[i]]

    if length >= 2:
        state = "".join(t[:2])
        if state not in INDIAN_STATE_CODES and state in INVALID_STATE_MAP:
            corrected_state = INVALID_STATE_MAP[state]
            t[0], t[1] = corrected_state[0], corrected_state[1]

    # NEW: O→K, Z→L state code recovery
    if length >= 2:
        state = "".join(t[:2])
        if state not in INDIAN_STATE_CODES:
            ocr_state_swap = {'O': 'K', 'Z': 'L', '0': 'K', '2': 'L'}
            new_t = list(t)
            changed = False
            for i in range(min(2, len(new_t))):
                if new_t[i] in ocr_state_swap:
                    new_t[i] = ocr_state_swap[new_t[i]]
                    changed = True
            new_state = "".join(new_t[:2])
            if changed and new_state in INDIAN_STATE_CODES:
                t = new_t

    corrected = "".join(t)
    if corrected != raw_text:
        print(f"  [OCR FIX] raw={raw_text} corrected={corrected}")
    
    return corrected


# ══════════════════════════════════════════════════════
#  TESTS
# ══════════════════════════════════════════════════════

def test_sharpness():
    print("\n--- Test: Sharpness Calculation ---")
    
    sharp = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(sharp, "KA03AB1234", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    blurry = cv2.GaussianBlur(sharp, (31, 31), 10)
    
    ss = calculate_sharpness(sharp)
    bs = calculate_sharpness(blurry)
    
    print(f"  Sharp={ss:.1f}  Blurry={bs:.1f}")
    assert ss > bs, f"Sharp ({ss}) must be > Blurry ({bs})"
    assert ss > 100, f"Sharp ({ss}) must be > threshold 100"
    print("  ✅ Sharp > Blurry > threshold")
    
    assert calculate_sharpness(np.array([])) == 0.0
    print("  ✅ Empty image = 0")


def test_ocr_fix_OZ_to_KL():
    print("\n--- Test: OZ → KL (user's specific example) ---")
    r = position_aware_correct("OZ29X5950")
    print(f"  OZ29X5950 → {r}")
    assert r[:2] == "KL", f"Expected KL, got {r[:2]}"
    print("  ✅ PASSED")


def test_valid_states_unchanged():
    print("\n--- Test: Valid states stay unchanged ---")
    for inp, expected_state in [
        ("KL29X5950", "KL"),
        ("DL01YZ2046", "DL"),
        ("KA03MV0927", "KA"),
        ("MH12AB1234", "MH"),
        ("TN01AB1234", "TN"),
    ]:
        r = position_aware_correct(inp)
        print(f"  {inp} → {r} (state={r[:2]})")
        assert r[:2] == expected_state, f"State should be {expected_state}, got {r[:2]}"
    print("  ✅ All valid states preserved")


def test_digit_position_correction():
    print("\n--- Test: Digit position corrections ---")
    r = position_aware_correct("KA03MV092Z")
    print(f"  KA03MV092Z → {r}")
    assert r[-1] == "2", f"Trailing Z should become 2, got {r[-1]}"
    print("  ✅ Z → 2 in digit position")


def test_KL_in_codes():
    print("\n--- Test: KL in INDIAN_STATE_CODES ---")
    assert "KL" in INDIAN_STATE_CODES
    print("  ✅ KL present")


if __name__ == "__main__":
    test_sharpness()
    test_ocr_fix_OZ_to_KL()
    test_valid_states_unchanged()
    test_digit_position_correction()
    test_KL_in_codes()
    print("\n" + "="*50)
    print("ALL TESTS PASSED ✅")
