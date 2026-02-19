import os
import httpx
import time
import cv2
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID")
API_URL     = f"https://api.telegram.org/bot{BOT_TOKEN}"

if BOT_TOKEN and CHAT_ID:
    print("✅ Telegram service ready.")
else:
    print("⚠️ WARNING: Telegram credentials missing. Alerts will be disabled.")

def create_combined_image(vehicle_path: str, plate_path: str, output_path: str):
    """
    Combines REAL vehicle image and REAL plate crop into a single picture-in-picture image.
    Plate crop is placed in the bottom-right corner with a border.
    """
    try:
        if not os.path.exists(vehicle_path) or not os.path.exists(plate_path):
            print(f"ERROR: Missing source images for combination. Vehicle: {os.path.exists(vehicle_path)}, Plate: {os.path.exists(plate_path)}")
            return False

        # Load images
        vehicle = cv2.imread(vehicle_path)
        plate = cv2.imread(plate_path)

        if vehicle is None or plate is None:
            print("ERROR: cv2 failed to load source images.")
            return False

        vh, vw = vehicle.shape[:2]
        
        # Area-based scaling: 80/20 rule.
        # If plate area = 0.2 * background area, then:
        # (pw * ph) = 0.2 * (vw * vh)
        # Since aspect ratio (AR) = ph / pw => ph = pw * AR
        # pw * pw * AR = 0.2 * vw * vh
        # pw = sqrt(0.2 * vw * vh / AR)
        
        background_area = vw * vh
        target_plate_area = background_area * 0.20
        
        plate_h, plate_w = plate.shape[:2]
        aspect_ratio = plate_h / plate_w
        
        target_pw = int(np.sqrt(target_plate_area / aspect_ratio))
        target_ph = int(target_pw * aspect_ratio)
        
        # Safety check for size
        target_pw = min(target_pw, int(vw * 0.5))
        target_ph = min(target_ph, int(vh * 0.5))

        # Resize plate
        plate_resized = cv2.resize(plate, (target_pw, target_ph), interpolation=cv2.INTER_LANCZOS4)
        
        # Add white border to plate
        border_size = 4
        plate_with_border = cv2.copyMakeBorder(
            plate_resized, 
            border_size, border_size, border_size, border_size, 
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        pbh, pbw = plate_with_border.shape[:2]

        # Define overlay position (bottom-right)
        margin = 20
        x_offset = vw - pbw - margin
        y_offset = vh - pbh - margin

        # Final composition
        combined = vehicle.copy()
        combined[y_offset:y_offset+pbh, x_offset:x_offset+pbw] = plate_with_border
        
        # Save to job-specific path
        cv2.imwrite(output_path, combined)
        return True
    except Exception as e:
        print(f"ERROR: Failed to create combined image: {e}")
        return False

def send_local_violation(combined_path: str, plate_text: str, 
                         job_id: str = "N/A", case_type: str = "ANPR", violation_count: int = 1,
                         status: str = "Success", timestamp: str = "N/A"):
    """
    Sends a single combined image (PIP) with the REAL detection results.
    STRICT: One message only. No links, no URLs.
    """
    if not BOT_TOKEN or not CHAT_ID:
        print("WARNING: Telegram credentials not set. Skipping notification.")
        return

    # EXACT Caption Format per Requirements
    caption = (
        f"Title: Traffic AI Detection\n"
        f"Plate: {plate_text}\n"
        f"Type: {case_type.upper()}\n"
        f"Violations: {violation_count}\n"
        f"Status: {status}\n"
        f"Date/Time: {timestamp}"
    )

    try:
        if os.path.exists(combined_path):
            with open(combined_path, "rb") as photo:
                resp = httpx.post(
                    f"{API_URL}/sendPhoto",
                    data={"chat_id": CHAT_ID, "caption": caption, "parse_mode": "Markdown"},
                    files={"photo": photo},
                    timeout=40.0 # Increased timeout for larger files
                )
                if resp.status_code != 200:
                    print(f"ERROR: Telegram send combined photo failed: {resp.text}")
                else:
                    print(f"✅ Telegram alert delivered successfully.")
        else:
            print(f"ERROR: Final alert image not found at {combined_path}")

    except Exception as e:
        print(f"ERROR: Telegram delivery failed: {e}")
