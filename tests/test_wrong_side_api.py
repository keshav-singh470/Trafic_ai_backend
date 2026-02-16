import requests
import time
import pandas as pd
import json
import os

API_URL = "http://localhost:8000"
VIDEO_PATH = r"d:\Python\Smart-Traffic-Analytics\test-video\Hulimavu_Circle_FIX_2_Helmet.mp4"
REPORT_FILE = "wrong_side_test_report.csv"

def test_wrong_side_detection():
    # 1. Upload Video
    print(f"Uploading video: {VIDEO_PATH}")
    with open(VIDEO_PATH, "rb") as f:
        files = {"file": f}
        try:
            response = requests.post(f"{API_URL}/api/wrong_side", files=files)
            response.raise_for_status()
            job_id = response.json()["job_id"]
            print(f"Job started with ID: {job_id}")
        except Exception as e:
            print(f"An error occurred: {e}")
            return

    # 2. Poll Status
    while True:
        status_res = requests.get(f"{API_URL}/status/{job_id}")
        status = status_res.json()["status"]
        print(f"Status: {status}")
        
        if status in ["completed", "failed", "error"]:
            break
        time.sleep(2)

    if status != "completed":
        print(f"Job failed: {status_res.json().get('error')}")
        return

    # 3. Get Report
    print("Fetching report...")
    report_res = requests.get(f"{API_URL}/report/{job_id}")
    report = report_res.json()
    
    print(f"Total violations: {len(report)}")
    if len(report) > 0:
        print("Sample Violation:")
        print(json.dumps(report[0], indent=2))
    
    # 4. Save to CSV
    df = pd.DataFrame(report)
    df.to_csv(REPORT_FILE, index=False)
    print(f"Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Video file not found: {VIDEO_PATH}")
    else:
        test_wrong_side_detection()
