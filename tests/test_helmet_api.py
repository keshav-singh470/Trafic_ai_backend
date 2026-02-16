import requests
import time
import sys
import os
import pandas as pd

API_URL = "http://localhost:8000"
VIDEO_PATH = r"d:\Python\Smart-Traffic-Analytics\test-video\Hulimavu_Circle_FIX_2_Helmet.mp4"
REPORT_FILE = "helmet_test_report.csv"

def test_helmet_detection():
    print(f"Uploading video: {VIDEO_PATH}")
    
    try:
        with open(VIDEO_PATH, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{API_URL}/api/helmet", files=files)
            
        if response.status_code != 200:
            print(f"Failed to start job: {response.text}")
            return

        job_id = response.json()["job_id"]
        print(f"Job started with ID: {job_id}")
        
        while True:
            status_res = requests.get(f"{API_URL}/status/{job_id}")
            if status_res.status_code != 200:
                print("Error checking status")
                break
                
            status_data = status_res.json()
            status = status_data.get("status")
            print(f"Status: {status}", end="\r")
            
            if status == "completed":
                print("\nProcessing complete!")
                break
            elif status == "error":
                print(f"\nJob failed: {status_data.get('error')}")
                return
            
            time.sleep(1)

        # Get Report
        report_res = requests.get(f"{API_URL}/report/{job_id}")
        report_data = report_res.json()
        
        if not report_data:
            print("No violations detected.")
            return

        # Save to CSV
        df = pd.DataFrame(report_data)
        df.to_csv(REPORT_FILE, index=False)
        print(f"Report saved to {REPORT_FILE}")
        print(df.head())

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_helmet_detection()
