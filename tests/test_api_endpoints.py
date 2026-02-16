import requests
import time
import os

API_BASE = "http://localhost:8000"
VIDEO_FILE = r"d:\Python\Smart-Traffic-Analytics\ANPR\new.mp4"

def test_api():
    print("--- Starting API Integration Test ---")
    
    # 1. Health Check
    try:
        resp = requests.get(f"{API_BASE}/")
        print(f"Health Check: {resp.status_code} - {resp.json()}")
    except Exception as e:
        print(f"Health Check Failed: {e}")
        return

    # 2. Upload Video
    if not os.path.exists(VIDEO_FILE):
        print(f"Test video not found: {VIDEO_FILE}")
        return

    print(f"Uploading video: {os.path.basename(VIDEO_FILE)}...")
    with open(VIDEO_FILE, 'rb') as f:
        files = {'file': (os.path.basename(VIDEO_FILE), f, 'video/mp4')}
        data = {'case_type': 'anpr'} # Testing ANPR case
        resp = requests.post(f"{API_BASE}/upload", files=files, data=data)
    
    if resp.status_code != 200:
        print(f"Upload Failed: {resp.status_code} - {resp.text}")
        return
    
    job_id = resp.json().get("job_id")
    print(f"Upload Success! Job ID: {job_id}")

    # 3. Poll Status
    print("Polling status...")
    status = "pending"
    while status in ["pending", "processing"]:
        time.sleep(5)
        resp = requests.get(f"{API_BASE}/status/{job_id}")
        if resp.status_code == 200:
            status = resp.json().get("status")
            print(f"Current Status: {status}")
        else:
            print(f"Status Check Failed: {resp.status_code}")
            break
            
    if status == "completed":
        # 4. Fetch Report
        print("Fetching violation report...")
        resp = requests.get(f"{API_BASE}/report/{job_id}")
        if resp.status_code == 200:
            report = resp.json()
            print(f"Report Received: {len(report)} detections found.")
            if len(report) > 0:
                print("First detection sample:", report[0])
        else:
            print(f"Report Fetch Failed: {resp.status_code}")

    print("--- API Test Completed ---")

if __name__ == "__main__":
    test_api()
