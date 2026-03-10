import requests
import time

BASE_URL = "http://localhost:8000"

def trigger_job():
    print("Uploading test.mp4...")
    with open("test.mp4", "rb") as f:
        files = {"file": ("test.mp4", f, "video/mp4")}
        data = {"case_type": "anpr"}
        response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
    
    if response.status_code == 200:
        job_id = response.json().get("job_id")
        print(f"Job started: {job_id}")
        return job_id
    else:
        print(f"Error starting job: {response.text}")
        return None

def monitor_job(job_id):
    while True:
        response = requests.get(f"{BASE_URL}/status/{job_id}")
        status = response.json().get("status")
        print(f"Job Status: {status}")
        if status in ["completed", "failed", "error"]:
            break
        time.sleep(5)
    
    if status == "completed":
        print("Job completed! Checking report...")
        report_res = requests.get(f"{BASE_URL}/report/{job_id}")
        print(f"Report length: {len(report_res.json())}")
        
        print("Checking analytics...")
        ana_res = requests.get(f"{BASE_URL}/traffic-analytics", params={"job_id": job_id})
        print(f"Analytics: {ana_res.json()}")

if __name__ == "__main__":
    jid = trigger_job()
    if jid:
        monitor_job(jid)
