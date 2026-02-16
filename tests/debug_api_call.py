import sys
import os
import uuid

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set env var
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from api import process_video_task, jobs

# Verify jobs dict exists (it's imported)
print("Imported process_video_task.")

# Find latest video in uploads
upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
files = [os.path.join(upload_dir, f) for f in os.listdir(upload_dir) if f.endswith(".mp4")]
if not files:
    print("No videos found.")
    sys.exit(1)

# Sort by modification time
latest_video = max(files, key=os.path.getmtime)
print(f"Testing with video: {latest_video}")

job_id = "debug_test_job"
output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', f"debug_output_{uuid.uuid4().hex[:8]}.mp4")

# Mock job entry
jobs[job_id] = {"job_id": job_id, "status": "pending", "case_type": "triple", "report": []}

try:
    print("Starting process_video_task...")
    process_video_task(job_id, latest_video, output_path, "triple")
    print("Task completed successfully.")
    print("Status:", jobs[job_id]["status"])
    if jobs[job_id]["status"] == "error":
        print("Error in job:", jobs[job_id].get("error"))
except Exception as e:
    print("CRITICAL EXCEPTION:", e)
    import traceback
    traceback.print_exc()
