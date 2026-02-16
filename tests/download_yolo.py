import requests
import os

url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
output = "yolov8n.pt"

if not os.path.exists(output):
    print(f"Downloading {output}...")
    try:
        r = requests.get(url, allow_redirects=True)
        open(output, 'wb').write(r.content)
        print("Download complete.")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Model already exists.")
