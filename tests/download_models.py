import urllib.request
import os

def download_file(url, filename):
    print(f"Downloading {url} to {filename}...")
    try:
        # User-agent header to avoid simple blocking
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, filename)
        file_size = os.path.getsize(filename)
        if file_size < 1000: # Very small file means it probably failed/redirected to login
            print(f"Warning: {filename} is suspiciously small ({file_size} bytes). It might be an error page.")
        else:
            print(f"Successfully downloaded {filename} ({file_size} bytes)")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Public GitHub raw links for specialized models
configs = [
    {
        "url": "https://github.com/meryemsakin/helmet-detection-yolov8/raw/main/best.pt",
        "name": "models/helmet_specialized.pt"
    },
    {
        "url": "https://github.com/HayaAbdullahM/Seat-Belt-Detection/raw/YOLOv8/best.pt", # Adjusted based on common structure
        "name": "models/seatbelt_specialized.pt"
    }
]

for cfg in configs:
    download_file(cfg["url"], cfg["name"])
