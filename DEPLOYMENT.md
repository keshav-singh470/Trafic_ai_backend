# Deployment & Setup Guide - ANPR System

This guide provides instructions for setting up and running the Smart Traffic AI backend (ANPR-System) in both local development and production environments.

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **FFmpeg**: Required for video processing
- **GPU (Optional but Recommended)**: NVIDIA GPU with CUDA support for faster inference (YOLOv8/v11)

## 🛠️ Local Setup (Windows/Linux/Mac)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ranjeetray6120/anpr-system.git
   cd ANPR-System
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Models**:
   Ensure your `.pt` model files are in the `models/` directory or the root directory as specified in `api.py`.

5. **Run the API Server**:
   ```bash
   # Using the batch file (Windows):
   .\run_api.bat
   
   # Or manually:
   python api.py
   ```
   The API will be available at `http://localhost:8000`.

---

## 🚀 Production Deployment (Ubuntu/Linux)

For production (e.g., AWS EC2), it is recommended to run the API using `uvicorn` and `Gunicorn` as a background service, with `Nginx` as a reverse proxy.

### 1. Nginx Configuration
Update your Nginx config (`/etc/nginx/sites-available/default`) to allow large video uploads:

```nginx
server {
    server_name ai.ranjeetdev.online;
    client_max_body_size 100M; # Crucial for video uploads

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```
Restart Nginx: `sudo systemctl restart nginx`

### 2. Background Service (Systemd)
Create a service file: `sudo nano /etc/systemd/system/anpr-api.service`

```ini
[Unit]
Description=ANPR API Service
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/anpr-system
Environment="PATH=/home/ubuntu/anpr-system/venv/bin"
ExecStart=/home/ubuntu/anpr-system/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl enable anpr-api
sudo systemctl start anpr-api
```

---

## 📁 Directory Structure
The system automatically creates these folders when needed:
- `uploads/`: Temporary storage for uploaded videos.
- `outputs/`: Processed videos.
- `outputs/assets/`: Frames and cropped images for violations.

## 🔍 Troubleshooting

- **413 Request Entity Too Large**: Check the `client_max_body_size` in Nginx.
- **CORS Errors**: Ensure the frontend `API_BASE` matches your backend URL (e.g., `https://ai.ranjeetdev.online`).
- **Memory Errors**: Ensure your server has enough RAM/VRAM for YOLO processing. Use smaller models (e.g., `yolo11n.pt`) for low-memory environments.
