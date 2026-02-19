"""
S3 Service — Presigned URL Architecture
========================================
Strategy:
  - Upload files to S3 (private bucket)
  - Store only the S3 *key* in MongoDB (not a public URL)
  - Generate presigned URLs on-demand when frontend needs to load media
  - Presigned URLs expire after PRESIGNED_EXPIRY_SECONDS (default 1 hour)
"""

import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from datetime import datetime

AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.getenv("AWS_REGION", "ap-south-1")
AWS_S3_BUCKET        = os.getenv("AWS_S3_BUCKET")  # Standardized name

# Presigned URL expiry — 1 hour (3600 seconds)
PRESIGNED_EXPIRY_SECONDS = 3600

_s3_client = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET]):
            print("⚠️ WARNING: S3 credentials or AWS_S3_BUCKET not fully configured. S3 operations will be skipped.")
            return None
        try:
            _s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION,
            )
            print(f"✅ S3 client initialized. Bucket: {AWS_S3_BUCKET}, Region: {AWS_REGION}")
        except Exception as e:
            print(f"❌ ERROR: Failed to create S3 client: {e}")
    return _s3_client


# ── Upload Functions (return S3 key, NOT public URL) ──────────────────────────

def upload_video(local_path: str, filename: str) -> str | None:
    """
    Upload a video to S3.
    Returns the S3 object KEY (e.g. 'videos/20250218_abc.mp4'), NOT a URL.
    """
    client = get_s3_client()
    if client is None:
        return None
    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        s3_key    = f"videos/{timestamp}_{filename}"
        client.upload_file(
            local_path,
            AWS_S3_BUCKET,
            s3_key,
            ExtraArgs={"ContentType": "video/mp4"}
        )
        print(f"✅ S3 video uploaded. Key: {s3_key}")
        return s3_key
    except NoCredentialsError:
        print("❌ ERROR: S3 credentials invalid.")
        return None
    except Exception as e:
        print(f"❌ ERROR: S3 video upload failed: {e}")
        return None


def upload_image(local_path: str, s3_key: str) -> str | None:
    client = get_s3_client()
    if client is None:
        return None
    try:
        client.upload_file(
            local_path,
            AWS_S3_BUCKET,
            s3_key,
            ExtraArgs={"ContentType": "image/jpeg"}
        )
        print(f"✅ S3 image uploaded. Key: {s3_key}")
        return s3_key
    except Exception as e:
        print(f"❌ ERROR: S3 image upload failed for {s3_key}: {e}")
        return None


def upload_file_to_s3(local_path: str, s3_key: str) -> str | None:
    """
    Generic upload function for any file.
    """
    client = get_s3_client()
    if client is None:
        return None
    try:
        content_type = "application/octet-stream"
        if local_path.endswith(".jpg") or local_path.endswith(".jpeg"):
            content_type = "image/jpeg"
        elif local_path.endswith(".mp4"):
            content_type = "video/mp4"

        client.upload_file(
            local_path,
            AWS_S3_BUCKET,
            s3_key,
            ExtraArgs={"ContentType": content_type}
        )
        print(f"✅ S3 file uploaded. Key: {s3_key}")
        return s3_key
    except Exception as e:
        print(f"❌ ERROR: Generic S3 upload failed for {s3_key}: {e}")
        return None


# ── Presigned URL Generator ───────────────────────────────────────────────────

def get_presigned_url(s3_key: str, expires_in: int = PRESIGNED_EXPIRY_SECONDS) -> str | None:
    """
    Generate a temporary presigned URL for a given S3 key.
    """
    if not s3_key:
        return None
    client = get_s3_client()
    if client is None:
        return None
    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": AWS_S3_BUCKET, "Key": s3_key},
            ExpiresIn=expires_in,
        )
        return url
    except Exception as e:
        print(f"❌ ERROR: Failed to generate presigned URL for key={s3_key}: {e}")
        return None


# Alias for backward compatibility if needed, but we should switch to get_presigned_url
generate_presigned_url = get_presigned_url


def generate_presigned_urls_for_report(report: list) -> list:
    """
    Given a report list (from jobs[job_id]['report']),
    replace all S3 keys with fresh presigned URLs.
    Fields processed: vehicle_image, plate_image, FullImgUrl, CropImgUrl
    Uses _s3_vehicle_key / _s3_plate_key if available to regenerate fresh URLs.
    Returns a new list with URLs filled in.
    """
    enriched = []
    for row in report:
        row = dict(row)  # copy so we don't mutate original

        # Determine keys for vehicle and plate images
        # Prefer _s3_... keys stored in the row
        s3_vehicle = row.get("_s3_vehicle_key")
        s3_plate   = row.get("_s3_plate_key")
        
        # Fallback: if _s3 keys missing, check if value itself looks like a key (no http)
        if not s3_vehicle:
             val = row.get("vehicle_image") or row.get("FullImgUrl")
             if val and not val.startswith("http") and not val.startswith("/"):
                 s3_vehicle = val
        
        if not s3_plate:
             val = row.get("plate_image") or row.get("CropImgUrl")
             if val and not val.startswith("http") and not val.startswith("/"):
                 s3_plate = val

        # Generate fresh URLs if keys found
        if s3_vehicle:
            url = get_presigned_url(s3_vehicle)
            if url:
                row["vehicle_image"] = url
                row["FullImgUrl"]    = url
        
        if s3_plate:
            url = get_presigned_url(s3_plate)
            if url:
                row["plate_image"] = url
                row["CropImgUrl"]  = url

        enriched.append(row)
    return enriched
