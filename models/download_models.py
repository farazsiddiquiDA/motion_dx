"""
Model downloader for MediaPipe Tasks API models.
Downloads pose_landmarker and face_landmarker models if not already cached.
"""

import os
import urllib.request

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_URLS = {
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
}

def ensure_models_exist():
    """Ensure all required MediaPipe task models are downloaded."""
    paths = {}
    for filename, url in MODEL_URLS.items():
        target_path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(target_path) or os.path.getsize(target_path) < 1000:
            print(f"Downloading {filename} from {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req) as resp, open(target_path, "wb") as f:
                f.write(resp.read())
            print(f"Successfully downloaded {filename} ({os.path.getsize(target_path)} bytes).")
        paths[filename] = target_path
    return paths

if __name__ == "__main__":
    ensure_models_exist()
