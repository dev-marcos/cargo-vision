import os
import urllib.request

def download_yolov8m_if_missing():
    """Downloads yolov8m.pt into models/ if it does not exist."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
    os.makedirs(models_dir, exist_ok=True)

    dest_path = os.path.join(models_dir, "yolov8m.pt")

    if os.path.exists(dest_path):
        print(f"yolov8m.pt already exists at {dest_path}")
        return dest_path

    print("Downloading yolov8m.pt into models/ ...")

    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"
    urllib.request.urlretrieve(url, dest_path)

    print(f"Downloaded yolov8m.pt to: {dest_path}")
    return dest_path
