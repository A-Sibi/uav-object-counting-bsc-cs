# src/utils/io.py

import yaml
from pathlib import Path
import cv2

def load_config(path: str) -> dict:
    """
    Load YAML configuration from a file.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    cfg["paths"] = {k: str(Path(v)) for k, v in cfg.get("paths", {}).items()}
    return cfg


def ensure_dir(path: str):
    """
    Create directory if it doesn't exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_np_image(image, path: str):
    """
    Save an image to the specified path.
    
    :param image: Image to save (numpy array).
    :param path: Path where the image will be saved.
    """
    ensure_dir(path)
    cv2.imwrite(path, image)
    return None


def extract_frames(video_path: str, out_dir: str, step: int = 5) -> list[str]:
    """
    Extract every `step`-th frame from the video and save it as JPEG in `out_dir`.
    Returns a list of saved image paths.
    """
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    saved = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            out_path = Path(out_dir) / f"frame_{idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append(str(out_path))
        idx += 1

    cap.release()
    return saved
