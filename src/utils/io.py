# src/utils/io.py

import datetime
import json
import numpy as np
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


def load_np_image(image_path: str) -> np.ndarray:
    """
    Load an image from the specified path using OpenCV.
    
    :param image_path: Path to the image file.
    :return: Loaded image as a numpy array.
    """

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def save_np_image(image: np.ndarray, path: str) -> None:
    """
    Save an image to the specified path.
    
    :param image: Image to save (numpy array).
    :param path: Path where the image will be saved.
    """
    ensure_dir(Path(path).parent)
    parent = Path(path).parent
    ensure_dir(str(parent))
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


def save_homographies_json(H_list, img_paths, out_dir, meta=None, filename=None):
    """
    Save a list of 3x3 homographies to JSON with the corresponding image order.
    Args:
        H_list: List[np.ndarray] of shape (3,3)
        img_paths: List[pathlib.Path] of input frames, same ordering used for H_list
        out_dir: Directory to save JSON into
        meta: Optional dict of extra metadata (resize, stitch params, mosaic size, etc.)
        filename: Optional filename; default is timestamped
    Returns:
        pathlib.Path to the saved JSON file
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": 1,
        "created_at": datetime.datetime.now().isoformat(),
        "images_order": [p.name for p in img_paths],
        "homographies": [H.astype(float).tolist() for H in H_list],
        "meta": meta or {}
    }

    if filename is None:
        filename = f"h_list_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    out_path = out_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def load_homographies_json(path):
    """
    Load homographies JSON written by save_homographies_json().
    Returns:
        H_list: List[np.ndarray] of shape (3,3)
        images_order: List[str]
        meta: dict
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    H_list = [np.array(H, dtype=np.float64) for H in data["homographies"]]
    images_order = data.get("images_order", [])
    meta = data.get("meta", {})
    return H_list, images_order, meta


def latest_json_file(dir_path: str | Path) -> Path:
    dir_path = Path(dir_path)
    cand = sorted(dir_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cand:
        raise FileNotFoundError(f"No JSON files found in {dir_path}")
    return cand[0]


def coerce_dets_schema(data) -> list[dict]:
    """
    Accept either:
      - a list of detection dicts, or
      - {"detections": [...]}
    Ensure keys x1,y1,x2,y2 exist; set conf=1.0 if missing.
    """
    if isinstance(data, dict) and "detections" in data:
        data = data["detections"]
    if not isinstance(data, list):
        raise ValueError("Unsupported detections JSON schema; expected list or {'detections': [...]}.")

    clean = []
    for d in data:
        if not all(k in d for k in ("x1", "y1", "x2", "y2")):
            continue
        dd = {
            "x1": float(d["x1"]),
            "y1": float(d["y1"]),
            "x2": float(d["x2"]),
            "y2": float(d["y2"]),
            "conf": float(d.get("conf", 1.0)),
        }
        clean.append(dd)
    return clean
