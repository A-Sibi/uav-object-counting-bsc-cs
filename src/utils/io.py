# src/utils/io.py

import yaml
from pathlib import Path
import cv2

def load_config(path: str) -> dict:
    """
    Naloži YAML konfiguracijo iz datoteke.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    """
    Ustvari mapo, če še ne obstaja.
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def extract_frames(video_path: str, out_dir: str, step: int = 5) -> list[str]:
    """
    Iz videa izlušči vsak `step`-ti frejm in ga shrani kot JPEG v `out_dir`.
    Vrne seznam poti do shranjenih slik.
    """
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Ne morem odpreti videa: {video_path}")

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
