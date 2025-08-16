# src/detection/__init__.py
from typing import List
from .types import Detection


def detect(image_path: str, cfg: dict) -> List[Detection]:
    """
    Dispatcher that selects a detector from cfg['detect']['model'].
    Supported: 'rb_vehicle', 'yolo'
    """
    model_key = str(cfg.get("detect").get("model", "rb_vehicle"))

    if model_key in ("rb_vehicle", "rbv", "rb"):
        from .detectors.rb_vehicle import detect as rb_detect
        return rb_detect(image_path, cfg)
    elif model_key in ("yolo", "yolov8", "ultralytics"):
        from .detectors.yolo import detect as yolo_detect
        return yolo_detect(image_path, cfg)
    else:
        raise ValueError(f"Unsupported detector '{model_key}'. Use one of: rb_vehicle, yolo.")
