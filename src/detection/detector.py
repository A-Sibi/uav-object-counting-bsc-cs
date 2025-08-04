# src/detection/detector.py
from typing import List, TypedDict, Dict
import numpy as np
from ultralytics import YOLO
from PIL import Image
from rfdetr import RFDETRBase


class Detection(TypedDict):
    """Single detection: bounding box (xyxy) and confidence score."""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float


def detect_cars_YOLO(image_path: str, cfg: dict) -> List[Detection]:
    """
    Run YOLO object detection on a single image.

    Args:
      image_path: Path to the input image file.
      cfg: detection config dict containing:
        - model: path or name of YOLO weights
        - conf: confidence threshold
        - iou: IoU threshold for NMS
        - car_class (optional): class index for 'car' in your model (default 2)
    Returns:
        List[Detection]: Each detection dict has keys:
            - x1, y1, x2, y2: bounding box coordinates (float)
            - conf: confidence score (float)
    """
    # Load model
    model = YOLO(cfg["model"])

    # Perform inference on the single image
    results = model(image_path, conf=cfg.get("conf", 0.25), iou=cfg.get("iou", 0.45))

    # There will be one Results object for this image
    res = results[0]
    detections = []
    car_cls = cfg.get("car_class", 2)

    # Iterate over each detected box
    for box in res.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        # Filter for cars only
        if cls == car_cls:
            detections.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": conf
            })

    return detections


def detect_cars_rb_vehicle(image_path: str, cfg: dict) -> List[Detection]:
    """
    Run RB vehicle detection (rebotnix/rb_vehicle)

    Args:
      image_path: Path to the input image file.
      cfg: detection config dict containing:
        - model: path or name of RB vehicle weights
        - conf: confidence threshold
        - iou: IoU threshold for NMS
    Returns:
        List[Detection]: Each detection dict has keys:
            - x1, y1, x2, y2: bounding box coordinates (float)
            - conf: confidence score (float)
    """

    model_path= "./rb_vehicle.pth"
    CLASS_NAMES = ["vehicle"]
    model = RFDETRBase(pretrain_weights=model_path,num_classes=len(CLASS_NAMES))

    image = Image.open(image_path)

    result = model.predict(image, threshold=0.15)

    def to_dicts(result, conf_threshold=0.0):
      xyxy = getattr(result, "xyxy")
      confs = getattr(result, "confidence")  # or result.conf if that’s what your wrapper uses
      return [
          {
              "x1": float(x1),
              "y1": float(y1),
              "x2": float(x2),
              "y2": float(y2),
              "conf": float(conf),
          }
          for (x1, y1, x2, y2), conf in zip(xyxy, confs)
          if conf >= conf_threshold
      ]

    detections = to_dicts(result)

    # filter confidence based on config
    threshold = float(cfg.get("detect", {}).get("conf", 0.0))
    detections = [d for d in detections if d.get("conf", 0.0) >= threshold]

    return detections