# src/detection/detectors/yolo.py
from typing import List
from ultralytics import YOLO

from src.detection.types import Detection, detection_id_counter

def detect(image_path: str, cfg: dict) -> List[Detection]:
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
                "id": detection_id_counter.next_id(),  # Generate a unique ID for each detection
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": conf
            })

    return detections
