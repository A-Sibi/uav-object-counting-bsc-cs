# src/detection/detector.py

from ultralytics import YOLO
import numpy as np

def detect_cars(image, cfg):
    """
    Run YOLO on a single image and return a list of detected cars:
      [
        {'x1':..., 'y1':..., 'x2':..., 'y2':..., 'conf':...},
        ...
      ]

    Args:
      image: a NumPy array or image path supported by YOLO
      cfg: detection config dict containing:
        - model: path or name of YOLO weights
        - conf: confidence threshold
        - iou: IoU threshold for NMS
        - car_class (optional): class index for 'car' in your model (default 2)
    """
    # Load model
    model = YOLO(cfg["model"])

    # Perform inference on the single image
    results = model(image, conf=cfg.get("conf", 0.25), iou=cfg.get("iou", 0.45))

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