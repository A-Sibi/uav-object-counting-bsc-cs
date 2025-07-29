# src/detection/detector.py

from ultralytics import YOLO
import numpy as np

def detect_cars(frames, cfg):
    """
    Run YOLO on a list of frames and return a list of list-of-dicts:
      [
        [ {'x1':..., 'y1':..., 'x2':..., 'y2':..., 'conf':...},  ... ],
        [ ... ],
        ...
      ]
    """
    model = YOLO(cfg["model"])
    results = model(frames, conf=cfg["conf"], iou=cfg["iou"])

    all_dets = []
    for res in results:  # one Results object per frame
        dets = []
        # res.boxes is a Boxes object with .xyxy, .conf, .cls
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            # if you only want class 'car' in COCO (typically cls==2)
            if cls == 2:
                dets.append({
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "conf": conf
                })
        all_dets.append(dets)
    return all_dets
