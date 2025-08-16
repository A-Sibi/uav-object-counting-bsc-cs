from typing import List
from PIL import Image
from rfdetr import RFDETRBase
import os

from src.detection.types import Detection, detection_id_counter


_RBV_MODEL = None  # Global variable to cache the model
def detect(image_path: str, cfg: dict) -> List[Detection]:
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
    global _RBV_MODEL
    CLASS_NAMES = ["vehicle"]
    if _RBV_MODEL is None:
        model_path = "./rb_vehicle.pth"
        if not os.path.exists(model_path):  
            raise FileNotFoundError(f"Model file not found: {model_path}")
        _RBV_MODEL = RFDETRBase(pretrain_weights=model_path, num_classes=len(CLASS_NAMES), class_names=CLASS_NAMES)
        if hasattr(_RBV_MODEL, "optimize_for_inference"):
            _RBV_MODEL.optimize_for_inference()
        # optionally: _RBV_MODEL.eval()

    model = _RBV_MODEL
    model_path= "./rb_vehicle.pth"

    image = Image.open(image_path)

    result = model.predict(image, threshold=0.15)

    def to_dicts(result, conf_threshold=0.0):
      xyxy = getattr(result, "xyxy")
      confs = getattr(result, "confidence")
      return [
          {
            "id": detection_id_counter.next_id(),  # Generate a unique ID for each detection
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