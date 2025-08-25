# src/eval/gt_to_mosaic.py
from __future__ import annotations
import numpy as np
import cv2
from typing import List, Dict

def warp_boxes(boxes: List[Dict], H: np.ndarray) -> List[Dict]:
    """
    Pretvori GT bokse iz koordinat GT-slike v mozaik s homog. matriko H (GT->mosaic).
    """
    pts = []
    for b in boxes:
        pts.extend([(b["x1"],b["y1"]), (b["x2"],b["y2"])])
    pts = np.array(pts, dtype=np.float32).reshape(-1,1,2)
    wpts = cv2.perspectiveTransform(pts, H).reshape(-1,2)

    out = []
    for i in range(0, len(wpts), 2):
        (x1,y1) = wpts[i]; (x2,y2) = wpts[i+1]
        x1,x2 = sorted([x1,x2]); y1,y2 = sorted([y1,y2])
        out.append({"x1":float(x1), "y1":float(y1), "x2":float(x2), "y2":float(y2)})
    return out
