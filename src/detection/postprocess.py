# src/utils/postprocess.py

from __future__ import annotations
import numpy as np
import cv2
from typing import Dict, List
from src.detection.detector import Detection

def _box_to_arr(d: Detection) -> np.ndarray:
    return np.array([d['x1'], d['y1'], d['x2'], d['y2']], dtype=float)

def _area(b):  # b: [x1,y1,x2,y2]
    return max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))

def _iou(a, b):
    xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
    w = max(0.0, xx2-xx1); h = max(0.0, yy2-yy1)
    inter = w*h
    return inter / max(_area(a) + _area(b) - inter, 1e-6)

def _center(b):
    return np.array([(b[0]+b[2])/2.0, (b[1]+b[3])/2.0], dtype=float)

def _diag(b):
    return float(np.hypot(b[2]-b[0], b[3]-b[1]))

def _weighted_fusion(boxes, scores):
    """Weighted Boxes Fusion on coordinates."""
    w = np.asarray(scores, dtype=float)
    w = w / (w.sum() + 1e-9)
    B = np.asarray(boxes, dtype=float)
    fused = (B * w[:, None]).sum(axis=0)
    fused_conf = float(np.mean(scores))           # or max(scores)
    return fused, fused_conf

def _crop_texture_var(image, b):
    x1,y1,x2,y2 = map(int, np.round(b))
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(image.shape[1]-1, x2); y2 = min(image.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = image[y1:y2, x1:x2]
    if roi.size == 0: return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Laplacian variance: low on flat asphalt
    return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())

def clean_projected_detections(
    mosaic_image: np.ndarray,
    detections: list[Detection],
    cfg: Dict[str, float],
    conf_min: float = 0.25,
    size_low_q: float = 0.10,
    size_high_q: float = 0.98,
    prox_factor: float = 0.45,
    iou_merge: float = 0.1,
    texture_min: float = 45.0,
) -> list[Detection]:
    """
    3-stage cleaner:
      1) conf + size range
      2) proximity clustering + Weighted Boxes Fusion
      3) texture rejection
    Args:
        mosaic_image: BGR mosaic for texture check
        detections: list of {'x1','y1','x2','y2','conf', ...}
        conf_min: drop boxes below this
        size_low_q/size_high_q: keep boxes whose area is within [q10,q98] of the set
        prox_factor: cluster radius = prox_factor * median box diagonal
        iou_merge: boxes inside a cluster are merged regardless of IoU; IoU is used to
                   help avoid joining far overlaps when centers are close
        texture_min: minimum Laplacian variance to accept as non-flat object

    Returns:
        list of fused, filtered detection dicts (same keys; extra kept if present)
    """
    if not detections:
        return []

    # --- 1) Basic filter by confidence + robust size range
    dets = [d for d in detections if d.get('conf', 0.0) >= conf_min]
    if not dets:
        return []

    boxes = np.array([_box_to_arr(d) for d in dets])
    areas = np.array([_area(b) for b in boxes])
    # robust size gates to kill tiny partials + huge blown boxes
    lo = np.quantile(areas, size_low_q)
    hi = np.quantile(areas, size_high_q)
    keep = (areas >= max(1.0, lo)) & (areas <= hi)
    dets = [dets[i] for i in np.where(keep)[0]]
    if not dets:
        return []

    # --- 2) Proximity clustering (greedy, size-aware)
    boxes = np.array([_box_to_arr(d) for d in dets])
    scores = np.array([d.get('conf', 1.0) for d in dets])
    centers = np.array([_center(b) for b in boxes])
    diags = np.array([_diag(b) for b in boxes])
    radius = prox_factor * np.median(diags)  # pixels

    order = np.argsort(-scores)  # high conf first
    visited = np.zeros(len(dets), dtype=bool)
    fused = []

    for idx in order:
        if visited[idx]:
            continue
        # start a new cluster with this seed
        c_members = [idx]
        visited[idx] = True

        # add neighbors close in center OR with decent IoU
        for j in order:
            if visited[j]:
                continue
            if np.linalg.norm(centers[j] - centers[idx]) <= radius or _iou(boxes[idx], boxes[j]) >= iou_merge:
                c_members.append(j)
                visited[j] = True

        # fuse members
        B = [boxes[i] for i in c_members]
        S = [scores[i] for i in c_members]
        fused_box, fused_conf = _weighted_fusion(B, S)

        # carry over optional metadata from best member
        best = c_members[int(np.argmax([scores[i] for i in c_members]))]
        out = {k: dets[best][k] for k in dets[best].keys() if k not in ('x1','y1','x2','y2','conf')}
        out.update({'x1': float(fused_box[0]), 'y1': float(fused_box[1]),
                    'x2': float(fused_box[2]), 'y2': float(fused_box[3]),
                    'conf': float(fused_conf)})
        fused.append(out)

    # --- 3) Texture rejection (cuts stray boxes on flat asphalt)
    cleaned = []
    for d in fused:
        b = _box_to_arr(d)
        # small sanity: drop if still gigantic (after fusion)
        if _area(b) < 4.0:
            continue
        tex = _crop_texture_var(mosaic_image, b)
        if tex >= texture_min:
            cleaned.append(d)

    return cleaned
