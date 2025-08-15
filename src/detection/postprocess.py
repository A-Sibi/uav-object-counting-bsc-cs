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

def clean_projected_detections(mosaic_image: np.ndarray, detections: list[Detection], cfg: Dict[str, any]) -> list[Detection]:
    """
    Clean detections projected into mosaic coordinates using config-driven knobs.

    Config keys (under cfg['postprocess']):
        conf_min: float            # min confidence to keep
        size_low_q: float          # lower area quantile (0..1)
        size_high_q: float         # upper area quantile (0..1)
        prox_factor: float         # cluster radius = prox_factor * median diag
        iou_merge: float           # IoU threshold to cluster even if centers close
        texture_min: float         # min Laplacian variance to accept

    Returns:
        List[Detection]: fused and filtered detections.
    """
    start_num = len(detections)
    print(f"[INFO] Postprocessing: {start_num} detections on start")
    if not detections:
        return []
    
    # Extract config parameters
    post = cfg.get("postprocess", {})
    conf_min   = float(post.get("conf_min",   0.25))
    size_low_q = float(post.get("size_low_q", 0.10))
    size_high_q= float(post.get("size_high_q",0.98))
    prox_factor= float(post.get("prox_factor",0.45))
    iou_merge  = float(post.get("iou_merge",  0.10))
    texture_min= float(post.get("texture_min",45.0))

    # --- 1) Basic filter by confidence
    dets = [d for d in detections if d.get('conf', 0.0) >= conf_min]
    if not dets:
        return []
    
    print(f"[INFO] Postprocessing: {len(dets)} detections after confidence filter (min conf={conf_min})")

    # --- 2) Filter by size quantiles to kill tiny partials + huge blown boxes
    boxes = np.array([_box_to_arr(d) for d in dets])
    areas = np.array([_area(b) for b in boxes])
    # robust size gates
    
    lo = np.quantile(areas, size_low_q)
    hi = np.quantile(areas, size_high_q)
    keep = (areas >= max(1.0, lo)) & (areas <= hi)
    dets = [dets[i] for i in np.where(keep)[0]]
    if not dets:
        return []
    
    print(f"[INFO] Postprocessing: {len(dets)} detections after size quantile filter (low={lo:.2f}, high={hi:.2f})")

    # --- 3) Proximity clustering (greedy, size-aware)
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

        member_frames = [dets[i].get('frame_idx', None) for i in c_members]
        support_frames = len({f for f in member_frames if f is not None})
        support_members = len(c_members)

        # fuse members
        B = [boxes[i] for i in c_members]
        S = [scores[i] for i in c_members]
        fused_box, fused_conf = _weighted_fusion(B, S)

        # carry over optional metadata from best member
        best = c_members[int(np.argmax([scores[i] for i in c_members]))]
        out = {k: dets[best][k] for k in dets[best].keys() if k not in ('x1','y1','x2','y2','conf')}
        out.update({'x1': float(fused_box[0]), 'y1': float(fused_box[1]),
                    'x2': float(fused_box[2]), 'y2': float(fused_box[3]),
                    'conf': float(fused_conf),
                    'support_members': int(support_members),
                    'support_frames': int(support_frames),
                    })
        fused.append(out)

    print(f"[INFO] Postprocessing: {len(fused)} detections after proximity clustering (radius={radius:.2f}, IoU={iou_merge:.2f})")

    # --- 4) Filter by support frames (number of unique frames that contributed detections to a fused cluster)
    min_support_frames = int(post.get("min_support_frames", 3))
    fused = [d for d in fused if d.get('support_frames', 1) >= min_support_frames]

    print(f"[INFO] Postprocessing: {len(fused)} detections after support frames filter (min={min_support_frames})")

    # --- 5) Texture rejection (cuts stray boxes on flat asphalt)
    cleaned = []
    for d in fused:
        b = _box_to_arr(d)
        # small sanity: drop if still gigantic (after fusion)
        if _area(b) < 4.0:
            continue
        tex = _crop_texture_var(mosaic_image, b)
        if tex >= texture_min:
            cleaned.append(d)

    print(f"[INFO] Postprocessing: {len(cleaned)} detections after texture filter (min texture={texture_min})")

    return cleaned
