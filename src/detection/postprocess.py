# src/utils/postprocess.py

from __future__ import annotations
import numpy as np
import cv2
from typing import Dict, List
from src.detection.types import Detection

# ---------- small filter helpers ----------

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

def _shrink_box(b, frac=0.10):
    x1,y1,x2,y2 = map(float, b)
    w = x2-x1; h = y2-y1
    dx = frac * w; dy = frac * h
    return np.array([x1+dx, y1+dy, x2-dx, y2-dy], dtype=float)

def _crop_gray(image_bgr, b):
    x1,y1,x2,y2 = map(int, np.round(b))
    H, W = image_bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = image_bgr[y1:y2, x1:x2]
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

def _edge_frac_auto(gray_u8):
    """
    Fraction of edge pixels using auto Canny (median heuristic).
    Lower fraction ~ empty/flat regions (asphalt); higher ~ objects/lines.
    """
    v = float(np.median(gray_u8))
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray_u8, lo, hi)
    return float((edges > 0).mean())   # 0..1

# popular choice for some reason, should check in more detail.
def _nms(dets, iou_thresh=0.5):
    if not dets: return []
    boxes = np.array([_box_to_arr(d) for d in dets])
    scores = np.array([d.get('conf', 0.0) for d in dets])
    order = scores.argsort()[::-1]
    keep = []
    x1,y1,x2,y2 = boxes.T
    areas = (x2-x1)*(y2-y1)
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1)*np.maximum(0, yy2-yy1)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return [dets[i] for i in keep]

# ---------- process helpers ----------

def _should_run(block: Dict[str, any] | None, default: bool = True) -> bool:
    """Return True/False depending on block['run'], with safe defaults."""
    return bool((block or {}).get("run", default))


def _log(stage: str, count: int, extra: str = ""):
    """Aligned logging for cleaner stages."""
    print(f"[INFO] Postprocessing: {count:5d} detections | {stage}{extra}")


# ---------- main cleaner ----------

def clean_projected_detections(
    mosaic_image: np.ndarray,
    detections: List[Detection],
    cfg: Dict[str, any]
) -> List[Detection]:
    """
    Clean detections projected into mosaic coordinates using config-driven knobs.

    Each stage has a `run:` flag in YAML. Default is True.
    """

    start_num = len(detections)
    _log("on start", start_num)
    if not detections:
        return []

    post = cfg.get("postprocess", {}) or {}
    cf   = post.get("confidence", {}) or {}
    sz   = post.get("size_quantiles", {}) or {}
    prox = post.get("proximity_clustering", {}) or {}
    sup  = post.get("support", {}) or {}
    tex  = post.get("texture_rejection", {}) or {}

    # thresholds
    conf_min   = float(cf.get('conf_min', 0.25))
    size_low_q = float(sz.get("size_low_q",  0.10))
    size_high_q= float(sz.get("size_high_q", 0.95))
    prox_factor= float(prox.get("prox_factor", 0.45))
    iou_merge  = float(prox.get("iou_merge", 0.50))
    min_support_frames  = int(sup.get("min_support_frames", 4))
    min_support_members = int(sup.get("min_support_members", 6))
    inner_shrink   = float(tex.get("inner_shrink", 0.10))
    empty_std_max  = float(tex.get("empty_std_max", 15.0))
    empty_edge_max = float(tex.get("empty_edge_max", 0.02))

    dets: List[Detection] = list(detections)

    # --- 1) Confidence ---
    if _should_run(cf, True):
        dets = [d for d in dets if d.get('conf', 0.0) >= conf_min]
        _log("after confidence filter", len(dets), f" (min={conf_min})")
        if not dets: return []
    else:
        print("[SKIP] Confidence filter disabled")

    # --- 2) Size quantiles ---
    if _should_run(sz, True):
        boxes = np.array([_box_to_arr(d) for d in dets])
        areas = np.array([_area(b) for b in boxes]) if len(dets) else np.array([])
        if areas.size == 0: return []
        lo = float(np.quantile(areas, size_low_q))
        hi = float(np.quantile(areas, size_high_q))
        keep = (areas >= max(1.0, lo)) & (areas <= hi)
        dets = [dets[i] for i in np.where(keep)[0]]
        _log("after size filter", len(dets), f" (low={lo:.1f}, high={hi:.1f})")
        if not dets: return []
    else:
        print("[SKIP] Size filter disabled")

    # --- 3) Proximity clustering ---
    if _should_run(prox, True):
        boxes = np.array([_box_to_arr(d) for d in dets])
        scores = np.array([d.get('conf', 1.0) for d in dets])
        centers = np.array([_center(b) for b in boxes])
        diags = np.array([_diag(b) for b in boxes])
        radius = prox_factor * (np.median(diags) if diags.size else 0.0)

        order = np.argsort(-scores)
        visited = np.zeros(len(dets), dtype=bool)
        fused: List[Detection] = []

        for idx in order:
            if visited[idx]:
                continue
            c_members = [idx]
            visited[idx] = True
            for j in order:
                if visited[j]: continue
                close = np.linalg.norm(centers[j] - centers[idx]) <= radius
                iouok = _iou(boxes[idx], boxes[j]) >= iou_merge
                if close or iouok:
                    c_members.append(j)
                    visited[j] = True

            member_frames   = [dets[i].get('frame_idx', None) for i in c_members]
            support_frames  = len({f for f in member_frames if f is not None})
            support_members = len(c_members)

            B = [boxes[i] for i in c_members]
            S = [scores[i] for i in c_members]
            fused_box, fused_conf = _weighted_fusion(B, S)

            best = c_members[int(np.argmax([scores[i] for i in c_members]))]
            out = {k: dets[best][k] for k in dets[best].keys()
                   if k not in ('x1','y1','x2','y2','conf')}
            out.update({
                'x1': float(fused_box[0]), 'y1': float(fused_box[1]),
                'x2': float(fused_box[2]), 'y2': float(fused_box[3]),
                'conf': float(fused_conf),
                'support_members': int(support_members),
                'support_frames': int(support_frames),
            })
            fused.append(out)

        dets = fused
        _log("after clustering", len(dets),
             f" (radius={radius:.1f}, IoU={iou_merge:.2f})")
        if not dets: return []
    else:
        print("[SKIP] Clustering disabled")

    # --- 4) Support ---
    if _should_run(sup, True):
        before = len(dets)
        dets = [
            d for d in dets
            if d.get('support_frames', 0) >= min_support_frames
            or d.get('support_members', 0) >= min_support_members
        ]
        _log("after support filter", len(dets),
             f" (frames≥{min_support_frames} OR members≥{min_support_members}, dropped {before-len(dets)})")
        if not dets: return []
    else:
        print("[SKIP] Support filter disabled")

    # --- 5) Texture ---
    if _should_run(tex, True):
        kept: List[Detection] = []
        for d in dets:
            b = _box_to_arr(d)
            if _area(b) < 16.0:
                continue
            b_in = _shrink_box(b, inner_shrink)
            patch_gray = _crop_gray(mosaic_image, b_in)
            if patch_gray is None or patch_gray.size < 25:
                continue
            std = float(patch_gray.std())
            efr = _edge_frac_auto(patch_gray)
            if (std <= empty_std_max) and (efr <= empty_edge_max):
                continue
            d['tex_std'] = std
            d['tex_edge_frac'] = efr
            kept.append(d)

        dets = kept
        _log("after texture filter", len(dets),
             f" (std>{empty_std_max}, edges>{empty_edge_max:.3f})")
    else:
        print("[SKIP] Texture filter disabled")

    return dets
