# src/mapping/project.py
from typing import List, TypedDict
import numpy as np
from src.detection.detector import Detection, TranslatedDetection

def project_detections(dets_per_frame, H_list, mosaic_shape=None):
    projected = []
    H, W = (mosaic_shape[:2] if mosaic_shape is not None else (None, None))

    for i, dets in enumerate(dets_per_frame):
        Hf = H_list[i]
        if Hf is None:  # skip frames whose H failed
            continue
        for d in dets:
            x1_b, y1_b, x2_b, y2_b = d['x1'], d['y1'], d['x2'], d['y2']
            corners = np.array([[x1_b, y1_b, 1.0],
                                [x2_b, y1_b, 1.0],
                                [x2_b, y2_b, 1.0],
                                [x1_b, y2_b, 1.0]], dtype=float)

            mapped = (Hf @ corners.T).T
            w = mapped[:, 2:3]
            if not np.all(np.isfinite(mapped)) or np.any(np.abs(w) < 1e-6):
                continue  # numerical trash

            mapped_xy = mapped[:, :2] / w
            xs, ys = mapped_xy[:, 0], mapped_xy[:, 1]
            xm1, ym1 = float(xs.min()), float(ys.min())
            xm2, ym2 = float(xs.max()), float(ys.max())

            # sanity (clamp + size check)
            if W is not None:
                xm1, xm2 = np.clip([xm1, xm2], 0, W-1)
                ym1, ym2 = np.clip([ym1, ym2], 0, H-1)
            if (xm2 - xm1) < 3 or (ym2 - ym1) < 3:
                continue  # too tiny
            if W is not None and ((xm2 - xm1) > W * 0.5 or (ym2 - ym1) > H * 0.5):
                continue  # blown up

            projected.append({
                'x1': xm1, 'y1': ym1, 'x2': xm2, 'y2': ym2,
                'conf': d.get('conf', 1.0), 'frame_idx': i,
                'x1_b': x1_b, 'y1_b': y1_b, 'x2_b': x2_b, 'y2_b': y2_b
            })
    return projected


def _project_detections(dets_per_frame: List[Detection], H_list: List[np.ndarray]) -> List[TranslatedDetection]:
    """
    Apply homography transformation to a list of detections.

    Attributes
    ----------
    dets_per_frame : List[Detection]
        List of detections for each frame, where each detection is a dict with keys:
          - x1, y1, x2, y2: float (bounding box coordinates)
          - conf: float (confidence score)
    H_list : List[np.ndarray]
        List of 3x3 homography np.ndarray, one per frame
    Returns
    -------
    List[TranslatedDetection]
        List of dicts each with transformed box data, saved original coordinates, and frame index:
          - x1, y1, x2, y2: float (transformed bounding box coordinates)
          - conf: float (confidence score) 
          - frame_idx: int (index of the frame)
          - x1_b, y1_b, x2_b, y2_b: float (original bounding box coordinates)
    """
    projected: List[TranslatedDetection] = []
    for i, dets in enumerate(dets_per_frame):
        H = H_list[i]
        for d in dets:
            # original box coords
            x1_b, y1_b = d['x1'], d['y1']
            x2_b, y2_b = d['x2'], d['y2']
            # corners of the box in homogeneous coords
            corners = np.array([
                [x1_b, y1_b, 1.0],
                [x2_b, y1_b, 1.0],
                [x2_b, y2_b, 1.0],
                [x1_b, y2_b, 1.0]
            ], dtype=float)
            # apply homography
            mapped = (H @ corners.T).T  # shape (4,3)
            # normalize to get (x,y)
            mapped_xy = mapped[:, :2] / mapped[:, 2:3]
            xs, ys = mapped_xy[:, 0], mapped_xy[:, 1]
            # new bbox in mosaic coords
            x1_m, y1_m = float(xs.min()), float(ys.min())
            x2_m, y2_m = float(xs.max()), float(ys.max())

            # build translated detection
            td: TranslatedDetection = {
                'x1': x1_m, 'y1': y1_m,
                'x2': x2_m, 'y2': y2_m,
                'conf': d['conf'],
                'frame_idx': i,
                'x1_b': x1_b, 'y1_b': y1_b,
                'x2_b': x2_b, 'y2_b': y2_b
            }
            projected.append(td)
    return projected