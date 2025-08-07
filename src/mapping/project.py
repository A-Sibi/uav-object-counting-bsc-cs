# src/mapping/project.py
from typing import List, TypedDict
import numpy as np
from src.detection.detector import Detection, TranslatedDetection

def project_detections(dets_per_frame: List[Detection], H_list: List[np.ndarray]) -> List[TranslatedDetection]:
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