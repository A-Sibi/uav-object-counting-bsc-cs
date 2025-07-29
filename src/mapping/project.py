# src/mapping/project.py
import numpy as np

def project_points(dets_per_frame, H_list):
    """
    Premakni detekcije iz koordinat posameznih frejmov v koordinate mozaika.

    Args:
        dets_per_frame: list of lists of detection dicts
        H_list: list of 3x3 homography np.ndarray, one per frame
    Returns:
        List of dicts each with original box data plus:
          - frame_idx: int
          - x_m, y_m: float  (coordinates on the mosaic)
    """
    projected = []
    for i, dets in enumerate(dets_per_frame):
        H = H_list[i]
        for d in dets:
            # centroid of the box
            x = (d["x1"] + d["x2"]) / 2.0
            y = (d["y1"] + d["y2"]) / 2.0
            p = np.array([x, y, 1.0])
            p_m = H @ p
            x_m, y_m = p_m[0] / p_m[2], p_m[1] / p_m[2]
            projected.append({**d, "frame_idx": i, "x_m": x_m, "y_m": y_m})
    return projected
