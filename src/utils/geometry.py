# src/utils/geometry.py
from typing import Dict, List
import cv2
import numpy as np


def _resolve_method_flag(method: str) -> int:
    m = method.upper()
    if m == "RANSAC":
        return cv2.RANSAC
    if m == "LMEDS":
        return cv2.LMEDS
    # USAC variants (OpenCV >= 4.5)
    if hasattr(cv2, m):
        return getattr(cv2, m)
    raise ValueError(f"Unsupported method: {method}")


def compute_homography(img1, img2, cfg: dict[str: any]) -> np.ndarray:
    """
    Compute a homography mapping points from img1 -> img2 using config-driven
    feature matching and robust estimation.

    Expected YAML (cfg['homography']):
    homography:
      feature: SIFT            # SIFT | ORB
      nfeatures: 4000          # detector feature budget
      matcher: AUTO            # AUTO | BF | FLANN
      ratio: 0.7               # Lowe's ratio threshold (0..1)
      method: USAC_MAGSAC      # RANSAC | LMEDS | USAC_MAGSAC | USAC_FAST | USAC_ACCURATE
      reproj_thresh: 3.0       # px threshold for RANSAC/USAC
      confidence: 0.999        # estimator confidence
      max_iters: 10000         # estimator iterations
      min_inliers: 10          # reject H if inliers below this

    Returns:
      H (3x3 float64): homography mapping img1 coords -> img2 coords
    Raises:
      RuntimeError/ValueError when matching or estimation fails.
    """

    # Extracting config parameters
    hcfg = cfg.get("homography", {})
    f = hcfg["feature"]; m = hcfg["match"]; e = hcfg["estimate"]

    feature = str(f.get("type", "SIFT")).upper()
    nfeatures = int(f.get("nfeatures", 4000))
    matcher = str(m.get("strategy", "AUTO")).upper()
    knn_k = int(m.get("knn_k", 2))
    if knn_k < 2:
        # Lowe's ratio needs at least two neighbors; force to 2
        knn_k = 2
    ratio = float(m.get("ratio", 0.7))
    method = str(e.get("method", "USAC_MAGSAC"))
    reproj_thresh = float(e.get("reproj_thresh", 4.0))
    confidence = float(e.get("confidence", 0.999))
    max_iters = int(e.get("max_iters", 10000))
    min_inliers = int(e.get("min_inliers", 10))

    # --- Detector & descriptor norm
    if feature == 'ORB':
        detector = cv2.ORB_create(nfeatures=nfeatures)  # more features
        norm = cv2.NORM_HAMMING
    elif feature == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=nfeatures)
        norm = cv2.NORM_L2
    else:
        raise ValueError(f"Unsupported feature type: {feature}")

    # Converting images to grayscale for robust feature finding?

    k1, d1 = detector.detectAndCompute(img1, None)
    k2, d2 = detector.detectAndCompute(img2, None)

    if d1 is None or d2 is None:
        raise RuntimeError("No descriptors")

    # --- Matcher selection
    use_flann = False
    if matcher == "AUTO":
        if feature == "ORB":
            bf = cv2.BFMatcher(norm)
        else:  # SIFT -> FLANN
            index_params = dict(algorithm=1, trees=5)  # KDTree
            search_params = dict(checks=64)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            use_flann = True
    elif matcher == "BF":
        bf = cv2.BFMatcher(norm)
    elif matcher == "FLANN":
        if norm != cv2.NORM_L2:
            raise ValueError("FLANN requires L2/float descriptors (e.g., SIFT)")
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=64)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        use_flann = True
    else:
        raise ValueError(f"Unsupported matcher: {matcher}")

    # --- KNN + Lowe's ratio
    raw = (flann.knnMatch(d1, d2, k=knn_k) if use_flann else bf.knnMatch(d1, d2, k=knn_k))

    good = []
    for m, n in raw:
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < 4:
        raise RuntimeError("Not enough matches")

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt  for m in good]).reshape(-1, 1, 2)

    # --- Robust estimator
    method_flag = _resolve_method_flag(method)
    if method_flag in (cv2.RANSAC, getattr(cv2, "USAC_MAGSAC", -1), getattr(cv2, "USAC_FAST", -1), getattr(cv2, "USAC_ACCURATE", -1)):
        H, mask = cv2.findHomography(src, dst, method_flag, reproj_thresh, maxIters=max_iters, confidence=confidence)
    elif method_flag == cv2.LMEDS:
        H, mask = cv2.findHomography(src, dst, method_flag)
    else:
        # Fallback (should not happen with supported flags)
        H, mask = cv2.findHomography(src, dst, method_flag, reproj_thresh)   


    if H is None:
        raise RuntimeError("findHomography failed")

    inliers = int(mask.sum()) if mask is not None else 0
    if inliers < min_inliers:
        raise RuntimeError(f"Too few inliers: {inliers}")

    return H


def apply_homography(points, H):
    """
    Apply a homography transformation to a set of points.

    Parameters
    ----------
    points : sequence of tuple of float
        List of (x, y) points to transform.
    H : ndarray
        Homography matrix.

    Returns
    -------
    ndarray
        Transformed points as an array of shape (n, 2).
    """
    points = np.array(points, dtype=np.float32)
    points_homogeneous = cv2.convertPointsToHomogeneous(points)
    transformed_points = cv2.perspectiveTransform(points_homogeneous, H)
    return cv2.convertPointsFromHomogeneous(transformed_points).reshape(-1, 2)


# useless for now, but might be useful later
def wrap_image(image, H, output_shape):
    """
    Wrap an image using a homography matrix.

    Parameters
    ----------
    image : ndarray
        Input image.
    H : ndarray
        Homography matrix.
    output_shape : tuple of int
        Shape of the output image as (height, width).

    Returns
    -------
    ndarray
        Warped image.
    """
    return cv2.warpPerspective(image, H, (output_shape[1], output_shape[0]))

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