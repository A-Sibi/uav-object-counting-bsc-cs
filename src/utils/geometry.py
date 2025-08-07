# src/utils/geometry.py
import cv2
import numpy as np


def compute_homography(img1, img2, feature='ORB', reproj_thresh=4.0) -> np.ndarray:
    """
    Compute the homography matrix between two images using feature matching.

    Parameters
    ----------
    img1 : ndarray
        Starting image.
    img2 : ndarray
        Image to project coordinates onto.
    feature : {'ORB', 'SIFT'}, optional
        Feature detection method, by default 'ORB'.
    reproj_thresh : float, optional
        RANSAC reprojection threshold, by default 4.0.

    Returns
    -------
    ndarray
        3x3 homography matrix.

    Raises
    ------
    ValueError
        If an unsupported feature type is provided.
    RuntimeError
        If not enough matches are found for homography estimation.
    """
    # 1. Create detector
    if feature == 'ORB':
        detector = cv2.ORB_create()
    elif feature == 'SIFT':
        detector = cv2.SIFT_create()
    else:
        raise ValueError(f"Unsupported feature type: {feature}")

    # 2. Detect & compute
    kpts1, desc1 = detector.detectAndCompute(img1, None)
    kpts2, desc2 = detector.detectAndCompute(img2, None)

    # 3. Match descriptors
    if feature == 'ORB':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2)  # for SIFT (L2)

    raw_matches = matcher.knnMatch(desc1, desc2, k=2)

    # 4. Filter with Lowe’s ratio test
    good = []
    ratio = 0.75
    for m,n in raw_matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    # 5. Build point arrays
    if len(good) < 4:
        raise RuntimeError("Not enough matches for homography")
    src_pts = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # 6. Estimate H
    H, mask = cv2.findHomography(src_pts, dst_pts,
                                 cv2.RANSAC,
                                 reproj_thresh)
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