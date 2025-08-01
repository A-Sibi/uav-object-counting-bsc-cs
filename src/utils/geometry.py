# src/utils/vis.py
import cv2
import numpy as np


def compute_homography():
    raise NotImplementedError("This function is not implemented yet.")


def apply_homography(points, H):
    """
    Apply a homography transformation to a set of points.
    
    :param points: List of points to transform, each point is a tuple (x, y).
    :param H: Homography matrix.
    :return: Transformed points as a numpy array.
    """
    points = np.array(points, dtype=np.float32)
    points_homogeneous = cv2.convertPointsToHomogeneous(points)
    transformed_points = cv2.perspectiveTransform(points_homogeneous, H)
    return cv2.convertPointsFromHomogeneous(transformed_points).reshape(-1, 2)


def wrap_image(image, H, output_shape):
    """
    Wrap an image using a homography matrix.
    
    :param image: Input image (numpy array).
    :param H: Homography matrix.
    :param output_shape: Shape of the output image (height, width).
    :return: Wrapped image.
    """
    return cv2.warpPerspective(image, H, (output_shape[1], output_shape[0]))