# src/utils/vis.py
import cv2
import numpy as np
from src.detection.detector import Detection, TranslatedDetection

def draw_boxes(image: np.ndarray, detections: list[Detection], color=(0, 255, 0), thickness=2) -> np.ndarray:
    """
    Draw bounding boxes on the image.
    
    :param image: Input image (numpy array).
    :param detections: List of detections, each detection is a dict (x1, y1, x2, y2, conf).
    :param color: Color of the bounding box in BGR format.
    :param thickness: Thickness of the bounding box lines.
    :return: Image with drawn bounding boxes.
    """
    for d in detections:
        x1, y1, x2, y2 = int(d['x1']), int(d['y1']), int(d['x2']), int(d['y2'])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

def draw_rich_boxes(image: np.ndarray, detections: list[Detection], color=(0, 255, 0), thickness=2) -> np.ndarray:
    """
    Draw bounding boxes with confidence on the image.
    
    :param image: Input image (numpy array).
    :param detections: List of detections, each detection is a dict (x1, y1, x2, y2, conf).
    :param color: Color of the bounding box in BGR format.
    :param thickness: Thickness of the bounding box lines.
    :return: Image with drawn bounding boxes and confidence.
    """
    for i, d in enumerate(detections, start=1):
        x1, y1, x2, y2 = int(d['x1']), int(d['y1']), int(d['x2']), int(d['y2'])
        conf = d.get('conf', 0.0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, f"#{i}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def plot_points_on_mosaic(mosaic: np.ndarray, points: list[tuple], color=(0, 0, 255), radius=5) -> np.ndarray:
    """
    Plot points on the mosaic image.
    
    :param mosaic: Input mosaic image (numpy array).
    :param points: List of points to plot, each point is a tuple (x, y).
    :param color: Color of the points in BGR format.
    :param radius: Radius of the points to draw.
    :return: Mosaic image with plotted points.
    """
    for (x, y) in points:
        cv2.circle(mosaic, (int(x), int(y)), radius, color, -1)
    return mosaic


def show_image(image: np.ndarray, title="Image") -> None:
    """
    Display an image using OpenCV.
    
    :param image: Image to display (numpy array).
    :param title: Title of the window.
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()