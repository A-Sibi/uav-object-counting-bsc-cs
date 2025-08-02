# src/utils/vis.py
import cv2


def draw_boxes(image, detections, color=(0, 255, 0), thickness=2):
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


def plot_points_on_mosaic(mosaic, points, color=(0, 0, 255), radius=5):
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


def show_image(image, title="Image"):
    """
    Display an image using OpenCV.
    
    :param image: Image to display (numpy array).
    :param title: Title of the window.
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()