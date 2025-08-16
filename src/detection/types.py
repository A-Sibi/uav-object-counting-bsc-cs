class Detection:
    """
    Single detection with bounding box coordinates (xyxy) and confidence score.

    Attributes
    ----------
    id : str
        Unique identifier for the detection.
    x1 : float
        Left x-coordinate of the bounding box.
    y1 : float
        Top y-coordinate of the bounding box.
    x2 : float
        Right x-coordinate of the bounding box.
    y2 : float
        Bottom y-coordinate of the bounding box.
    conf : float
        Confidence score of the detection (0.0-1.0).
    """
    id: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

class TranslatedDetection(Detection):
    """
    Detection with additional fields for frame index and original coordinates.

    Attributes
    ----------
    frame_idx : int
        Index of the frame where the detection was made.
    x1_b : float
        Left x-coordinate of the bounding box in the original image.
    y1_b : float
        Top y-coordinate of the bounding box in the original image.
    x2_b : float
        Right x-coordinate of the bounding box in the original image.
    y2_b : float
        Bottom y-coordinate of the bounding box in the original image.
    """
    frame_idx: int
    x1_b: float
    y1_b: float
    x2_b: float
    y2_b: float


class DetectionIDCounter:
    """
    Simple counter to generate unique IDs for detections.
    """
    def __init__(self):
        self._counter = 0

    def next_id(self) -> int:
        """
        Generate a new unique ID.
        """
        self._counter += 1
        return self._counter

# Global counter instance
detection_id_counter = DetectionIDCounter()