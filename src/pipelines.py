# src/pipelines.py
from pathlib import Path
import cv2

from src.utils.io import extract_frames, save_np_image
from src.stitching.mosaic import build_mosaic
from src.detection.detector import detect_cars

def run_extract(video_path: str, cfg: dict[str, any]) -> None:
    """
    Extract frames from the video and save them as JPEG images.
    :param video_path: Path to the input video file.
    :param cfg: Configuration dictionary containing paths and parameters.
    """

    print("Running Frame Extraction Pipeline...")
    print(f"Extracting frames from video: {video_path}")
    extract_frames(
        video_path,
        cfg["paths"]["interim_frames"],
        cfg["video"]["frame_step"]
    )
    print(f"Frames extracted to: {cfg['paths']['interim_frames']}")
    return None

def run_stitch(images_dir: Path, cfg) -> None:
    """
    Run the stitching pipeline on a set of frames.
    :param images: List of image paths to stitch.
    :param cfg: Configuration dictionary containing stitching parameters.
    """

    print("Running Stitching Pipeline...")
    print(f"Stitching images in: {images_dir}")

    frames = [cv2.imread(fp) for fp in images_dir.glob("*.jpg")]
    mosaic, H_list = build_mosaic(frames, cfg)
    save_np_image(mosaic, cfg["paths"]["interim_mosaic"])
    print(f"Mosaic saved to {cfg["paths"]["interim_mosaic"]}")
    return None


def run_single_image_detect(image_path: str, cfg: dict[str, any]) -> None:
    """
    Run object detection on a single image or mosaic and print the results.
    """
    print("Running Detection Pipeline on: ", image_path)
    
    detect_cfg = cfg.get("detect", {}).copy()


    # 3) Load image via OpenCV
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image as CV2 matrix: {image_path}")

    # 4) Run the core detector on the image
    detections = detect_cars(image, detect_cfg)

    # 5) Draw bounding boxes on a copy of the image
    image_with_boxes = image.copy()
    for det in detections:
        x1, y1 = int(det['x1']), int(det['y1'])
        x2, y2 = int(det['x2']), int(det['y2'])
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 6) Print results
    print(f"Detected {len(detections)} cars in '{image_path}'")
    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        conf = det['conf']
        print(f"{idx:02d}. bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) conf={conf:.2f}")

    # 7) Save annotated image to processed path
    processed_dir = Path(cfg.get("paths", {}).get("processed", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "test_detect_result.jpg"
    cv2.imwrite(str(output_path), image_with_boxes)
    print(f"Annotated image saved to '{output_path}'")

    return detections, image_with_boxes


def run_pipeline1(video_path, cfg: dict[str, any]) -> None:
    print("Running Pipeline 1...")
    # 1. Extract frames
    frame_paths = extract_frames(
        video_path,
        cfg["paths"]["interim_frames"],
        cfg["video"]["frame_step"]
    )
    frames = [cv2.imread(fp) for fp in frame_paths]
    # 2. Stitch frames into a mosaic
    # 3. Detect cars in the mosaic
    # 4. Save results
    raise NotImplementedError("Pipeline 1 is not implemented yet.")


def run_pipeline2(video_path, cfg: dict[str, any]) -> None:
    print("Running Pipeline 2...")
    # 1. Extract frames
    # 2. Detect cars in each frame
    # 3. Stitch frames into a mosaic
    # 4. Map detections to the mosaic
    # 5. Save results
    raise NotImplementedError("Pipeline 2 is not implemented yet.")


def save_data() -> None:
    """
    Save processed data as a new experiment.
    """

    raise NotImplementedError("Data saving is not implemented yet.")
