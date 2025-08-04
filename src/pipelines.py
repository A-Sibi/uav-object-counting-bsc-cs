# src/pipelines.py
from pathlib import Path
import cv2

from src.utils.io import *
from src.utils.vis import *
from src.stitching.mosaic import build_mosaic
from src.detection.detector import *

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
    :param images_dir: Path to directory of frame images to stitch.
    :param cfg: Configuration dictionary containing stitching parameters.
    """

    print("Running Stitching Pipeline...")
    print(f"Stitching images in: {images_dir}")
    mosaic, H_list = build_mosaic(images_dir, cfg)
    print(f"Mosaic built with {len(H_list)} images.")
    save_np_image(mosaic, cfg["paths"]["interim_mosaic"])
    print(f"Mosaic saved to {cfg['paths']['interim_mosaic']}")
    return None


def run_single_image_detect(image_path: str, cfg: dict[str, any]) -> None:
    """
    Run object detection on a single image or mosaic and print the results.
    """
    print("Running Detection Pipeline on: ", image_path)

    # detections = detect_cars_YOLO(image_path, detect_cfg)
    detections = detect_cars_rb_vehicle(image_path, cfg)

    image_with_boxes =  draw_rich_boxes(load_np_image(image_path), detections)


    # Print results
    print(f"Detected {len(detections)} cars in '{image_path}'")
    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        conf = det['conf']
        print(f"{idx:02d}. bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) conf={conf:.2f}")

    save_np_image(image_with_boxes, cfg["paths"]["processed_image"])
    print(f"Annotated image saved to '{cfg['paths']['processed_image']}'")

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
