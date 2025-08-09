# src/pipelines.py
import json
from pathlib import Path
import cv2
import torch

from src.utils.io import *
from src.utils.vis import *
from src.stitching.mosaic import build_mosaic
from src.mapping.project import project_detections
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


def run_stitch(images_dir: Path, cfg, compute_homographies: bool) -> None:
    """
    Run the stitching pipeline on a set of frames.
    :param images_dir: Path to directory of frame images to stitch.
    :param cfg: Configuration dictionary containing stitching parameters.
    """

    print("Running Stitching Pipeline...")
    print(f"Stitching images in: {images_dir}")
    mosaic, _ = build_mosaic(images_dir, cfg, compute_homographies)
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


def run_batch_detect(images_dir: str, cfg: dict[str, any]) -> None:
    """
    Run object detection on a batch of images and save the results.
    """
    print("Running Batch Detection Pipeline...")
    print(f"Processing images in: {images_dir}")

    dets_dir = Path(cfg.get('paths', {}).get('interim_detections', 'data/interim/detections'))
    dets_dir.mkdir(parents=True, exist_ok=True)
    batch_dets_dir = Path(cfg.get('paths', {}).get('interim_detections_batch', 'data/interim/detections_batch'))
    batch_dets_dir.mkdir(parents=True, exist_ok=True)

    # Get all image paths
    image_paths = sorted(Path(images_dir).glob("*.jpg"))
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    for image_path in image_paths:
        dets = detect_cars_rb_vehicle(str(image_path), cfg)
        image_with_boxes = draw_rich_boxes(load_np_image(str(image_path)), dets)
        save_np_image(image_with_boxes, batch_dets_dir / f"{image_path.stem}_annotated.jpg")

        out_path = dets_dir / f"{image_path.stem}_detections.json"
        clean = [{k: float(v) for k,v in d.items()} for d in dets]
        with open(out_path, 'w') as f:
            json.dump(clean, f, indent=2)
    print(f"Detections saved in: {out_path}")
    print("Batch detection completed.")
    return None


def run_batch_map(dets_dir: str, homography_dir: str, cfg: dict[str, any]) -> None:
    """
    Run batch mapping of frame detections to a mosaic.

    Parameters
    ----------
    mosaic_path: str
        Path to the mosaic image.
    dets_dir: str
        Directory containing detection JSON files.
    homography_dir: str
        Directory containing homography JSON files.
    cfg: dict
        Configuration dictionary containing paths and parameters.
    """
    print("Running Batch Map Pipeline...")

    # 1. Load mosaic
    mosaic = load_np_image(cfg["paths"]["interim_mosaic"])
    print(f"Loaded mosaic from: {cfg['paths']['interim_mosaic']}")

    # 2. Load homographies
    h_json = latest_json_file(homography_dir)
    H_list, images_order, meta = load_homographies_json(h_json)
    print(f"Loaded {len(H_list)} homographies from: {h_json}")

    # 3) Load all detections into a mapping by stem
    dets_dir = Path(dets_dir)
    det_files = sorted(dets_dir.glob("*.json"))
    if not det_files:
        raise FileNotFoundError(f"No detection JSONs found in {dets_dir}")

    dets_by_stem: dict[str, list[dict]] = {}
    for dp in det_files:
        with open(dp, "r", encoding="utf-8") as f:
            raw = json.load(f)
        dets_by_stem[dp.stem] = coerce_dets_schema(raw)

    
    # 4) Build dets_per_frame aligned with images_order (critical!)
    #    If a frame has no detection file, use an empty list.
    dets_per_frame: list[list[dict]] = []
    missing = []
    for name in images_order:
        stem = Path(name).stem
        dets = dets_by_stem.get(stem, [])
        if not dets and stem not in dets_by_stem:
            missing.append(stem)
        dets_per_frame.append(dets)
    if missing:
        print(f"[WARN] No detection file for {len(missing)} frames. Examples: {missing[:5]}")

    # 5) Project detections onto the mosaic
    projected_detections = project_detections(dets_per_frame, H_list)
    print(f"[INFO] Projected {len(projected_detections)} boxes onto mosaic.")

    # 6) (Optional) merge duplicates here later (DBSCAN / distance NMS)

    # 7) Draw and save the results
    image_with_boxes = draw_rich_boxes(mosaic, projected_detections)
    save_np_image(image_with_boxes, cfg["paths"]["processed_image"])
    print(f"Annotated mosaic saved to '{cfg['paths']['processed_image']}'")
    
    return None


def run_pipeline1(video_path, cfg: dict[str, any]) -> None:
    print("Running Pipeline 1...")
    print(f"Processing video: {video_path}")

    # 1. Extract frames
    frame_paths = extract_frames(
        video_path,
        cfg["paths"]["interim_frames"],
        cfg["video"]["frame_step"]
    )
    # 2. Stitch frames into a mosaic
    images_dir = cfg["paths"]["interim_frames"]
    mosaic, H_list = build_mosaic(images_dir, cfg)
    save_np_image(mosaic, cfg["paths"]["interim_mosaic"])

    # 3. Detect cars in the mosaic
    image_path = cfg["paths"]["interim_mosaic"]
    detections = detect_cars_rb_vehicle(image_path, cfg)
    image_with_boxes =  draw_rich_boxes(load_np_image(image_path), detections)

    # 4. Save results
    save_np_image(image_with_boxes, cfg["paths"]["processed_image"])
    print(f"Annotated image saved to '{cfg['paths']['processed_image']}'")
    print("Pipeline 1 completed successfully.")
    return None


def run_pipeline2(video_path, cfg: dict[str, any]) -> None:
    print("Running Pipeline 2...")
    # 1. Extract frames (only if not already done)
    frames_dir = Path(cfg["paths"]["interim_frames"])
    frames_dir.mkdir(parents=True, exist_ok=True)
    if not any(frames_dir.iterdir()):
        print(f"Extracting frames from video: {video_path}")
        frame_paths = extract_frames(
            video_path,
            cfg["paths"]["interim_frames"],
            cfg["video"]["frame_step"]
        )
    else:
        print(f"Using existing frames in: {frames_dir}")
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in {frames_dir}")
    # 2. Detect cars in each frame
    frame_detections = []
    with torch.no_grad():
        for frame_path in frame_paths:
            detections = detect_cars_rb_vehicle(frame_path, cfg)
            frame_detections.append(detections)

    # 3. Stitch frames into a mosaic
    images_dir = cfg["paths"]["interim_frames"]
    mosaic, H_list = build_mosaic(images_dir, cfg, True)
    save_np_image(mosaic, cfg["paths"]["interim_mosaic"])

    s = cfg.get('stitch', {}).get('resize_factor', 1.0)
    if s != 1.0:
        S = np.array([[s, 0, 0],
                    [0, s, 0],
                    [0, 0, 1]], dtype=np.float64)
        H_list = [H @ S for H in H_list]
    
    # 4. Map detections to the mosaic
    projected_detections = project_detections(frame_detections, H_list, mosaic_shape=mosaic.shape[:2])
    image_with_boxes = draw_rich_boxes(mosaic, projected_detections)

    # 5. Save results
    save_np_image(image_with_boxes, cfg["paths"]["processed_image"])
    print(f"Annotated image saved to '{cfg['paths']['processed_image']}'")
    print("Pipeline 2 completed successfully.")
    return None


def save_data() -> None:
    """
    Save processed data as a new experiment.
    """

    raise NotImplementedError("Data saving is not implemented yet.")
