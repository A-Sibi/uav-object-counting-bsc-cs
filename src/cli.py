# src/cli.py

import argparse
import cv2
import json
import numpy as np
from pathlib import Path

from src.utils.io import load_config, extract_frames, ensure_dir
from src.stitching.mosaic import build_mosaic
from src.detection.detector import detect_cars
from src.mapping.project import project_points

def main():
    p = argparse.ArgumentParser(description="UAV Object Counting Pipeline")
    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file"
    )
    p.add_argument(
        "--mode",
        choices=["pipeline1", "stitch", "detect"],
        default="pipeline1",
        help="Which stage(s) to run"
    )
    p.add_argument(
        "--video",
        required=True,
        help="Path to input video file"
    )
    args = p.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # 1) Extract frames
    raw_frames_dir = Path(cfg["paths"]["raw"]) / "frames"
    frame_paths = extract_frames(
        args.video,
        str(raw_frames_dir),
        cfg["video"]["frame_step"]
    )
    frames = [cv2.imread(fp) for fp in frame_paths]

    # 2) (Temporary) dummy homographies for each frame
    homos = [np.eye(3) for _ in frames]

    # 3) Stitching (if requested)
    if args.mode in ["stitch", "pipeline1"]:
        mosaic_img, H_list = build_mosaic(frames, cfg["stitch"])
    else:
        H_list = homos

    # 4) Detection (if requested)
    if args.mode in ["detect", "pipeline1"]:
        dets_per_frame = detect_cars(frames, cfg["detect"])
    else:
        dets_per_frame = []

    # 5) Mapping & counting (only for detect or pipeline1)
    points_on_mosaic = []
    if args.mode in ["detect", "pipeline1"]:
        points_on_mosaic = project_points(dets_per_frame, H_list)

    # 6) Save results
    ensure_dir(cfg["paths"]["processed"])
    processed_dir = Path(cfg["paths"]["processed"])

    # Save mosaic image
    if args.mode in ["stitch", "pipeline1"]:
        cv2.imwrite(str(processed_dir / "mosaic.png"), mosaic_img)

    # Save detections JSON
    if args.mode in ["detect", "pipeline1"]:
        with open(processed_dir / "cars.json", "w") as f:
            json.dump(points_on_mosaic, f, indent=2)

    print("Done. Results saved to:", processed_dir)

if __name__ == "__main__":
    main()
