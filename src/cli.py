# src/cli.py
import sys
import argparse
from pathlib import Path
from src.pipelines import *
from src.utils.io import load_config


def main():
    parser = argparse.ArgumentParser(prog="uav-counter", description="UAV Object Counting CLI")
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default="configs/default.yaml",
        help="Path to YAML config file"
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Choose a pipeline mode")

    # Pipeline 1 -> run1
    p1 = subparsers.add_parser("run1", help="Mosaic-first detection (pipeline 1)")
    p1.add_argument(
        "video",
        nargs="?",
        default=None,
        type=Path,
        help="Input video file (default: from config.paths.raw/video.mp4)"
    )

    # Pipeline 2 -> run2
    p2 = subparsers.add_parser("run2", help="Frame-first detection (Pipeline 2)")
    p2.add_argument(
        "video",
        nargs="?",
        default=None,
        type=Path,
        help="Input video file (default: from config.paths.raw/video.mp4)"
    )

    # extract frames
    pe = subparsers.add_parser("extract", help="Extract frames from video")
    pe.add_argument(
        "video",
        nargs="?",
        default=None,
        type=Path,
        help="Input video file (default: from config.paths.raw/video.mp4)"
    )

    # stitch frames
    ps = subparsers.add_parser("stitch", help="Stitch frames into mosaic")
    ps.add_argument(
        "images_dir",
        nargs="?",
        default=None,
        type=Path,
        help="Input images directory (default: from config.paths.interim_frames)"
    )

    # detect on single image
    pd = subparsers.add_parser("detect", help="Run detection on single image")
    pd.add_argument(
        "image",
        nargs="?",
        default=None,
        type=Path,
        help="Input image file (default: from config.paths.raw_image)"
    )

    # save_data
    psd = subparsers.add_parser("save_data", help="Archive current processed data into experiments/")


    args = parser.parse_args()
    cfg = load_config(args.config)
    
    # Set default paths if not provided
    if hasattr(args, 'video') and args.video is None:
        args.video = Path(cfg["paths"]["raw_video"])
    if hasattr(args, 'image') and args.image is None:
        args.image = Path(cfg["paths"]["raw_image"])
    if hasattr(args, 'images_dir') and args.images_dir is None:
        args.images_dir = Path(cfg["paths"]["interim_frames"])

    if args.command == 'run1':
        run_pipeline1(args.video, cfg)
    elif args.command == 'run2':
        run_pipeline2(args.video, cfg)
    elif args.command == 'extract':
        run_extract(args.video, cfg)
    elif args.command == 'stitch':
        images_dir = args.images_dir
        run_stitch(images_dir, cfg)
    elif args.command == 'detect':
        run_single_image_detect(args.image, cfg)
    elif args.command == 'save_data':
        save_data()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
