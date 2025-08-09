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

    # (p1 & p2) extract frames
    pe = subparsers.add_parser("extract", help="Extract frames from video")
    pe.add_argument(
        "video",
        nargs="?",
        default=None,
        type=Path,
        help="Input video file (default: from config.paths.raw/video.mp4)"
    )

    # (p1 & p2) stitch frames
    ps = subparsers.add_parser("stitch", help="Stitch frames into mosaic")
    ps.add_argument(
        "images_dir",
        nargs="?",
        default=None,
        type=Path,
        help="Input images directory (default: from config.paths.interim_frames)"
    )
    ps.add_argument(
        "-H", "--include_homography",
        action="store_true",
        help="If set, computes and returns homography matrices for each frame"
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

    # (p2) batch detect on folder of images, saves detections
    pdb = subparsers.add_parser("batch_detect", help="Run detection on batch of images and save detections")
    pdb.add_argument(
        "images_dir",
        nargs="?",
        default=None,
        type=Path,
        help="Input images directory (default: from config.paths.interim_frames)"
    )

    # batch map & stitch
    pbm = subparsers.add_parser(
        "batch_map",
        help="For given frames, partials and detections on frames, compute homographies, map saved detections, and save mosaic"
    )
    pbm.add_argument(
        "dets_dir",
        nargs="?",
        default=None,
        type=Path,
        help="Directory of saved detections (default from config)"
    )
    pbm.add_argument(
        "homographies_dir",
        nargs="?",
        default=None,
        type=Path,
        help="Directory of homographies (default from config)"
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
    if hasattr(args, 'dets_dir') and args.dets_dir is None:
        args.dets_dir = Path(cfg["paths"]["interim_detections"])
    if hasattr(args, 'homographies_dir') and args.homographies_dir is None:
        args.homographies_dir = Path(cfg["paths"]["interim_homographies"])

    if args.command == 'run1':
        run_pipeline1(args.video, cfg)
    elif args.command == 'run2':
        run_pipeline2(args.video, cfg)
    elif args.command == 'extract':
        run_extract(args.video, cfg)
    elif args.command == 'stitch':
        run_stitch(args.images_dir, cfg, args.include_homography)
    elif args.command == 'detect':
        run_single_image_detect(args.image, cfg)
    elif args.command == 'batch_detect':
        run_batch_detect(args.images_dir, cfg)
    elif args.command == 'batch_map':
        run_batch_map(args.dets_dir, args.homographies_dir, cfg)
    elif args.command == 'save_data':
        save_data()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
