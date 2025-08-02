# src/cli.py
import sys
import argparse
from pathlib import Path
from src.pipelines import *
from src.utils.io import load_config


def main():
    parser = argparse.ArgumentParser(description="UAV Object Counting Pipeline")
    parser.add_argument(
        "-m", "--mode",
        choices=[
            "pipeline1",
            "pipeline2",
            "extract",
            "stitch",
            "detect",
            "save_data",
        ],
        help=(
            'Which stage to run.'
            'Choose "pipeline1" or "pipeline2" for complete processing from video to output;'
            '"extract", "stitch" and "detect" run specific parts of a pipeline, use for testing;'
            '"save_data" saves processed data to a new experiment folder.'
        ),
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default="configs/default.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "-v", "--video",
        type= Path,
        default= Path("data/raw/video.mp4"),
        help="Path to input video file"
    )
    parser.add_argument(
        "-i", "--image",
        type=Path,
        default="data/raw/test_image.jpg",
        help="Path to test image/mosaic for detection"
    )
    parser.add_argument(
        "-s", "--stitch-folder",
        type=Path,
        help="Folder containing frames to stitch (defaults to config paths.interim_frames)"
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    if   args.mode == 'pipeline1':
        run_pipeline1(args.video, cfg)
    elif args.mode == 'pipeline2':
        run_pipeline2(args.video, cfg)
    elif args.mode == 'extract':
        run_extract(args.video, cfg)
    elif args.mode == 'stitch':
        images_dir = args.stitch_folder or cfg["paths"]["interim_frames"]
        run_stitch(images_dir, cfg)
    elif args.mode == 'detect':
        run_single_image_detect(args.image, cfg)
    elif args.mode == 'save_data':
        save_data()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
