# src/stitching/mosaic.py
import cv2
import time
import numpy as np
from pathlib import Path

from src.utils.geometry import compute_homography


def build_mosaic(images_dir: str, cfg: dict, include_homography: bool = False) -> tuple[np.ndarray, list]:
    """
    Build a stitched mosaic from a directory of images, with optional resizing,
    chunked stitching, and stitching mode selection based on configuration.

    Args:
        images_dir (str): Path to directory containing frame images (JPEGs).
        cfg (dict): Configuration dictionary. Supported keys under 'stitch':
            resize_factor (float): scale factor to downsample images (default 1.0)
            chunk_size (int): number of images to stitch per batch (default: all)
            mode (str): 'PANORAMA' or 'SCANS' to choose stitching algorithm (default: 'PANORAMA')
    Returns:
        mosaic (np.ndarray): Stitched panorama image (BGR).
        H_list (List[np.ndarray]): Identity homography matrices for each input image.
    """
    # Start timing
    start_time = time.perf_counter()

    img_paths = sorted(Path(images_dir).glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    # Load and optionally downsample images
    resize_factor = cfg.get('stitch', {}).get('resize_factor', 1.0)
    images = []
    for p in img_paths:
        im = cv2.imread(str(p))
        if im is None:
            raise RuntimeError(f"Failed to read image {p}")
        if resize_factor != 1.0:
            width = int(im.shape[1] * resize_factor)
            height = int(im.shape[0] * resize_factor)
            im = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
        images.append(im)

    # Determine stitching mode
    mode_str = cfg.get('stitch', {}).get('mode', 'PANORAMA').upper()
    stitch_mode = cv2.Stitcher_PANORAMA if mode_str == 'PANORAMA' else cv2.Stitcher_SCANS

    # Chunked stitching for performance
    chunk_size = cfg.get('stitch', {}).get('chunk_size', len(images))
    partials = []
    for i in range(0, len(images), chunk_size):
        batch = images[i:i + chunk_size]
        stitcher = cv2.Stitcher_create(stitch_mode)
        status, pano = stitcher.stitch(batch)
        if status != cv2.Stitcher_OK:
            raise RuntimeError(f"Stitching batch {i}-{i+chunk_size} failed with status {status}")
        partials.append(pano)

    # Combine partial mosaics if multiple batches
    if len(partials) > 1:
        stitcher = cv2.Stitcher_create(stitch_mode)
        status, mosaic = stitcher.stitch(partials)
        if status != cv2.Stitcher_OK:
            raise RuntimeError(f"Final stitching failed with status {status}")
    else:
        mosaic = partials[0]

    # End timing
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"[INFO] build_mosaic: stitched {len(images)} images in {elapsed:.2f} seconds.")

    H_list: list[np.ndarray] = []
    if include_homography:
        # 1) compute partial → mosaic for each chunk
        H_pm: list[np.ndarray] = []
        for p_idx, partial in enumerate(partials):
            try:
                H_p2m = compute_homography(
                    partial, mosaic,
                    feature=cfg['stitch'].get('feature', 'ORB'),
                    reproj_thresh=cfg['stitch'].get('reproj_thresh', 4.0)
                )
            except Exception as e:
                print(f"Error computing homography for partial {p_idx}: {e}")
                H_p2m = np.eye(3, dtype=np.float64)
            H_pm.append(H_p2m)

        # 2) for each original frame, find frame → partial and chain
        for chunk_idx, start in enumerate(range(0, len(images), chunk_size)):
            end = min(start + chunk_size, len(images))
            batch = images[start:end]
            for i, frame in enumerate(batch):
                frame_idx = start + i
                try:
                    H_f2p = compute_homography(
                        frame, partials[chunk_idx],
                        feature=cfg['stitch'].get('feature', 'ORB'),
                        reproj_thresh=cfg['stitch'].get('reproj_thresh', 4.0)
                    )
                    # chain: frame → mosaic = (partial→mosaic) ∘ (frame→partial)
                    H = H_pm[chunk_idx] @ H_f2p
                except Exception as e:
                    print(f"Error computing homography for frame {frame_idx}: {e}")
                    H = np.eye(3, dtype=np.float64)
                H_list.append(H)

    return mosaic, H_list


