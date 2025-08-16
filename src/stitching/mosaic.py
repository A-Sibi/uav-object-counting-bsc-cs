# src/stitching/mosaic.py
import cv2
import time
import numpy as np
from typing import List, Optional
from pathlib import Path

from src.utils.geometry import compute_homography
from src.utils.io import ensure_dir, save_homographies_json


def _stitch_batch(images: List[np.ndarray], mode: int) -> np.ndarray:
    """Stitch a batch of images with OpenCV high-level Stitcher."""
    stitcher = cv2.Stitcher_create(mode)
    status, pano = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"Stitching failed (status {status})")
    return pano


def build_mosaic(images_dir: str, cfg: dict, include_homography: bool = False) -> tuple[np.ndarray, list]:
    """
    Build a stitched mosaic from a directory of images, with cfg-driven speed/quality knobs and
    optional homography computation. Homography computation strategy is controlled via YAML.

    Args:
        images_dir: Folder containing frame images (e.g., JPGs).
        cfg: Configuration dictionary. Expected keys:
            paths:
              interim_partials: str  # where partials are saved (if enabled)
              interim_homographies: str  # where to write homography JSON
            stitch:
              mode: PANORAMA|SCANS (default SCANS)
              resize_factor: float in (0,1] (default 1.0)
              chunk_size: int (default: all)
              save_partials: bool (default False)
            homography:
              compute: off|direct|chain (default: off)
              feature: SIFT|ORB|AKAZE (default: SIFT)
              reproj_thresh: float  # pixels (fallback to ransac_reproj_thresh if absent)
              ransac_reproj_thresh: float  # backward-compat name
              # additional keys like min_inliers/confidence can be added and used in compute_homography

        include_homography: If True, homographies are computed according to cfg['homography']['compute'].

    Returns:
        mosaic: Stitched panorama image (BGR).
        H_list: List of 3x3 homographies mapping *frame coords → mosaic coords* (or None on failure per frame).
    """
    print(f"Building mosaic from {images_dir}, include_homography={include_homography}")
    total_start = time.perf_counter()

    # Extracting config parameters
    s_cfg = cfg.get('stitch', {})
    mode_str = s_cfg.get('mode', 'SCANS').upper()
    stitch_mode = cv2.Stitcher_PANORAMA if mode_str == 'PANORAMA' else cv2.Stitcher_SCANS
    resize_factor = s_cfg.get('resize_factor', 1.0)


    # preparing paths
    img_paths = sorted(Path(images_dir).glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    partials_dir = Path(cfg.get('paths', {}).get('interim_partials', 'data/interim/partials'))
    ensure_dir(partials_dir)

    # load and optionally downsample
    images = []
    for p in img_paths:
        im = cv2.imread(str(p))
        if im is None:
            raise RuntimeError(f"Failed to read image {p}")
        if resize_factor != 1.0:
            w = int(im.shape[1] * resize_factor)
            h = int(im.shape[0] * resize_factor)
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        images.append(im)

    n = len(images)
    chunk_size = s_cfg.get('chunk_size', n)


    # --- Stitching timer start ---
    stitch_start = time.perf_counter()

    # chunked stitching
    print(f"[INFO] build_mosaic: stitching {n} images in mode '{mode_str}', chunk_size={chunk_size}, resize_factor={resize_factor}")
    partials = []
    for batch_idx, start in enumerate(range(0, n, chunk_size)):
        batch = images[start:start + chunk_size]
        partial = _stitch_batch(batch, stitch_mode)
        cv2.imwrite(str(partials_dir / f"partial_{batch_idx:03d}.jpg"), partial)
        partials.append(partial)

        if n > 10 and batch_idx % 5 == 0:
            print(f"[INFO] build_mosaic: stitched {batch_idx * len(batch)}/{n} frames into partials")

    # Final merge
    mosaic = partials[0] if len(partials) == 1 else _stitch_batch(partials, stitch_mode)


    stitch_elapsed = time.perf_counter() - stitch_start
    minutes = int(stitch_elapsed // 60)
    seconds = int(stitch_elapsed % 60)
    print(f"[TIMER] build_mosaic: stitched {n} images in {minutes:02d}:{seconds:02d}s")

    # --- Homography computation (used in pipeline 2) ---
    
    H_list: List[Optional[np.ndarray]] = []
    if include_homography:
        h_cfg = cfg.get('homography', {})
        compute_mode = str(h_cfg.get('compute', 'direct')).lower()  # 'direct' | 'chain'
        feature = h_cfg.get('feature', {}).get('type', 'SIFT')  # prefer SIFT here
        reproj = h_cfg.get('estimate', {}).get('reproj_thresh', 3.0)

        homography_start = time.perf_counter()
        if compute_mode == 'direct':
            # Direct: frame → mosaic
            print("[INFO] build_mosaic: computing frame->mosaic homographies (direct method)")

            for idx, frame in enumerate(images):  # 'images' are the (possibly resized) frames
                try:
                    H_fm = compute_homography(frame, mosaic, cfg)
                    # adjust for original-size detections
                    if resize_factor != 1.0:
                        s = resize_factor
                        S = np.array([[s,0,0],[0,s,0],[0,0,1]], dtype=np.float64)
                        H_fm = H_fm @ S
                    H_list.append(H_fm)
                except Exception as e:
                    print(f"[WARN] frame {idx}: homography failed ({e}); marking with identity matrix I.")
                    H_list.append(np.eye(3, dtype=np.float64))
                
                if n > 10 and  idx % 20 == 0:
                    print(f"[INFO] build_mosaic (homography): processed {idx}/{n} frames")

        elif compute_mode == 'chain':
            # Chain: (frame → partial) then (partial → mosaic)

            # First get H for each partial to mosaic
            print("[INFO] build_mosaic: computing frame->partials->mosaic homographies (chain method)")
            H_pm = []
            for p_idx, partial in enumerate(partials):
                try:
                    H_pm.append(compute_homography(partial, mosaic, cfg))
                except Exception as e:
                    print(f"[WARN] Error computing partial homography {p_idx}: {e}")
                    H_pm.append(np.eye(3, dtype=np.float64))
            
            # Then for each frame within its batch: frame → partial → mosaic
            for chunk_idx, start in enumerate(range(0, n, chunk_size)):
                end = min(start + chunk_size, n)
                for idx in range(start, end):
                    frame = images[idx]
                    try:
                        H_fp = compute_homography(partial, mosaic, cfg)
                        H_list.append(H_pm[chunk_idx] @ H_fp)
                    except Exception as e:
                        print(f"Error computing homography for frame {idx}: {e}")
                        H_list.append(np.eye(3, dtype=np.float64))
        else:
            raise ValueError("Unsupported homograpy computing method.")


        # save homographies to JSON
        meta = {
            "resize_factor": resize_factor,
            "stitch_mode": mode_str,
            "chunk_size": chunk_size,
            "feature": feature,
            "reproj_thresh": reproj,
            "mosaic_shape": mosaic.shape,  # (H, W, C)
            "compute_mode": compute_mode
        }
        out_json = save_homographies_json(H_list, img_paths, cfg["paths"]["interim_homographies"], meta=meta)
        print(f"[INFO] build_mosaic: saved homographies to {out_json}")

        homography_elapsed = time.perf_counter() - homography_start
        h_minutes = int(homography_elapsed // 60)
        h_seconds = int(homography_elapsed % 60)
        print(f"[TIMER] build_mosaic: computed {len(H_list)} homographies in {h_minutes:02d}:{h_seconds:02d}s")
    else:
        print("[INFO] build_mosaic: skipping homography computation")

    total_elapsed = time.perf_counter() - total_start
    total_minutes = int(total_elapsed // 60)
    total_seconds = int(total_elapsed % 60)
    print(f"[TIMER] build_mosaic: total time {total_minutes:02d}:{total_seconds:02d}s ")

    return mosaic, H_list
