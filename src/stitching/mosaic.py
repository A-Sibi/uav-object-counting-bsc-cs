# src/stitching/mosaic.py
import cv2
import time
import numpy as np
from pathlib import Path

from src.utils.geometry import compute_homography
from src.utils.io import save_homographies_json


def build_mosaic(images_dir: str, cfg: dict, include_homography: bool = False) -> tuple[np.ndarray, list]:
    """
    Build a stitched mosaic from a directory of images, with optional resizing,
    chunked stitching, saving intermediate partial panoramas, and stitching mode selection.

    Args:
        images_dir (str): Path to directory containing frame images (JPEGs).
        cfg (dict): Configuration dictionary. Supported keys under 'stitch':
            resize_factor (float): scale factor to downsample images (default 1.0)
            chunk_size (int): number of images to stitch per batch (default: all)
            mode (str): 'PANORAMA' or 'SCANS' to choose stitching algorithm (default: 'PANORAMA')
        include_homography (bool): If True, computes and returns H_list for each frame.
    Returns:
        mosaic (np.ndarray): Stitched panorama image (BGR).
        H_list (List[np.ndarray]): Homography matrices for each input image (empty if include_homography=False).
    """
    print("[INFO] build_mosaic: starting stitching process")
    start_time = time.perf_counter()
    img_paths = sorted(Path(images_dir).glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    # prepare output folder for partials
    partials_dir = Path(cfg.get('paths', {}).get('interim_partials', 'data/interim/partials'))
    partials_dir.mkdir(parents=True, exist_ok=True)

    # load and optionally downsample
    resize_factor = cfg.get('stitch', {}).get('resize_factor', 1.0)
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

    # choose stitch mode
    mode_str = cfg.get('stitch', {}).get('mode', 'PANORAMA').upper()
    stitch_mode = cv2.Stitcher_PANORAMA if mode_str == 'PANORAMA' else cv2.Stitcher_SCANS

    # chunked stitching
    chunk_size = cfg.get('stitch', {}).get('chunk_size', len(images))
    partials = []
    for batch_idx, start in enumerate(range(0, len(images), chunk_size)):
        batch = images[start:start + chunk_size]
        stitcher = cv2.Stitcher_create(stitch_mode)
        status, pano = stitcher.stitch(batch)
        if status != cv2.Stitcher_OK:
            raise RuntimeError(f"Stitching batch {start}-{start+chunk_size} failed (status {status})")
        # save each partial
        out_path = partials_dir / f"partial_{batch_idx:03d}.jpg"
        cv2.imwrite(str(out_path), pano)
        partials.append(pano)

    # final merge
    if len(partials) > 1:
        stitcher = cv2.Stitcher_create(stitch_mode)
        status, mosaic = stitcher.stitch(partials)
        if status != cv2.Stitcher_OK:
            raise RuntimeError(f"Final stitching failed (status {status})")
    else:
        mosaic = partials[0]

    elapsed = time.perf_counter() - start_time
    print(f"[INFO] build_mosaic: stitched {len(images)} images in {elapsed:.2f}s")

    # PARTIAL CHANING METHOD
    # H_list: list[np.ndarray] = []
    # if include_homography:
    #     print("[INFO] build_mosaic: computing homographies")
    #     # compute partial->mosaic homographies
    #     H_pm = []
    #     for p_idx, partial in enumerate(partials):
    #         try:
    #             H_pm.append(compute_homography(
    #                 partial, mosaic,
    #                 feature=cfg['stitch'].get('feature', 'ORB'),
    #                 reproj_thresh=cfg['stitch'].get('reproj_thresh', 4.0)
    #             ))
    #         except Exception as e:
    #             print(f"Error computing partial homography {p_idx}: {e}")
    #             H_pm.append(np.eye(3, dtype=np.float64))
    #     # chain frame->partial then -> mosaic
    #     for chunk_idx, start in enumerate(range(0, len(images), chunk_size)):
    #         end = min(start + chunk_size, len(images))
    #         for idx in range(start, end):
    #             frame = images[idx]
    #             try:
    #                 H_fp = compute_homography(
    #                     frame, partials[chunk_idx],
    #                     feature=cfg['stitch'].get('feature', 'ORB'),
    #                     reproj_thresh=cfg['stitch'].get('reproj_thresh', 4.0)
    #                 )
    #                 H_list.append(H_pm[chunk_idx] @ H_fp)
    #             except Exception as e:
    #                 print(f"Error computing homography for frame {idx}: {e}")
    #                 H_list.append(np.eye(3, dtype=np.float64))

    # DIRETCT METHOD
    H_list = []
    if include_homography:
        print("[INFO] build_mosaic: computing frame->mosaic homographies (direct)")
        feature = cfg['stitch'].get('feature', 'SIFT')  # prefer SIFT here
        reproj = cfg['stitch'].get('reproj_thresh', 3.0)
        resize_factor = cfg.get('stitch', {}).get('resize_factor', 1.0)

        for idx, frame in enumerate(images):  # 'images' are the (possibly resized) frames
            try:
                H_fm = compute_homography(frame, mosaic, feature=feature, reproj_thresh=reproj)
                # adjust for original-size detections
                if resize_factor != 1.0:
                    s = resize_factor
                    S = np.array([[s,0,0],[0,s,0],[0,0,1]], dtype=np.float64)
                    H_fm = H_fm @ S
                H_list.append(H_fm)
            except Exception as e:
                print(f"[WARN] frame {idx}: homography failed ({e}); marking invalid")
                H_list.append(None)  # mark invalid; we’ll skip these later

        # save homographies to JSON
        meta = {
            "resize_factor": cfg.get("stitch", {}).get("resize_factor", 1.0),
            "stitch_mode": cfg.get("stitch", {}).get("mode", "PANORAMA"),
            "chunk_size": cfg.get("stitch", {}).get("chunk_size", len(img_paths)),
            "feature": cfg.get("stitch", {}).get("feature", "ORB"),
            "reproj_thresh": cfg.get("stitch", {}).get("reproj_thresh", 4.0),
            "mosaic_shape": mosaic.shape,  # (H, W, C)
        }
        out_json = save_homographies_json(H_list, img_paths, cfg["paths"]["interim_homographies"], meta=meta)
        print(f"[INFO] build_mosaic: saved homographies to {out_json}")
    else:
        print("[INFO] build_mosaic: skipping homography computation")
    return mosaic, H_list
