from pathlib import Path
import cv2
import numpy as np

def build_mosaic(images_dir: str, cfg: dict):
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

    # Placeholder homographies for pipeline 2 mapping
    H_list = [np.eye(3, dtype=np.float64) for _ in img_paths]
    return mosaic, H_list


def _build_mosaic(images_dir, cfg):
    """
    Build a stitched mosaic from a directory of images.

    Args:
        images_dir (str): Path to folder containing frame images (e.g. JPEGs).
        cfg (dict): Configuration mapping, expects keys:
            cfg['stitch']['feature']       # e.g., 'ORB', 'SIFT'
            cfg['stitch']['reproj_thresh'] # e.g., 4.0

    Returns:
        mosaic (np.ndarray): The stitched panorama image (BGR).
        H_list (List[np.ndarray]): Homography matrices (3x3) for each input image.
    """
    
    # 1. Collect and load images
    img_paths = sorted(Path(images_dir).glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    images = [cv2.imread(str(p)) for p in img_paths]
    if any(im is None for im in images):
        raise RuntimeError("Failed to read one or more images for stitching.")

    # 2. Use OpenCV's high-level Stitcher
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    # Optionally tune internal parameters via detailed API in future
    status, mosaic = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"Stitching failed (status code: {status})")

    # 3. Placeholder homographies (identities) – for pipeline2 mapping later
    H_list = [np.eye(3, dtype=np.float64) for _ in images]

    return mosaic, H_list
