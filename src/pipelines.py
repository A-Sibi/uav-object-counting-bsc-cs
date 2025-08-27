# src/pipelines.py
import json
from pathlib import Path
import cv2
import torch
import time
import shutil

from src.detection.postprocess import clean_projected_detections
from src.eval.annotate_gt import annotate_image
from src.eval.eval_mosaic import evaluate_mosaic
from src.eval.metrics import load_boxes, pr_curve_and_ap50
from src.eval.plot import save_pr_curve
from src.utils.geometry import compute_homography, warp_boxes
from src.utils.io import *
from src.utils.vis import *
from src.stitching.mosaic import build_mosaic
from src.mapping.project import project_detections
from src.detection import detect

def run_extract(video_path: str, cfg: dict[str, any]) -> None:
    """
    Extract frames from the video and save them as JPEG images.
    :param video_path: Path to the input video file.
    :param cfg: Configuration dictionary containing paths and parameters.
    """

    print("Running Frame Extraction Pipeline...")
    extract_frames(
        video_path,
        cfg["paths"]["interim_frames"],
        cfg["video"]["frame_step"]
    )
    print("Frame extraction pipeline completed.")
    return None


def run_stitch(images_dir: Path, cfg, compute_homographies: bool) -> None:
    """
    Run the stitching pipeline on a set of frames.
    :param images_dir: Path to directory of frame images to stitch.
    :param cfg: Configuration dictionary containing stitching parameters.
    """

    print("Running Stitching Pipeline...")
    mosaic, _ = build_mosaic(images_dir, cfg, compute_homographies)
    save_np_image(mosaic, cfg["paths"]["interim_mosaic"])
    print("Stitching pipeline completed.")
    return None


def run_single_image_detect(image_path: str, cfg: dict[str, any]) -> None:
    """
    Run object detection on a single image or mosaic and print the results.
    """
    print("Running Detection Pipeline on: ", image_path)

    # detections = detect_cars_YOLO(image_path, detect_cfg)
    detections = detect(image_path, cfg)
    save_detections_json(detections, cfg["paths"]["processed_detections"])
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

    for idx, image_path in enumerate(image_paths):
        dets = detect(str(image_path), cfg)
        image_with_boxes = draw_rich_boxes(load_np_image(str(image_path)), dets)
        save_np_image(image_with_boxes, batch_dets_dir / f"{image_path.stem}_annotated.jpg")

        out_path = dets_dir / f"{image_path.stem}.json"
        clean = [{k: float(v) for k,v in d.items()} for d in dets]
        with open(out_path, 'w') as f:
            json.dump(clean, f, indent=2)

        if idx % 10 == 0:
            print(f"[INFO] detection: processed {idx + 1}/{len(image_paths)} images")
        
    print(f"Detections saved in: {out_path}")
    print("Batch detection completed.")
    return None


def run_batch_map(dets_dir: str, homography_dir: str, cfg: dict[str, any], filter=False) -> None:
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

    total_start = time.perf_counter()

    # 1. Load mosaic
    mosaic = load_np_image(cfg["paths"]["interim_mosaic"])

    print(f"Running Batch Map Pipeline for detections in {dets_dir} on mosaic '{cfg['paths']['interim_mosaic']}'")
    # 2. Load homographies
    h_json = latest_json_file(homography_dir)
    H_list, images_order, meta = load_homographies_json(h_json)
    print(f"[INFO] Loaded {len(H_list)} homographies from: {h_json}")

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
    projected_detections = project_detections(dets_per_frame, H_list, mosaic_shape=mosaic.shape[:2])
    print(f"[INFO] Projected {len(projected_detections)} detections to new coordinates.")

    # 6) (Optional) merge duplicates here later (DBSCAN / distance NMS)
    if filter:
        print("Running filter on projected detections...")
        projected_detections = clean_projected_detections(mosaic, projected_detections, cfg)
        ...

    # 7) Draw and save the results
    save_detections_json(projected_detections, cfg["paths"]["processed_detections"])
    print(f"Processed detections saved to: {cfg['paths']['processed_detections']}")
    image_with_boxes = draw_translated_boxes(mosaic, projected_detections)
    out_path = cfg["paths"]["all_detections_mapped"] if not filter else cfg["paths"]["p2_result"]
    save_np_image(image_with_boxes, out_path)

    # Timer for the entire pipeline
    total_elapsed = time.perf_counter() - total_start
    total_minutes = int(total_elapsed // 60)
    total_seconds = int(total_elapsed % 60)
    print(f"[TIMER] Batch map: total time {total_minutes:02d}:{total_seconds:02d}s ")

    print(f"Detected {len(projected_detections)} cars in the mosaic.")
    print(f"Annotated mosaic saved to '{out_path}'")
    print("Batch mapping completed.")
    
    return None


def run_pipeline1(video_path, cfg: dict[str, any]) -> None:
    print("Running Pipeline 1...")

    total_start = time.perf_counter()

    # 1. Extract frames
    extract_frames(
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
    detections = detect(image_path, cfg)
    image_with_boxes =  draw_rich_boxes(load_np_image(image_path), detections)

    # 4. Save results
    save_detections_json(detections, cfg["paths"]["p1_detections"])
    print(f"Detections saved to '{cfg['paths']['p1_detections']}'")
    save_np_image(image_with_boxes, cfg["paths"]["p1_result"])
    print(f"Annotated image saved to '{cfg['paths']['p1_result']}'")

    # Timer for the entire pipeline
    total_elapsed = time.perf_counter() - total_start
    total_minutes = int(total_elapsed // 60)
    total_seconds = int(total_elapsed % 60)
    print(f"[TIMER] Pipeline 1: total time {total_minutes:02d}:{total_seconds:02d}s ")

    print(f"Detected {len(detections)} cars in the mosaic.")
    print("Pipeline 1 completed successfully.")
    return None


def run_pipeline2(video_path, cfg: dict[str, any]) -> None:
    print("Running Pipeline 2...")

    total_start = time.perf_counter()


    # 1. Extract frames (only if not already done)
    frames_dir = Path(cfg["paths"]["interim_frames"])
    frames_dir.mkdir(parents=True, exist_ok=True)
    if not any(frames_dir.iterdir()):
        print(f"Extracting frames from video: {video_path}")
        extract_frames(
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
    dets_dir = Path(cfg["paths"]["interim_detections"])
    dets_dir.mkdir(parents=True, exist_ok=True)
    
    frame_detections = []
    with torch.no_grad():
        for i, frame_path in enumerate(frame_paths, start=1):
            detections = detect(frame_path, cfg)
            frame_detections.append(detections)

            out_path = Path(cfg["paths"]["interim_detections"]) / f"{frame_path.stem}.json"
            clean = [{k: float(v) for k,v in d.items()} for d in detections]
            with open(out_path, 'w') as f:
                json.dump(clean, f, indent=2)


            if i % 10 == 0:
                print(f"[INFO] detection: processed {i + 1}/{len(frame_paths)} images")


    # 3. Stitch frames into a mosaic
    images_dir = cfg["paths"]["interim_frames"]
    mosaic, H_list = build_mosaic(images_dir, cfg, True)
    save_np_image(mosaic, cfg["paths"]["interim_mosaic"])

    
    # 4. Map detections to the mosaic
    projected_detections = project_detections(frame_detections, H_list, mosaic_shape=mosaic.shape[:2])
    save_detections_json(projected_detections, cfg["paths"]["interim_all_detections"])
    all_dets_image = draw_translated_boxes(mosaic.copy(), projected_detections)
    save_np_image(all_dets_image, cfg["paths"]["all_detections_mapped"])



    # remove excessive detections
    projected_detections = clean_projected_detections(mosaic, projected_detections, cfg)
    save_detections_json(projected_detections, cfg["paths"]["p2_detections"])
    print(f"Detections saved to '{cfg['paths']['p2_detections']}'")
    image_with_boxes = draw_translated_boxes(mosaic, projected_detections)

    # 5. Save results
    save_np_image(image_with_boxes, cfg["paths"]["p2_result"])
    print(f"Annotated image saved to '{cfg['paths']['p2_result']}'")

    
    # Timer for the entire pipeline
    total_elapsed = time.perf_counter() - total_start
    total_minutes = int(total_elapsed // 60)
    total_seconds = int(total_elapsed % 60)
    print(f"[TIMER] Pipeline 2: total time {total_minutes:02d}:{total_seconds:02d}s ")

    print(f"Detected {len(projected_detections)} cars in the mosaic.")
    print("Pipeline 2 completed successfully.")
    return None


def save_data(exp_name: str, cfg: dict, force: bool = False) -> None:
    """
    Copy experiment-related files into experiments/<exp_name>.
    Warn if destination exists. Prompt unless --force is given.
    """
    keys_to_save = [
        "interim_mosaic",
        "processed_detections",
        "all_detections_mapped",
        "gt_image",
        "gt_raw_annotations",
        "gt_annotations",
        "p1_result",
        "p2_result",
        "p1_detections",
        "p2_detections"
    ]

    paths_cfg = (cfg or {}).get("paths", {})
    exp_root = Path("experiments") / exp_name
    exp_root.mkdir(parents=True, exist_ok=True)

    # Special handling for homographies - only save latest
    if "interim_homographies" in paths_cfg:
        homog_dir = Path(paths_cfg["interim_homographies"])
        if homog_dir.exists():
            latest_homog = latest_json_file(homog_dir)
            if latest_homog:
                dst = exp_root / latest_homog.name
                if dst.exists():
                    if force:
                        print(f"[OVERWRITE] {dst}")
                    else:
                        ans = input(f"[WARN] destination file exists: {dst}. Overwrite? [y/N] ")
                        if ans.lower() != "y":
                            print("[SKIP] kept existing file")
                        else:
                            shutil.copy2(latest_homog, dst)
                            print(f"[INFO] saved latest homography: {latest_homog} -> {dst}")
                else:
                    shutil.copy2(latest_homog, dst)
                    print(f"[INFO] saved latest homography: {latest_homog} -> {dst}")

    for key in keys_to_save:
        src_str = paths_cfg.get(key)
        if not src_str:
            print(f"[WARN] config missing key: {key}")
            continue

        src = Path(src_str)
        if not src.exists():
            print(f"[WARN] no file: {src}")
            continue

        if src.is_dir():
            dst = exp_root / src.name
            if dst.exists():
                if force:
                    shutil.rmtree(dst)
                    print(f"[OVERWRITE] {dst}")
                else:
                    ans = input(f"[WARN] destination folder exists: {dst}. Overwrite? [y/N] ")
                    if ans.lower() != "y":
                        print("[SKIP] kept existing folder")
                        continue
                    shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"[INFO] saved: {src} -> {dst}")
        else:
            # --- JSON files go into subfolder json/ ---
            if src.suffix.lower() == ".json":
                dst_dir = exp_root / "json"
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src.name
            else:
                dst = exp_root / src.name
            if dst.exists():
                if force:
                    print(f"[OVERWRITE] {dst}")
                else:
                    ans = input(f"[WARN] destination file exists: {dst}. Overwrite? [y/N] ")
                    if ans.lower() != "y":
                        print("[SKIP] kept existing file")
                        continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"[INFO] saved: {src} -> {dst}")
    print(f"[INFO] All relevant files copied to experiments/{exp_name}/")

def run_clear(cfg: dict, keep_interim=False, keep_processed=False) -> None:
    """
    Clear the interim and processed data directories unless flagged to keep.
    """
    print("Clearing data...")
    interim_dir = Path("data/interim")
    processed_dir = Path("data/processed")

    if not keep_interim and interim_dir.exists():
        shutil.rmtree(interim_dir, ignore_errors=True)
        print("[INFO] Removed data/interim")
    else:
        print("[INFO] Kept data/interim")

    if not keep_processed and processed_dir.exists():
        shutil.rmtree(processed_dir)
        print("[INFO] Removed data/processed")
    else:
        print("[INFO] Kept data/processed")

    print("Clear command finished.")

def run_annotate(
    mosaic: Path,
    cfg: dict,
    on_mosaic: bool = False,
    gt_image: Path | None = None,
    out_path: Path | None = None,
) -> None:
    """
    - on_mosaic=False: annotate on GT image, auto-compute H=compute_homography(GT, mosaic, cfg), warp to mosaic coords.
    - on_mosaic=True: annotate directly on mosaic (no homography).
    Always writes final JSON in MOSAIC coordinates (what eval expects).
    Also writes a raw JSON of the clicked boxes for traceability.
    """

    final_out = cfg["paths"].get("gt_annotations", "data/processed/gt_detections.json") if out_path is None else out_path

    if on_mosaic:
        print("Annotating directly on the mosaic (no homography).")
        # click on mosaic, save as-is (already in mosaic coords)
        raw_json = cfg["paths"].get("gt_annotations", "data/processed/gt_detections.json")
        saved = annotate_image(str(mosaic), str(raw_json))
        with open(saved, "r", encoding="utf-8") as f_in, open(final_out, "w", encoding="utf-8") as f_out:
            json.dump(json.load(f_in), f_out, indent=2)
        print(f"[EVAL] GT saved → {final_out}")
        return

    print("Annotating on the GT image (warping to mosaic coords).")
    # click on GT image
    if gt_image is None:
        raise RuntimeError("GT image not provided (use --gt-image or set cfg.paths.gt_image).")
    raw_json = cfg["paths"].get("gt_raw_image", "data/interim/gt_raw_image.json")
    saved_raw = annotate_image(str(gt_image), str(raw_json))
    print(f"[EVAL] Raw GT (image coords) → {saved_raw}")

    # auto-compute H (GT -> mosaic) using your function
    gt_img = cv2.imread(str(gt_image))
    mosaic_img = cv2.imread(str(mosaic))
    if gt_img is None or mosaic_img is None:
        raise FileNotFoundError("Could not read GT or mosaic image.")
    H = compute_homography(gt_img, mosaic_img, cfg)  # <- your function

    # warp boxes to mosaic coords and save
    with open(saved_raw, "r", encoding="utf-8") as f:
        boxes = json.load(f)
    boxes_mosaic = warp_boxes(boxes, H)
    with open(final_out, "w", encoding="utf-8") as f:
        json.dump(boxes_mosaic, f, indent=2)
    print(f"[EVAL] GT (warped to mosaic) saved → {final_out}")

def run_eval(pairs: Path, iou: float | None, cfg: dict[str, any]) -> None:
    """
    Standardized wrapper: reads IoU from cfg if not provided, prints metrics via eval module,
    and creates per-pair overlays on the current mosaic.
    YAML:
      eval:
        iou: 0.5
      paths:
        results: experiments
        interim_mosaic: data/interim/mosaic.jpg
    """
    iou_val = float(iou if iou is not None else (cfg.get("eval") or {}).get("iou", 0.5))
    results_dir = "data/processed"

    print(f"Running evaluation with IoU={iou_val:.2f} and pairs file: {pairs}")
    # 1) metrics
    _ = evaluate_mosaic(str(pairs), iou_val)  # prints detailed report

    # === PR curves ===
    pr_dir = Path(results_dir + "/" + "pr_curves")
    pr_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(pairs, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for idx, ln in enumerate(lines, start=1):
            parts = [s.strip() for s in ln.split(";")]
            pred_json, gt_json = parts[0], parts[1]
            # preberi detekcije
            preds = load_boxes(pred_json)
            gts   = load_boxes(gt_json)
            # izračun PR in AP@IoU (isti kot eval)
            AP, precisions, recalls, best = pr_curve_and_ap50(preds, gts, iou_val)
            # varnost: če seznami prazni, preskoči
            if not recalls or not precisions:
                print(f"[EVAL] PR skipped (no points) for line {idx}")
                continue
            # shranimo krivuljo
            # optional meta: video/pipeline iz parsa (če imaš; tukaj ga lahko izluščiš kot pri overlay)
            title = f"AP@{iou_val:.2f}={AP:.3f}"
            out_path = pr_dir / f"pr_{idx:02d}.png"
            saved = save_pr_curve(recalls, precisions, AP, best, str(out_path), title=title)
            print(f"[EVAL] PR curve saved → {saved}")
    except Exception as e:
        print(f"[WARN] PR curve generation skipped: {e}")