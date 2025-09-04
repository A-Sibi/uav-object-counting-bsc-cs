# UAV Object Counting — BSc Thesis Project

This project aims to detect, count, and localize vehicles in parking lots from drone footage using two different image processing pipelines. It is part of a Bachelor thesis in Computer Science.

---

## 🧠 Project Overview

**Two pipelines will be developed and compared:**

* **Pipeline 1:** Extract frames → build mosaic → detect & count on mosaic
* **Pipeline 2:** Extract frames → detect on frames → build mosaic → project detections → count

Each approach has its pros and cons in terms of accuracy, complexity, and robustness against double-counting.

---

## Command Reference

All commands accept the global option: `-c, --config PATH`

* Path to YAML config (default: *configs/default.yaml*)

---

### run1 — Mosaic-first detection

Usage:

```bash
    uav-counter run1 [-c CONFIG] [VIDEO]
```

Description:
Extracts frames → stitches them into a mosaic → runs detection on the mosaic → writes annotated mosaic to `paths.processed_image`.

Args & defaults:

* VIDEO (optional): if omitted, uses `paths.raw_video` from the config.

Outputs:

* `paths.interim_frames/` extracted frames
* `paths.interim_mosaic` stitched mosaic
* `paths.processed_image` annotated mosaic

---

### run2 — Frame-first detection + projection

Usage:

```bash
    uav-counter run2 [-c CONFIG] [VIDEO]
```

Description:
Extracts frames → runs detection per-frame → stitches a mosaic (with frame→mosaic homographies) → projects frame detections into mosaic space → writes annotated mosaic to `paths.processed_image`.

Args & defaults:

* VIDEO (optional): if omitted, uses `paths.raw_video`.

Outputs:

* As in `run1`, plus homography JSON if enabled during stitch step.

---

### extract — Extract frames from video

Usage:

```bash
    uav-counter extract [-c CONFIG] [VIDEO]
```

Description:
Saves every `video.frame_step`-th frame as JPEGs to `paths.interim_frames`.

Args & defaults:

* VIDEO (optional): if omitted, uses `paths.raw_video`.

Outputs:

* `paths.interim_frames/` with `frame_00000.jpg`, …

---

### stitch — Build mosaic (optionally export homographies)

Usage:

```bash
    uav-counter stitch [-c CONFIG] [IMAGES_DIR] [-H|--include_homography]
```

Description:
Stitches frames from IMAGES_DIR (default `paths.interim_frames`) into a mosaic at `paths.interim_mosaic`. With `--include_homography`, computes frame→mosaic homographies and writes JSON to `paths.interim_homographies`.

Args & flags:

* IMAGES_DIR (optional): defaults to `paths.interim_frames`
* -H, --include_homography: compute and export homographies

Outputs:

* `paths.interim_partials/` partial mosaics (chunked)
* `paths.interim_mosaic` final mosaic
* `paths.interim_homographies/*.json` (when `-H` is set)

---

### detect — Detect on a single image

Usage:

```bash
    uav-counter detect [-c CONFIG] [IMAGE]
```

Description:
Runs the default detector on IMAGE (or `paths.raw_image` if omitted) and saves an annotated image to `paths.processed_image`. Detection results are also printed to stdout.

Args & defaults:

* IMAGE (optional): defaults to `paths.raw_image`

Outputs:

* `paths.processed_image` annotated image

---

### batch_detect — Detect on a directory of frames

Usage:

```bash
    uav-counter batch_detect [-c CONFIG] [IMAGES_DIR]
```

Description:
Runs detection on all *.jpg in IMAGES_DIR (default `paths.interim_frames`) and writes per-frame JSON + annotated previews.

Args & defaults:

* IMAGES_DIR (optional): defaults to `paths.interim_frames`

Outputs:

* `paths.interim_detections/*.json` per-frame detections
* `paths.interim_detections_batch/*_annotated.jpg` previews

---

### batch_map — Map saved frame detections into the mosaic

Usage:

```bash
    uav-counter batch_map [-c CONFIG] [DETECTIONS_DIR] [HOMOGRAPHIES_DIR]
```

Description:
Loads the stitched mosaic from `paths.interim_mosaic`, reads the latest homographies JSON from HOMOGRAPHIES_DIR, projects detections from DETECTIONS_DIR into mosaic space, and writes the annotated mosaic to `paths.processed_image`.

Outputs:

* `paths.processed_image` annotated mosaic

---

### save_data — Archive current processed data

Status:

* Not implemented yet (calling this will raise NotImplementedError)

---

## Environment Setup

This project uses **Poetry** and targets **Python 3.10–3.12**. The CLI is exposed as `uav-counter`.

### 1) Prerequisites

* Python 3.10, 3.11, or 3.12
* Git
* FFmpeg (recommended for robust video I/O)
* (Linux) OpenCV runtime libs:
    Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y ffmpeg libgl1 libglib2.0-0
* (Windows) If a Torch/OpenCV DLL is missing, install the Microsoft Visual C++ Redistributable (2022+)

### 2) Install Poetry

Option A (pipx):

```bash
    pipx install poetry
```

Option B (official installer):

```bash
    curl -sSL <https://install.python-poetry.org> | python3 -
```

Verify:

```bash
    poetry --version
```

### 3) Clone the repository

```bash
    git clone https://github.com/YOUR_USERNAME/uav-object-counting-bsc-cs.git
    cd uav-object-counting-bsc-cs
```

### 4) (Optional) Pin a Python version with pyenv

```powershell
    pyenv install -s 3.12.3
    pyenv local 3.12.3
    python --version
```

### 5) Install dependencies

```bash
    poetry install
```

This reads `pyproject.toml` and creates a virtualenv with the project’s dependencies.

Headless tip: if you hit Qt/GUI errors with OpenCV on servers, swap `opencv-python` for `opencv-python-headless` in `pyproject.toml`, then:
    poetry lock --no-update && poetry install

### 6) Activate the virtual environment

```bash
    poetry shell
```

After this, `uav-counter` is on your PATH and can be called directly.

### 7) (Optional) GPU-enabled Torch

If you have NVIDIA CUDA and want GPU acceleration, install a matching CUDA wheel **inside the active venv**:

```bash
    pip install --index-url <https://download.pytorch.org/whl/cu121> torch torchvision --upgrade
```

Check:

```bash
    python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
```

### 8) Prepare model weights

The default RB Vehicle detector expects: `./rb_vehicle.pth`
Place the file at the project root (or wire YOLO yourself; pipelines default to RB Vehicle).

### 9) Configure paths and parameters

Create or edit a config:

```bash
    cp configs/default.yaml configs/local.yaml
```

Key fields:

* *paths.raw_video / paths.raw_image*
* *paths.interim_* / paths.processed_image*
* *video.frame_step*
* *stitch.{feature,reproj_thresh,resize_factor,chunk_size,mode}*
* *detect.{conf,model,iou} (YOLO keys matter only if you wire the YOLO path)*

### 10) Verify the CLI

```bash
    uav-counter --help
    uav-counter stitch --help
```

### 11) Quick smoke tests

Full pipeline 1:

```bash
    uav-counter run1 data/raw/video.mp4
```

Full pipeline 2:

```bash
    uav-counter run2 data/raw/video.mp4
```

### 12) Troubleshooting

* OpenCV “libGL.so.1” / GTK/Qt errors (Linux):

```bash
    sudo apt-get install -y libgl1 libglib2.0-0
```

  Or use the headless wheel (see step 5).

* Torch not using GPU:

    Ensure `python -c "import torch; print(torch.cuda.is_available())"` prints True.
    Reinstall a Torch wheel matching your CUDA toolkit if needed.
* Videos won’t open:
    Install FFmpeg (Linux: `sudo apt-get install ffmpeg`) and verify `paths.raw_video`.
* No annotated outputs:
    Confirm `paths.processed_image` exists in your config and intermediates are written to `paths.interim_*`.

### 13) Usage reminder

With the venv active (`poetry shell`), run:

```bash
    uav-counter <subcommand> [options]
```

(If you prefer not to activate the venv, you can always prefix commands with `poetry run`.)

## 🧱 Source layout

```bash
src/                    
├── cli.py              # CLI entry point
├── pipelines.py        # Orchestrates both pipelines and module runs
├── detection/          # Detectors and related modules
│   ├── detectors/      # Specific detector implementations (e.g. YOLO, RB Vehicle)
│   │   ├── rb_vehicle.py
│   │   └── yolo.py
│   ├── __init__.py     # Entry: choose RB Vehicle (default) or YOLO
│   ├── detect.py       # Main detection call (imported in pipelines)
│   ├── types.py        # Type definitions (Detection, TranslatedDetection)
│   └── postprocess.py  # Filters: merging duplicates, support, texture ...
├── eval/               # (Not essential) Evaluation metrics and comparison logic
│   ├── annotate_gt.py  # Draws ground truth boxes on mosaic for visual comparison
│   ├── eval_mosaic.py  # Evaluation routines (precision, recall, F1)
│   ├── metrics.py      # Evaluation metrics (e.g. IoU, precision)
│   └── plot.py         # Evaluation plots (e.g. precision-recall curves)
├── mapping/
│   └── project.py      # Projects frame detections onto the mosaic via homographies
├── stitching/
│   └── mosaic.py       # Stitch frames; optional homography computation & JSON export
└── utils/
    ├── io.py           # I/O helpers, config, frame extraction, (de)serialization
    ├── vis.py          # Box drawing and simple visualization
    └── geometry.py     # Geometric utilities for transforms
```

## Data & results

```bash
configs/                # YAML configuration files for setting up pipeline behavior

data/
├── raw/                # Input videos
├── interim/            # Extracted frames, mosaics, temporary files
└── processed/          # Final outputs, visualizatons, results

experiments/            # Archived results and logs, grouped by experiment
├── exp_001/
│   ├── json/                   # Detection results and annotations
│   │   ├── p1_detections.json  # Raw detections from pipeline 1
│   │   ├── p2_detections.json  # Filtered detections from pipeline 2
│   │   ├── gt_detections.json  # Ground-truth boxes (mosaic coords)
│   │   └── h_list.json         # Latest frame→mosaic homographies
│   ├── ground_truth.jpg        # Reference image
│   ├── mosaic.jpg              # Final stitched mosaic
│   ├── overlay_p1.jpg          # Mosaic with overlayed p1 and gt boxes
│   ├── overlay_p2.jpg          # Mosaic with overlayed p2 and gt
│   ├── RESULT_p1.jpg           # Annotated mosaic from pipeline 1
│   ├── RESULT_p2.jpg           # Annotated mosaic from pipeline 2
│   ├── p1_logs.txt             # Stdout log from pipeline 1 run
│   ├── p2_logs.txt             # Stdout log from pipeline 2 run
│   ├── README.md               # Notes about this experiment
│   ├── p2_all_detections_mapped.jpg  # Mosaic with all p2 detections (no filtering)
└── exp_002/
    └── ...
```

---

## ⚙️ Configuration Example (`configs/default.yaml`)

```yaml
paths:
  raw_video: data/raw/video.mp4
  raw_image: data/raw/image.jpg
  interim_frames: data/interim/frames
  interim_partials: data/interim/partials
  interim_detections: data/interim/detections
  interim_detections_batch: data/interim/detections_batch
  interim_all_detections: data/interim/all_detections.json
  interim_mosaic: data/interim/mosaic.jpg
  interim_homographies: data/interim/homography
  processed_image: data/processed/processed_image.jpg
  processed_detections: data/processed/processed_detections.json
  all_detections_mapped: data/processed/p2_all_detections_mapped.jpg
  gt_image: data/raw/ground_truth.jpg
  gt_raw_annotations: data/interim/gt_raw_annotations.json
  gt_annotations: data/processed/gt_detections.json
  p1_result: data/processed/RESULT_p1.jpg
  p2_result: data/processed/RESULT_p2.jpg
  p1_detections: data/processed/p1_detections.json
  p2_detections: data/processed/p2_detections.json
  interim_pairs: experiments/pairs.txt

video:
  frame_step: 30       # take every 30th frame

detect:
  model: rb_vehicle     # {rb_vehicle, YOLO}
  conf: 0.5             # detector confidence treshold
  iou: 0.5
  YOLO:
    weight: yolov8n.pt

stitch: # edit in diploma
  mode: SCANS           # final stitching mode is PANORAMA
  chunk_size: 5
  chunk_overlap: 1
  resize_factor: 0.5

homography:
  compute: direct        # direct | chain
  feature:               # --- keypoint/descriptor extraction ---
    type: SIFT            # ORB | SIFT
    nfeatures: 4000
  match:                 # --- descriptor matching + ratio test ---
    strategy: FLANN       # AUTO | BF | FLANN  (AUTO: BF+Hamming for ORB, FLANN for SIFT)
    knn_k: 2             # 2 for Lowe's ratio test
    ratio: 0.7           # Lowe's ratio threshold (lower = stricter)
  estimate:              # --- robust H estimation ---
    method: USAC_MAGSAC  # RANSAC | LMEDS | USAC_MAGSAC | USAC_ACCURATE | USAC_FAST
    reproj_thresh: 3.0   # px
    confidence: 0.999
    max_iters: 10000
    min_inliers: 10      # reject H if too few inliers

postprocess:
  confidence:
    run: true
    conf_min: 0.78

  size_quantiles:
    run: false
    size_low_q: 0.1
    size_high_q: 0.95

  proximity_clustering:
    run: true
    prox_factor: 0.45
    iou_merge: 0.5

  support:
    run: true
    min_support_frames: 4
    min_support_members: 6

  texture_rejection:
    run: true
    texture_min: 40
    inner_shrink: 0.10
    empty_std_max: 15
    empty_edge_max: 0.02

eval:
  iou: 0.5
```

Alternate config files can be used for experiments.

---

## Notes

* `stitch.mode` selects `cv2.Stitcher_PANORAMA` or `cv2.Stitcher_SCANS`.
* When `--include_homography` is used, frame→mosaic homographies are computed (default **SIFT** in code) and exported to JSON under `paths.interim_homographies` with metadata (image order, stitch params, mosaic shape, etc.).

## 📤 Outputs

* `RESULT_p1.jpg` — annotated mosaic from pipeline 1
* `RESULT_p2.jpg` — annotated mosaic from pipeline 2
* `p1_detections.json` — raw detections from pipeline 1
* `p2_detections.json` — filtered detections from pipeline 2

---

## 🔍 Implementation Highlights

* **Stitching** (`stitching/mosaic.py`)
  * Chunked stitching (configurable `chunk_size`) with final merge
  * `SCANS` or `PANORAMA` mode via OpenCV Stitcher
  * Optional homography export: direct frame→mosaic estimation via feature matches and RANSAC
* **Homographies** are saved with image order; later used to **project** boxes from frame space to mosaic space.
* **Detection**
  * Default: **RB Vehicle** (`rfdetr`) with cached model and configurable confidence filter
  * Optional: **YOLO** (`ultralytics`) helper is included but not wired by default in pipelines
* **Projection** (`mapping/project.py`)
  * Robustness checks (finite, reasonable scale; size clamps) before accepting mapped boxes
* **Visualization** writes numbered boxes with confidences to the output image

---

## 👤 Author

**Aleksa Sibinović**
University of Ljubljana
Email: [as1871@student.uni-lj.si](mailto:as1871@student.uni-lj.si)
