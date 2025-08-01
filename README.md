# UAV Object Counting — BSc Thesis Project

This project aims to detect, count, and localize vehicles in parking lots from drone footage using two different image processing pipelines. It is part of a Bachelor thesis in Computer Science.

---

## 🧠 Project Overview

**Two pipelines will be developed and compared:**

* **Pipeline 1:** Extract frames → build mosaic → detect & count on mosaic
* **Pipeline 2:** Extract frames → detect on frames → build mosaic → project detections → count

Each approach has its pros and cons in terms of accuracy, complexity, and robustness against double-counting.

---

## 📁 Folder Structure

```
configs/                # YAML configuration files for setting up pipeline behavior

data/
├── raw/                # Input videos
├── interim/            # Extracted frames, mosaics, temporary files
├── processed/          # Final outputs, visualizatons, results

experiments/            # Stored results and logs (e.g. annotated mosaics, CSVs)
├── exp_001/            # Grouped by experiments

src/                    # Source code
├── cli.py              # CLI entry point for executing pipelines
├── pipelines.py        # Pipeline logic implementation
├── detection/
│   ├── detector.py     # YOLOv8-based car detector
│   ├── postprocess.py  # Optional post-filtering logic (e.g. NMS adjustments)
├── eval/               # (Planned) Evaluation metrics and comparison logic
├── mapping/
│   ├── project.py      # (Planned) Mapping frame detections to mosaic coordinates
├── stitching/
│   ├── mosaic.py       # Builds stitched image from multiple frames
├── utils/
│   ├── io.py           # YAML config loader, frame extractor
│   ├── vis.py          # Visualization helpers
│   ├── geometry.py     # Geometric utilities for transforms
```

---

## ⚙️ Environment Setup

> ✅ This project uses **Poetry** for dependency management.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/uav-object-counting-bsc-cs.git
cd uav-object-counting-bsc-cs
```

### 2. Setup the environment

```bash
make init
```

### 3. (Optional) Enter virtual environment for interactive debugging

```bash
poetry shell
```

Use this if you want to run Python or tools manually within the Poetry-managed environment. For all project operations, `make` and `poetry run` are sufficient.

---

## ▶️ Running the Pipeline

### 1. Default full pipeline (Pipeline 1):

```bash
make run1
```

This will:

* Extract frames from `data/raw/video.mp4`
* Build a mosaic
* Detect cars using YOLOv8
* Save results in `experiments/`

### 2. Format code

```bash
make format
```

### 3. Clean results

```bash
make clean
```

---

## ⚙️ Configuration Example (`configs/default.yaml`)

```yaml
video:
  frame_step: 5
stitch:
  feature: ORB
  reproj_thresh: 4.0
detect:
  model: yolov8n.pt
  conf: 0.4
  iou: 0.5
```

Alternate config files can be used for experiments.

---

## 🗺 Planned Features

* [ ] Pipeline 1 implementation
* [ ] Pipeline 2 implementation
* [ ] Vehicle localization via projection
* [ ] Mosaic annotation (bounding boxes, IDs)
* [ ] Accuracy and efficiency comparison of both pipelines
* [ ] Optional: zone-based counting, density heatmaps

---

## 📌 Notes for Future Dev

* [ ] Add automatic saving of counts to `.csv` or `.json`
* [ ] Abstract out common utility functions
* [ ] Add logging or progress output
* [ ] Add support for switching pipelines via `--mode` flag

---

## 👤 Author

**Aleksa Sibinović**
BSc Computer Science, University of Ljubljana
Email: [as1871@student.uni-lj.si](mailto:as1871@student.uni-lj.si)

> *This README is a living document. Update as features evolve.*
