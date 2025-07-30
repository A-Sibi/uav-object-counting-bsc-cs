# 📘 Project Development Plan


## 🎯 Objective

To create a modular and configurable computer vision system that processes aerial video footage to output the number and positions of vehicles in a parking lot. The system should explore and compare two distinct processing pipelines.

---

## 📂 Input & Output

* **Input:** A video file captured by a drone flying over a parking lot where the camera angle is positioned orthogonally to the ground, hovering over all relevant parking areas.
* **Output:**

  * Total vehicle count
  * Visual mosaic with localized vehicle positions
  * Structured data of detected objects (optional)

---

## 🔁 Pipeline Architectures

### Pipeline 1: Mosaic-First Detection

```
1. Video →
2. Frame Extraction →
3. Image Stitching (Mosaic) →
4. Object Detection on Mosaic →
5. Counting + Position Labeling →
6. Result
```

**Focus:** Build a complete visual representation first, then detect objects on the unified image.

**Challenges:**

* Partial cars on seams
* Detection errors due to blending artifacts

---

### Pipeline 2: Frame-First Detection

```
1. Video →
2. Frame Extraction →
3. Object Detection on Frames →
4. Image Stitching (Mosaic) with Stored Transforms →
5. Project Frame Detections onto Mosaic →
6. Counting + Position Labeling →
7. Result
```

**Focus:** Detect cars while still on individual frames, then stitch and map results to mosaic.

**Challenges:**

* Avoiding duplicate counts across overlapping frames
* Accurately transforming detections to mosaic space

---

## ⚖️ Comparison Criteria

* Detection accuracy (count precision)
* Localization quality
* Stitching clarity and usefulness
* Computational efficiency
* Robustness to overlapping cars / duplicate detections

---

## 🧱 Core Modules to Build

* `cli.py` — orchestrate pipeline based on CLI and config
    * `run_pipeline1()`
    * `run_pipeline2()`
* `detector.py` — run YOLOv8 detection (on an mosaic or frame)
* `mosaic.py` — stitch frames into a single image
* `detection_projector.py` — map detections to mosaic (pipeline 2)
* `result_writer.py` — save visual and numerical outputs

---

## 📋 Output Expectations

* Mosaic image with bounding boxes and IDs
* Detection summary file (CSV/JSON)
* Comparison table of both pipelines

---

## 🔧 Configuration

All parameters (paths, model settings, detection thresholds) will be stored in YAML config files and loaded dynamically per run.

---

## ✅ Final Deliverables

* Fully working implementation of both pipelines
* Evaluation results and visuals
* Thesis write-up with methodology, results, and analysis
