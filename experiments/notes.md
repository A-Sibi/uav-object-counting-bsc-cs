# Random notes

## Example experiment output layout

Each pipeline run should create a new experiment directory and record the configuration used. A recommended structure:

```text
exp_001/
├── config.yaml        # copy of the config used (paths, thresholds, model, frame_step, etc.)
├── notes.md          # observations or changes made in this experiment
├── mosaic.png        # output mosaic from this run
├── cars.json         # detections & projected points
├── metrics.json      # evaluation metrics (MAE, precision/recall, etc.)
└── logs.txt          # console output or training logs
```

Optionally include a timestamp in the folder name (e.g., `exp_20250801_1500/`) and a `README.md` summarizing the experiment.

## Tuning tips

* Speed first: feature=ORB, matcher=BF, method=RANSAC, reproj_thresh=4.0, ratio=0.75.

* Robustness first: feature=SIFT, matcher=FLANN, method=USAC_MAGSAC, reproj_thresh=3.0, ratio=0.7.

* If you’re getting too few matches, increase nfeatures or lighten the ratio (e.g., 0.8).

* If you get bad homographies, decrease ratio (stricter), reduce reproj_thresh, or switch to USAC.

This keeps your homography computation fully aligned with the YAML config, and matches the knobs you care about for your thesis experiments.

### transitive clustering for dual detections?

Why your result bent by ~10°:

Stitcher (even in SCANS) still estimates camera parameters pairwise and performs bundle-like adjustments that can over-fit on outliers, especially when textures repeat (rows of cars, lane lines). Once one partial tilts slightly, the next aligns to the tilted one and the error propagates. The global anchor + similarity model removes degrees of freedom that cause that bend.

Moving cars are ignored on videos compared to a single image, additional time variance information.
