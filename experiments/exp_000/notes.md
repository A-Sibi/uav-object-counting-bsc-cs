### Example experiment output layout

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
```
