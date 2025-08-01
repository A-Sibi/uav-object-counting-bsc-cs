### this folder is an example of how experiments and their results should be stored, with relevant data written for analyisis

*  maybe generate this folder each time you run a pipeline (note which pipeline was used)?
*  something like this could be a good output:
├── exp_001/
│   ├── config.yaml           # copy of the config used (paths, thresholds, model, frame_step…)
│   ├── notes.md              # any observations or changes made in this experiment
│   ├── mosaic.png            # output mosaic from this run
│   ├── cars.json             # detections & projected points
│   ├── metrics.json          # computed evaluation metrics (MAE, precision/recall, etc.)
│   └── logs.txt              # console output or training logs, if any