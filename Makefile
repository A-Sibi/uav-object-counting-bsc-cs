# Makefile for UAV Object Counting Project

# Configuration variables
CONFIG   = configs/default.yaml
WEIGHTS  = yolov8n.pt
VIDEO    = data/raw/video1.mp4
IMG      = data/raw/test_image3.jpg
FRAMES   = data/interim/frames

.PHONY: init run1 run2 detect stitch logs format clean

# Download YOLOv8n weights to project root if missing
$(WEIGHTS):
	@echo "Downloading YOLOv8n weights to $(WEIGHTS)..."
	@wget -O $(WEIGHTS) https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
	@echo "Downloaded YOLOv8n weights to $(WEIGHTS)"


# 1. Initialize environment & install dependencies
init:
	poetry env use python3.11
	poetry install


# 2. Run Pipeline 1 (mosaic-first detection)
run1:
	poetry run python -m src.cli \
		--mode pipeline1 \
		--video $(VIDEO) \
		--config $(CONFIG)

# 3. Run Pipeline 2 (frame-first detection)
run2:
	poetry run python -m src.cli \
		--mode pipeline2 \
		--video $(VIDEO) \
		--config $(CONFIG)

# 4. Run frame extraction (saves frames to data/interim)
extract:
	poetry run python -m src.cli \
		--mode extract \
		--video $(VIDEO) \
		--config $(CONFIG)

# 5. Run stitching on a folder of frames
stitch:
	poetry run python -m src.cli \
		--mode stitch \
		--config $(CONFIG)

# 6. Run single-image detection (saves annotated output to data/processed)
detect:
	poetry run python -m src.cli \
		--mode detect \
		--image $(IMG) \
		--config $(CONFIG)


format:
	poetry run black src

clean:
	rm -rf data/interim/* data/processed/* experiments/*
