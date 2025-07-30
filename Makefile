CONFIG=configs/default.yaml

init:
  poetry env use python3.11
  poetry install

run1:
  poetry run python -m src.cli --mode pipeline1 --video data/raw/video.mp4 --config $(CONFIG)

format:
  poetry run black src

clean:
  rm -rf data/interim/* experiments/*
