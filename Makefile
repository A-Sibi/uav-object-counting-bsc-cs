ENV=parking
CONFIG=configs/default.yaml

init:
\tconda create -n $(ENV) python=3.10 -y

run1:
\tpython -m src.cli --mode pipeline1 --video data/raw/video.mp4 --config $(CONFIG)

format:
\tblack src

clean:
\trm -rf data/interim/* experiments/*
