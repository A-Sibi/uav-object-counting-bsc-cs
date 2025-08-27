# Experiment 004

* We tried to create a scenario with a lot of moving objects to test the robustness of our detection pipeline.
* The test was to see if we could accurately detect only parked vehicles and ignore moving ones.
* The test was rough, due to misfortunate traffic jam event which p2 shows, leading to many false positives.
* On the other hand it is interesting to see time-based patterns in the detection results.

* This test also had an issue with proper marking on the ground truth annotations, which affected the evaluation metrics (cars are rotated and semi secluded).
