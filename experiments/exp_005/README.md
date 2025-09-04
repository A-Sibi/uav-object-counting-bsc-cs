# Experiment 005

* This time we flew  more in a zig zag pattern instead of having 2 or 3 straight passes to check how it affects the detection results.

* It did create a black spot on a pivot point on mosaic, which caused some detections to be missed in pipeline 1.
* Pipeline 2 did catch it as it is not affected by the same stitching artifacts, but the toll on imprecise homographies still impacted the results.

* A cluster of missed detections in the top right of the mosaic (pipeline 2) was observed in the area at the end of the flight path, indicating that there were not enough support frames to mark them as valid, video should not start and stop abruptly to avoid this in the future.

---

**IMPORTANT:**

* At the end, I decided not to include this experiment in the final thesis as it is heavily affected by the low input video quality and pathing, making the results not very representative of the actual performance of the system.
