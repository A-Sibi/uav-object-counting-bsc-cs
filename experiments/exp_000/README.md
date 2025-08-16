I realized some insights could be made from testing so I am uploading some unstructured results here.
This is done after numerous stitching tuning, so only the best result from pipeline 1 was saved.

Logs are not yet standardized.

# Pipeline 1

mode scans is better for 2d, but for some reason, final stitching on panorama is better?
mode panorama + chunk_size 5 + frame_step 60 gave best results but one weird partial


# Pipeline 2

In p2 all detections, 4 horizontally oriented detections in the middle of the bottom right quadrant that seem out of place are actually good detections, since a car was moving through that area at that time.