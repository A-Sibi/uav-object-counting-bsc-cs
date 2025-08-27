# This is a test on a lot of true positives

Both pipelines doing good, we enconter problem with moving cars, cars parked on the street on the side not translated correctly casuing misses in true localization.

## p2 postprocess has problems with diagonally parked cars which AABB overlap a lot

The "problem" was actually having good homographies so our defensive filter for quantile sizes did more harm than good. It effectively removed good true detections on diagonal cars just because their AABB was the lagerst in the set.
