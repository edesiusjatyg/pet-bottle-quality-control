# Progress 2 classical ML — validation summary

Configuration selection used validation macro-F1. The test split has not been evaluated by this run.

| Model | Pipeline | Parameters | Validation macro-F1 | Accuracy | Correct |
|---|---|---|---:|---:|---:|
| svm | center_roi__clahe | `{"C": 0.1, "class_weight": "balanced", "gamma": "scale", "kernel": "linear"}` | 0.625 | 0.600 | 3/5 |
| xgboost | center_roi__gray | `{"learning_rate": 0.05, "max_depth": 2, "n_estimators": 80}` | 0.625 | 0.600 | 3/5 |

Validation contains five original photos from bottle 03. Each class except normal has one validation photo, so one error changes that class recall by 1.0.
The primary dataset has 20 original photos from four physical bottles. The 290 train augmentations are photometric descendants and do not add independent observations.
The subtle dents on bottles 01 and 03 remain `penyok`; the confirmed local defect is on the upper-right body. Side-specific profile features compare that zone with a train-normal reference.
Background varies by bottle; normal images usually show the label back and tilted-label images usually show the front. Results may reflect acquisition shortcuts.

Inspect `pcd_roi_comparison_train_originals.png` and `feature_detection_summary.csv`; threshold and Canny did not yield a clean bottle mask on these transparent, reflective photos.
