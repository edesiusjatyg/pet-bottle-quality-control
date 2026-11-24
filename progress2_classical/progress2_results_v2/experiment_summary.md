# Progress 2 classical ML — validation summary

Dataset version: v2. Features and search grid are unchanged from the initial experiment.

Configuration selection used validation macro-F1. The test split has not been evaluated by this run.

| Model | Pipeline | Parameters | Validation macro-F1 | Accuracy | Correct |
|---|---|---|---:|---:|---:|
| svm | center_roi__gray | `{"C": 0.1, "class_weight": "balanced", "gamma": "scale", "kernel": "linear"}` | 0.375 | 0.400 | 2/5 |
| xgboost | center_roi__gray | `{"learning_rate": 0.05, "max_depth": 2, "n_estimators": 80}` | 0.350 | 0.400 | 2/5 |

Validation contains five original photos from bottle 03. Each class except normal has one validation photo, so one error changes that class recall by 1.0.
The primary dataset has 20 original photos from four physical bottles. The 290 train augmentations are photometric descendants and do not add independent observations.
Version v2 uses retaken penyok01 (train) and penyok03 (validation) with more visible defects. The original subtle-dent photos and their old descendants are excluded. The retake followed initial evaluation, so score changes do not establish improved generalization or subtle-dent sensitivity.
Side-specific upper-right features are retained from the initial experiment alongside whole-body profile features; a local feature is not proof of defect detection.
Background varies by bottle; normal images usually show the label back and tilted-label images usually show the front. Results may reflect acquisition shortcuts.

Inspect `pcd_roi_comparison_train_originals.png` and `feature_detection_summary.csv`; threshold and Canny did not yield a clean bottle mask on these transparent, reflective photos.
