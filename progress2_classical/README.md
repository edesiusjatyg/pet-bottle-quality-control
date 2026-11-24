# Progress 2 v2: PCD + handcrafted features + classical ML

This package contains the initial EDA and the two classical baselines for the
Aquviva 700 ml visual-defect project. The main notebook is
`progress2_classical_ml.ipynb`; `experiment.py` holds the reusable functions.

`progress2_classical_ml_v2_executed.ipynb` contains the saved local Jupyter
execution, with all seven code cells completed without errors and test disabled.
Kaggle execution has not been performed. CPU is sufficient; no GPU is required.
Read `progress2_results_v2/v2_analysis.md` for the error analysis and the
descriptive v1/v2 comparison. Validation macro-F1 is 0.375 for SVM and 0.350
for XGBoost, each with 2/5 correct photos.

## Dataset and fixed split

Attach or extract the provided `comvis_dataset_v2.zip` so that the dataset
root contains `train.csv`, `train_original.csv`, `val.csv`, `test.csv`,
`metadata_original.csv`, `augmentation_parameters.json`, `img/`, and
`augmented/`. The original archive layout has `dataset_prep/` at its root.

The split is fixed by physical bottle ID: bottles 01 and 02 for training,
bottle 03 for validation, and bottle 04 for test. The notebook does not create
a random split or generate more augmentations. It trains on the supplied 300
rows (10 primary training photos plus 290 photometric descendants), while
reporting the 20 primary photos and four physical bottles separately.

On Kaggle, add the unzipped data as a Kaggle Dataset and add this folder as a
second Kaggle Dataset (or place `experiment.py` beside the notebook). The
notebook searches `/kaggle/input/**/dataset_prep` and also accepts an explicit
`AQUVIVA_DATASET_ROOT` path. It writes artifacts to
`/kaggle/working/progress2_results_v2/`. The notebook checks that its input is
version v2. If multiple dataset versions are attached, set
`AQUVIVA_DATASET_ROOT` explicitly in the first code cell.

The existing `progress2_results/` contains the initial v1 results and is retained.
The v2 run writes new artifacts to `progress2_results_v2/`. Input CSV fingerprints,
revision metadata, package versions, and the script checksum are saved with the
selected configuration. An output folder from another dataset version is rejected.

## Run

1. Install the packages in `requirements.txt` if the runtime does not already
   include them. Kaggle usually includes NumPy, pandas, OpenCV, scikit-learn,
   matplotlib, and XGBoost.
2. Open and run `progress2_classical_ml.ipynb` from top to bottom.
3. Review `pcd_roi_comparison_train_originals.png` and the detector diagnostics
   before treating the PCD as a useful representation.
4. Use validation macro-F1 to select the ROI/contrast variant and model
   settings. The selected models/configuration and validation predictions are
   written to the output directory.
5. The final notebook cell has `RUN_FINAL_TEST = False`. After the selected
   configurations are locked, set it to `True` once to run the separate test
   evaluation. Do not use those five test photos to revise preprocessing,
   features, or hyperparameters.

To run the same pipeline as a script from the project root:

```bash
AQUVIVA_DATASET_ROOT="$PWD/comvis_dataset_v2/dataset_prep" \
AQUVIVA_OUTPUT_DIR="$PWD/progress2_classical/progress2_results_v2" \
python progress2_classical/experiment.py
```

## PCD and feature notes

The exploratory pipeline compares a fixed center crop with a cap-anchored ROI,
grayscale with CLAHE, Otsu thresholding, Canny, and 3×3 morphological closing.
The bottle axis is estimated from paired outer-edge evidence inside the ROI;
the label is never straightened independently. The visual comparison uses only
the ten original training photos.

Features include an edge-derived silhouette-width and left/right edge profile,
deviations from a normal profile fitted only on original train normals, and a
separate right-upper-body inward-deviation feature. The local feature was designed
for the original subtle dents in v1 and is retained alongside the full-body profile
for this v2 run. The retake is not assumed to affect only the upper-right body. The
pipeline also includes edge density, GLCM contrast/homogeneity/energy, and a
white low-saturation label-region detector with estimated angle. A failed label measurement remains missing (`NaN`) and
is handled by an imputer fitted on train; the exported feature CSV preserves
those missing values and the `label_found` diagnostic. It is not replaced by a
guessed label area or angle. The width-profile detector is also diagnostic; a
low valid-row fraction means its measurements should be interpreted with care.

`feature_overlay_train_*.png` and `feature_overlay_validation_*.png` show the
actual sampled edge pairs and detected label boxes. Strong edges may belong
to printed labels, ribs or reflections; a valid-row count is not segmentation
accuracy. On `normal01_light.jpg`, for example, the label box selects a lower
reflection instead of the label. The extractor remains an exploratory baseline
and needs more reliable boundary and label detection.

## Scope and limitations

Ground truth comes from the physical condition confirmed by the photographer.
Dataset v2 replaces `penyok01.jpg` (train) and `penyok03.jpg` (validation) with
retakes showing more visible defects. The 37 descendants of `penyok01.jpg` were
also replaced; all other train images and all test images are unchanged. The
augmentation settings and the feature/model search remain the same. The retake
followed initial evaluation, so v1 versus v2 scores describe different data
conditions and do not establish improved generalization or subtle-dent detection.
Validation and test each contain five original photos from one physical bottle,
so all scores are preliminary and do not establish generalization.

Background differs by bottle ID, and normal photos generally show the back of
the label while tilted-label photos show the front. Reflections, droplets,
orientation, and lighting also vary. These may act as shortcuts or distractors;
a high score is not proof that the model recognizes the defect itself. The
photometric descendants increase training rows but do not add independent
bottles or photo scenes.

The notebook intentionally does not run the raw-photo or PCD + MobileNetV3
experiments; those remain the next project stage.
