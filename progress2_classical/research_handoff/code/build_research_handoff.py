"""Package both completed experiments for research; no fitting or test prediction."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
PACKAGE_ROOT = CODE_ROOT / "research_handoff"
DATA_ROOTS = {
    "v1": PROJECT_ROOT / "comvis_dataset_ready_v1/dataset_prep",
    "v2": PROJECT_ROOT / "comvis_dataset_v2/dataset_prep",
}
RESULT_ROOTS = {
    "v1": CODE_ROOT / "progress2_results",
    "v2": CODE_ROOT / "progress2_results_v2",
}


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def training_diagnostics(results_root, target):
    """Descriptive in-sample scores from existing fitted models only."""
    config = json.loads((results_root / "selected_config.json").read_text())
    metrics, predictions = [], []
    for model_name, settings in config["models"].items():
        model = joblib.load(results_root / f"{model_name}_pipeline.joblib")
        frame = pd.read_csv(results_root / f"features_{settings['selected_variant']}.csv")
        frame = frame[frame["dataset_role"] == "train"].copy()
        predicted = model.predict(frame[settings["features"]])
        frame["predicted_label"] = [config["classes_in_order"][int(value)] for value in predicted] if model_name == "xgboost" else predicted
        for role, rows in (("train_original", frame[frame["is_augmented"] == 0]), ("train_including_augmentation", frame)):
            metrics.append({
                "model": model_name, "data_role": role, "sample_count": len(rows),
                "accuracy": accuracy_score(rows["label"], rows["predicted_label"]),
                "macro_f1": f1_score(rows["label"], rows["predicted_label"], labels=config["classes_in_order"], average="macro", zero_division=0),
            })
        originals = frame[frame["is_augmented"] == 0][["filename", "source_filename", "bottle_id", "label", "predicted_label"]].copy()
        originals.insert(0, "model", model_name)
        predictions.append(originals)
    pd.DataFrame(metrics).to_csv(target / "training_diagnostic_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(target / "training_original_predictions.csv", index=False)


def main():
    photo_records = []
    metadata_names = [
        "README.md", "train_original.csv", "train.csv", "val.csv", "test.csv",
        "metadata_original.csv", "augmentation_parameters.json", "augment_train.py",
        "dataset_summary.json", "dataset_revision.json", "verification.json",
    ]
    for dataset_version, data_root in DATA_ROOTS.items():
        destination = PACKAGE_ROOT / "versions" / dataset_version
        for name in metadata_names:
            if (data_root / name).is_file():
                copy_file(data_root / name, destination / "dataset_metadata" / name)
        metadata = pd.read_csv(data_root / "metadata_original.csv", dtype={"bottle_id": str})
        review = metadata[metadata["split"].isin(["train", "validation"])]
        assert len(review) == 15
        for row in review.to_dict("records"):
            source = data_root / row["filename"]
            assert hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"]
            target = destination / "review_photos" / row["filename"]
            copy_file(source, target)
            photo_records.append({"version": dataset_version, "split": row["split"], "bottle_id": row["bottle_id"], "filename": target.relative_to(PACKAGE_ROOT).as_posix(), "sha256": row["sha256"]})
        shutil.copytree(RESULT_ROOTS[dataset_version], destination / "results", dirs_exist_ok=True)
        training_diagnostics(RESULT_ROOTS[dataset_version], destination / "results")

    for name in ["experiment.py", "requirements.txt", "README.md", "build_notebook.py", "build_research_handoff.py", "progress2_classical_ml.ipynb", "progress2_classical_ml_v2_executed.ipynb"]:
        copy_file(CODE_ROOT / name, PACKAGE_ROOT / "code" / name)
    comparison = PACKAGE_ROOT / "comparison"
    comparison.mkdir(exist_ok=True)
    for name in ["validation_comparison_v1_v2.csv", "audit_v1_v2.json", "feature_reproducibility_v1_v2.json"]:
        copy_file(RESULT_ROOTS["v2"] / name, comparison / name)

    old_metadata = pd.read_csv(DATA_ROOTS["v1"] / "metadata_original.csv")
    new_metadata = pd.read_csv(DATA_ROOTS["v2"] / "metadata_original.csv")
    changes = old_metadata[["filename", "split", "label", "sha256"]].merge(
        new_metadata[["filename", "split", "label", "sha256"]], on="filename", suffixes=("_v1", "_v2"), validate="one_to_one")
    changes["image_bytes_changed"] = changes["sha256_v1"] != changes["sha256_v2"]
    changes.to_csv(comparison / "primary_photo_changes_v1_v2.csv", index=False)
    assert changes["image_bytes_changed"].sum() == 2
    old_predictions = pd.read_csv(RESULT_ROOTS["v1"] / "validation_predictions.csv")
    new_predictions = pd.read_csv(RESULT_ROOTS["v2"] / "validation_predictions.csv")
    columns = ["model", "source_filename", "true_label", "predicted_label", "correct"]
    old_predictions[columns].merge(new_predictions[columns], on=["model", "source_filename", "true_label"], suffixes=("_v1", "_v2"), validate="one_to_one").to_csv(comparison / "validation_predictions_side_by_side.csv", index=False)

    paths = sorted(path for path in PACKAGE_ROOT.rglob("*") if path.is_file() and path.name != "MANIFEST.json")
    photo_paths = {item["filename"] for item in photo_records}
    image_paths = {path.relative_to(PACKAGE_ROOT).as_posix() for path in paths if "review_photos" in path.parts}
    assert image_paths == photo_paths and len(photo_records) == 30
    assert not any("04" in Path(item["filename"]).stem for item in photo_records)
    assert not any(path.name.startswith("test_predictions") for path in paths)
    manifest = {
        "purpose": "Research review of both v1 and v2; not a complete runnable dataset",
        "included_primary_review_photos": photo_records,
        "omitted": ["test image pixels", "offline augmented JPEGs (metadata and features included)"],
        "files": [{"path": path.relative_to(PACKAGE_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
    }
    (PACKAGE_ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    archive_path = CODE_ROOT / "aquviva_research_handoff_v1_v2.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in paths + [PACKAGE_ROOT / "MANIFEST.json"]:
            archive.write(path, path.relative_to(PACKAGE_ROOT))
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for item in manifest["files"]:
            assert hashlib.sha256(archive.read(item["path"])).hexdigest() == item["sha256"]
    print(json.dumps({"archive": str(archive_path), "size_mb": round(archive_path.stat().st_size / 1_000_000, 1), "files": len(paths) + 1, "review_photos": len(photo_records), "test_images_included": 0}, indent=2))


if __name__ == "__main__":
    main()
