"""Progress 2 classical ML experiments for the Aquviva bottle dataset.

All learned preprocessing is fitted on training data only. Validation selects
the candidate pipeline and model settings. Test data is loaded only by the
separate final-evaluation function.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import zipfile
from importlib.metadata import version
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSES = ["normal", "label_miring", "label_hilang", "penyok"]
SEED = 42
MAX_SIDE = 768
PROFILE_Y = np.array([0.28, 0.34, 0.40, 0.46, 0.52, 0.58, 0.64, 0.70, 0.76, 0.82])
PROFILE_COLS = [f"profile_width_{int(round(y * 100)):02d}" for y in PROFILE_Y]
PROFILE_LEFT_COLS = [f"profile_left_x_{int(round(y * 100)):02d}" for y in PROFILE_Y]
PROFILE_RIGHT_COLS = [f"profile_right_x_{int(round(y * 100)):02d}" for y in PROFILE_Y]
PROFILE_REFERENCE_COLS = PROFILE_COLS + PROFILE_LEFT_COLS + PROFILE_RIGHT_COLS
UPPER_RIGHT_COLS = [PROFILE_RIGHT_COLS[index] for index, y in enumerate(PROFILE_Y) if y in (0.34, 0.40, 0.46)]
DIAGNOSTIC_COLS = {
    "filename", "label", "bottle_id", "split", "source_filename", "is_augmented",
    "view_id", "qc_status", "qc_note", "roi_mode", "contrast_mode",
    "cap_found", "axis_found", "axis_angle_deg", "axis_fit_r2", "roi_fallback",
    "label_box_x0", "label_box_y0", "label_box_x1", "label_box_y1", "label_angle_fit_r2",
}


def read_csv(root: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(root / name, dtype={"bottle_id": str})
    frame["bottle_id"] = frame["bottle_id"].str.zfill(2)
    return frame


def dataset_identity(root: str | Path) -> dict:
    """Identify input bytes so artifacts cannot silently cross dataset versions."""
    root = Path(root)
    revision_path = root / "dataset_revision.json"
    revision = json.loads(revision_path.read_text()) if revision_path.is_file() else {"version": "v1"}
    names = ["train.csv", "train_original.csv", "val.csv", "test.csv", "metadata_original.csv", "augmentation_parameters.json"]
    return {
        "version": revision["version"],
        "revision": revision,
        "input_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
    }


def audit_dataset(root: str | Path) -> dict:
    """Read and validate every CSV row, all file paths, and source lineage."""
    root = Path(root)
    train_original = read_csv(root, "train_original.csv")
    train = read_csv(root, "train.csv")
    val = read_csv(root, "val.csv")
    metadata = read_csv(root, "metadata_original.csv")
    test = read_csv(root, "test.csv")

    expected_splits = {"train": {"01", "02"}, "validation": {"03"}, "test": {"04"}}
    test_ids = set(test["bottle_id"])
    split_ids = {
        "train": set(train_original["bottle_id"]),
        "validation": set(val["bottle_id"]),
        "test": test_ids,
    }
    for split, expected in expected_splits.items():
        if split_ids[split] != expected:
            raise ValueError(f"Unexpected {split} bottle IDs: {split_ids[split]}")
    if any(split_ids[a] & split_ids[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Bottle IDs overlap across splits")

    if len(train_original) != 10 or len(train) != 300 or len(val) != 5 or len(test) != 5 or len(test_ids) != 1:
        raise ValueError("Dataset counts do not match the documented locked split")
    if set(train["label"]) != set(CLASSES) or set(val["label"]) != set(CLASSES) or set(test["label"]) != set(CLASSES):
        raise ValueError("Unexpected or missing class names")
    train_counts = train["label"].value_counts().reindex(CLASSES, fill_value=0)
    if not (train_counts == 75).all():
        raise ValueError(f"Training data is not 75 per class: {train_counts.to_dict()}")
    if set(train_original["filename"]) - set(train["filename"]):
        raise ValueError("train.csv is missing original training rows")

    for split_name, frame in {"train": train, "validation": val}.items():
        missing = [name for name in frame["filename"] if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {split_name} files: {missing[:5]}")
    if not set(metadata["filename"]) == set(train_original["filename"]) | set(val["filename"]) | set(test["filename"]):
        raise ValueError("metadata_original.csv does not contain exactly the 20 primary photos")
    if not (train["split"] == "train").all() or not (val["split"] == "validation").all() or not (test["split"] == "test").all():
        raise ValueError("CSV split fields do not match the locked split files")
    confirmed_dents = metadata[metadata["source_filename"].isin(["penyok01.jpg", "penyok03.jpg"])]
    if set(confirmed_dents["label"]) != {"penyok"}:
        raise ValueError("Confirmed subtle dents were relabeled unexpectedly")

    params = json.loads((root / "augmentation_parameters.json").read_text())
    if any(not (0.85 <= item["brightness"] <= 1.15 and 0.90 <= item["contrast"] <= 1.10 and 0.0 <= item["blur"] <= 0.35) for item in params):
        raise ValueError("Augmentation parameters are outside the ranges documented in README.md")
    augmented = train[train["is_augmented"].astype(int) == 1]
    param_files = {item["filename"] for item in params}
    if len(params) != 290 or set(augmented["filename"]) != param_files:
        raise ValueError("Augmentation parameter records do not match train.csv")
    source_labels = train_original.set_index("source_filename")["label"].to_dict()
    source_bottles = train_original.set_index("source_filename")["bottle_id"].to_dict()
    for item in params:
        source = item["source_filename"]
        if source not in source_labels:
            raise ValueError(f"Augmentation source is not in training originals: {source}")
        target = augmented.loc[augmented["filename"] == item["filename"]].iloc[0]
        if target["label"] != source_labels[source] or target["bottle_id"] != source_bottles[source]:
            raise ValueError(f"Augmentation source metadata mismatch: {item['filename']}")
        if not (root / item["filename"]).is_file():
            raise FileNotFoundError(root / item["filename"])

    # Confirm hashes for every CSV sample and every primary image.
    hash_checks = 0
    for row in train.to_dict("records"):
        digest = hashlib.sha256((root / row["filename"]).read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Training image checksum mismatch: {row['filename']}")
        hash_checks += 1
    for row in metadata.to_dict("records"):
        path = root / row["filename"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Original image checksum mismatch: {row['filename']}")
        if row["filename"] not in set(train["filename"]):
            hash_checks += 1

    archive_path = root.parent.parent / f"{root.parent.name}.zip"
    archive_audit = {"available": archive_path.is_file()}
    if archive_path.is_file():
        with zipfile.ZipFile(archive_path) as archive:
            archive_names = {name for name in archive.namelist() if not name.endswith("/")}
            extracted_paths = {path.relative_to(root.parent).as_posix(): path for path in root.parent.rglob("*") if path.is_file()}
            if archive_names != set(extracted_paths):
                raise ValueError("ZIP contents and extracted dataset file lists differ")
            mismatches = []
            for name in sorted(archive_names):
                archive_hash = hashlib.sha256(archive.read(name)).digest()
                extracted_hash = hashlib.sha256(extracted_paths[name].read_bytes()).digest()
                if archive_hash != extracted_hash:
                    mismatches.append(name)
            if mismatches:
                raise ValueError(f"ZIP/unzipped byte mismatch: {mismatches[:5]}")
            archive_audit.update({"file_count": len(archive_names), "uncompressed_bytes": sum(item.file_size for item in archive.infolist() if not item.is_dir()), "byte_mismatches": 0})

    summary = {
        "dataset": dataset_identity(root),
        "primary_photos": len(metadata),
        "physical_bottles": int(metadata["bottle_id"].nunique()),
        "train_original": len(train_original),
        "train_total_including_augmentation": len(train),
        "train_augmented": int(augmented.shape[0]),
        "validation_original": len(val),
        "test_original_reserved": len(test),
        "train_class_counts": train_counts.to_dict(),
        "validation_class_counts": val["label"].value_counts().reindex(CLASSES, fill_value=0).to_dict(),
        "test_class_counts_reserved": test["label"].value_counts().reindex(CLASSES, fill_value=0).to_dict(),
        "bottle_ids_by_split": {key: sorted(value) for key, value in split_ids.items()},
        "original_hashes_checked": hash_checks,
        "augmentation_parameters_checked": len(params),
        "augmentation_source_count": int(augmented["source_filename"].nunique()),
        "augmentation_parameter_ranges": {
            key: [float(min(item[key] for item in params)), float(max(item[key] for item in params))]
            for key in ("brightness", "contrast", "blur")
        },
        "zip_vs_unzipped": archive_audit,
    }
    return summary


def run_eda(root: str | Path, output_dir: str | Path, audit_summary: dict | None = None) -> dict:
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = audit_summary or audit_dataset(root)
    originals = read_csv(root, "metadata_original.csv")
    train_original = read_csv(root, "train_original.csv")
    train = read_csv(root, "train.csv")
    params = pd.DataFrame(json.loads((root / "augmentation_parameters.json").read_text()))

    originals.to_csv(output_dir / "original_photo_inventory.csv", index=False)
    lineage = train.groupby(["label", "source_filename", "bottle_id"], as_index=False).agg(
        total_train_rows=("filename", "size"),
        augmented_rows=("is_augmented", lambda values: int((values.astype(int) == 1).sum())),
    )
    lineage.to_csv(output_dir / "augmentation_lineage.csv", index=False)
    params.to_csv(output_dir / "augmentation_parameters.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    split_order = ["train", "validation", "test"]
    counts = originals.groupby(["label", "split"]).size().unstack(fill_value=0).reindex(CLASSES, fill_value=0)
    counts = counts.reindex(columns=split_order, fill_value=0)
    counts.plot(kind="bar", ax=axes[0], color=["#4263eb", "#2f9e44", "#f08c00"])
    axes[0].set_title("20 foto primer menurut split")
    axes[0].set_xlabel("Kelas")
    axes[0].set_ylabel("Foto asli")
    axes[0].tick_params(axis="x", rotation=20)
    train_orig = train[train["is_augmented"].astype(int) == 0]["label"].value_counts().reindex(CLASSES, fill_value=0)
    train_aug = train[train["is_augmented"].astype(int) == 1]["label"].value_counts().reindex(CLASSES, fill_value=0)
    x = np.arange(len(CLASSES))
    axes[1].bar(x - 0.18, train_orig.values, 0.36, label="Foto train asli", color="#4263eb")
    axes[1].bar(x + 0.18, train_aug.values, 0.36, label="Turunan augmentasi", color="#adb5bd")
    axes[1].set_xticks(x, CLASSES, rotation=20)
    axes[1].set_title("Train: sumber primer dipisahkan dari augmentasi")
    axes[1].set_ylabel("Baris CSV")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "eda_class_counts.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    for ax, row in zip(axes.flat, train_original.sort_values(["bottle_id", "label", "view_id"]).to_dict("records")):
        image = load_rgb(root / row["filename"], max_side=512)
        ax.imshow(image)
        ax.set_title(f"{row['filename'].split('/')[-1]}\n{row['label']} | botol {row['bottle_id']} | {row['split']}", fontsize=7)
        ax.axis("off")
    fig.suptitle("EDA visual 10 foto asli train; augmentasi dan holdout test tidak ditampilkan", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_original_photos.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, column, color in zip(axes, ["brightness", "contrast", "blur"], ["#4263eb", "#2f9e44", "#f08c00"]):
        ax.hist(params[column], bins=18, color=color, alpha=0.85)
        ax.set_title(column)
        ax.set_ylabel("Jumlah turunan")
    fig.suptitle("Parameter augmentasi offline, hanya untuk turunan train")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_augmentation_parameters.png", dpi=180)
    plt.close(fig)

    (output_dir / "dataset_audit.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_rgb(path: str | Path, max_side: int = MAX_SIDE) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image)


def find_cap_center(rgb: np.ndarray) -> tuple[tuple[int, int] | None, bool]:
    """Find a blue cap candidate near the top; never use the class label."""
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([88, 35, 35]), np.array([138, 255, 255]))
    prior = np.zeros_like(mask)
    prior[: max(1, int(height * 0.23)), int(width * 0.18):int(width * 0.82)] = 255
    mask = cv2.bitwise_and(mask, prior)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        ratio = w / max(h, 1)
        area_fraction = area / (height * width)
        if 1.05 <= ratio <= 5.0 and 0.00025 <= area_fraction <= 0.035:
            cx, cy = centroids[index]
            score = area * (1.0 - min(cy / max(height * 0.23, 1), 1.0) * 0.35)
            candidates.append((score, (int(round(cx)), int(round(cy)))))
    if not candidates:
        return None, False
    return max(candidates, key=lambda item: item[0])[1], True


def _estimate_body_axis(gray: np.ndarray) -> tuple[float | None, float, np.ndarray, np.ndarray]:
    """Estimate a near-vertical body axis from paired outer-edge evidence."""
    height, width = gray.shape
    edges = cv2.Canny(gray, 35, 95)
    y_values, centers, strengths = [], [], []
    start, stop = int(height * 0.22), int(height * 0.88)
    left_band = (int(width * 0.14), int(width * 0.46))
    right_band = (int(width * 0.54), int(width * 0.86))
    for y in range(start, stop, max(1, height // 180)):
        y0, y1 = max(0, y - 2), min(height, y + 3)
        left_sum = edges[y0:y1, left_band[0]:left_band[1]].sum(axis=0)
        right_sum = edges[y0:y1, right_band[0]:right_band[1]].sum(axis=0)
        if left_sum.size == 0 or right_sum.size == 0:
            continue
        lx = left_band[0] + int(np.argmax(left_sum))
        rx = right_band[0] + int(np.argmax(right_sum))
        left_strength, right_strength = int(left_sum.max()), int(right_sum.max())
        if 0.2 <= (rx - lx) / width <= 0.7 and 0.25 <= (lx + rx) / (2 * width) <= 0.75:
            y_values.append(y)
            centers.append((lx + rx) / 2)
            strengths.append(min(left_strength, right_strength))
    if len(y_values) < 25:
        return None, 0.0, np.asarray(y_values), np.asarray(centers)
    ys, xs, weights = np.asarray(y_values), np.asarray(centers), np.asarray(strengths, dtype=float)
    keep = weights >= max(2.0, np.percentile(weights, 25))
    ys, xs = ys[keep], xs[keep]
    if len(ys) < 20:
        return None, 0.0, ys, xs
    slope, intercept = np.polyfit(ys, xs, 1)
    predicted = slope * ys + intercept
    residual = xs - predicted
    robust_keep = np.abs(residual - np.median(residual)) <= max(3.0, np.percentile(np.abs(residual), 70))
    if robust_keep.sum() >= 15:
        ys, xs = ys[robust_keep], xs[robust_keep]
        slope, intercept = np.polyfit(ys, xs, 1)
        predicted = slope * ys + intercept
    total = float(np.sum((xs - xs.mean()) ** 2))
    r2 = 1.0 - float(np.sum((xs - predicted) ** 2)) / total if total > 1e-6 else 0.0
    angle = float(np.degrees(np.arctan(slope)))
    if r2 < 0.15 or abs(angle) > 8:
        return None, r2, ys, xs
    return angle, r2, ys, xs


def prepare_roi(rgb: np.ndarray, roi_mode: str = "cap_roi") -> tuple[np.ndarray, dict]:
    height, width = rgb.shape[:2]
    cap_center, cap_found = find_cap_center(rgb)
    if roi_mode == "cap_roi" and cap_found:
        center_x = cap_center[0]
        fallback = False
    elif roi_mode in ("cap_roi", "center_roi"):
        center_x = width / 2
        fallback = roi_mode == "cap_roi" and not cap_found
    else:
        raise ValueError(f"Unknown ROI mode: {roi_mode}")
    crop_width = min(width, int(round(width * 0.62)))
    x0 = int(round(center_x - crop_width / 2))
    x0 = min(max(0, x0), width - crop_width)
    crop = rgb[:, x0:x0 + crop_width].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    angle, fit_r2, axis_y, axis_x = _estimate_body_axis(gray)
    axis_found = angle is not None
    if axis_found:
        matrix = cv2.getRotationMatrix2D((crop.shape[1] / 2, crop.shape[0] / 2), -angle, 1.0)
        crop = cv2.warpAffine(crop, matrix, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    details = {
        "cap_found": int(cap_found),
        "axis_found": int(axis_found),
        "axis_angle_deg": angle if axis_found else np.nan,
        "axis_fit_r2": fit_r2,
        "roi_fallback": int(fallback),
        "cap_center_x": cap_center[0] if cap_found else np.nan,
        "cap_center_y": cap_center[1] if cap_found else np.nan,
        "crop_x0": x0,
        "crop_x1": x0 + crop_width,
        "axis_y": axis_y,
        "axis_x": axis_x,
    }
    return crop, details


def clahe_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def otsu_and_close(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return binary, closed


def render_pcd_comparison(root: str | Path, output_dir: str | Path) -> dict:
    """Visual comparison on the ten original training photos only."""
    root, output_dir = Path(root), Path(output_dir)
    train_original = read_csv(root, "train_original.csv").sort_values(["label", "bottle_id"])
    fig, axes = plt.subplots(len(train_original), 6, figsize=(17, 22))
    headers = ["Raw + ROI box", "Cap ROI + axis", "Otsu threshold", "Canny", "Canny + closing", "CLAHE"]
    for column, header in enumerate(headers):
        axes[0, column].set_title(header, fontsize=10)
    diagnostics = []
    for row_index, row in enumerate(train_original.to_dict("records")):
        rgb = load_rgb(root / row["filename"])
        gray_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        roi, details = prepare_roi(rgb, "cap_roi")
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        binary, closed_binary = otsu_and_close(gray)
        edges = cv2.Canny(gray, 35, 95)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        enhanced = clahe_gray(gray)
        raw_with_roi = rgb.copy()
        cv2.rectangle(raw_with_roi, (details["crop_x0"], 0), (details["crop_x1"] - 1, raw_with_roi.shape[0] - 1), (255, 40, 40), 3)
        if details["cap_found"]:
            cv2.circle(raw_with_roi, (int(details["cap_center_x"]), int(details["cap_center_y"])), 7, (255, 230, 0), -1)
        image_views = [raw_with_roi, roi, binary, edges, closed_edges, enhanced]
        for column, image in enumerate(image_views):
            axes[row_index, column].imshow(image, cmap="gray" if image.ndim == 2 else None)
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(f"{row['filename'].split('/')[-1]}\n{row['label']}", fontsize=7)
        axis_label = f"cap={details['cap_found']} axis={details['axis_found']}"
        axes[row_index, 1].text(0.02, 0.02, axis_label, color="yellow", fontsize=7, transform=axes[row_index, 1].transAxes, bbox={"facecolor": "black", "alpha": 0.55, "pad": 2})
        diagnostics.append({"filename": row["filename"], "label": row["label"], **{key: value for key, value in details.items() if key not in ("axis_x", "axis_y")}})
    fig.suptitle("PCD exploratory comparison — train originals only\nThreshold/Canny outputs are edge candidates, not a validated bottle mask", y=0.998)
    fig.tight_layout(rect=[0.02, 0, 1, 0.985])
    path = output_dir / "pcd_roi_comparison_train_originals.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    result = pd.DataFrame(diagnostics)
    result.to_csv(output_dir / "pcd_roi_diagnostics_train_originals.csv", index=False)
    return {"visual_path": str(path), "cap_detection_rate": float(result["cap_found"].mean()), "axis_detection_rate": float(result["axis_found"].mean()), "rows": len(result)}


def detect_label(rgb: np.ndarray, axis_found: bool) -> tuple[dict, tuple[int, int, int, int] | None]:
    """Measure a broad white-label band; leave geometric measurements missing on failure.

    The 0.50 detection threshold is fixed from train-original inspection: train
    label-absent photos peak below 0.42, while train photos with a visible label
    peak above 0.61 in this cap-centred ROI. It is not fitted using validation.
    """
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    white = (hsv[:, :, 1] < 112) & (hsv[:, :, 2] > 128)
    x0, x1 = int(width * 0.18), int(width * 0.82)
    y0, y1 = int(height * 0.28), int(height * 0.75)
    row_coverage = white[y0:y1, x0:x1].mean(axis=1)
    smoothed = cv2.GaussianBlur(row_coverage.astype(np.float32).reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=2).ravel()
    peak_index = int(np.argmax(smoothed))
    peak = float(smoothed[peak_index])
    peak_y = y0 + peak_index
    result = {
        "label_found": int(peak >= 0.50),
        "label_white_coverage_peak": peak,
        "label_white_coverage_mean": float(row_coverage.mean()),
        "label_peak_y_ratio": float(peak_y / height),
        "label_area_ratio": np.nan,
        "label_width_ratio": np.nan,
        "label_height_ratio": np.nan,
        "label_angle_deg": np.nan,
        "label_angle_fit_r2": np.nan,
        "label_box_x0": np.nan,
        "label_box_y0": np.nan,
        "label_box_x1": np.nan,
        "label_box_y1": np.nan,
    }
    if peak < 0.50:
        return result, None

    # Keep the band local to the coverage peak so bright background above or
    # below the label cannot become a fabricated rectangle measurement.
    band_threshold = max(0.38, peak * 0.55)
    active_rows = np.flatnonzero(smoothed >= band_threshold)
    if len(active_rows) < height * 0.035:
        return result, None
    y_low = max(y0 + int(active_rows.min()), peak_y - int(height * 0.13))
    y_high = min(y0 + int(active_rows.max()) + 1, peak_y + int(height * 0.13))
    if y_high - y_low < height * 0.035:
        return result, None
    band = white[y_low:y_high, x0:x1]
    column_coverage = band.mean(axis=0)
    active_columns = np.flatnonzero(column_coverage >= max(0.10, peak * 0.12))
    if len(active_columns) < width * 0.16:
        return result, None
    box_x0 = x0 + int(active_columns.min())
    box_x1 = x0 + int(active_columns.max()) + 1
    box_width = box_x1 - box_x0
    band = white[y_low:y_high, box_x0:box_x1]
    if box_width < width * 0.18:
        return result, None

    angle = np.nan
    fit_r2 = np.nan
    if axis_found:
        # Use the same fixed y-window around the coverage peak across views.
        # The earlier connected-component PCA collapsed to one large reflection
        # component on these images, so fit the row-centres of white pixels.
        angle_y0 = max(0, peak_y - int(height * 0.10))
        angle_y1 = min(height, peak_y + int(height * 0.10))
        angle_x0, angle_x1 = int(width * 0.12), int(width * 0.88)
        window = white[angle_y0:angle_y1, angle_x0:angle_x1]
        x_values, y_values = [], []
        minimum_pixels = max(4, int(window.shape[0] * 0.10))
        for column_index in range(window.shape[1]):
            column = np.flatnonzero(window[:, column_index])
            if len(column) >= minimum_pixels:
                x_values.append(angle_x0 + column_index)
                y_values.append(float(np.median(column + angle_y0)))
        if len(x_values) >= 12:
            x_values, y_values = np.asarray(x_values), np.asarray(y_values)
            slope, intercept = np.polyfit(x_values, y_values, 1)
            predicted = slope * x_values + intercept
            total = float(np.sum((y_values - y_values.mean()) ** 2))
            fit_r2 = 1.0 - float(np.sum((y_values - predicted) ** 2)) / total if total > 1e-6 else 0.0
            if fit_r2 >= 0.20:
                angle = float(np.degrees(np.arctan(slope)))

    result.update({
        "label_area_ratio": float(np.count_nonzero(band) / (height * width)),
        "label_width_ratio": float(box_width / width),
        "label_height_ratio": float((y_high - y_low) / height),
        "label_angle_deg": angle,
        "label_angle_fit_r2": fit_r2,
        "label_box_x0": box_x0 / width,
        "label_box_y0": y_low / height,
        "label_box_x1": box_x1 / width,
        "label_box_y1": y_high / height,
    })
    return result, (box_x0, y_low, box_width, y_high - y_low)


def _glcm_features(gray: np.ndarray, mask: np.ndarray) -> dict:
    levels = 16
    quantized = np.clip((gray.astype(np.uint16) * levels) // 256, 0, levels - 1).astype(np.uint8)
    values = {}
    pairs = []
    for dy, dx, suffix in [(0, 1, "h"), (1, 0, "v")]:
        a = quantized[: gray.shape[0] - dy or None, : gray.shape[1] - dx or None]
        b = quantized[dy:, dx:]
        valid = mask[: gray.shape[0] - dy or None, : gray.shape[1] - dx or None] & mask[dy:, dx:]
        if valid.sum() < 20:
            values[f"glcm_contrast_{suffix}"] = np.nan
            values[f"glcm_homogeneity_{suffix}"] = np.nan
            values[f"glcm_energy_{suffix}"] = np.nan
            continue
        encoded = a[valid].astype(np.int32) * levels + b[valid].astype(np.int32)
        matrix = np.bincount(encoded, minlength=levels * levels).reshape(levels, levels).astype(float)
        matrix = matrix + matrix.T
        matrix /= matrix.sum()
        i, j = np.indices(matrix.shape)
        values[f"glcm_contrast_{suffix}"] = float(np.sum(matrix * (i - j) ** 2) / (levels - 1) ** 2)
        values[f"glcm_homogeneity_{suffix}"] = float(np.sum(matrix / (1 + np.abs(i - j))))
        values[f"glcm_energy_{suffix}"] = float(np.sum(matrix**2))
        pairs.append(int(valid.sum()))
    values["glcm_pair_count"] = float(min(pairs)) if pairs else 0.0
    return values


def extract_features(rgb: np.ndarray, roi_mode: str = "cap_roi", contrast_mode: str = "gray") -> dict:
    roi, details = prepare_roi(rgb, roi_mode)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    feature_gray = clahe_gray(gray) if contrast_mode == "clahe" else gray.copy()
    if contrast_mode not in ("gray", "clahe"):
        raise ValueError(f"Unknown contrast mode: {contrast_mode}")
    edges = cv2.Canny(feature_gray, 35, 95)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    height, width = gray.shape
    features = {
        "cap_found": float(details["cap_found"]),
        "axis_found": float(details["axis_found"]),
        "axis_angle_deg": details["axis_angle_deg"],
        "axis_fit_r2": details["axis_fit_r2"],
        "roi_fallback": float(details["roi_fallback"]),
        "edge_density": float(np.count_nonzero(edges) / edges.size),
        "closed_edge_density": float(np.count_nonzero(closed_edges) / closed_edges.size),
    }

    left_low, left_high = int(width * 0.12), int(width * 0.46)
    right_low, right_high = int(width * 0.54), int(width * 0.88)
    row_edges = []
    row_left_edges = []
    row_right_edges = []
    row_strengths = []
    for fraction in PROFILE_Y:
        y = int(round(height * fraction))
        y0, y1 = max(0, y - 3), min(height, y + 4)
        band = edges[y0:y1]
        left_profile = band[:, left_low:left_high].sum(axis=0)
        right_profile = band[:, right_low:right_high].sum(axis=0)
        if left_profile.size == 0 or right_profile.size == 0:
            row_edges.append(np.nan)
            row_left_edges.append(np.nan)
            row_right_edges.append(np.nan)
            row_strengths.append(0.0)
            continue
        lx = left_low + int(np.argmax(left_profile))
        rx = right_low + int(np.argmax(right_profile))
        strength = float(min(left_profile.max(), right_profile.max()))
        valid = 0.22 <= (rx - lx) / width <= 0.72 and strength >= 2 * 255
        row_edges.append(float((rx - lx) / width) if valid else np.nan)
        row_left_edges.append(float(lx / width) if valid else np.nan)
        row_right_edges.append(float(rx / width) if valid else np.nan)
        row_strengths.append(strength)
    features.update({col: value for col, value in zip(PROFILE_COLS, row_edges)})
    features.update({col: value for col, value in zip(PROFILE_LEFT_COLS, row_left_edges)})
    features.update({col: value for col, value in zip(PROFILE_RIGHT_COLS, row_right_edges)})
    finite_widths = np.asarray([value for value in row_edges if np.isfinite(value)], dtype=float)
    features["profile_valid_fraction"] = float(len(finite_widths) / len(PROFILE_Y))
    upper_right_indices = [index for index, y_value in enumerate(PROFILE_Y) if y_value in (0.34, 0.40, 0.46)]
    features["profile_right_upper_valid_fraction"] = float(np.isfinite([row_right_edges[index] for index in upper_right_indices]).mean())
    features["profile_width_median"] = float(np.median(finite_widths)) if len(finite_widths) else np.nan
    features["profile_width_iqr"] = float(np.percentile(finite_widths, 75) - np.percentile(finite_widths, 25)) if len(finite_widths) >= 2 else np.nan

    label, label_box = detect_label(roi, bool(details["axis_found"]))
    features.update(label)
    body_mask = np.zeros((height, width), dtype=bool)
    body_mask[int(height * 0.18):int(height * 0.90), int(width * 0.22):int(width * 0.78)] = True
    if label_box is not None:
        x, y, w, h = label_box
        body_mask[max(0, y - 4):min(height, y + h + 4), max(0, x - 4):min(width, x + w + 4)] = False
    features.update(_glcm_features(feature_gray, body_mask))
    features["middle_gray_mean"] = float(np.mean(feature_gray[int(height * .25):int(height * .78), int(width * .28):int(width * .72)]))
    features["middle_gray_std"] = float(np.std(feature_gray[int(height * .25):int(height * .78), int(width * .28):int(width * .72)]))
    return features


def _load_features(frame: pd.DataFrame, root: Path, roi_mode: str, contrast_mode: str) -> pd.DataFrame:
    records = []
    for row in frame.to_dict("records"):
        rgb = load_rgb(root / row["filename"])
        features = extract_features(rgb, roi_mode, contrast_mode)
        records.append({**row, **features, "roi_mode": roi_mode, "contrast_mode": contrast_mode})
    return pd.DataFrame(records)


def add_normal_reference(train_features: pd.DataFrame, other_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    originals = train_features[
        (train_features["is_augmented"].astype(int) == 0) & (train_features["label"] == "normal")
    ]
    reference = {column: float(originals[column].mean()) for column in PROFILE_REFERENCE_COLS if originals[column].notna().any()}
    for frame in (train_features, other_features):
        _apply_profile_reference(frame, reference)
    return train_features, other_features, reference


def _apply_profile_reference(frame: pd.DataFrame, reference: dict) -> None:
    for column, value in reference.items():
        frame[f"{column}_raw_dev_from_train_normal"] = frame[column] - value

    width_deviation_cols = [f"{column}_raw_dev_from_train_normal" for column in PROFILE_COLS if column in reference]
    frame["profile_dev_mean_abs"] = frame[width_deviation_cols].abs().mean(axis=1) if width_deviation_cols else np.nan
    frame["profile_dev_max_abs"] = frame[width_deviation_cols].abs().max(axis=1) if width_deviation_cols else np.nan
    frame["profile_dev_fraction_over_03"] = (frame[width_deviation_cols].abs() > 0.03).mean(axis=1) if width_deviation_cols else np.nan

    left_raw = [f"{column}_raw_dev_from_train_normal" for column in PROFILE_LEFT_COLS if column in reference]
    right_raw = [f"{column}_raw_dev_from_train_normal" for column in PROFILE_RIGHT_COLS if column in reference]
    lower_pairs = []
    for index, y in enumerate(PROFILE_Y):
        if y >= 0.64:
            left_col = f"{PROFILE_LEFT_COLS[index]}_raw_dev_from_train_normal"
            right_col = f"{PROFILE_RIGHT_COLS[index]}_raw_dev_from_train_normal"
            if left_col in frame and right_col in frame:
                lower_pairs.append((frame[left_col] + frame[right_col]) / 2)
    translation = pd.concat(lower_pairs, axis=1).median(axis=1) if lower_pairs else pd.Series(np.nan, index=frame.index)
    for source_columns, side in ((left_raw, "left"), (right_raw, "right")):
        aligned_cols = []
        for raw_column in source_columns:
            aligned_column = raw_column.replace("_raw_dev_from_train_normal", "_aligned_dev_from_train_normal")
            frame[aligned_column] = frame[raw_column] - translation
            aligned_cols.append(aligned_column)
        frame[f"profile_{side}_dev_mean_abs"] = frame[aligned_cols].abs().mean(axis=1) if aligned_cols else np.nan
        frame[f"profile_{side}_dev_max_abs"] = frame[aligned_cols].abs().max(axis=1) if aligned_cols else np.nan

    upper_right_deviation_cols = []
    for column in UPPER_RIGHT_COLS:
        raw_column = f"{column}_raw_dev_from_train_normal"
        if raw_column in frame:
            inward_column = f"{column}_inward_dev_from_train_normal"
            frame[inward_column] = -(frame[raw_column] - translation)
            upper_right_deviation_cols.append(inward_column)
    frame["profile_right_upper_inward_mean"] = frame[upper_right_deviation_cols].mean(axis=1) if upper_right_deviation_cols else np.nan
    frame["profile_right_upper_inward_max"] = frame[upper_right_deviation_cols].max(axis=1) if upper_right_deviation_cols else np.nan


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    absolute_side_profiles = set(PROFILE_LEFT_COLS + PROFILE_RIGHT_COLS)
    raw_side_deviations = {f"{column}_raw_dev_from_train_normal" for column in PROFILE_LEFT_COLS + PROFILE_RIGHT_COLS}
    return [column for column in frame.select_dtypes(include=[np.number]).columns if column not in DIAGNOSTIC_COLS and column not in {"width", "height"} and column not in absolute_side_profiles and column not in raw_side_deviations]


def render_feature_overlays(root: str | Path, output_dir: str | Path) -> None:
    """Show actual sampled edge points and detected label regions."""
    from matplotlib.patches import Rectangle

    root, output_dir = Path(root), Path(output_dir)
    config = json.loads((output_dir / "selected_config.json").read_text())
    variants = sorted({item["selected_variant"] for item in config["models"].values()})
    for variant in variants:
        frame = pd.read_csv(output_dir / f"features_{variant}.csv")
        for role in ("train", "validation"):
            rows = frame[(frame["dataset_role"] == role) & (frame["is_augmented"] == 0)]
            fig, axes = plt.subplots(2 if role == "train" else 1, 5, figsize=(16, 9 if role == "train" else 5), squeeze=False)
            for ax, row in zip(axes.flat, rows.to_dict("records")):
                roi, _ = prepare_roi(load_rgb(root / row["filename"]), variant.split("__")[0])
                height, width = roi.shape[:2]
                ax.imshow(roi)
                for y, left_column, right_column in zip(PROFILE_Y, PROFILE_LEFT_COLS, PROFILE_RIGHT_COLS):
                    if np.isfinite(row[left_column]) and np.isfinite(row[right_column]):
                        ax.plot([row[left_column] * width, row[right_column] * width], [y * height] * 2, "o-", color="#00ffff", markersize=3, linewidth=0.7)
                if row["label_found"]:
                    x0, y0 = row["label_box_x0"] * width, row["label_box_y0"] * height
                    box_width = (row["label_box_x1"] - row["label_box_x0"]) * width
                    box_height = (row["label_box_y1"] - row["label_box_y0"]) * height
                    ax.add_patch(Rectangle((x0, y0), box_width, box_height, fill=False, edgecolor="yellow", linewidth=1.2))
                ax.set_title(f"{row['source_filename']}\naxis={int(row['axis_found'])}, profile={row['profile_valid_fraction']:.1f}", fontsize=8)
                ax.axis("off")
            fig.suptitle(f"{role} — {variant}\nCyan: sampled edge pairs (may hit label/reflections); yellow: label-region box", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            fig.savefig(output_dir / f"feature_overlay_{role}_{variant}.png", dpi=140)
            plt.close(fig)


def _evaluate(y_true: list[str], y_pred: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(CLASSES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    }


def _plot_confusion(metrics: dict, model_name: str, output_dir: Path) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    ax.set_title(f"Validation confusion matrix — {model_name}\nmacro-F1={metrics['macro_f1']:.3f}; n=5")
    for y in range(len(CLASSES)):
        for x in range(len(CLASSES)):
            ax.text(x, y, str(matrix[y, x]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(output_dir / f"validation_confusion_{model_name}.png", dpi=180)
    plt.close(fig)


def _plot_train_profiles(train_features: pd.DataFrame, variant: str, output_dir: Path) -> None:
    originals = train_features[train_features["is_augmented"].astype(int) == 0]
    colors = {"normal": "#4263eb", "label_miring": "#f08c00", "label_hilang": "#2f9e44", "penyok": "#c92a2a"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for row in originals.to_dict("records"):
        color = colors[row["label"]]
        width_deviation = [row.get(f"{column}_raw_dev_from_train_normal", np.nan) for column in PROFILE_COLS]
        axes[0].plot(PROFILE_Y, width_deviation, marker="o", linewidth=1.1, alpha=0.8, color=color, label=row["label"])
        right_deviation = [row.get(f"{column}_inward_dev_from_train_normal", np.nan) for column in UPPER_RIGHT_COLS]
        upper_y = [PROFILE_Y[PROFILE_RIGHT_COLS.index(column)] for column in UPPER_RIGHT_COLS]
        axes[1].plot(upper_y, right_deviation, marker="o", linewidth=1.1, alpha=0.8, color=color, label=row["label"])
    axes[0].axhline(0, color="#495057", linewidth=0.8)
    axes[0].axvspan(0.32, 0.48, color="#ffd43b", alpha=0.18, label="zona badan atas")
    axes[0].set_title("Deviasi lebar siluet dari referensi normal train")
    axes[0].set_xlabel("Ketinggian relatif ROI")
    axes[0].set_ylabel("Deviasi lebar / lebar ROI")
    axes[1].axhline(0, color="#495057", linewidth=0.8)
    axes[1].set_title("Perpindahan tepi kanan ke dalam, badan atas")
    axes[1].set_xlabel("Ketinggian relatif ROI")
    axes[1].set_ylabel("Referensi normal − posisi tepi kanan")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=5, frameon=False)
    fig.suptitle(f"Profil bentuk foto train asli saja — {variant}\nReferensi dihitung dari foto normal train; tepi yang gagal tetap kosong", y=1.02)
    fig.tight_layout(rect=[0, 0.12, 1, 0.98])
    fig.savefig(output_dir / f"silhouette_profile_deviation_{variant}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def select_models_and_save(root: str | Path, output_dir: str | Path) -> dict:
    """Compare two ROI modes × two contrast modes and select by validation macro-F1."""
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = dataset_identity(root)
    config_path = output_dir / "selected_config.json"
    if config_path.is_file():
        previous_identity = json.loads(config_path.read_text()).get("dataset")
        if previous_identity != identity:
            raise ValueError("Output folder contains a different dataset version. Use a new output folder.")
    train = read_csv(root, "train.csv")
    val = read_csv(root, "val.csv")
    variants = [(roi, contrast) for roi in ("center_roi", "cap_roi") for contrast in ("gray", "clahe")]
    search_rows, candidate_features, saved_models = [], {}, {}

    svm_settings = [
        {"C": c, "kernel": kernel, "gamma": "scale", "class_weight": "balanced"}
        for kernel, values in (("linear", [0.1, 1.0, 10.0]), ("rbf", [0.5, 2.0]))
        for c in values
    ]
    xgb_settings = [
        {"n_estimators": n, "max_depth": depth, "learning_rate": 0.05}
        for n in (80, 160) for depth in (2, 3)
    ]

    for roi_mode, contrast_mode in variants:
        variant = f"{roi_mode}__{contrast_mode}"
        train_features = _load_features(train, root, roi_mode, contrast_mode)
        val_features = _load_features(val, root, roi_mode, contrast_mode)
        train_features, val_features, reference = add_normal_reference(train_features, val_features)
        combined = pd.concat([train_features.assign(dataset_role="train"), val_features.assign(dataset_role="validation")], ignore_index=True)
        combined.to_csv(output_dir / f"features_{variant}.csv", index=False)
        _plot_train_profiles(train_features, variant, output_dir)
        train_original_features = train_features[train_features["is_augmented"].astype(int) == 0]
        for role, feature_frame in (("train_original", train_original_features), ("validation", val_features)):
            candidate_features.setdefault("detector_rows", []).append({
                "variant": variant,
                "data_role": role,
                "sample_count": len(feature_frame),
                "cap_found_rate": float(feature_frame["cap_found"].mean()),
                "axis_found_rate": float(feature_frame["axis_found"].mean()),
                "label_found_rate": float(feature_frame["label_found"].mean()),
                "profile_valid_fraction_mean": float(feature_frame["profile_valid_fraction"].mean()),
                "profile_right_upper_valid_fraction_mean": float(feature_frame["profile_right_upper_valid_fraction"].mean()),
                "label_angle_available_count": int(feature_frame["label_angle_deg"].notna().sum()),
            })
        candidate_features[variant] = (train_features, val_features, reference, _feature_columns(train_features))

        # Keep explicit extractor diagnostics separate from measurements: failed detections remain NaN.
        for model_name, settings_list in (("svm", svm_settings), ("xgboost", xgb_settings)):
            for setting in settings_list:
                columns = _feature_columns(train_features)
                if model_name == "svm":
                    estimator = Pipeline([
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                        ("classifier", SVC(random_state=SEED, **setting)),
                    ])
                else:
                    from xgboost import XGBClassifier
                    estimator = Pipeline([
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("classifier", XGBClassifier(
                            objective="multi:softprob", num_class=len(CLASSES), eval_metric="mlogloss",
                            tree_method="hist", random_state=SEED, n_jobs=2, subsample=0.9,
                            colsample_bytree=0.9, reg_lambda=2.0, min_child_weight=1.0,
                            verbosity=0, **setting,
                        )),
                    ])
                x_train = train_features[columns]
                y_train = train_features["label"].astype(str)
                y_train_encoded = pd.Categorical(y_train, categories=CLASSES, ordered=True).codes
                y_val = val_features["label"].astype(str).tolist()
                estimator.fit(x_train, y_train if model_name == "svm" else y_train_encoded)
                predicted = estimator.predict(val_features[columns])
                if model_name == "xgboost":
                    predicted = [CLASSES[int(label)] for label in predicted]
                else:
                    predicted = [str(label) for label in predicted]
                metrics = _evaluate(y_val, predicted)
                record = {
                    "model": model_name,
                    "roi_mode": roi_mode,
                    "contrast_mode": contrast_mode,
                    "settings": json.dumps(setting, sort_keys=True),
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "validation_predictions": json.dumps(predicted),
                }
                search_rows.append(record)
                key = (model_name, metrics["macro_f1"], metrics["accuracy"], variant)
                previous = saved_models.get(model_name)
                if previous is None or (key[1], key[2], key[3]) > (previous["key"][1], previous["key"][2], previous["key"][3]):
                    saved_models[model_name] = {
                        "key": key,
                        "estimator": estimator,
                        "setting": setting,
                        "variant": variant,
                        "metrics": metrics,
                        "predictions": predicted,
                        "feature_columns": columns,
                        "reference": reference,
                        "train_features": train_features,
                        "val_features": val_features,
                    }

    search = pd.DataFrame(search_rows).sort_values(["model", "macro_f1", "accuracy"], ascending=[True, False, False])
    search.to_csv(output_dir / "model_search_validation.csv", index=False)
    detector_rows = candidate_features.get("detector_rows", [])
    pd.DataFrame(detector_rows).to_csv(output_dir / "feature_detection_summary.csv", index=False)
    predictions = []
    model_summary = {}
    for model_name, chosen in saved_models.items():
        variant = chosen["variant"]
        val_features = chosen["val_features"]
        predictions.extend([
            {
                "model": model_name,
                "filename": row["filename"],
                "source_filename": row["source_filename"],
                "bottle_id": row["bottle_id"],
                "view_id": row["view_id"],
                "qc_note": row["qc_note"],
                "true_label": row["label"],
                "predicted_label": prediction,
                "correct": row["label"] == prediction,
                "roi_mode": val_features.iloc[index]["roi_mode"],
                "contrast_mode": val_features.iloc[index]["contrast_mode"],
                "cap_found": val_features.iloc[index]["cap_found"],
                "axis_found": val_features.iloc[index]["axis_found"],
                "label_found": val_features.iloc[index]["label_found"],
                "label_angle_deg": val_features.iloc[index]["label_angle_deg"],
                "profile_valid_fraction": val_features.iloc[index]["profile_valid_fraction"],
                "profile_dev_mean_abs": val_features.iloc[index]["profile_dev_mean_abs"],
                "profile_right_upper_inward_mean": val_features.iloc[index]["profile_right_upper_inward_mean"],
                "profile_right_upper_inward_max": val_features.iloc[index]["profile_right_upper_inward_max"],
            }
            for index, (row, prediction) in enumerate(zip(val.to_dict("records"), chosen["predictions"]))
        ])
        _plot_confusion(chosen["metrics"], model_name, output_dir)
        joblib.dump(chosen["estimator"], output_dir / f"{model_name}_pipeline.joblib")
        model_summary[model_name] = {
            "selected_variant": variant,
            "selected_parameters": chosen["setting"],
            "features": chosen["feature_columns"],
            "train_normal_profile_reference": chosen["reference"],
            "validation": chosen["metrics"],
        }
    prediction_frame = pd.DataFrame(predictions)
    prediction_frame.to_csv(output_dir / "validation_predictions.csv", index=False)
    prediction_frame[prediction_frame["correct"] == False].to_csv(output_dir / "validation_error_analysis.csv", index=False)  # noqa: E712
    (output_dir / "selected_config.json").write_text(json.dumps({
        "dataset": identity,
        "environment": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "pandas", "Pillow", "scikit-learn", "xgboost", "joblib", "matplotlib")},
            "opencv": cv2.__version__,
            "experiment_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "seed": SEED,
        "classes_in_order": CLASSES,
        "selection_metric": "validation_macro_f1",
        "test_evaluated": False,
        "models": model_summary,
        "limitations": [
            "Dataset v2 replaces penyok01 and penyok03 after initial evaluation; changes in scores cannot be attributed solely to model quality or generalization." if identity["version"] == "v2" else "Original subtle-dent photos are used in dataset v1.",
            "Only four independent bottles are represented in the primary dataset.",
            "Validation contains five primary photos from bottle 03 only.",
            "Normal photos usually show the back of the label while tilted-label photos show the front; this may be a shortcut.",
            "Background, reflections, droplets, and orientation vary between source photos.",
            "Photometric augmentations are descendants of ten training photos and are not independent bottles or scenes.",
        ],
    }, indent=2))

    rows = []
    for model_name, chosen in saved_models.items():
        for label, values in chosen["metrics"]["per_class"].items():
            rows.append({"model": model_name, "class": label, **values, "macro_f1": chosen["metrics"]["macro_f1"], "accuracy": chosen["metrics"]["accuracy"]})
    pd.DataFrame(rows).to_csv(output_dir / "validation_metrics_by_class.csv", index=False)
    summary_rows = []
    for model_name, chosen in saved_models.items():
        summary_rows.append({
            "model": model_name,
            "selected_variant": chosen["variant"],
            "selected_parameters": json.dumps(chosen["setting"], sort_keys=True),
            "validation_macro_f1": chosen["metrics"]["macro_f1"],
            "validation_accuracy": chosen["metrics"]["accuracy"],
            "correct_validation_photos": int(sum(a == b for a, b in zip(val["label"].astype(str).tolist(), chosen["predictions"]))),
            "validation_photo_count": len(val),
        })
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_dir / "validation_metrics_summary.csv", index=False)
    summary_lines = [
        "# Progress 2 classical ML — validation summary",
        "",
        f"Dataset version: {identity['version']}. Features and search grid are unchanged from the initial experiment.",
        "",
        "Configuration selection used validation macro-F1. The test split has not been evaluated by this run.",
        "",
        "| Model | Pipeline | Parameters | Validation macro-F1 | Accuracy | Correct |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        summary_lines.append(
            f"| {row['model']} | {row['selected_variant']} | `{row['selected_parameters']}` | "
            f"{row['validation_macro_f1']:.3f} | {row['validation_accuracy']:.3f} | "
            f"{row['correct_validation_photos']}/{row['validation_photo_count']} |"
        )
    summary_lines += [
        "",
        "Validation contains five original photos from bottle 03. Each class except normal has one validation photo, so one error changes that class recall by 1.0.",
        "The primary dataset has 20 original photos from four physical bottles. The 290 train augmentations are photometric descendants and do not add independent observations.",
        "Version v2 uses retaken penyok01 (train) and penyok03 (validation) with more visible defects. The original subtle-dent photos and their old descendants are excluded. The retake followed initial evaluation, so score changes do not establish improved generalization or subtle-dent sensitivity." if identity["version"] == "v2" else "The subtle dents on bottles 01 and 03 remain `penyok`; the confirmed local defect is on the upper-right body.",
        "Side-specific upper-right features are retained from the initial experiment alongside whole-body profile features; a local feature is not proof of defect detection.",
        "Background varies by bottle; normal images usually show the label back and tilted-label images usually show the front. Results may reflect acquisition shortcuts.",
        "",
        "Inspect `pcd_roi_comparison_train_originals.png` and `feature_detection_summary.csv`; threshold and Canny did not yield a clean bottle mask on these transparent, reflective photos.",
    ]
    (output_dir / "experiment_summary.md").write_text("\n".join(summary_lines) + "\n")
    return {"search": search, "predictions": prediction_frame, "selected": model_summary}


def evaluate_test(root: str | Path, output_dir: str | Path) -> pd.DataFrame:
    """Evaluate previously locked configurations; call only after selection is final."""
    root, output_dir = Path(root), Path(output_dir)
    config_path = output_dir / "selected_config.json"
    config = json.loads(config_path.read_text())
    if config.get("dataset") != dataset_identity(root):
        raise ValueError("Dataset does not match the locked model input fingerprints.")
    if not config.get("models"):
        raise ValueError("No locked model configurations are available")
    test = read_csv(root, "test.csv")
    results = []
    all_predictions = []
    for model_name, model_config in config["models"].items():
        roi_mode = model_config["selected_variant"].split("__")[0]
        contrast_mode = model_config["selected_variant"].split("__")[1]
        test_features = _load_features(test, root, roi_mode, contrast_mode)
        reference = model_config["train_normal_profile_reference"]
        _apply_profile_reference(test_features, reference)
        columns = model_config["features"]
        estimator = joblib.load(output_dir / f"{model_name}_pipeline.joblib")
        predicted = estimator.predict(test_features[columns])
        if model_name == "xgboost":
            predicted = [CLASSES[int(label)] for label in predicted]
        else:
            predicted = [str(label) for label in predicted]
        metrics = _evaluate(test["label"].astype(str).tolist(), predicted)
        results.append({"model": model_name, **{k: metrics[k] for k in ("macro_f1", "accuracy")}})
        for row, prediction in zip(test.to_dict("records"), predicted):
            all_predictions.append({
                "model": model_name, "filename": row["filename"], "bottle_id": row["bottle_id"],
                "true_label": row["label"], "predicted_label": prediction,
                "correct": row["label"] == prediction,
                "cap_found": test_features.iloc[len(all_predictions) % len(test)]["cap_found"],
                "axis_found": test_features.iloc[len(all_predictions) % len(test)]["axis_found"],
                "label_found": test_features.iloc[len(all_predictions) % len(test)]["label_found"],
            })
        _plot_confusion(metrics, f"{model_name}_test", output_dir)
        (output_dir / f"test_metrics_{model_name}.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(all_predictions).to_csv(output_dir / "test_predictions.csv", index=False)
    summary = pd.DataFrame(results)
    summary.to_csv(output_dir / "test_metrics_summary.csv", index=False)
    config["test_evaluated"] = True
    config["test_metrics"] = results
    config_path.write_text(json.dumps(config, indent=2))
    return summary


def run_all(root: str | Path, output_dir: str | Path) -> dict:
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(root)
    eda = run_eda(root, output_dir, audit_summary=audit)
    pcd = render_pcd_comparison(root, output_dir)
    models = select_models_and_save(root, output_dir)
    render_feature_overlays(root, output_dir)
    return {"eda": eda, "pcd": pcd, "selected": models["selected"]}


def find_dataset_root() -> Path:
    env_root = os.environ.get("AQUVIVA_DATASET_ROOT")
    if env_root:
        root = Path(env_root)
        if not (root / "train.csv").is_file():
            raise FileNotFoundError(f"AQUVIVA_DATASET_ROOT has no train.csv: {root}")
        return root.resolve()
    candidates = [
        Path("../comvis_dataset_v2/dataset_prep"),
        Path("comvis_dataset_v2/dataset_prep"),
        Path("../comvis_dataset_ready/dataset_prep"),
        Path("comvis_dataset_ready/dataset_prep"),
        Path("dataset_prep"),
    ]
    candidates += sorted(Path("/kaggle/input").glob("**/dataset_prep")) if Path("/kaggle/input").exists() else []
    for candidate in candidates:
        if (candidate / "train.csv").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Set AQUVIVA_DATASET_ROOT or attach the dataset containing dataset_prep/train.csv")


if __name__ == "__main__":
    dataset_root = find_dataset_root()
    dataset_version = dataset_identity(dataset_root)["version"]
    result_dir = Path(os.environ.get("AQUVIVA_OUTPUT_DIR", f"progress2_results_{dataset_version}"))
    outputs = run_all(dataset_root, result_dir)
    print(json.dumps(outputs, indent=2))
