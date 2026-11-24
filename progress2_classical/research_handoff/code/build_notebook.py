import json
from pathlib import Path

def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.splitlines(keepends=True),
    }

cells = [
    markdown("""# Progress 2 v2 — PCD + handcrafted features + ML klasik

Eksperimen awal klasifikasi empat kelas: `normal`, `label_miring`, `label_hilang`, dan `penyok`.

Notebook ini membandingkan ROI tengah tetap dan ROI berjangkar tutup, grayscale dan CLAHE, lalu mengekstrak profil siluet, fitur label, tepi, dan tekstur untuk baseline SVM dan XGBoost.

**Split tetap:** botol 01–02 train, 03 validation, 04 test. Tidak ada random split dan tidak ada augmentasi ulang. Hanya validation yang dipakai memilih konfigurasi. Sel evaluasi test terpisah dan tidak aktif secara default.

Hasil masih preliminary: 20 foto primer dari empat botol; validation/test masing-masing lima foto dari satu botol. Dataset v2 mengganti penyok01 (train) dan penyok03 (validation) dengan foto retake yang cacatnya lebih terlihat; turunannya diperbarui. Retake dilakukan setelah evaluasi awal, sehingga perbandingan v1/v2 bukan bukti peningkatan generalisasi atau kemampuan mendeteksi penyok subtle. Fitur dan grid konfigurasi tetap sama dengan eksperimen awal.
"""),
    code("""# Jika dependency belum tersedia, install dari requirements.txt di runtime Kaggle.
# Biasanya Kaggle sudah menyediakan OpenCV, scikit-learn, dan XGBoost.
# %pip install -q -r /kaggle/input/progress2-classical/requirements.txt

import json
import os
import sys
from pathlib import Path
from IPython.display import display, Image as DisplayImage

code_candidates = [Path.cwd(), Path.cwd() / "progress2_classical"]
if Path("/kaggle/input").exists():
    code_candidates += sorted({path.parent for path in Path("/kaggle/input").glob("**/experiment.py")})
CODE_DIR = next((path.resolve() for path in code_candidates if (path / "experiment.py").is_file()), None)
if CODE_DIR is None:
    raise FileNotFoundError("Letakkan folder progress2_classical (berisi experiment.py) di samping notebook atau sebagai input Kaggle Dataset.")
sys.path.insert(0, str(CODE_DIR))

from experiment import (
    audit_dataset, dataset_identity, evaluate_test, find_dataset_root, read_csv,
    render_feature_overlays, render_pcd_comparison, run_eda, select_models_and_save,
)

# Jika input Kaggle memuat lebih dari satu versi, tentukan folder v2 secara eksplisit:
# os.environ["AQUVIVA_DATASET_ROOT"] = "/kaggle/input/NAMA-DATASET/dataset_prep"
DATASET_ROOT = find_dataset_root()
DATASET_VERSION = dataset_identity(DATASET_ROOT)["version"]
if DATASET_VERSION != "v2":
    raise ValueError("Notebook ini untuk dataset v2. Arahkan AQUVIVA_DATASET_ROOT ke folder v2.")
output_base = Path("/kaggle/working") if Path("/kaggle/working").exists() else CODE_DIR
OUTPUT_DIR = Path(os.environ.get("AQUVIVA_OUTPUT_DIR", str(output_base / "progress2_results_v2")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Dataset:", DATASET_ROOT)
print("Outputs:", OUTPUT_DIR)
"""),
    markdown("## 1. Audit split, CSV, dan asal augmentasi\n\nAudit ini membaca metadata test hanya untuk memeriksa split dan checksum. Foto test tidak dipakai untuk memilih PCD atau model."),
    code("""audit = audit_dataset(DATASET_ROOT)
print(json.dumps(audit, indent=2))
"""),
    markdown("## 2. EDA: foto primer dan turunan augmentasi\n\nGrafik pertama menghitung 20 foto primer per split. Grafik kedua memisahkan 10 foto train asli dari 290 turunan augmentasi. Grid foto visual hanya menampilkan 10 foto train asli agar pemeriksaan visual tidak membuka holdout test sebelum konfigurasi dikunci."),
    code("""eda_summary = run_eda(DATASET_ROOT, OUTPUT_DIR, audit_summary=audit)
print(json.dumps(eda_summary, indent=2))
for filename in ["eda_class_counts.png", "eda_original_photos.png", "eda_augmentation_parameters.png"]:
    display(DisplayImage(filename=str(OUTPUT_DIR / filename)))
display(read_csv(DATASET_ROOT, "metadata_original.csv").groupby(["split", "label"]).size().rename("primary_photos").to_frame())
display(read_csv(DATASET_ROOT, "train.csv").groupby(["label", "is_augmented"]).size().rename("train_rows").to_frame())
"""),
    markdown("## 3. Bandingkan PCD dan ROI dengan foto train\n\nPanel membandingkan foto mentah, ROI yang dipusatkan pada tutup dan sumbu badan, Otsu threshold, Canny, Canny + closing 3×3, dan CLAHE. Threshold dan edge maps adalah kandidat visual, bukan mask botol yang sudah tervalidasi."),
    code("""pcd_summary = render_pcd_comparison(DATASET_ROOT, OUTPUT_DIR)
print(json.dumps(pcd_summary, indent=2))
display(DisplayImage(filename=str(OUTPUT_DIR / "pcd_roi_comparison_train_originals.png")))
display(__import__("pandas").read_csv(OUTPUT_DIR / "pcd_roi_diagnostics_train_originals.csv"))
"""),
    markdown("## 4. Ekstraksi fitur dan pemilihan baseline\n\nFitur yang tidak berhasil diukur disimpan sebagai nilai kosong dan diimputasi hanya dari train. Referensi profil normal dibentuk dari foto train asli kelas normal. Fitur lokal kanan atas berasal dari rancangan untuk foto subtle versi awal dan tetap dipakai bersama profil badan keseluruhan pada retake v2; lokasi cacat baru tidak diasumsikan hanya di kanan atas. SVM memakai imputer + scaler yang di-fit pada train; XGBoost memakai imputer train. Grid kecil dipilih berdasarkan macro-F1 validation."),
    code("""model_results = select_models_and_save(DATASET_ROOT, OUTPUT_DIR)
render_feature_overlays(DATASET_ROOT, OUTPUT_DIR)
search = model_results["search"]
display(search.groupby("model", group_keys=False).head(6).reset_index(drop=True))
display(model_results["predictions"])
display(__import__("pandas").read_csv(OUTPUT_DIR / "validation_metrics_by_class.csv"))
selected_variants = sorted({item["selected_variant"] for item in model_results["selected"].values()})
for variant in selected_variants:
    display(DisplayImage(filename=str(OUTPUT_DIR / f"silhouette_profile_deviation_{variant}.png")))
    for role in ("train", "validation"):
        display(DisplayImage(filename=str(OUTPUT_DIR / f"feature_overlay_{role}_{variant}.png")))
for filename in ["validation_confusion_svm.png", "validation_confusion_xgboost.png"]:
    display(DisplayImage(filename=str(OUTPUT_DIR / filename)))
display(__import__("pandas").read_csv(OUTPUT_DIR / "feature_detection_summary.csv"))
print((OUTPUT_DIR / "experiment_summary.md").read_text())
"""),
    markdown("## 5. Konfigurasi terkunci dan kesalahan validation\n\n`selected_config.json` menyimpan varian ROI/contrast, parameter tiap model, referensi normal train, dan metrik validation. Tinjau setiap prediksi validation yang salah bersama status deteksi ROI/label dan catatan akuisisi."),
    code("""selected = json.loads((OUTPUT_DIR / "selected_config.json").read_text())
print(json.dumps(selected, indent=2))
errors = __import__("pandas").read_csv(OUTPUT_DIR / "validation_error_analysis.csv")
display(errors)
"""),
    markdown("""## 6. Evaluasi test terpisah — jalankan setelah konfigurasi dikunci

Sel ini sengaja nonaktif. Jangan ubah konfigurasi setelah membaca hasil test. Untuk evaluasi final satu kali, ubah `RUN_FINAL_TEST` menjadi `True` dan jalankan sel ini setelah pemilihan validation selesai.
"""),
    code("""RUN_FINAL_TEST = False
if RUN_FINAL_TEST:
    test_summary = evaluate_test(DATASET_ROOT, OUTPUT_DIR)
    display(test_summary)
else:
    print("Test belum dibaca untuk prediksi. Konfigurasi masih menggunakan hasil validation saja.")
"""),
    markdown("""## Catatan interpretasi

- Independensi primer adalah empat botol dan 20 foto. Augmentasi brightness, contrast, dan blur hanya menghasilkan turunan dari 10 foto train.
- Validation dan test masing-masing hanya berisi lima foto dari satu botol; satu kesalahan mengubah recall kelas sebesar satu penuh pada kelas yang hanya punya satu foto.
- Background berubah antar-ID. Normal umumnya memperlihatkan belakang label, sedangkan label miring memperlihatkan depan label. Model dapat memanfaatkan shortcut ini.
- Foto memuat variasi pantulan, tetesan air, orientasi, dan pencahayaan. Skor tinggi tidak membuktikan model memahami cacat.
- Periksa file `pcd_roi_comparison_train_originals.png`, `pcd_roi_diagnostics_train_originals.csv`, dan kolom `label_found` / `profile_valid_fraction` pada tabel fitur sebelum memakai fitur sebagai bukti deteksi fisik.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for index, cell in enumerate(cells, start=1):
    cell["id"] = f"progress2-{index:02d}"
target = Path(__file__).with_name("progress2_classical_ml.ipynb")
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
print(target)
