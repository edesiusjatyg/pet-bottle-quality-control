# Hasil Progress 3 — dataset v3

Run aktual menggunakan device `cpu`, PyTorch/Torchvision untuk DL dan scikit-learn/XGBoost untuk ML. Detail lingkungan tersedia pada environment.json.

## Data dan metode

102 foto primer: train 61, validation 20, test 21. Random split berstrata per foto, seed 2026. Tidak dilakukan split ulang.
ML memakai 63 fitur eksplisit dari 61 foto train asli. DL memakai 244 view dari sumber yang sama (asli + 3 variasi photometric per sumber).
DL adalah transfer learning backbone frozen; yang dilatih hanya head dropout + linear. Seluruh backbone ImageNet tidak di-fine-tune.

## Tuning dan konfigurasi terpilih

- 162 kandidat ML: 3 PCD × 3 nilai d × (10 SVM + 8 XGBoost).
- PCD bersama terpilih: `center_gray`; GLCM **d=4** piksel pada ROI sisi maksimal 768 px. Arah 0°/90°, 16 level.
- SVM: `{"C": 10, "gamma": "scale", "kernel": "rbf"}`.
- XGBoost: `{"learning_rate": 0.1, "max_depth": 2, "n_estimators": 160}`.
- 16 trial DL: delapan konfigurasi berpasangan Raw/PCD.
- Parameter DL bersama: `{"dropout": 0.2, "lr": 0.003, "weight_decay": 0.0}`; Adam, batch 32, maksimal 20 epoch, patience 5.
- Epoch checkpoint terbaik: Raw 18, PCD 18.

## Skor aktual

| Model | Split | n | Macro-F1 | Accuracy |
|---|---|---:|---:|---:|
| svm | train | 61 | 1.0000 | 100.00% |
| svm | val | 20 | 0.5528 | 55.00% |
| svm | test | 21 | 0.3794 | 38.10% |
| xgboost | train | 61 | 1.0000 | 100.00% |
| xgboost | val | 20 | 0.6042 | 60.00% |
| xgboost | test | 21 | 0.3873 | 38.10% |
| dl_raw | train | 61 | 1.0000 | 100.00% |
| dl_raw | val | 20 | 0.9495 | 95.00% |
| dl_raw | test | 21 | 0.7175 | 76.19% |
| dl_pcd | train | 61 | 1.0000 | 100.00% |
| dl_pcd | val | 20 | 0.9495 | 95.00% |
| dl_pcd | test | 21 | 0.7262 | 71.43% |

## Pengaruh PCD pada DL

- Δ macro-F1 PCD − Raw pada val: **+0.0000**.
- Δ macro-F1 PCD − Raw pada test: **+0.0087**.

Perbandingan tersebut memakai split, backbone, bobot awal, hyperparameter, dan kebijakan augmentasi yang sama. Efeknya adalah paket PCD yang dipilih, bukan CLAHE saja.

## Perbaikan implementasi

- Fitur profil yang seluruhnya gagal tetap NaN. Denominator proporsi deviasi hanya menghitung titik valid.
- Target numerik dan metadata tidak masuk fitur; daftar fitur eksplisit mencegah kebocoran label.
- ASM, energy, dan homogeneity mengikuti rumus yang dibedakan secara benar. Jarak d dapat dituning dan disimpan.
- Referensi normal, imputer, scaler, dan model hanya di-fit pada train. Tuning memakai validation.
- Seluruh kandidat ROI adaptif dilaporkan walaupun tidak dipilih; ada fallback untuk deteksi yang gagal.

## Verifikasi

Checkpoint reload check: lulus; prediksi 20 foto validation direproduksi untuk keempat model.
Test dievaluasi setelah konfigurasi serta checksum dikunci. Receipt dan selection_lock disertakan. Tidak ada tuning ulang berdasarkan hasil test.

## Batas kesimpulan

- Split per foto tidak menjamin botol fisik berbeda antarbagian. Skor tidak membuktikan generalisasi ke botol baru.
- Validation hanya 20 foto dan dipakai untuk banyak kandidat; test hanya 21 foto. Satu foto mengubah accuracy test sekitar 4,76 poin persentase.
- Skor v2 tidak dapat dibandingkan langsung untuk menyimpulkan efek bug fix, karena dataset dan split berubah.
- ROI, sumbu, dan label masih heuristik; tidak tersedia anotasi segmentasi untuk mengukur akurasi kontur.
- Backbone frozen memperkecil biaya komputasi. Paket belum membuktikan hasil full-backbone fine-tuning.

Semua tabel, confusion matrix, prediksi per foto, lineage augmentasi, dan log trial tersedia di folder results. Petunjuk retraining GPU ada di README.md.

## Diagnosis representasi fitur

Linear SVM juga dilatih pada 960 fitur MobileNet frozen dari 61 foto train asli, tanpa memasukkan turunan augmentasi. Pemeriksaan ini hanya menggunakan validation dan tidak mengubah empat model utama.

- SVM pada embedding raw: macro-F1 validation 0.9000, accuracy 90.00%.
- SVM pada embedding pcd: macro-F1 validation 0.8540, accuracy 85.00%.

Jika SVM dengan embedding ikut tinggi, hasil ini mendukung bahwa representasi visual lebih informatif pada split ini daripada fitur manual. Ini bukan bukti bahwa dimensi tinggi sendiri menyebabkan kesulitan, atau bahwa botol independen akan memperoleh skor serupa.

## Feature importance dan Grad-CAM

Analisis post-hoc menggunakan 20 foto validation, 20 pengacakan per fitur, tanpa refit atau penggunaan test.

- svm: kelompok dengan penurunan macro-F1 rata-rata terbesar adalah `bentuk_dan_deviasi_normal` (+0.2903).
- xgboost: kelompok dengan penurunan macro-F1 rata-rata terbesar adalah `tekstur_dan_intensitas` (+0.2400).

Importance individual dan kelompok tersimpan sebagai CSV/grafik. DL dilengkapi ranking permutation importance 960 channel visual serta Grad-CAM pada contoh validation per kelas dan kesalahan prediksi. Overlay menampilkan ruang input masing-masing jalur, bukan mask cacat ground truth.
Ranking bergantung pada korelasi fitur, model, dan hanya 20 foto validation. Error bar adalah variasi pengacakan, bukan confidence interval. Peta Grad-CAM 7×7 bersifat kasar dan tidak membuktikan sebab-akibat. Lihat importance/README.md.

Catatan inspeksi visual khusus run ini tersedia pada `importance/VISUAL_REVIEW.md`.

## Stratified 5-Fold tambahan

Analisis tambahan memakai 61 foto train saja, konfigurasi tetap, preprocessing dipelajari ulang per fold, dan augmentasi mengikuti sumbernya. Validation 20 serta test 21 tidak dimasukkan. DL memakai 18 epoch tetap tanpa pemilihan checkpoint pada fold validasi. Ini analisis kestabilan bersyarat, bukan nested CV.

| Model | Macro-F1 mean | Std antar-fold |
|---|---:|---:|
| svm | 0.3307 | 0.1910 |
| xgboost | 0.4217 | 0.1921 |
| dl_raw | 0.7208 | 0.0982 |
| dl_pcd | 0.7221 | 0.0939 |

Hasil utama dan checkpoint test tidak diubah. Detail ada pada `cross_validation/README.md`, `fold_metrics.csv`, dan `out_of_fold_predictions.csv`.
