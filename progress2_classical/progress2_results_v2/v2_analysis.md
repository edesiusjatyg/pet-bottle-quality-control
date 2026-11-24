# Analisis eksperimen awal Progress 2 — dataset v2

## Data dan pelaksanaan

Dataset v2 tetap memuat 20 foto primer dari empat botol fisik, dengan split train 01–02, validation 03, test 04. Train berisi 10 foto asli dan 290 turunan offline, masing-masing kelas 75. Tidak ada augmentasi baru atau random split selama eksperimen.

Audit terhadap v1 menemukan tepat dua foto primer berubah: penyok01 (train) dan penyok03 (validation). Sebanyak 37 turunan penyok01 juga diperbarui, sedangkan parameter augmentasi dan seluruh file test tetap sama. Subagent mereproduksi 37 turunan itu di memori dan mendapatkan byte JPEG identik dengan file v2, menggunakan Pillow 11.1.0. Lihat audit_v1_v2.json.

Fitur, empat kandidat ROI/contrast, grid 20 konfigurasi SVM dan 16 konfigurasi XGBoost, seed 42, dan aturan seleksi tetap seperti eksperimen awal. Referensi normal hanya memakai empat foto normal train asli. Imputer dan scaler dipelajari dari train. Pada 266 gambar train/validation yang tidak berubah, seluruh 129 kolom numerik hasil ekstraksi cocok dengan v1 untuk keempat varian; lihat feature_reproducibility_v1_v2.json. Skor dihitung dari lima foto validation asli, bukan sampel augmentasi.

## Hasil validation

| Model | Macro-F1 v1 | Macro-F1 v2 | Accuracy v2 | Benar v2 |
|---|---:|---:|---:|---:|
| SVM | 0,625 | 0,375 | 0,400 | 2/5 |
| XGBoost | 0,625 | 0,350 | 0,400 | 2/5 |

Kedua model memilih center_roi + grayscale. SVM memilih kernel linear, C=0.1 dan class_weight=balanced. XGBoost memilih 80 trees, max_depth=2 dan learning_rate=0.05. Pada nilai macro-F1 dan accuracy yang sama, pemilihan mengikuti aturan deterministik pipeline yang sudah ada; data validation sekecil ini tidak cukup untuk menyatakan satu konfigurasi unggul secara stabil.

Urutan kelas confusion matrix: normal, label_miring, label_hilang, penyok. Precision, recall, F1 dan support tiap kelas tersimpan di validation_metrics_by_class.csv; matriks lengkap ada di selected_config.json dan gambar validation_confusion_*.png.

Kedua model benar pada labelhilang03 dan labelmiring03. Keduanya menganggap normal03 dan normal03_light sebagai label_miring. SVM menganggap penyok03 sebagai normal, sementara XGBoost menganggapnya label_miring. Recall normal dan penyok sama-sama 0 pada validation v2. Prediksi benar untuk dua kelas lainnya hanya didukung satu foto per kelas.

## Diagnosis dan batas kesimpulan

Pada foto train retake penyok01, deformasi terlihat besar dan melibatkan beberapa bagian badan. Keterangan lokal kanan atas berlaku untuk foto subtle v1, sehingga fitur kanan atas dipertahankan sebagai bagian rancangan awal tanpa mengasumsikan lokasi cacat v2 sama.

Otsu dan Canny + closing masih menangkap tekstur background, label, dan pantulan selain batas botol. CLAHE meningkatkan detail sekaligus gangguan. Keduanya belum menjadi mask siluet yang tervalidasi. Fitur profil memakai pasangan tepi pada rentang posisi yang tetap; pemeriksaan validitas baris menguji kekuatan tepi dan rentang lebar, bukan kecocokan dengan siluet ground truth. Karena itu profile_valid_fraction tinggi tidak berarti konturnya benar. Periksa feature_overlay_train_center_roi__gray.png dan feature_overlay_validation_center_roi__gray.png untuk posisi titik pengukuran aktual.

Pada varian terpilih, estimasi sumbu badan tersedia pada 6/10 foto train asli dan 3/5 validation. Kegagalan sumbu ditandai axis_found=0 dan ROI tetap tanpa rotasi; label tidak diluruskan secara terpisah. Detektor menemukan region label pada 8/10 train dan 4/5 validation, konsisten dengan jumlah foto berlabel, tetapi itu tidak membuktikan segmentasi label tepat. Overlay train memperlihatkan contoh kegagalan pada normal01_light: kotak label justru memilih pantulan di bawah label. Sejumlah pasangan titik tepi juga mengenai bagian dalam botol, bukan batas siluet. Sudut label tersedia hanya 1/10 train dan 2/5 validation. Nilai yang gagal diukur tetap NaN sebelum imputer train.

Deviasi lebar profil rata-rata pada penyok03 adalah 0,0893; dua normal validation masing-masing 0,1289 dan 0,0947. Ukuran ini tidak memisahkan penyok dari normal pada validation. Penyok03 juga menghasilkan sudut region label sekitar -8,48 derajat. Deformasi kemasan dapat memengaruhi fitur label; angka tersebut bukan anotasi sudut label ground truth. Penyebab tiap keputusan model belum dapat ditentukan secara kausal dari skor atau fitur ini saja.

Penurunan skor setelah retake menunjukkan bahwa visibilitas cacat bagi manusia belum cukup membuat baseline fitur ini mengenalinya. Hasil v1 dan v2 berasal dari kondisi data berbeda, termasuk penggantian satu foto validation setelah evaluasi awal, sehingga selisih skor bukan ukuran peningkatan/penurunan generalisasi yang terkontrol. Eksperimen ini juga belum membuktikan kemampuan mendeteksi penyok subtle.

Background berbeda antar-ID, normal biasanya memperlihatkan belakang label sedangkan label_miring memperlihatkan depan. Variasi tetesan air, pantulan, orientasi dan pencahayaan tetap ada. Empat botol dan satu botol validation tidak mendukung klaim generalisasi kuat.

## Artefak dan langkah berikutnya

Notebook progress2_classical_ml_v2_executed.ipynb memuat hasil eksekusi lokal. Notebook progress2_classical_ml.ipynb merupakan versi bersih untuk dijalankan ulang di Kaggle CPU; belum dijalankan di Kaggle. Model tersimpan sebagai svm_pipeline.joblib dan xgboost_pipeline.joblib. Konfigurasi menyimpan versi dependency, checksum kode, fingerprint CSV, dan referensi normal train.

Test belum dievaluasi. Sel RUN_FINAL_TEST=False tetap nonaktif. Evaluasi final terpisah menggunakan model dan fitur yang terkunci, serta memeriksa kecocokan fingerprint dataset. Jangan memakai hasil test untuk mengubah konfigurasi.

Perbaikan representasi berikutnya perlu berangkat dari overlay foto train: pasangan tepi yang mengikuti kontur badan, penanganan ketidakpastian sumbu badan, serta pemeriksaan label yang terdeformasi. Tambahan foto asli dari botol baru dengan sisi label dan background yang lebih konsisten tetap diperlukan. Eksperimen deep learning belum dijalankan pada tahap ini.
