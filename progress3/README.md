# Progress 3 — Aquviva bottle quality control

Kode lengkap untuk dataset v3: perbaikan fitur ML, tuning SVM/XGBoost, serta Raw + DL dan PCD + DL menggunakan MobileNetV3-Large. Folder `results` berisi hasil eksekusi aktual dan checkpoint terpilih. Dataset dan pembagiannya tidak diacak ulang oleh kode ini.

## Jalankan ulang di komputer GPU

1. Ekstrak paket lengkap. Buka terminal pada folder `progress3`.
2. Gunakan Python 3.10–3.12; eksekusi yang disertakan memakai Python 3.12.
3. Buat virtual environment jika diperlukan. Pasang PyTorch yang sesuai GPU menggunakan pemilih instalasi resmi: https://pytorch.org/get-started/locally/ . Untuk NVIDIA gunakan build CUDA, untuk AMD ikuti dukungan ROCm, dan untuk Mac gunakan MPS bila tersedia. Jangan menyalin environment CPU dari komputer lain. requirements.txt mengizinkan build PyTorch yang lebih baru bila diperlukan GPU; requirements-tested.txt mencatat versi run CPU yang disertakan.
4. Instal dependensi proyek:

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print('CUDA/ROCm:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

Paket Drive menyertakan dataset siap pakai di `data/comvis_dataset_v3`. Bila memakai source dari GitHub, ekstrak `comvis_dataset_v3_ready.zip` ke direktori tersebut atau ganti nilai `--data`.

```bash
python run.py all --data data/comvis_dataset_v3 --device auto --output runs/retrain_gpu --epochs 20 --patience 5 --batch-size 32 --augmentation-copies 3 --glcm-distances 1 2 4
```

Satu perintah ini menjalankan audit, tuning ML, pemilihan PCD, ekstraksi fitur backbone DL, tuning head DL, penyimpanan model, dan laporan validation. Default `auto` memilih CUDA/ROCm bila tersedia, kemudian MPS, lalu CPU. Gunakan `--device cuda:0` untuk meminta GPU tertentu; kode akan berhenti dengan pesan jelas jika CUDA tidak tersedia. Jika GPU kehabisan memori, kecilkan `--batch-size` menjadi 16 atau 8 dan gunakan output baru.

Hasil baru selalu diarahkan ke direktori output baru. Jangan memakai ulang folder `results` bawaan untuk eksperimen baru. Kode menolak menimpa eksperimen ML yang sudah memiliki konfigurasi terpilih.

Setelah training, alur utama juga memverifikasi checkpoint yang dimuat ulang dan menjalankan diagnosis tambahan SVM pada embedding MobileNet. Diagnosis ini memakai validation dan tidak mengubah empat model utama. Laporan Markdown dibuat otomatis dari hasil aktual.

Untuk menjalankan bagian tertentu:

```bash
python run.py ml --data data/comvis_dataset_v3 --output runs/eksperimen_baru --glcm-distances 1 2 3 4
python run.py dl --data data/comvis_dataset_v3 --output runs/eksperimen_baru --device auto --epochs 30 --patience 7 --batch-size 32 --augmentation-copies 3
python run.py report --data data/comvis_dataset_v3 --output runs/eksperimen_baru
```

Tahap DL membaca PCD yang terpilih dari tahap ML pada direktori output yang sama. `Progress3_GPU_Retraining.ipynb` menyediakan alur yang sama dalam notebook. Notebook tidak memerlukan modifikasi isi modul Python untuk mulai retraining.

## Checkpoint dan kebutuhan internet

`results/models/backbone_imagenet_v2.pt` berisi backbone pretrained. Kode otomatis memakai checkpoint bawaan saat retraining sehingga bobot tidak perlu diunduh lagi. Untuk lokasi lain:

```bash
python run.py all --data data/comvis_dataset_v3 --output runs/run2 --device auto --backbone results/models/backbone_imagenet_v2.pt
```

Tanpa checkpoint lokal, Torchvision mengunduh bobot resmi ImageNet pada penggunaan pertama. Pemasangan dependensi tetap memerlukan internet. Checkpoint head disimpan terpisah dari backbone agar paket tidak menyimpan bobot backbone berulang. Untuk memindahkan model, bawa **seluruh** `results/models` beserta dua file `*_selected_config.json`, bukan file head saja.

## Batasan training DL yang dijalankan

Run yang disertakan menggunakan **transfer learning dengan backbone frozen**: MobileNetV3-Large → global average pooling → dropout → linear empat kelas. Yang dilatih adalah classifier head. BatchNorm backbone berada dalam mode evaluasi. Ini bukan fine-tuning seluruh backbone.

Fitur backbone disimpan dalam cache agar delapan kandidat hyperparameter head dapat dilatih tanpa mengulang forward pass gambar yang sama. Cache hanya sah untuk backbone, gambar, parameter augmentasi, dan PCD yang sama. Raw dan PCD memiliki cache terpisah. GPU digunakan saat ekstraksi fitur dan training head bila tersedia. SVM dan XGBoost pada implementasi ini tetap memakai CPU. Menambah epoch pada perintah di atas menambah epoch **head**, bukan membuka backbone secara otomatis.

Normalisasi PyTorch: RGB → float 0–1 → mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`. Ini berbeda dari pengaturan input Keras MobileNetV3 pada rancangan awal. Tidak ada center crop 224 yang membuang bagian botol; gambar diperkecil proporsional lalu diberi padding abu-abu menjadi 224×224.

## Dataset, split, dan kelas

| Bagian | Foto primer |
|---|---:|
| Train | 61 |
| Validation | 20 |
| Test | 21 |

Split v3 merupakan random split berstrata **per foto**, seed 2026, sesuai keputusan pemilik dataset. Semua model memakai pembagian yang sama. Angka `48xx` adalah nomor foto; ID botol fisik tidak diinferensikan. Evaluasi ini tidak menjamin botol fisik yang berbeda antarbagian.

Urutan kelas tunggal untuk semua model adalah `label_hilang`, `label_miring`, `normal`, `penyok` dengan ID 0–3. Jangan memakai urutan ID dari notebook v2 tanpa konversi.

ML dilatih pada 61 foto train asli. DL memakai 61 foto yang sama dan tiga variasi brightness/contrast per foto: total **244 view train**, bukan 244 foto primer atau botol independen. Tidak ada turunan validation/test. Parameter augmentasi Raw dan PCD sama dan tercatat pada `dl_augmentation_lineage.csv`.

## Perbaikan dari kode Progress 2

- Loader memahami `metadata.csv` dan `splits/*.csv` v3; tidak lagi mewajibkan botol 01–04 atau jumlah 300/5/5.
- Kolom fitur memakai daftar eksplisit di `features.py`; `class_id`, nomor foto, ukuran file, serta label tidak pernah menjadi predictor.
- Proporsi penyimpangan profil hanya memakai pengukuran valid. Jika semua gagal, hasilnya `NaN`, bukan 0.
- Referensi profil normal dihitung dari median foto normal **train**. Imputer median dan scaler SVM hanya di-fit pada train. Kolom seluruhnya kosong dipertahankan dan ditandai missing, dengan fallback imputer yang tercatat pada model.
- ASM dan energy dipisahkan; energy adalah akar ASM. Homogeneity memakai denominator `1 + (i-j)^2`. Contrast dinormalisasi dengan 225 untuk 16 level grayscale.
- Ada kandidat ROI adaptif dan pelurusan sumbu PCA dengan fallback. Detektor masih merupakan heuristik, bukan segmentasi ground truth.
- Judul grafik dan jumlah sampel mengikuti split aktual.
- `legacy_reference.py` menyimpan kode v2 sebagai referensi. Pipeline aktif tidak memanggil audit, pencarian model, atau pemilih kolom numerik dari file tersebut; hanya fungsi ROI lama dipakai untuk kandidat pembanding `center_gray`.

## Tuning ML dan parameter GLCM d

`d` adalah jarak pasangan piksel pada **ROI setelah resize sisi maksimal 768 px**, bukan jarak pada foto resolusi penuh. Sudut GLCM tetap 0° dan 90°, level grayscale 16. Histogram disimetriskan dan dinormalisasi.

Default `--glcm-distances 1 2 4` membandingkan d=1,2,4. Ubah menjadi `--glcm-distances 1 2 3 4` atau nilai positif lain untuk eksperimen baru. Nilai terpilih ada pada `ml_selected_config.json` dan digunakan kembali pada prediksi foto baru.

Tiga kandidat PCD: center ROI + grayscale, adaptive ROI + grayscale, dan adaptive ROI + CLAHE. SVM membandingkan empat nilai C linear dan enam kombinasi C/gamma RBF; XGBoost membandingkan delapan kombinasi jumlah pohon, depth, serta learning rate. Dengan tiga kandidat d, total **162 konfigurasi ML** dievaluasi pada validation. Grid dapat diubah di fungsi `grids()` pada `ml.py` sebelum membuat run baru.

Untuk menjaga satu pipeline PCD dan satu vektor fitur yang sama bagi SVM/XGBoost, tiap pasangan PCD/d mendapat konfigurasi SVM terbaik dan XGBoost terbaik. Rata-rata macro-F1 keduanya menentukan satu pasangan PCD/d yang digunakan pada model final. Tie-break: mean accuracy, urutan varian yang sudah ditentukan, d lebih kecil. Pada skor seri dalam model/varian yang sama, kandidat pertama dipertahankan. Semua trial disimpan, termasuk kandidat yang tidak terpilih.

## Tuning DL dan perbandingan PCD

Delapan kandidat berasal dari learning rate `[0.001, 0.003]`, dropout `[0.2, 0.4]`, dan weight decay `[0, 0.001]`. Setiap kandidat dijalankan pada Raw dan PCD, sehingga ada 16 trial head. Grid dapat diubah pada `tuning_grid()` di `deep_learning.py` sebelum run baru.

Dipilih **satu set hyperparameter bersama** yang memaksimalkan rata-rata macro-F1 validation Raw dan PCD; tie-break memakai mean validation loss lalu urutan trial. Ini mencegah perbandingan PCD tercampur oleh pemilihan hyperparameter yang berbeda pada kedua jalur. Backbone awal, seed, batch size, augmentasi, batas epoch, dan aturan early stopping sama. Checkpoint masing-masing jalur dipilih dengan macro-F1 validation, kemudian cross-entropy validation; epoch terbaik boleh berbeda karena aturan pemilihannya sama.

## Feature importance ML dan DL

Perintah `all` otomatis menghitung permutation importance pada validation setelah checkpoint terpilih: per fitur dan per kelompok untuk SVM/XGBoost, serta per channel visual hasil pooling untuk DL. Default pengacakan 20 kali; ubah dengan `--importance-repeats 30` pada run baru.

Grad-CAM dibuat untuk contoh validation per kelas dan semua kesalahan validation DL. Peta menampilkan area input yang memengaruhi kelas prediksi, bukan mask cacat. Channel DL tidak mempunyai arti semantik yang terjamin. Ranking fitur berkorelasi juga perlu dibaca bersama importance kelompok.

```bash
python run.py explain --data data/comvis_dataset_v3 --output runs/retrain_gpu --device auto --importance-repeats 20
```

Tabel, grafik, heatmap, dan keterbatasannya ada di `results/importance/`. Analisis ini tidak melakukan training ulang, tidak membaca test, dan tidak dipakai untuk mengubah konfigurasi terpilih.

## Stratified K-Fold tambahan

Paket ini juga memuat pemeriksaan kestabilan 5-fold pada **61 foto train**, dengan konfigurasi model utama yang tetap. Validation 20 dan test 21 tidak digabungkan ke dalam fold.

```bash
python analysis/cross_validate.py --data data/comvis_dataset_v3 --results results --device auto --folds 5
```

SVM/XGBoost dilatih ulang per fold dengan imputer, scaler, dan referensi normal dari training fold. DL hanya memakai turunan augmentasi dari foto induk training fold; foto validation fold memakai view asli. Epoch DL ditetapkan dari konfigurasi utama sebelum fold dijalankan, sehingga held-out fold tidak dipakai memilih checkpoint atau early stopping.

Ini pemeriksaan konfigurasi tetap setelah pemilihan melalui holdout utama, bukan nested CV. Mean dan std antar-fold bersifat deskriptif; pembagian tetap per foto dan tidak menjamin botol fisik independen. Output ada pada `results/cross_validation/`, termasuk skor fold, prediksi out-of-fold, model fold, dan riwayat training.

## Evaluasi test setelah konfigurasi dikunci

Perintah training default berhenti pada validation. Setelah konfigurasi diputuskan, jalankan:

```bash
python run.py lock --data data/comvis_dataset_v3 --output runs/retrain_gpu
python run.py test --data data/comvis_dataset_v3 --output runs/retrain_gpu --device auto
```

Untuk run lengkap yang langsung mengunci dan mengevaluasi sekali di akhir:

```bash
python run.py all --data data/comvis_dataset_v3 --device auto --output runs/final_gpu --evaluate-test --glcm-distances 1 2 4
```

Kode menyimpan hash dataset, konfigurasi, kode, dan checkpoint; evaluasi menolak perubahan setelah lock. `test_evaluation_receipt.json` mencegah evaluasi ulang pada output yang sama. Jangan melakukan tuning tambahan berdasarkan skor test. Mengevaluasi banyak run berbeda pada 21 foto test yang sama juga mengurangi independensi evaluasi; gunakan validation untuk perbandingan eksperimen.

Dataset audit dapat membaca byte gambar test untuk checksum tanpa memakai hasilnya untuk fitting atau pemilihan konfigurasi. Test tidak digunakan untuk memilih PCD, d GLCM, referensi normal, imputer, scaler, atau hyperparameter.

## Prototipe prediksi foto baru

```bash
python predict.py --image /path/foto_botol.jpg --results results --device auto
python predict.py --image /path/foto_botol.jpg --results runs/retrain_gpu --model dl_pcd --device auto
```

Output mencantumkan prediksi empat model. Softmax DL bukan probabilitas terkalibrasi. Prototipe ini berupa CLI; tidak ada klaim sistem inspeksi produksi.

## Isi hasil

| File / folder | Fungsi |
|---|---|
| `results/comparison.csv`, `comparison.png` | Perbandingan skor keempat model per split. |
| `results/*_metrics.json` | Macro-F1, accuracy, precision/recall/F1 per kelas, confusion matrix. |
| `results/*_predictions.csv`, `*_confusion.png` | Prediksi individual dan confusion matrix. |
| `results/ml_tuning_trials.csv`, `dl_tuning_trials.csv` | Seluruh kandidat tuning dan hasil validation. |
| `results/dl_training_history.csv` | Loss dan skor validation per epoch per trial. |
| `results/features/` | Pengukuran, vektor fitur model, serta diagnostik ROI setiap kandidat. |
| `results/models/` | SVM, XGBoost, backbone pretrained, dan dua checkpoint head DL. |
| `results/cache/` | Fitur backbone yang benar-benar digunakan pada transfer learning. |
| `results/*_selected_config.json` | PCD/d/hyperparameter dan checksum model terpilih. |
| `results/environment.json`, `command.json`, `run.log` | Lingkungan, perintah dan log pelaksanaan. |
| `results/selection_lock.json`, `test_evaluation_receipt.json` | Rekaman pemisahan tuning dan test. |
| `tests/` | Regresi bug missing values, kebocoran metadata, GLCM d/energy, dan bentuk input. |

Jalankan pemeriksaan kode:

```bash
python -m unittest discover -s tests -v
```

Checkpoint sklearn/joblib sebaiknya dimuat dengan versi dependency yang sama; lihat `requirements-tested.txt` dan `results/environment.json`. Untuk retraining baru boleh memakai lingkungan kompatibel, lalu gunakan checkpoint yang dihasilkan lingkungan tersebut. Hasil GPU dan CPU dapat sedikit berbeda walaupun seed sama.

## Batas interpretasi

Jumlah test hanya 21 foto dan random split dapat menempatkan foto botol yang sama di beberapa bagian. Validation dipakai untuk banyak kandidat sehingga skor validation bisa optimistis. Perbandingan skor v2 dan v3 tidak mengisolasi efek perbaikan kode karena dataset dan split juga berubah. ROI/label tetap heuristik; peningkatan skor tidak otomatis membuktikan kontur botol sudah benar. Run yang disertakan menggunakan pretrained ImageNet; izin bobot pretrained mengikuti ketentuan tugas kelompok.

Rujukan implementasi: Torchvision MobileNetV3-Large 0.21, scikit-learn SimpleImputer dan Pipeline, scikit-image graycoprops untuk definisi fitur GLCM. Detail tautan ada di `docs/REFERENCES.md`.
