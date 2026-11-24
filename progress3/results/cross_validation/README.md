# Stratified K-Fold tambahan

5 fold pada 61 foto train asli. Validation 20 dan test 21 tidak dimasukkan.
Konfigurasi tetap mengikuti model utama; tidak ada tuning ulang dan tidak ada perubahan hasil test utama.
Referensi normal, imputer, scaler, dan classifier dilatih ulang di dalam setiap fold. Augmentasi DL hanya berasal dari sumber training fold.
DL dilatih selama 18 epoch tetap. Bagian validasi fold tidak dipakai untuk early stopping atau pemilihan epoch.

| Model | Macro-F1 mean | Std antar-fold | OOF accuracy |
|---|---:|---:|---:|
| svm | 0.3307 | 0.1910 | 34.43% |
| xgboost | 0.4217 | 0.1921 | 42.62% |
| dl_raw | 0.7208 | 0.0982 | 72.13% |
| dl_pcd | 0.7221 | 0.0939 | 72.13% |

Std antar-fold bukan confidence interval. Karena konfigurasi sudah dipilih melalui protokol holdout utama, ini analisis kestabilan bersyarat dan bukan nested cross-validation. Folds per foto tidak menjamin botol fisik independen.
