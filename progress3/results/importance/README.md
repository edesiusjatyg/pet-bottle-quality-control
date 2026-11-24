# Feature importance dan Grad-CAM

Semua analisis memakai 20 foto validation dari model yang sudah terpilih, tanpa refit dan tanpa test.

- `*_feature_importance.csv/png`: penurunan macro-F1 ketika satu fitur ML diacak. Nilai positif berarti pengacakan menurunkan skor, nilai negatif berarti skor malah meningkat pada sampel kecil ini. Error bars adalah std antar-permutasi, bukan confidence interval.
- `*_group_importance.csv/png`: satu kelompok bentuk, label, atau tekstur/intensitas diacak bersama; hubungan internal kelompok dipertahankan dengan urutan pengacakan yang sama.
- `dl_*_channel_importance.csv/png`: importance 960 channel visual setelah global pooling. Nomor channel tidak berarti konsep seperti penyok atau label.
- `gradcam/`, `dl_*_gradcam_contact.jpg`: input 224 di kiri dan overlay Grad-CAM di kanan. Targetnya kelas prediksi. Raw dan PCD memiliki ruang koordinat input masing-masing; overlay PCD bukan peta pada foto mentah. Layer akhir beresolusi 7×7 sehingga detail kecil tidak dapat dilokalisasi presisi.
- `gradcam_index.csv` mencatat sampel dan heatmap yang degenerat. Heatmap konstan tidak diberi warna palsu.

Ranking ini bergantung pada model, data dan korelasi fitur. Importance bukan bukti sebab-akibat, tidak menggantikan anotasi cacat, dan tidak digunakan untuk memilih ulang model setelah melihat test.
