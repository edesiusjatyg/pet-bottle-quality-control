Saya mengunggah paket riset computer vision kelompok 15 untuk klasifikasi cacat
kemasan botol Aquviva 700 ml. Tolong baca README.md dan RESEARCH_BRIEF_FOR_GPT.md,
lalu periksa kedua versi v1 dan v2: kode, README dataset, CSV, parameter/script
augmentasi, hasil, dan visual train. Paket sengaja tidak memuat foto test.

Saya ingin diagnosis yang berbasis bukti mengapa baseline SVM dan XGBoost
gagal pada validation, termasuk mengapa retake penyok yang lebih jelas pada v2
tidak memperbaiki skor. Jangan menganggap perbandingan ini eksperimen model
dengan data identik: train DAN satu foto validation berubah setelah evaluasi awal.

Tolong hasilkan:

1. Tabel perbedaan v1–v2 dan penjelasan apa yang sah dibandingkan.
2. Diagnosis berurutan menurut kekuatan bukti: pengukuran ROI/fitur, kekurangan
   data independen, pergeseran kondisi akuisisi, dan potensi shortcut. Kaitkan
   klaim ke berkas, fungsi, atau gambar tertentu. Pisahkan hipotesis dari temuan.
3. Audit kelemahan kode yang dapat memengaruhi pengukuran atau evaluasi.
4. Maksimal tiga kandidat perbaikan PCD/fitur yang sederhana, dengan cara
   memeriksa hasil, failure handling, dependency, dan perkiraan beban CPU.
5. Rencana ablation kecil dan urutan perbaikan sebelum mencoba hyperparameter
   lebih luas. Usulan detektor dibentuk dari train; validation dipakai memilih.
6. Rencana penambahan foto primer/botol: prioritas train, validation, test,
   kontrol sisi label/background, dan variasi penyok subtle sampai jelas.
7. Rujukan primer yang mendukung usulan, beserta keterbatasan penerapannya pada
   botol transparan dan dataset kecil ini. Jangan mengarang paper atau hasil.

Gunakan bahasa Indonesia yang jelas. Ground truth penyok dikonfirmasi secara
fisik dan tidak boleh diganti. Pertahankan split berdasarkan ID botol; jangan
menyarankan random split foto atau augmentasi val/test. Semua model terpilih
sudah dijalankan pada train/validation; belum ada evaluasi test maupun DL.
Notebook v2 dijalankan penuh di Jupyter lokal, tetapi belum di Kaggle.

Knowledge base proyek juga ada di:
https://docs.google.com/document/d/1i1DCU9CjWlozTVr2nM55yFRVL3Dkts4eufsJh63k5qU
Jika tidak bisa diakses, katakan demikian dan lanjutkan memakai paket terlampir.
