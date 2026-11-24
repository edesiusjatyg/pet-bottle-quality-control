# Handoff riset Progress 2 — Aquviva, v1 dan v2

Unggah `aquviva_research_handoff_v1_v2.zip` ke GPT bersama isi `PROMPT_FOR_GPT.md`.
Baca `RESEARCH_BRIEF_FOR_GPT.md` sebelum menafsirkan tabel skor.

Paket ini memuat kedua versi eksperimen, dengan struktur:

- `versions/v1/`: metadata dataset awal, 10 foto train asli dan 5 foto validation
  asli, seluruh hasil eksperimen v1, dan diagnostik train dari model tersimpan.
- `versions/v2/`: metadata dataset retake, 10 foto train asli dan 5 foto validation
  asli, seluruh hasil eksperimen v2, dan diagnostik train dari model tersimpan.
- `comparison/`: perubahan file/label per split, prediksi v1–v2 berdampingan,
  perbandingan skor, audit reproduksi augmentasi, dan audit konsistensi fitur.
- `code/`: kode pipeline bersama, requirements, builder notebook, notebook v2
  bersih, dan notebook v2 yang telah dieksekusi lokal dengan outputnya.
- `MANIFEST.json`: ukuran dan checksum seluruh berkas paket selain manifest itu
  sendiri, termasuk pemetaan setiap foto ke split dan versi.

Foto test tidak disertakan. Metadata test hanya disertakan untuk audit split.
Seluruh hasil model yang disertakan adalah train/validation; belum ada evaluasi
test. Foto validation disertakan agar kesalahan bisa ditinjau, tetapi revisi
aturan detektor perlu didasarkan pada train dan dinilai kembali di validation.

Untuk menjaga ukuran paket, 290 JPEG augmentasi per versi tidak disertakan.
`train.csv`, `augmentation_parameters.json`, `augmentation_lineage.csv`, fitur
seluruh 300 sampel train, dan `augment_train.py` tersedia untuk setiap versi.
Ini paket riset, bukan pengganti ZIP dataset lengkap untuk Run All. Gunakan ZIP
dataset lengkap yang sesuai ketika hendak menjalankan ulang pipeline; jangan
menganggap folder `dataset_metadata/` sebagai dataset lengkap.

Model `.joblib` disertakan untuk diagnostik. Pemakaiannya memerlukan versi
dependency yang sesuai; lingkungan v2 tercatat dalam `selected_config.json`.
Snapshot kode/notebook v1 asli tidak tersedia terpisah. Kode bersama sekarang
memiliki tambahan identitas versi dan visual diagnostik; algoritma ekstraksi
fitur serta grid model tetap seperti eksperimen awal. Kesamaan fitur pada
gambar yang tidak berubah diverifikasi pada kedua versi.

Notebook v1 dahulu diperiksa format/sintaks, sedangkan eksperimennya dijalankan
sebagai skrip. Notebook v2 benar-benar dijalankan melalui kernel Jupyter lokal:
7 sel kode selesai tanpa error, test dinonaktifkan. Belum dijalankan di Kaggle.

Knowledge base proyek:
https://docs.google.com/document/d/1i1DCU9CjWlozTVr2nM55yFRVL3Dkts4eufsJh63k5qU

Jika tautan knowledge base tidak dapat dibaca, gunakan brief dan berkas lokal
sebagai sumber. Nyatakan keterbatasan akses; jangan mengarang isinya.
