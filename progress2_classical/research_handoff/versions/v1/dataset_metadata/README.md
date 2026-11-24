# Dataset Kelompok 15, siap untuk eksperimen awal

## Isi dan split

- img berisi 20 foto asli tanpa perubahan byte.
- train_original.csv berisi 10 foto asli botol 01 dan 02.
- train.csv berisi 300 sampel training, yaitu 10 asli dan 290 augmentasi. Setiap kelas memiliki 75 sampel.
- val.csv berisi 5 foto asli botol 03, terdiri dari 2 normal serta 1 masing-masing label miring, label hilang, dan penyok.
- test.csv berisi 5 foto asli botol 04 dengan komposisi kelas yang sama seperti validation.
- metadata_original.csv mencatat seluruh 20 foto asli.
- augmented berisi 290 foto turunan training.
- augmentation_parameters.json mencatat foto sumber dan parameter setiap turunan.
- augment_train.py menghasilkan ulang augmentasi dari train_original.csv dengan seed 42.
- contact_sheet.jpg dan augmentation_preview.jpg membantu pemeriksaan visual.
- dataset_summary.json merangkum jumlah data dan pemeriksaan.

Gunakan path CSV relatif terhadap folder dataset_prep. Simpan bottle_id sebagai string agar nol di depan tidak hilang. Kelas adalah normal, label_miring, label_hilang, dan penyok. Jangan melakukan random split ulang terhadap train.csv, val.csv, dan test.csv.

## Konfirmasi ground truth

Pada 17 November 2026, pengambil foto mengonfirmasi bahwa semua foto berlabel penyok memang memperlihatkan botol yang secara fisik penyok. Penyok botol 01 dan 03 bersifat subtle. Label penyok tetap dipakai dan konfirmasi ini dicatat di metadata serta diwariskan ke turunan augmentasi botol 01. Kondisi fisik bersumber dari konfirmasi pengambil foto, bukan dari tebakan visual.

## Augmentasi yang sudah dijalankan

Training diperbanyak menjadi 300 sampel, 75 per kelas. Ini target praktis, bukan minimum ilmiah. Sumber dipilih bergiliran dalam setiap kelas. Foto normal lebih banyak, sehingga tiap foto normal mendapatkan lebih sedikit turunan dibanding kelas lain.

Transformasi awal menggunakan brightness 0.85 sampai 1.15, contrast 0.90 sampai 1.10, dan Gaussian blur radius 0 sampai 0.35 piksel. Turunan disimpan dengan sisi terpanjang maksimal 1024 piksel dan JPEG quality 95. Foto asli tidak diubah. Rentang transformasi merupakan setelan implementasi awal, bukan angka optimal dari paper. Contoh hasil diperiksa melalui preview. Tidak setiap turunan diperiksa manual satu per satu.

Tidak digunakan crop, penghilangan label, deformasi, flip, atau rotasi lokal label. Transformasi global di atas bertujuan mempertahankan bukti cacat. Validation dan test tidak diaugmentasi. Resize input model 224 piksel dilakukan nanti secara konsisten di pipeline model.

Untuk menghasilkan ulang, install Pillow kemudian jalankan python augment_train.py. Skrip menulis train.csv dan file turunan yang sama secara deterministik pada environment yang sama. Jangan menambah augmentasi online tanpa mencatatnya jika memakai dataset offline ini.

## Pemeriksaan yang sudah selesai

Seluruh 310 sampel train, validation, dan test dapat dibaca dan memiliki path valid. Checksum setiap berkas sesuai CSV. Semua checksum unik. ID botol antar-split tidak tumpang tindih, augmentasi hanya bersumber dari train, dan jumlah training seimbang pada empat kelas.

## Batasan akuisisi dan tindak lanjut

Foto normal umumnya memperlihatkan belakang label, sementara label miring memperlihatkan depan label. Perbedaan ini berpotensi menjadi petunjuk selain cacat yang perlu dikontrol saat menambah foto. Beberapa foto juga memiliki tetesan air, variasi orientasi, dan pantulan. Metadata standard/light mengikuti nama berkas, bukan hasil pengukuran pencahayaan terkontrol.

Jumlah data tetap berasal dari 20 foto primer dan 4 botol fisik. Augmentasi tidak menambah objek independen. Validation dan test masing-masing hanya 5 foto dari 1 botol, sehingga hasil masih bersifat eksperimen awal. Tambah foto asli untuk validation dan test, prioritaskan botol baru, dan pertahankan seluruh foto satu botol dalam satu split.

Langkah berikutnya adalah EDA pada data asli, pembandingan PCD dan ROI, ekstraksi fitur, kemudian baseline SVM dan XGBoost. Pelajari konfigurasi dan parameter dari train dan gunakan validation untuk pemilihan konfigurasi. Test dipakai untuk evaluasi akhir. Gunakan split yang sama untuk empat eksperimen proyek.
