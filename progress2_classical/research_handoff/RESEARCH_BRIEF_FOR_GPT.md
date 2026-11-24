# Brief riset: diagnosis baseline PCD + ML klasik, kelompok 15

## Tujuan

Menemukan perbaikan sederhana yang dapat diperiksa untuk klasifikasi cacat visual
kemasan botol Aquviva 700 ml. Saat ini fokus pada PCD + fitur manual + SVM dan
PCD + fitur manual + XGBoost. Foto raw + MobileNetV3-Large dan foto PCD +
MobileNetV3-Large merupakan eksperimen lanjutan, belum dijalankan.

## Data, ground truth, dan split yang dikunci

Empat kelas, dalam urutan metrik yang tetap:
`normal`, `label_miring`, `label_hilang`, `penyok`.

Ada empat botol fisik, masing-masing lima foto: normal standar, normal dengan
pencahayaan berbeda, label miring, label hilang, dan penyok dengan label terpasang.
Totalnya 20 foto primer. Train memakai botol 01–02, validation botol 03, test
botol 04. Dilarang random split ulang. Turunan selalu mengikuti split sumber.

Train berisi 10 foto asli + 290 augmentasi offline, total 300 baris dengan 75
per kelas. Validation dan test masing-masing lima foto asli, dengan normal dua
foto dan tiga kelas lain masing-masing satu. Jangan menganggap 300 sampel train
sebagai 300 observasi primer independen.

Ground truth ditentukan dari kondisi fisik yang dikonfirmasi pengambil foto.
Semua foto penyok memang penyok, termasuk foto subtle v1. Jangan mengganti
label berdasarkan keterlihatan lekukan.

## Perbedaan v1 dan v2

| Aspek | v1 | v2 |
|---|---|---|
| penyok01, train | Penyok subtle, dikonfirmasi hanya di kanan atas | Retake dengan deformasi jauh lebih jelas, melibatkan beberapa bagian badan |
| penyok03, validation | Penyok subtle, dikonfirmasi hanya di kanan atas | Retake dengan deformasi lebih jelas |
| Turunan penyok01 | 37 turunan dari sumber lama | 37 turunan diperbarui dari foto retake |
| Foto primer lain | Foto awal | 18 foto primer lain identik byte demi byte |
| Test | Lima foto botol 04 | Seluruh foto dan CSV test identik dengan v1 |
| Split, jumlah, kelas | Dikunci | Tidak berubah |
| Parameter augmentasi | Setelan awal | JSON parameter sama persis |

Penggantian dilakukan pada 17 November 2026, setelah hasil awal ditinjau.
Ini revisi data dan tingkat keterlihatan cacat, bukan intervensi model saja.
Skor v1–v2 tidak boleh ditafsirkan sebagai bukti peningkatan/penurunan generalisasi
model akibat satu faktor. V2 tidak membuktikan kemampuan mendeteksi penyok subtle.

Audit menemukan 38 sampel train berubah: satu original penyok01 dan 37
turunannya; validation berubah satu foto, penyok03. Seluruh 37 turunan v2
direproduksi di memori memakai sumber retake dan parameter tercatat, dengan
byte JPEG identik pada Pillow 11.1.0. Tidak dilakukan augmentasi tambahan.

Augmentasi: seed 42, brightness 0.85–1.15, contrast 0.90–1.10, Gaussian blur
radius 0–0.35; sisi terpanjang maksimal 1024, JPEG quality 95. Tidak ada flip,
rotasi lokal label, atau deformasi geometris. Val/test tidak diaugmentasi.

## Pipeline yang benar-benar dijalankan

1. Membaca gambar dengan koreksi orientasi EXIF, RGB, sisi maksimal 768.
2. Membandingkan empat varian: center ROI / cap-anchored ROI × gray / CLAHE.
3. ROI crop horizontal 62% lebar gambar; cap dicari menggunakan warna biru HSV.
4. Estimasi sumbu badan dari pasangan tepi. Jika lolos pemeriksaan, rotasi ROI
   terhadap sumbu badan. Jika gagal, pertahankan ROI tanpa rotasi dan tandai
   kegagalannya. Label tidak pernah diluruskan terpisah.
5. Perbandingan visual pada train asli: Otsu threshold, Canny, closing 3×3,
   CLAHE. Threshold/Canny belum menghasilkan mask botol yang tervalidasi.
6. Fitur mencakup profil lebar pada 10 ketinggian ROI, deviasi terhadap normal
   train, profil sisi lokal, edge density, fitur label berbasis putih dengan
   saturasi rendah, serta GLCM horizontal/vertikal. Tinjau kode untuk definisi
   tepatnya. Fitur ini bukan pengukuran kontur ground truth.
7. Referensi normal dibentuk dari empat foto normal train asli saja. Sudut/area
   yang gagal terukur disimpan NaN; imputer median dan missing indicators
   dipelajari dari train. SVM juga memakai StandardScaler train.
8. Grid per varian: SVM linear C=0.1/1/10 dan RBF C=0.5/2, gamma='scale',
   class_weight='balanced'; XGBoost n_estimators=80/160, depth=2/3, lr=0.05.
   Total 20 konfigurasi SVM dan 16 XGBoost. Seed 42, XGBoost CPU hist, 2 thread.
9. Pilih berdasarkan macro-F1 validation empat kelas. Untuk nilai yang sama,
   implementasi membandingkan accuracy lalu string nama varian; parameter yang
   seri dalam varian yang sama mempertahankan kandidat pertama. Jangan
   menganggap selisih ranking kecil pada lima foto sebagai bukti kuat.

Varian, algoritma fitur, dan grid sama antarversi. Pada 266 gambar train/val
yang tidak berubah, 129 kolom numerik hasil ekstraksi cocok untuk keempat
varian. Sebanyak 75 fitur masuk ke model terpilih; kolom identitas, ukuran foto,
dan beberapa diagnostik tidak menjadi input model. Audit kembali pemilihan
kolom pada kode bila menilai potensi shortcut atau kebocoran.

## Hasil aktual dan prediksi

| Dataset | Model | Varian terpilih | Macro-F1 val | Accuracy val |
|---|---|---|---:|---:|
| v1 | SVM linear C=0.1 | center ROI + CLAHE | 0.625 | 0.60, 3/5 |
| v1 | XGBoost depth=2, n=80 | center ROI + gray | 0.625 | 0.60, 3/5 |
| v2 | SVM linear C=0.1 | center ROI + gray | 0.375 | 0.40, 2/5 |
| v2 | XGBoost depth=2, n=80 | center ROI + gray | 0.350 | 0.40, 2/5 |

V1: kedua model benar pada label_hilang dan label_miring, tetapi memprediksi
kedua foto normal DAN foto penyok sebagai penyok. Jadi keberhasilan v1 pada
penyok subtle bukan bukti model mampu membedakannya dari normal.

V2: kedua model tetap benar pada label_hilang dan label_miring. Kedua normal
diprediksi label_miring. Penyok03 diprediksi normal oleh SVM dan label_miring
oleh XGBoost. Recall normal dan penyok sama-sama 0 pada validation v2.

V2: kedua model mencapai accuracy dan macro-F1 train 1.0, baik pada 10 foto
asli maupun seluruh 300 sampel train. Ini skor pada data yang dipakai fit,
bukan evaluasi independen. Polanya konsisten dengan overfitting dan/atau
pergeseran kondisi akuisisi; penyebab dominannya belum dibuktikan.

## Temuan visual, keterbatasan, dan hipotesis

**Teramati pada artefak:** pasangan tepi profil dapat menangkap cetakan label,
rusuk plastik, atau pantulan. Pada normal01_light, kotak label justru memilih
pantulan di bawah label. Karena itu status label_found=1 atau valid-row fraction
tinggi tidak membuktikan segmentasi benar.

Pada varian v2 terpilih, sumbu tersedia pada 6/10 train asli dan 3/5 validation;
region label ditemukan 8/10 dan 4/5, tetapi batas region belum tentu tepat.
Sudut label tersedia hanya pada 1/10 train dan 2/5 validation. Lihat overlay,
CSV diagnostik, dan nilai NaN untuk penilaian tiap gambar.

Deviasi lebar profil rata-rata penyok03 v2 adalah 0.0893, sedangkan dua normal
validation 0.1289 dan 0.0947. Ukuran ini belum spesifik terhadap penyok.

Background berbeda antar-ID botol. Normal umumnya menunjukkan belakang label,
sedangkan label_miring menunjukkan depan. Foto memiliki variasi tetesan air,
orientasi, pantulan dan pencahayaan. Protokol akuisisi terkontrol adalah rencana,
bukan kondisi yang dipenuhi seluruh foto aktual.

**Hipotesis yang perlu diuji, belum terbukti:** fitur mengikuti detail interior
atau background; bentuk penyok train kurang mewakili validation; dua botol train
terlalu sedikit; deformasi label pada botol penyok mencampur sinyal antarkelas.
Jangan menyimpulkan shortcut tertentu hanya dari feature importance.

## Pertanyaan riset yang diminta

- Apa diagnosis paling masuk akal berdasarkan kode dan visual train, dan bukti
  apa yang mendukung atau membantah setiap hipotesis?
- Dua atau tiga alternatif sederhana apa yang layak untuk ROI, estimasi sumbu,
  batas botol transparan, dan label? Bandingkan kebutuhan anotasi, failure modes,
  dependency, serta kemampuan berjalan di Kaggle free CPU.
- Bagaimana mengukur kualitas ROI/fitur sebelum menilai klasifikasi? Usulkan
  anotasi/pemeriksaan pada train dan aturan abstain atau missing yang eksplisit.
- Bagaimana merancang ablation kecil untuk fitur bentuk, label, tekstur dan
  intensitas agar tidak sekadar menambah pencarian pada lima foto validation?
- Penambahan data apa yang paling membantu: botol baru pada train, validation,
  test, sisi label yang konsisten, background, dan tingkat keparahan penyok?
  Nyatakan target jumlah sebagai rencana praktis, bukan minimum ilmiah universal.
- Apakah rumus fitur, missing handling, ROI fallback, atau pemilihan kolom kode
  memiliki kelemahan implementasi? Laporkan lokasi fungsi dan dampaknya.

Gunakan sumber primer/paper atau dokumentasi resmi bila melakukan pencarian.
Bedakan temuan teramati, hipotesis, dan usulan yang belum dijalankan. Jangan
mengarang hasil perbaikan atau menjanjikan akurasi tertentu.

## Batas eksperimen berikutnya

Jangan ubah label penyok, random split, atau menambah augmentasi otomatis.
Jangan gunakan validation/test untuk fit scaler, referensi bentuk, seleksi fitur
berbasis fit, atau parameter detektor yang dipelajari. Validation boleh memilih
konfigurasi; test disimpan untuk konfigurasi yang sudah dikunci. Pengulangan
optimasi pada lima foto validation juga harus diakui sebagai keterbatasan.
Botol baru harus dialokasikan per ID sebelum dipakai; seluruh turunannya tetap
di split sumber. Jangan merombak split lama tanpa keputusan eksperimen baru.
