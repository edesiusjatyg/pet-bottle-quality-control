# Inspeksi visual run CPU yang disertakan

Overlay Grad-CAM Raw yang ditinjau meliputi contoh validation dari keempat kelas dan kesalahan prediksi. Pada `normal4807.JPG`, hotspot menonjol tampak di latar kanan atas. Pada beberapa foto lain, hotspot tampak pada area label atau badan botol.

Temuan tersebut menjadi alasan untuk memeriksa kemungkinan shortcut latar. Grad-CAM beresolusi kasar 7×7 tidak membuktikan bahwa suatu area adalah satu-satunya penyebab keputusan. Tidak ada parameter model atau PCD yang diubah berdasarkan inspeksi ini.

Catatan ini khusus checkpoint pada paket ini; setelah retraining, tinjau ulang overlay run yang baru. Ini bukan anotasi segmentasi atau validasi fisik cacat.
