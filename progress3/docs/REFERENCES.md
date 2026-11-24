# Referensi implementasi

- Kode Progress 2 sumber: https://github.com/edesiusjatyg/pet-bottle-quality-control/tree/main/progress2_classical
- Torchvision MobileNetV3-Large 0.21 dan bobot ImageNet: https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.mobilenet_v3_large.html
- PyTorch instalasi CPU/CUDA/ROCm: https://pytorch.org/get-started/locally/
- scikit-learn SimpleImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
- scikit-learn Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- Definisi GLCM contrast, homogeneity, ASM, energy: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycoprops
- Permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html
- Grad-CAM (Selvaraju dkk.): https://arxiv.org/abs/1610.02391
- Proyek tugas: https://docs.google.com/document/d/1i1DCU9CjWlozTVr2nM55yFRVL3Dkts4eufsJh63k5qU

Implementasi memakai PyTorch, berbeda dari contoh Keras pada rancangan awal. Arsitektur eksperimen tetap MobileNetV3-Large dengan backbone pretrained yang dibekukan, pooling, dropout dan classifier empat kelas. Resize/padding dipakai untuk mempertahankan seluruh botol; normalisasi mengikuti mean/std ImageNet Torchvision.
