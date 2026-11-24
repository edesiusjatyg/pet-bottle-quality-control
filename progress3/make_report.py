"""Create a readable Progress 3 report from actual saved artifacts."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from PIL import Image,ImageDraw
from common import load_json,save_json

def make_report(root,output):
    root,output=Path(root),Path(output);table=pd.read_csv(output/'comparison.csv')
    ml=load_json(output/'ml_selected_config.json');dl=load_json(output/'dl_selected_config.json')
    trialml=pd.read_csv(output/'ml_tuning_trials.csv');trialdl=pd.read_csv(output/'dl_tuning_trials.csv')
    audit=load_json(output/'dataset_audit.json');counts=audit['counts'];plan=load_json(output/'ml_search_plan.json')
    train_views=len(pd.read_csv(output/'dl_augmentation_lineage.csv'))
    lines=['# Hasil Progress 3 — dataset v3','',
        f'Run aktual menggunakan device `{dl["device_used"]}`, PyTorch/Torchvision untuk DL dan scikit-learn/XGBoost untuk ML. Detail lingkungan tersedia pada environment.json.',
        '', '## Data dan metode','',
        f'{sum(counts.values())} foto primer: train {counts["train"]}, validation {counts["val"]}, test {counts["test"]}. Random split berstrata per foto, seed 2026. Tidak dilakukan split ulang.',
        f'ML memakai {len(ml["features"])} fitur eksplisit dari {counts["train"]} foto train asli. DL memakai {train_views} view dari sumber yang sama (asli + {dl["augmentation_copies"]} variasi photometric per sumber).',
        'DL adalah transfer learning backbone frozen; yang dilatih hanya head dropout + linear. Seluruh backbone ImageNet tidak di-fine-tune.',
        '', '## Tuning dan konfigurasi terpilih','',
        f'- {len(trialml)} kandidat ML: {len(plan["variants"])} PCD × {len(plan["glcm_distances"])} nilai d × (10 SVM + 8 XGBoost).',
        f'- PCD bersama terpilih: `{ml["pcd_variant"]}`; GLCM **d={ml["glcm_distance"]}** piksel pada ROI sisi maksimal 768 px. Arah 0°/90°, 16 level.',
        f'- SVM: `{json.dumps(ml["models"]["svm"]["parameters"],sort_keys=True)}`.',
        f'- XGBoost: `{json.dumps(ml["models"]["xgboost"]["parameters"],sort_keys=True)}`.',
        f'- {len(trialdl)} trial DL: delapan konfigurasi berpasangan Raw/PCD.',
        f'- Parameter DL bersama: `{json.dumps(dl["parameters"],sort_keys=True)}`; Adam, batch {dl["batch_size"]}, maksimal {dl["epochs"]} epoch, patience {dl["patience"]}.',
        f'- Epoch checkpoint terbaik: Raw {dl["models"]["raw"]["best_epoch"]}, PCD {dl["models"]["pcd"]["best_epoch"]}.',
        '', '## Skor aktual','', '| Model | Split | n | Macro-F1 | Accuracy |','|---|---|---:|---:|---:|']
    for row in table.to_dict('records'):
        lines.append(f'| {row["model"]} | {row["split"]} | {row["n"]} | {row["macro_f1"]:.4f} | {row["accuracy"]:.2%} |')
    effect=load_json(output/'pcd_effect.json')
    lines += ['', '## Pengaruh PCD pada DL','']
    for split,e in effect.items():lines.append(f'- Δ macro-F1 PCD − Raw pada {split}: **{e["delta_macro_f1_pcd_minus_raw"]:+.4f}**.')
    lines += ['', 'Perbandingan tersebut memakai split, backbone, bobot awal, hyperparameter, dan kebijakan augmentasi yang sama. Efeknya adalah paket PCD yang dipilih, bukan CLAHE saja.',
        '', '## Perbaikan implementasi','',
        '- Fitur profil yang seluruhnya gagal tetap NaN. Denominator proporsi deviasi hanya menghitung titik valid.',
        '- Target numerik dan metadata tidak masuk fitur; daftar fitur eksplisit mencegah kebocoran label.',
        '- ASM, energy, dan homogeneity mengikuti rumus yang dibedakan secara benar. Jarak d dapat dituning dan disimpan.',
        '- Referensi normal, imputer, scaler, dan model hanya di-fit pada train. Tuning memakai validation.',
        '- Seluruh kandidat ROI adaptif dilaporkan walaupun tidak dipilih; ada fallback untuk deteksi yang gagal.',
        '', '## Verifikasi','']
    check=load_json(output/'checkpoint_verification.json')
    lines.append(f'Checkpoint reload check: {"lulus" if check["passed"] else "gagal"}; prediksi 20 foto validation direproduksi untuk keempat model.')
    if (output/'test_evaluation_receipt.json').exists():
        lines.append('Test dievaluasi setelah konfigurasi serta checksum dikunci. Receipt dan selection_lock disertakan. Tidak ada tuning ulang berdasarkan hasil test.')
    else:lines.append('Test belum dievaluasi pada run ini.')
    lines += ['', '## Batas kesimpulan','',
        '- Split per foto tidak menjamin botol fisik berbeda antarbagian. Skor tidak membuktikan generalisasi ke botol baru.',
        '- Validation hanya 20 foto dan dipakai untuk banyak kandidat; test hanya 21 foto. Satu foto mengubah accuracy test sekitar 4,76 poin persentase.',
        '- Skor v2 tidak dapat dibandingkan langsung untuk menyimpulkan efek bug fix, karena dataset dan split berubah.',
        '- ROI, sumbu, dan label masih heuristik; tidak tersedia anotasi segmentasi untuk mengukur akurasi kontur.',
        '- Backbone frozen memperkecil biaya komputasi. Paket belum membuktikan hasil full-backbone fine-tuning.',
        '', 'Semua tabel, confusion matrix, prediksi per foto, lineage augmentasi, dan log trial tersedia di folder results. Petunjuk retraining GPU ada di README.md.','']
    probe_path=output/'representation_probe.json'
    if probe_path.exists():
        probe=load_json(probe_path)
        lines += ['## Diagnosis representasi fitur','',
            'Linear SVM juga dilatih pada 960 fitur MobileNet frozen dari 61 foto train asli, tanpa memasukkan turunan augmentasi. Pemeriksaan ini hanya menggunakan validation dan tidak mengubah empat model utama.','']
        for branch,r in probe['results'].items():
            lines.append(f'- SVM pada embedding {branch}: macro-F1 validation {r["metrics"]["macro_f1"]:.4f}, accuracy {r["metrics"]["accuracy"]:.2%}.')
        lines += ['', 'Jika SVM dengan embedding ikut tinggi, hasil ini mendukung bahwa representasi visual lebih informatif pada split ini daripada fitur manual. Ini bukan bukti bahwa dimensi tinggi sendiri menyebabkan kesulitan, atau bahwa botol independen akan memperoleh skor serupa.','']
    importance_path=output/'importance/importance_summary.json'
    if importance_path.exists():
        explanation=load_json(importance_path)
        lines += ['## Feature importance dan Grad-CAM','',
            f'Analisis post-hoc menggunakan {explanation["samples"]} foto validation, {explanation["permutation_repeats"]} pengacakan per fitur, tanpa refit atau penggunaan test.','']
        for name in ['svm','xgboost']:
            groups=explanation['summary'][name]['grouped_importance']
            best=max(groups,key=lambda x:x['importance_mean'])
            lines.append(f'- {name}: kelompok dengan penurunan macro-F1 rata-rata terbesar adalah `{best["feature"]}` ({best["importance_mean"]:+.4f}).')
        lines += ['', 'Importance individual dan kelompok tersimpan sebagai CSV/grafik. DL dilengkapi ranking permutation importance 960 channel visual serta Grad-CAM pada contoh validation per kelas dan kesalahan prediksi. Overlay menampilkan ruang input masing-masing jalur, bukan mask cacat ground truth.',
            'Ranking bergantung pada korelasi fitur, model, dan hanya 20 foto validation. Error bar adalah variasi pengacakan, bukan confidence interval. Peta Grad-CAM 7×7 bersifat kasar dan tidak membuktikan sebab-akibat. Lihat importance/README.md.','']
        if (output/'importance/VISUAL_REVIEW.md').exists():
            lines += ['Catatan inspeksi visual khusus run ini tersedia pada `importance/VISUAL_REVIEW.md`.','']
    (output/'PROGRESS3_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    # Inspectable error panels are produced only after evaluation, not used for model selection.
    errors=[]
    for name in ['svm','xgboost','dl_raw','dl_pcd']:
        p=output/f'{name}_test_predictions.csv'
        if p.exists():
            f=pd.read_csv(p);f=f[f.class_name!=f.predicted_label].head(6)
            errors += [{**r,'model':name} for r in f.to_dict('records')]
    if errors:
        meta=pd.read_csv(root/'metadata.csv').set_index('image_id');cols=4;cw,ch=245,285
        canvas=Image.new('RGB',(cols*cw,int(np.ceil(len(errors)/cols))*ch),'white');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(errors):
            with Image.open(root/meta.loc[r['image_id'],'image_path']) as src:im=src.convert('RGB');im.thumbnail((230,220))
            x=(i%cols)*cw;y=(i//cols)*ch;canvas.paste(im,(x+(cw-im.width)//2,y))
            draw.text((x+4,y+223),r['model']+' | '+r['source_filename'],fill='black')
            draw.text((x+4,y+241),'GT: '+r['class_name'],fill='black');draw.text((x+4,y+258),'Pred: '+r['predicted_label'],fill='red')
        canvas.save(output/'test_error_examples.jpg',quality=90)
    return output/'PROGRESS3_REPORT.md'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(make_report(a.data,a.output))
