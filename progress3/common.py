"""Shared dataset contract, checksums, metrics, and reproducible outputs."""
from __future__ import annotations
import hashlib, json, os, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, f1_score

CLASSES = ['label_hilang', 'label_miring', 'normal', 'penyok']
SEED = 2026

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [clean(v) for v in x]
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        if isinstance(x,Path):return str(x)
        return x
    path.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def load_json(path):return json.loads(Path(path).read_text())

def dataset_root(value=None):
    if value or os.environ.get('AQUVIVA_DATASET_ROOT'):
        root=Path(value or os.environ['AQUVIVA_DATASET_ROOT']).expanduser().resolve()
        if not (root/'metadata.csv').is_file():raise FileNotFoundError(f'metadata.csv not found in {root}')
        return root
    candidates=[Path.cwd()/'comvis_dataset_v3',Path(__file__).resolve().parent.parent/'comvis_dataset_v3']
    if Path('/kaggle/input').exists():candidates+=sorted(p.parent for p in Path('/kaggle/input').glob('**/metadata.csv') if 'comvis' in str(p))
    for root in candidates:
        if (root/'metadata.csv').is_file():return root.resolve()
    raise FileNotFoundError('Use --data /path/to/comvis_dataset_v3 or AQUVIVA_DATASET_ROOT.')

def identity(root):
    root=Path(root)
    paths=['metadata.csv','class_names.json','splits/train.csv','splits/val.csv','splits/test.csv']
    return {'version':'v3','files':{p:digest(root/p) for p in paths},'class_names':CLASSES}

def read_split(root, split):
    if split not in {'train','val','test'}:raise ValueError(split)
    frame=pd.read_csv(Path(root)/'splits'/f'{split}.csv',dtype={'bottle_id':str,'photo_number':str})
    if not (frame['split']==split).all():raise ValueError('Inconsistent split fields')
    if set(frame['class_name'])!=set(CLASSES):raise ValueError('Every split must contain all four classes')
    if frame['image_id'].duplicated().any():raise ValueError('Duplicate image assignment')
    expected=frame['class_name'].map(dict(zip(CLASSES,range(4))))
    if not (expected==frame['class_id']).all():raise ValueError('Class mapping mismatch')
    return frame

def audit(root, verify_images=True):
    root=Path(root);splits={s:read_split(root,s) for s in ['train','val','test']}
    joined=pd.concat(splits.values(),ignore_index=True)
    metadata=pd.read_csv(root/'metadata.csv')
    if set(joined['image_id'])!=set(metadata['image_id']):raise ValueError('Split/metadata coverage mismatch')
    if joined['image_id'].duplicated().any() or joined['pixel_sha256'].duplicated().any():raise ValueError('Duplicate image or pixels across splits')
    checked=0
    for row in joined.to_dict('records'):
        if digest(root/row['image_path'])!=row['prepared_sha256']:raise ValueError(f'Checksum mismatch: {row["source_filename"]}')
        if verify_images:
            with Image.open(root/row['image_path']) as im:im.verify()
        checked+=1
    return {'dataset':identity(root),'counts':{s:len(f) for s,f in splits.items()},
        'class_counts':{s:f['class_name'].value_counts().to_dict() for s,f in splits.items()},
        'images_verified':checked,'split_method':'random_stratified_per_photo','split_seed':2026,
        'no_exact_pixel_duplicates':True,'physical_bottle_independence_guaranteed':False}

def load_rgb(path,max_side=768):
    with Image.open(path) as src:im=ImageOps.exif_transpose(src).convert('RGB')
    im.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
    return np.asarray(im).copy()

def metrics(y_true,y_pred):
    y_true=np.asarray(y_true);y_pred=np.asarray(y_pred)
    if y_true.dtype.kind in {'i','u'}:y_true=np.array(CLASSES)[y_true]
    if y_pred.dtype.kind in {'i','u'}:y_pred=np.array(CLASSES)[y_pred]
    p,r,f,n=precision_recall_fscore_support(y_true,y_pred,labels=CLASSES,zero_division=0)
    return {'macro_f1':float(f1_score(y_true,y_pred,labels=CLASSES,average='macro',zero_division=0)),
        'accuracy':float(accuracy_score(y_true,y_pred)),'n':len(y_true),
        'per_class':{c:{'precision':float(p[i]),'recall':float(r[i]),'f1':float(f[i]),'support':int(n[i])} for i,c in enumerate(CLASSES)},
        'confusion_matrix':confusion_matrix(y_true,y_pred,labels=CLASSES).tolist(),'class_order':CLASSES}

def report_predictions(output,model_name,split,frame,predicted):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    m=metrics(frame['class_name'],predicted)
    save_json(output/f'{model_name}_{split}_metrics.json',m)
    p=frame[['image_id','source_filename','class_name','class_id','split']].copy()
    p['predicted_label']=np.array(CLASSES)[predicted] if np.asarray(predicted).dtype.kind in {'i','u'} else predicted
    p['correct']=p['class_name']==p['predicted_label'];p.to_csv(output/f'{model_name}_{split}_predictions.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matrix=np.array(m['confusion_matrix']);fig,ax=plt.subplots(figsize=(6,5))
    ax.imshow(matrix,cmap='Blues',vmin=0)
    ax.set(xticks=range(4),yticks=range(4),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Prediksi',ylabel='Ground truth',
        title=f'{model_name} | {split} | n={len(frame)} | macro-F1={m["macro_f1"]:.3f}')
    plt.setp(ax.get_xticklabels(),rotation=25,ha='right')
    for y in range(4):
        for x in range(4):ax.text(x,y,str(matrix[y,x]),ha='center',va='center')
    fig.tight_layout();fig.savefig(output/f'{model_name}_{split}_confusion.png',dpi=140);plt.close(fig)
    return m
