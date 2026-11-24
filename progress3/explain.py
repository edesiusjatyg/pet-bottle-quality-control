"""Validation-only permutation importance and Grad-CAM for selected models."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,joblib,numpy as np,pandas as pd
from PIL import Image,ImageDraw
from common import CLASSES,SEED,read_split,load_json,save_json,load_rgb,metrics,digest
from features import SHAPE,LABEL,TEXTURE,DERIVED,FEATURE_COLUMNS,reference_apply
from ml import feature_table,predict_model
from preprocessing import prepare,letterbox

def plot_importance(frame,path,title,limit=15):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f=frame.sort_values('importance_mean',ascending=False).head(limit).iloc[::-1]
    fig,ax=plt.subplots(figsize=(10,max(3,len(f)*.32+1.5)))
    ax.barh(f['feature'],f['importance_mean'],xerr=f['importance_std'],color='#3274a1',capsize=2)
    ax.axvline(0,color='black',linewidth=.7);ax.set(title=title,xlabel='Penurunan macro-F1 validation setelah pengacakan (mean ± std)')
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def permutation_table(name,model,x,y,groups,repeats=20):
    baseline=metrics(y,predict_model(name,model,x))['macro_f1'];rows=[]
    for group,columns in groups.items():
        rng=np.random.default_rng(SEED);scores=[]
        for _ in range(repeats):
            shuffled=x.copy();order=rng.permutation(len(x))
            shuffled.loc[:,columns]=x.iloc[order][columns].to_numpy()
            scores.append(baseline-metrics(y,predict_model(name,model,shuffled))['macro_f1'])
        rows.append({'feature':group,'importance_mean':float(np.mean(scores)),'importance_std':float(np.std(scores)),
            'baseline_macro_f1':baseline,'repeats':repeats,'columns':json.dumps(columns)})
    return pd.DataFrame(rows)

def batched_macro_f1(predictions,truth):
    predictions=np.asarray(predictions);truth=np.asarray(truth)
    values=[]
    for c in range(len(CLASSES)):
        tp=((predictions==c)&(truth==c)).sum(axis=-1)
        denominator=(predictions==c).sum(axis=-1)+int((truth==c).sum())
        values.append(np.divide(2*tp,denominator,out=np.zeros_like(tp,dtype=float),where=denominator!=0))
    return np.stack(values,axis=-1).mean(axis=-1)

def channel_importance(x,weights,bias,y,repeats=20,seed=SEED):
    # Exact logit updates for a single linear head; no full CNN reruns or refitting.
    x=np.asarray(x);weights=np.asarray(weights);bias=np.asarray(bias);n,channels=x.shape
    base=x@weights.T+bias;baseline=metrics(y,base.argmax(1))['macro_f1']
    rng=np.random.default_rng(seed);deltas=np.empty((repeats,channels,n),dtype=np.float32)
    for r in range(repeats):
        for j in range(channels):deltas[r,j]=x[rng.permutation(n),j]-x[:,j]
    logits=base[None,None,:,:]+deltas[:,:,:,None]*weights.T[None,:,None,:]
    decreases=baseline-batched_macro_f1(logits.argmax(-1),y)
    return pd.DataFrame({'feature':[f'channel_{j:03d}' for j in range(channels)],'channel':np.arange(channels),
        'importance_mean':decreases.mean(0),'importance_std':decreases.std(0),'baseline_macro_f1':baseline,'repeats':repeats})

def gradcam(backbone,head,array,device):
    import torch
    from deep_learning import normalize
    activations=[]
    handle=backbone.features[-1].register_forward_hook(lambda module,args,out:activations.append(out))
    try:
        tensor=normalize(np.stack([array])).to(device).requires_grad_(True)
        backbone.eval();head.eval();logits=head(backbone(tensor));target=int(logits.argmax(1).item())
        activation=activations[0]
        gradient=torch.autograd.grad(logits[0,target],activation,retain_graph=False)[0]
        cam=(gradient.mean((2,3),keepdim=True)*activation).sum(1).relu()[0].detach().cpu().numpy()
    finally:handle.remove()
    span=float(cam.max()-cam.min());valid=span>1e-10
    if valid:cam=(cam-cam.min())/span
    else:cam=np.zeros_like(cam)
    full=cv2.resize(cam,(array.shape[1],array.shape[0]),interpolation=cv2.INTER_LINEAR)
    colors=cv2.cvtColor(cv2.applyColorMap(np.uint8(255*full),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
    overlay=np.clip(.6*array+.4*colors,0,255).astype(np.uint8) if valid else array.copy()
    return target,cam,full,overlay,valid

def run_explanations(root,output,device_request='auto',repeats=20,per_class=2):
    if repeats<1 or per_class<1:raise ValueError('Permutation repeats and examples per class must be positive.')
    import torch
    from deep_learning import choose_device,get_backbone,new_head
    output=Path(output);out=output/'importance';out.mkdir(exist_ok=True,parents=True)
    val=read_split(root,'val');ml=load_json(output/'ml_selected_config.json');dl=load_json(output/'dl_selected_config.json')
    raw=feature_table(root,val,ml['pcd_variant'],glcm_distance=ml['glcm_distance']);summary={}
    groups={'bentuk_dan_deviasi_normal':SHAPE+DERIVED,'label':LABEL,'tekstur_dan_intensitas':TEXTURE}
    for name,cfg in ml['models'].items():
        if digest(output/cfg['model_path'])!=cfg['model_sha256']:raise ValueError('Model changed before importance analysis')
        model=joblib.load(output/cfg['model_path']);x=reference_apply(raw,cfg['reference'])
        feature=permutation_table(name,model,x,val.class_name,{c:[c] for c in FEATURE_COLUMNS},repeats)
        grouped=permutation_table(name,model,x,val.class_name,groups,repeats)
        feature.to_csv(out/f'{name}_feature_importance.csv',index=False);grouped.to_csv(out/f'{name}_group_importance.csv',index=False)
        plot_importance(feature,out/f'{name}_feature_importance.png',f'{name}: individual features')
        plot_importance(grouped,out/f'{name}_group_importance.png',f'{name}: grouped features')
        summary[name]={'top_features':feature.sort_values('importance_mean',ascending=False).head(5).to_dict('records'),
            'grouped_importance':grouped.to_dict('records')}
        print(f'Importance complete: {name}',flush=True)
    device=choose_device(device_request);backbone,_=get_backbone(output,device)
    selected_indices=set(val.groupby('class_name',sort=False).head(per_class).index.tolist())
    for branch in ['raw','pcd']:
        pred=pd.read_csv(output/f'dl_{branch}_val_predictions.csv')
        selected_indices.update(pred.index[pred.class_name!=pred.predicted_label].tolist())
    selected=val.loc[sorted(selected_indices)]
    cam_records=[]
    for branch,cfg in dl['models'].items():
        path=output/cfg['head_path']
        if digest(path)!=cfg['head_sha256']:raise ValueError('DL checkpoint changed')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        head=new_head(checkpoint['parameters']['dropout']);head.load_state_dict(checkpoint['state_dict']);head.to(device).eval()
        data=np.load(output/'cache'/f'{branch}_val.npz',allow_pickle=False)
        x,y=data['features'],data['labels']
        feature=channel_importance(x,head[1].weight.detach().cpu().numpy(),head[1].bias.detach().cpu().numpy(),y,repeats)
        feature.to_csv(out/f'dl_{branch}_channel_importance.csv',index=False)
        plot_importance(feature,out/f'dl_{branch}_channel_importance.png',f'DL {branch}: pooled visual channels')
        summary['dl_'+branch]={'top_channels':feature.sort_values('importance_mean',ascending=False).head(5).to_dict('records')}
        branchdir=out/'gradcam'/branch;branchdir.mkdir(exist_ok=True,parents=True)
        panels=[]
        for row in selected.to_dict('records'):
            rgb=load_rgb(Path(root)/row['image_path'])
            if branch=='pcd':rgb,_=prepare(rgb,dl['pcd_variant'])
            prepared=letterbox(rgb);target,cam,full,overlay,valid=gradcam(backbone,head,prepared,device)
            filename=Path(row['source_filename']).stem
            np.save(branchdir/f'{filename}_cam.npy',full)
            panel=Image.new('RGB',(448,256),'white');panel.paste(Image.fromarray(prepared),(0,32));panel.paste(Image.fromarray(overlay),(224,32))
            draw=ImageDraw.Draw(panel);draw.text((4,3),f'{row["source_filename"]} | GT {row["class_name"]}',fill='black')
            draw.text((4,17),f'Pred {CLASSES[target]} | Grad-CAM valid={valid}',fill='black')
            panel.save(branchdir/f'{filename}.jpg',quality=92);panels.append(panel)
            cam_records.append({'branch':branch,'image_id':row['image_id'],'source_filename':row['source_filename'],
                'true_label':row['class_name'],'target_label':CLASSES[target],'target':'predicted_class',
                'coarse_height':cam.shape[0],'coarse_width':cam.shape[1],'nondegenerate_cam':valid,
                'coordinate_space':'model_input_224_after_branch_preprocessing'})
        canvas=Image.new('RGB',(896,int(np.ceil(len(panels)/2))*256),'white')
        for i,panel in enumerate(panels):canvas.paste(panel,((i%2)*448,(i//2)*256))
        canvas.save(out/f'dl_{branch}_gradcam_contact.jpg',quality=90)
        print(f'Importance complete: DL {branch}, {len(selected)} Grad-CAM examples',flush=True)
    pd.DataFrame(cam_records).to_csv(out/'gradcam_index.csv',index=False)
    result={'analysis_split':'val','samples':len(val),'test_used':False,'models_refitted':False,
        'metric':'macro_f1_decrease','permutation_repeats':repeats,'seed':SEED,
        'ml_method':'single and grouped feature permutation through fitted pipeline',
        'dl_global_method':'exact single-channel permutation of 960 pooled features through the frozen linear head',
        'dl_local_method':'Grad-CAM at final convolution, target predicted class, 224 input coordinate space',
        'gradcam_selection':'first examples per validation class plus all main-DL validation errors',
        'cautions':['Validation has only 20 photos; means/std are perturbation variability, not confidence intervals.',
            'Correlated features can mask single-feature importance; grouped results complement them.',
            'Channel IDs have no guaranteed human semantic meaning.',
            'Grad-CAM is coarse and is not a defect mask or causal explanation.',
            'No selected configuration is changed using these post-hoc importance results.'],
        'summary':summary}
    save_json(out/'importance_summary.json',result)
    (out/'README.md').write_text('''# Feature importance dan Grad-CAM

Semua analisis memakai 20 foto validation dari model yang sudah terpilih, tanpa refit dan tanpa test.

- `*_feature_importance.csv/png`: penurunan macro-F1 ketika satu fitur ML diacak. Nilai positif berarti pengacakan menurunkan skor, nilai negatif berarti skor malah meningkat pada sampel kecil ini. Error bars adalah std antar-permutasi, bukan confidence interval.
- `*_group_importance.csv/png`: satu kelompok bentuk, label, atau tekstur/intensitas diacak bersama; hubungan internal kelompok dipertahankan dengan urutan pengacakan yang sama.
- `dl_*_channel_importance.csv/png`: importance 960 channel visual setelah global pooling. Nomor channel tidak berarti konsep seperti penyok atau label.
- `gradcam/`, `dl_*_gradcam_contact.jpg`: input 224 di kiri dan overlay Grad-CAM di kanan. Targetnya kelas prediksi. Raw dan PCD memiliki ruang koordinat input masing-masing; overlay PCD bukan peta pada foto mentah. Layer akhir beresolusi 7×7 sehingga detail kecil tidak dapat dilokalisasi presisi.
- `gradcam_index.csv` mencatat sampel dan heatmap yang degenerat. Heatmap konstan tidak diberi warna palsu.

Ranking ini bergantung pada model, data dan korelasi fitur. Importance bukan bukti sebab-akibat, tidak menggantikan anotasi cacat, dan tidak digunakan untuk memilih ulang model setelah melihat test.
''',encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='auto');p.add_argument('--repeats',type=int,default=20);p.add_argument('--per-class',type=int,default=2)
    a=p.parse_args();run_explanations(a.data,a.output,a.device,a.repeats,a.per_class)
