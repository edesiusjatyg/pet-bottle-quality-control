"""Supplementary fixed-configuration Stratified K-Fold stability check on train only."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from common import CLASSES,SEED,identity,read_split,load_json,save_json,metrics,digest
from features import reference_fit,reference_apply,select_features
from ml import feature_table,estimator,fit_model,predict_model
from deep_learning import choose_device,seed_all,new_head

def fit_fixed_head(x,y,params,device,epochs,batch_size,seed):
    """Train for a fixed epoch budget; held-out fold is never used for stopping."""
    seed_all(seed);head=new_head(params['dropout']).to(device)
    optimizer=torch.optim.Adam(head.parameters(),lr=params['lr'],weight_decay=params['weight_decay'])
    counts=torch.bincount(y,minlength=4).float();weights=len(y)/(4*counts)
    loss_fn=torch.nn.CrossEntropyLoss(weight=weights.to(device));generator=torch.Generator().manual_seed(seed);history=[]
    for epoch in range(1,epochs+1):
        head.train();loss_sum=0.
        for indices in torch.randperm(len(y),generator=generator).split(batch_size):
            optimizer.zero_grad();loss=loss_fn(head(x[indices].to(device)),y[indices].to(device));loss.backward();optimizer.step()
            loss_sum+=float(loss.detach())*len(indices)
        history.append({'epoch':epoch,'train_loss':loss_sum/len(y)})
    head.eval();return head,history

def run(root,results,device_request='auto',folds=5):
    root,results=Path(root),Path(results);out=results/'cross_validation';out.mkdir(exist_ok=True,parents=True);(out/'models').mkdir(exist_ok=True)
    train=read_split(root,'train');ml=load_json(results/'ml_selected_config.json');dl=load_json(results/'dl_selected_config.json')
    if ml['dataset']!=identity(root) or dl['dataset']!=identity(root):raise ValueError('Dataset mismatch')
    if folds<2 or folds>train.class_name.value_counts().min():raise ValueError('Too many folds for smallest training class')
    device=choose_device(device_request);fixed_epochs=max(m['best_epoch'] for m in dl['models'].values())
    raw=feature_table(root,train,ml['pcd_variant'],glcm_distance=ml['glcm_distance'])
    lineage=pd.read_csv(results/'dl_augmentation_lineage.csv')
    original_positions={int(row.source_index):int(i) for i,row in lineage.iterrows() if row.augmentation_index==0}
    if len(original_positions)!=len(train):raise ValueError('Incorrect source lineage')
    embeddings={}
    for branch in ['raw','pcd']:
        cache=np.load(results/'cache'/f'{branch}_train.npz',allow_pickle=False)
        embeddings[branch]=(torch.tensor(cache['features'],dtype=torch.float32),torch.tensor(cache['labels'],dtype=torch.long))
    plan={'purpose':'post_hoc_fixed_configuration_stability_check','split_used':'train only','primary_photos':len(train),
        'folds':folds,'shuffle':True,'seed':SEED,'device':str(device),'dataset':identity(root),
        'pcd_variant':ml['pcd_variant'],'glcm_distance':ml['glcm_distance'],'ml_parameters':{n:c['parameters'] for n,c in ml['models'].items()},
        'dl_parameters':dl['parameters'],'dl_fixed_epochs':fixed_epochs,'dl_epoch_rule':'Fixed before folds from main validation-selected checkpoints; no held-out-fold early stopping.',
        'heldout_validation20_used_in_this_analysis':False,'test21_used':False,'main_models_changed':False,
        'normal_reference_imputer_scaler_refitted_inside_each_fold':True,'augmentation_parent_fold_exclusivity':True,
        'limitations':['Configurations and PCD were previously selected using the main holdout protocol; this is descriptive conditional CV, not nested CV or an unbiased new test.',
            'Folds are stratified per photo and do not establish independent physical bottles.']}
    save_json(out/'cv_plan.json',plan);rows=[];oof=[];history_rows=[];fold_records=[]
    splitter=StratifiedKFold(n_splits=folds,shuffle=True,random_state=SEED)
    for fold,(tr,va) in enumerate(splitter.split(np.zeros(len(train)),train.class_id),1):
        start=time.perf_counter();tr_meta=train.iloc[tr].reset_index(drop=True);va_meta=train.iloc[va].reset_index(drop=True)
        reference=reference_fit(raw.iloc[tr].reset_index(drop=True),tr_meta)
        xtr=reference_apply(raw.iloc[tr].reset_index(drop=True),reference);xva=reference_apply(raw.iloc[va].reset_index(drop=True),reference)
        record={'fold':fold,'train_image_ids':tr_meta.image_id.tolist(),'validation_image_ids':va_meta.image_id.tolist(),
            'train_class_counts':tr_meta.class_name.value_counts().to_dict(),'validation_class_counts':va_meta.class_name.value_counts().to_dict(),
            'normal_reference':reference}
        for name,cfg in ml['models'].items():
            model=fit_model(name,estimator(name,cfg['parameters']),xtr,tr_meta);pred=predict_model(name,model,xva)
            m=metrics(va_meta.class_name,pred);rows.append({'fold':fold,'model':name,'n_train':len(tr),'n_validation':len(va),'macro_f1':m['macro_f1'],'accuracy':m['accuracy']})
            joblib.dump(model,out/'models'/f'fold{fold:02d}_{name}.joblib')
            for r,p in zip(va_meta.to_dict('records'),pred):oof.append({'fold':fold,'model':name,'image_id':r['image_id'],'source_filename':r['source_filename'],'true_label':r['class_name'],'predicted_label':str(p)})
        train_views=np.flatnonzero(lineage.source_index.isin(tr).to_numpy())
        val_views=np.array([original_positions[int(i)] for i in va])
        assert not set(lineage.iloc[train_views].image_id)&set(lineage.iloc[val_views].image_id)
        record.update(dl_train_views=len(train_views),dl_validation_originals=len(val_views))
        for branch,(x,y) in embeddings.items():
            head,hist=fit_fixed_head(x[train_views],y[train_views],dl['parameters'],device,fixed_epochs,dl['batch_size'],SEED+fold)
            with torch.inference_mode():pred=head(x[val_views].to(device)).argmax(1).cpu().numpy()
            name='dl_'+branch;m=metrics(va_meta.class_id.to_numpy(),pred)
            rows.append({'fold':fold,'model':name,'n_train':len(tr),'n_validation':len(va),'macro_f1':m['macro_f1'],'accuracy':m['accuracy']})
            history_rows += [{'fold':fold,'model':name,**h} for h in hist]
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in head.state_dict().items()},'parameters':dl['parameters'],
                'epochs':fixed_epochs,'seed':SEED+fold,'class_order':CLASSES},out/'models'/f'fold{fold:02d}_{name}_head.pt')
            for r,p in zip(va_meta.to_dict('records'),pred):oof.append({'fold':fold,'model':name,'image_id':r['image_id'],'source_filename':r['source_filename'],'true_label':r['class_name'],'predicted_label':CLASSES[int(p)]})
        fold_records.append(record)
        print(f'CV fold {fold}/{folds} completed in {time.perf_counter()-start:.1f}s',flush=True)
    scores=pd.DataFrame(rows);oof_frame=pd.DataFrame(oof)
    scores.to_csv(out/'fold_metrics.csv',index=False);oof_frame.to_csv(out/'out_of_fold_predictions.csv',index=False)
    pd.DataFrame(history_rows).to_csv(out/'dl_fold_training_history.csv',index=False);save_json(out/'fold_assignments.json',fold_records)
    summary=[]
    for name,f in scores.groupby('model',sort=False):
        predictions=oof_frame[oof_frame.model==name]
        if len(predictions)!=len(train) or predictions.image_id.duplicated().any():raise ValueError('OOF coverage failure')
        m=metrics(predictions.true_label,predictions.predicted_label)
        summary.append({'model':name,'folds':folds,'macro_f1_mean':float(f.macro_f1.mean()),'macro_f1_std':float(f.macro_f1.std(ddof=1)),
            'accuracy_mean':float(f.accuracy.mean()),'accuracy_std':float(f.accuracy.std(ddof=1)),
            'oof_macro_f1':m['macro_f1'],'oof_accuracy':m['accuracy'],'oof_photos':len(predictions)})
    table=pd.DataFrame(summary);table.to_csv(out/'summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5));ax.bar(table.model,table.macro_f1_mean,yerr=table.macro_f1_std,capsize=5,color='#3274a1')
    ax.set(title=f'{folds}-fold stability check on {len(train)} train photos',ylabel='Macro-F1 mean ± fold std',ylim=(0,1));fig.tight_layout()
    fig.savefig(out/'cv_macro_f1.png',dpi=150);plt.close(fig)
    text=['# Stratified K-Fold tambahan','',f'{folds} fold pada {len(train)} foto train asli. Validation 20 dan test 21 tidak dimasukkan.',
        'Konfigurasi tetap mengikuti model utama; tidak ada tuning ulang dan tidak ada perubahan hasil test utama.',
        'Referensi normal, imputer, scaler, dan classifier dilatih ulang di dalam setiap fold. Augmentasi DL hanya berasal dari sumber training fold.',
        f'DL dilatih selama {fixed_epochs} epoch tetap. Bagian validasi fold tidak dipakai untuk early stopping atau pemilihan epoch.',
        '', '| Model | Macro-F1 mean | Std antar-fold | OOF accuracy |','|---|---:|---:|---:|']
    for r in summary:text.append(f'| {r["model"]} | {r["macro_f1_mean"]:.4f} | {r["macro_f1_std"]:.4f} | {r["oof_accuracy"]:.2%} |')
    text += ['', 'Std antar-fold bukan confidence interval. Karena konfigurasi sudah dipilih melalui protokol holdout utama, ini analisis kestabilan bersyarat dan bukan nested cross-validation. Folds per foto tidak menjamin botol fisik independen.','']
    (out/'README.md').write_text('\n'.join(text));print(table.to_string(index=False),flush=True);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--results',required=True)
    p.add_argument('--device',default='auto');p.add_argument('--folds',type=int,default=5)
    a=p.parse_args();run(a.data,a.results,a.device,a.folds)
