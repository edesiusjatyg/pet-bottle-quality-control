"""Paired MobileNetV3-Large transfer learning with reusable frozen features."""
from __future__ import annotations
import copy,json,random,time
from pathlib import Path
import numpy as np,pandas as pd
from PIL import Image,ImageEnhance
import torch
from torch import nn
from torchvision.models import mobilenet_v3_large,MobileNet_V3_Large_Weights
from common import CLASSES,SEED,read_split,load_rgb,identity,save_json,load_json,report_predictions,metrics,digest
from preprocessing import prepare,letterbox

MEAN=torch.tensor([.485,.456,.406]).view(1,3,1,1)
STD=torch.tensor([.229,.224,.225]).view(1,3,1,1)

def choose_device(request='auto'):
    torch.set_num_threads(min(4,max(1,torch.get_num_threads())))
    if request=='auto':
        if torch.cuda.is_available():return torch.device('cuda')
        if torch.backends.mps.is_available():return torch.device('mps')
        return torch.device('cpu')
    if request.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA not available; install the GPU PyTorch build first.')
    if request=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable.')
    return torch.device(request)

def seed_all(seed=SEED):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def normalize(batch):
    # Letterbox replaces the weights' center crop to preserve cap and bottle base.
    x=torch.as_tensor(np.asarray(batch).copy(),dtype=torch.float32).permute(0,3,1,2)/255
    return (x-MEAN)/STD

def build_model(pretrained=True):
    model=mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
    model.classifier=nn.Identity()
    return model

def get_backbone(output,device,backbone_source=None):
    path=Path(output)/'models'/'backbone_imagenet_v2.pt';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        model=build_model(False);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    elif backbone_source and Path(backbone_source).is_file():
        model=build_model(False);model.load_state_dict(torch.load(backbone_source,map_location='cpu',weights_only=True))
        torch.save(model.state_dict(),path)
    else:
        model=build_model(True);torch.save(model.state_dict(),path)
    for p in model.parameters():p.requires_grad=False
    return model.to(device).eval(),path

def augmentation_plan(frame,copies=3,seed=SEED):
    rng=np.random.default_rng(seed);rows=[]
    for i,row in enumerate(frame.to_dict('records')):
        for aug in range(copies+1):
            rows.append({'source_index':i,'image_id':row['image_id'],'augmentation_index':aug,
                'brightness':1. if aug==0 else float(rng.uniform(.85,1.15)),
                'contrast':1. if aug==0 else float(rng.uniform(.90,1.10))})
    return rows

def cache_features(root,frame,branch,pcd_variant,plan,backbone,device,batch_size,cache_path):
    cached=Path(cache_path)
    source_key={'dataset':identity(root),'image_ids':frame['image_id'].tolist(),'branch':branch,
        'pcd_variant':pcd_variant,'plan':plan,'normalization':'imagenet_rgb_0_1_mean_std','size':224,
        'backbone_sha256':digest(Path(cache_path).parents[1]/'models'/'backbone_imagenet_v2.pt')}
    key=json.dumps(source_key,sort_keys=True)
    if cached.exists():
        data=np.load(cached,allow_pickle=False)
        if str(data['key'].item())==key:
            return torch.tensor(data['features'],dtype=torch.float32),torch.tensor(data['labels'],dtype=torch.long)
        raise ValueError('Feature cache belongs to another configuration; use a fresh output directory.')
    base=[]
    for r in frame.to_dict('records'):
        image=load_rgb(Path(root)/r['image_path'])
        if branch=='pcd':image,_=prepare(image,pcd_variant)
        base.append(letterbox(image))
    outputs=[];targets=[];batch=[]
    def flush():
        if not batch:return
        with torch.inference_mode():outputs.append(backbone(normalize(np.stack(batch)).to(device)).cpu())
        batch.clear()
    for record in plan:
        im=Image.fromarray(base[record['source_index']])
        im=ImageEnhance.Brightness(im).enhance(record['brightness'])
        im=ImageEnhance.Contrast(im).enhance(record['contrast'])
        batch.append(np.asarray(im).copy());targets.append(int(frame.iloc[record['source_index']]['class_id']))
        if len(batch)>=batch_size:flush()
    flush();x=torch.cat(outputs);y=torch.tensor(targets,dtype=torch.long)
    cached.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cached,features=x.numpy(),labels=y.numpy(),key=np.array(key))
    return x,y

def new_head(dropout):return nn.Sequential(nn.Dropout(dropout),nn.Linear(960,len(CLASSES)))

def train_head(x,y,xv,yv,params,device,epochs,patience,batch_size,seed=SEED):
    seed_all(seed);head=new_head(params['dropout']).to(device)
    optimizer=torch.optim.Adam(head.parameters(),lr=params['lr'],weight_decay=params['weight_decay'])
    counts=torch.bincount(y,minlength=4).float();weights=y.numel()/(4*counts)
    criterion=nn.CrossEntropyLoss(weight=weights.to(device));history=[];best=None;stale=0
    generator=torch.Generator().manual_seed(seed)
    for epoch in range(1,epochs+1):
        head.train();order=torch.randperm(len(y),generator=generator);total=0.
        for indices in order.split(batch_size):
            xb=x[indices].to(device);yb=y[indices].to(device)
            optimizer.zero_grad();loss=criterion(head(xb),yb);loss.backward();optimizer.step();total+=float(loss.detach())*len(indices)
        head.eval()
        with torch.inference_mode():
            logits=head(xv.to(device));val_loss=float(nn.functional.cross_entropy(logits,yv.to(device)))
            pred=logits.argmax(1).cpu().numpy()
        m=metrics(yv.numpy(),pred);history.append({'epoch':epoch,'train_loss':total/len(y),'val_loss':val_loss,
            'val_macro_f1':m['macro_f1'],'val_accuracy':m['accuracy']})
        # This stopping rule is shared by B and C and declared before test evaluation.
        score=(m['macro_f1'],-val_loss)
        if best is None or score>best['score']:
            best={'score':score,'state':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
                'epoch':epoch,'metrics':m,'predictions':pred.tolist(),'val_loss':val_loss};stale=0
        else:stale+=1
        if stale>=patience:break
    return best,history

def tuning_grid():
    return [{'lr':lr,'dropout':d,'weight_decay':wd} for lr in [.001,.003] for d in [.2,.4] for wd in [0.,.001]]

def run_tuning(root,output,device_request='auto',epochs=20,patience=5,batch_size=32,augmentation_copies=3,backbone_source=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);(output/'models').mkdir(exist_ok=True)
    device=choose_device(device_request);torch.set_num_threads(min(4,max(1,torch.get_num_threads())))
    seed_all();train=read_split(root,'train');val=read_split(root,'val')
    pcd_variant=load_json(output/'ml_selected_config.json')['pcd_variant']
    backbone,backbone_path=get_backbone(output,device,backbone_source)
    tr_plan=augmentation_plan(train,augmentation_copies);va_plan=augmentation_plan(val,0)
    pd.DataFrame(tr_plan).to_csv(output/'dl_augmentation_lineage.csv',index=False)
    plan={'dataset':identity(root),'grid':tuning_grid(),'backbone':'torchvision MobileNetV3-Large IMAGENET1K_V2',
        'training':'frozen_backbone_cached_features','device':str(device),'epochs':epochs,'patience':patience,
        'batch_size':batch_size,'augmentation_copies':augmentation_copies,'normalization':'ImageNet mean/std after /255',
        'seed':SEED,'pcd_variant':pcd_variant,'selection':'One common hyperparameter set maximizes mean validation macro-F1 across raw and pcd.',
        'checkpoint_rule':'max validation macro-F1, then min validation cross-entropy; early stop on no improvement',
        'fine_tuning_backbone':False,'test_used':False}
    save_json(output/'dl_search_plan.json',plan)
    data={}
    for branch in ['raw','pcd']:
        print(f'Caching MobileNet features: {branch}, {len(tr_plan)} train views + {len(va_plan)} val images on {device}',flush=True)
        x,y=cache_features(root,train,branch,pcd_variant,tr_plan,backbone,device,batch_size,output/'cache'/f'{branch}_train.npz')
        xv,yv=cache_features(root,val,branch,pcd_variant,va_plan,backbone,device,batch_size,output/'cache'/f'{branch}_val.npz')
        data[branch]=(x,y,xv,yv)
    del backbone
    trials=[];histories=[];candidates=[]
    for trial,params in enumerate(tuning_grid()):
        pair={};start=time.perf_counter()
        for branch,(x,y,xv,yv) in data.items():
            best,history=train_head(x,y,xv,yv,params,device,epochs,patience,batch_size)
            pair[branch]=best
            trials.append({'trial':trial,'branch':branch,**params,'best_epoch':best['epoch'],
                'macro_f1':best['metrics']['macro_f1'],'accuracy':best['metrics']['accuracy'],'val_loss':best['val_loss']})
            histories += [{'trial':trial,'branch':branch,**h} for h in history]
        mean_f1=float(np.mean([b['metrics']['macro_f1'] for b in pair.values()]))
        mean_loss=float(np.mean([b['val_loss'] for b in pair.values()]))
        candidates.append({'trial':trial,'parameters':params,'pair':pair,'mean_macro_f1':mean_f1,'mean_val_loss':mean_loss})
        pd.DataFrame(trials).to_csv(output/'dl_tuning_trials.csv',index=False)
        pd.DataFrame(histories).to_csv(output/'dl_training_history.csv',index=False)
        print(f'DL trial {trial+1}/8: {params}; paired val macro-F1={mean_f1:.3f}; {time.perf_counter()-start:.1f}s',flush=True)
    selected=max(candidates,key=lambda c:(c['mean_macro_f1'],-c['mean_val_loss'],-c['trial']))
    config={'dataset':identity(root),'seed':SEED,'class_order':CLASSES,'parameters':selected['parameters'],
        'selected_trial':selected['trial'],'mean_validation_macro_f1':selected['mean_macro_f1'],'pcd_variant':pcd_variant,
        'backbone_path':str(backbone_path.relative_to(output)),'backbone_sha256':digest(backbone_path),
        'training_mode':'frozen_backbone_cached_features','backbone_parameters_updated':False,
        'epochs':epochs,'patience':patience,'batch_size':batch_size,'augmentation_copies':augmentation_copies,
        'device_used':str(device),'models':{},'test_evaluated':False}
    for branch,best in selected['pair'].items():
        path=output/'models'/f'dl_{branch}_head.pt'
        torch.save({'state_dict':best['state'],'parameters':selected['parameters'],'class_order':CLASSES,'best_epoch':best['epoch']},path)
        config['models'][branch]={'head_path':str(path.relative_to(output)),'head_sha256':digest(path),
            'best_epoch':best['epoch'],'validation':best['metrics']}
        report_predictions(output,f'dl_{branch}','val',val,np.array(best['predictions'],dtype=int))
        x,y,xv,yv=data[branch];head=new_head(selected['parameters']['dropout']);head.load_state_dict(best['state']);head.eval()
        original_indices=[i for i,r in enumerate(tr_plan) if r['augmentation_index']==0]
        with torch.inference_mode():pred=head(x[original_indices]).argmax(1).numpy()
        report_predictions(output,f'dl_{branch}','train',train,pred)
    save_json(output/'dl_selected_config.json',config)
    return config

def predict_images(root,frame,branch,output,device_request='auto',batch_size=32):
    output=Path(output);config=load_json(output/'dl_selected_config.json');device=choose_device(device_request)
    if config['dataset']!=identity(root):raise ValueError('Input dataset fingerprints differ from the locked configuration')
    if digest(output/config['backbone_path'])!=config['backbone_sha256']:raise ValueError('Backbone checkpoint changed')
    backbone,_=get_backbone(output,device)
    hpath=output/config['models'][branch]['head_path']
    if digest(hpath)!=config['models'][branch]['head_sha256']:raise ValueError('Head checkpoint changed')
    checkpoint=torch.load(hpath,map_location='cpu',weights_only=True)
    head=new_head(checkpoint['parameters']['dropout']);head.load_state_dict(checkpoint['state_dict']);head.to(device).eval()
    plan=augmentation_plan(frame,0)
    # Prediction computes pixels fresh; it never reuses train or validation cached logits.
    outputs=[];batch=[]
    for i,r in enumerate(frame.to_dict('records')):
        image=load_rgb(Path(root)/r['image_path'])
        if branch=='pcd':image,_=prepare(image,config['pcd_variant'])
        batch.append(letterbox(image))
        if len(batch)>=batch_size or i==len(frame)-1:
            with torch.inference_mode():outputs.append(head(backbone(normalize(np.stack(batch)).to(device))).argmax(1).cpu().numpy())
            batch=[]
    return np.concatenate(outputs)

def evaluate_final(root,output,device_request='auto'):
    frame=read_split(root,'test');summaries=[]
    for branch in ['raw','pcd']:
        pred=predict_images(root,frame,branch,output,device_request)
        summaries.append({'model':'dl_'+branch,**report_predictions(output,'dl_'+branch,'test',frame,pred)})
    return summaries
