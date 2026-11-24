"""Predict a new bottle image using the four selected Progress 3 models."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from common import CLASSES,load_rgb,load_json,digest
from features import extract,reference_apply,select_features
from ml import predict_model
from preprocessing import prepare,letterbox

def predict(image,results,model='all',device='auto'):
    results=Path(results);rgb=load_rgb(image);predictions={}
    if model in ['all','svm','xgboost']:
        cfg=load_json(results/'ml_selected_config.json')
        raw,diagnostics,_=extract(rgb,cfg['pcd_variant'],cfg['glcm_distance']);raw=pd.DataFrame([raw])
        for name,setting in cfg['models'].items():
            if model not in ['all',name]:continue
            path=results/setting['model_path']
            if digest(path)!=setting['model_sha256']:raise ValueError('ML model checksum changed')
            estimator=joblib.load(path);f=reference_apply(raw,setting['reference'])
            predictions[name]={'label':str(predict_model(name,estimator,f)[0]),'pcd_variant':cfg['pcd_variant'],'glcm_distance':cfg['glcm_distance']}
    if model in ['all','dl_raw','dl_pcd']:
        import torch
        from deep_learning import build_model,new_head,choose_device,normalize
        cfg=load_json(results/'dl_selected_config.json');dev=choose_device(device)
        path=results/cfg['backbone_path']
        if digest(path)!=cfg['backbone_sha256']:raise ValueError('Backbone checksum changed')
        backbone=build_model(False);backbone.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));backbone.to(dev).eval()
        for branch,setting in cfg['models'].items():
            name='dl_'+branch
            if model not in ['all',name]:continue
            p=results/setting['head_path']
            if digest(p)!=setting['head_sha256']:raise ValueError('DL head checksum changed')
            saved=torch.load(p,map_location='cpu',weights_only=True)
            head=new_head(saved['parameters']['dropout']);head.load_state_dict(saved['state_dict']);head.to(dev).eval()
            image_rgb=prepare(rgb,cfg['pcd_variant'])[0] if branch=='pcd' else rgb
            tensor=normalize(np.stack([letterbox(image_rgb)])).to(dev)
            with torch.inference_mode():prob=head(backbone(tensor)).softmax(1).cpu().numpy()[0]
            predictions[name]={'label':CLASSES[int(prob.argmax())],'softmax':dict(zip(CLASSES,map(float,prob)))}
    return predictions

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True);p.add_argument('--results',default=str(Path(__file__).resolve().parent/'results'))
    p.add_argument('--model',choices=['all','svm','xgboost','dl_raw','dl_pcd'],default='all');p.add_argument('--device',default='auto')
    args=p.parse_args();print(json.dumps(predict(args.image,args.results,args.model,args.device),indent=2))

if __name__=='__main__':main()
