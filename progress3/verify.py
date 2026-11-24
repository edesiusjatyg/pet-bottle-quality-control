"""Verify saved models reproduce validation predictions without retraining."""
from pathlib import Path
import argparse,json
import joblib,numpy as np,pandas as pd
from common import read_split,load_json,digest,save_json
from ml import feature_table,predict_model
from features import reference_apply
from deep_learning import predict_images

def verify(root,output,device='auto'):
    output=Path(output);val=read_split(root,'val');cfg=load_json(output/'ml_selected_config.json');checks=[]
    raw=feature_table(root,val,cfg['pcd_variant'],glcm_distance=cfg['glcm_distance'])
    for name,c in cfg['models'].items():
        assert digest(output/c['model_path'])==c['model_sha256']
        model=joblib.load(output/c['model_path']);pred=predict_model(name,model,reference_apply(raw,c['reference']))
        saved=pd.read_csv(output/f'{name}_val_predictions.csv')
        assert saved['image_id'].tolist()==val['image_id'].tolist()
        assert saved['predicted_label'].tolist()==pred.tolist(),name
        checks.append({'model':name,'validation_predictions_reproduced':len(val)})
    classes=load_json(output/'dl_selected_config.json')['class_order']
    for branch in ['raw','pcd']:
        pred=predict_images(root,val,branch,output,device)
        saved=pd.read_csv(output/f'dl_{branch}_val_predictions.csv')
        assert saved['image_id'].tolist()==val['image_id'].tolist()
        assert saved['predicted_label'].tolist()==np.array(classes)[pred].tolist(),branch
        checks.append({'model':'dl_'+branch,'validation_predictions_reproduced':len(val)})
    result={'passed':True,'checks':checks,'test_evaluated_by_verifier':False}
    save_json(output/'checkpoint_verification.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='auto')
    a=p.parse_args();print(json.dumps(verify(a.data,a.output,a.device),indent=2))
