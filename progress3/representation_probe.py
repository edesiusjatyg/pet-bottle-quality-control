"""Validation-only diagnostic: hold classifier family fixed, change representation."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from common import SEED,read_split,identity,load_json,metrics,save_json

def run_probe(root,output):
    output=Path(output);cfg=load_json(output/'dl_selected_config.json')
    if cfg['dataset']!=identity(root):raise ValueError('Dataset mismatch')
    train,val=read_split(root,'train'),read_split(root,'val')
    lineage=pd.read_csv(output/'dl_augmentation_lineage.csv')
    originals=np.flatnonzero(lineage.augmentation_index.to_numpy()==0)
    assert len(originals)==len(train)
    assert lineage.iloc[originals].image_id.tolist()==train.image_id.tolist()
    rows=[];summary={}
    for branch in ['raw','pcd']:
        a=np.load(output/'cache'/f'{branch}_train.npz',allow_pickle=False)
        b=np.load(output/'cache'/f'{branch}_val.npz',allow_pickle=False)
        x,y=a['features'][originals],a['labels'][originals];xv,yv=b['features'],b['labels']
        assert np.array_equal(y,train.class_id.to_numpy()) and np.array_equal(yv,val.class_id.to_numpy())
        best=None
        for c in [.01,.1,1,10]:
            model=make_pipeline(StandardScaler(),SVC(kernel='linear',C=c,class_weight='balanced',random_state=SEED))
            model.fit(x,y);pred=model.predict(xv);m=metrics(yv,pred)
            rows.append({'representation':'mobilenet_'+branch,'classifier':'linear_svm','C':c,
                'train_originals':len(train),'dimensions':x.shape[1],'macro_f1':m['macro_f1'],'accuracy':m['accuracy']})
            if best is None or (m['macro_f1'],m['accuracy'])>(best['metrics']['macro_f1'],best['metrics']['accuracy']):
                best={'C':c,'metrics':m,'predictions':pred.tolist()}
        summary[branch]=best
    pd.DataFrame(rows).to_csv(output/'representation_probe_trials.csv',index=False)
    result={'purpose':'validation_only_diagnostic','train_originals':len(train),'validation_originals':len(val),
        'no_augmented_samples_used':True,'test_read':False,'principal_models_changed':False,
        'backbone':'same frozen ImageNet MobileNetV3-Large as main DL experiments','feature_dimensions':960,
        'selection':'validation macro-F1 then accuracy; first C on tie','results':summary,
        'caution':'A better score supports a representation bottleneck on this split; it does not establish a causal explanation or independent-bottle generalization.'}
    save_json(output/'representation_probe.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run_probe(a.data,a.output),indent=2))
