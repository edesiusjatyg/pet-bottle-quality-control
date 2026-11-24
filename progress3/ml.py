"""Train-only fitting, bounded validation search, fixed test evaluation."""
from __future__ import annotations
import json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from common import CLASSES,SEED,identity,read_split,load_rgb,metrics,save_json,load_json,report_predictions,digest
from preprocessing import VARIANTS
from features import extract,reference_fit,reference_apply,select_features,FEATURE_COLUMNS

def feature_table(root,frame,variant,output=None,split=None,glcm_distance=1):
    rows=[];diagnostics=[]
    for row in frame.to_dict('records'):
        f,d,_=extract(load_rgb(Path(root)/row['image_path']),variant,glcm_distance)
        rows.append(f);diagnostics.append({'image_id':row['image_id'],'source_filename':row['source_filename'],**d})
    f=pd.DataFrame(rows)
    if output is not None:
        p=Path(output)/'features';p.mkdir(exist_ok=True,parents=True)
        pd.concat([frame[['image_id','source_filename','class_name']].reset_index(drop=True),f],axis=1).to_csv(p/f'{variant}_d{glcm_distance}_{split}_measurements.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(p/f'{variant}_d{glcm_distance}_{split}_diagnostics.csv',index=False)
    return f

def estimator(name,params):
    # Empty train columns are retained explicitly; missingness indicators are train-fitted.
    imputer=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)
    if name=='svm':
        return Pipeline([('imputer',imputer),('scaler',StandardScaler()),
            ('classifier',SVC(class_weight='balanced',random_state=SEED,**params))])
    return Pipeline([('imputer',imputer),('classifier',XGBClassifier(objective='multi:softprob',num_class=4,
        eval_metric='mlogloss',tree_method='hist',n_jobs=2,random_state=SEED,subsample=.85,
        colsample_bytree=.85,reg_lambda=3,min_child_weight=1,verbosity=0,**params))])

def grids():
    svm=[{'kernel':'linear','C':c} for c in [.01,.1,1,10]]
    svm += [{'kernel':'rbf','C':c,'gamma':g} for c in [.1,1,10] for g in ['scale',.01]]
    xgb=[{'n_estimators':n,'max_depth':d,'learning_rate':lr} for n in [80,160] for d in [2,3] for lr in [.03,.1]]
    return {'svm':svm,'xgboost':xgb}

def fit_model(name,model,x,frame):
    y=frame['class_name'] if name=='svm' else frame['class_id'].to_numpy()
    kw={} if name=='svm' else {'classifier__sample_weight':compute_sample_weight('balanced',y)}
    model.fit(select_features(x),y,**kw)
    return model

def predict_model(name,model,x):
    y=model.predict(select_features(x))
    return np.array(CLASSES)[np.asarray(y,dtype=int)] if name=='xgboost' else np.asarray(y)

def run_tuning(root,output,glcm_distances=(1,2,4)):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'models').mkdir(exist_ok=True);train=read_split(root,'train');val=read_split(root,'val')
    trials=[];best={};pcd_scores=[];cache={}
    glcm_distances=sorted(set(int(d) for d in glcm_distances))
    if not glcm_distances or min(glcm_distances)<1:raise ValueError('GLCM distances must be positive integers')
    save_json(output/'ml_search_plan.json',{'dataset':identity(root),'variants':VARIANTS,'glcm_distances':glcm_distances,'grids':grids(),
        'selection':'Mean of best SVM and best XGBoost validation macro-F1 chooses one shared PCD variant.',
        'tie_break':'mean accuracy, then declared variant order, then smaller d; within a family first grid entry on exact score ties',
        'training_originals':len(train),'offline_augmentation':False,'feature_columns':FEATURE_COLUMNS,
        'no_test_predictions_during_tuning':True})
    for variant,distance in [(v,d) for v in VARIANTS for d in glcm_distances]:
        t0=time.perf_counter()
        tr=feature_table(root,train,variant,output,'train',distance);va=feature_table(root,val,variant,output,'val',distance)
        reference=reference_fit(tr,train);tr=reference_apply(tr,reference);va=reference_apply(va,reference)
        cache[(variant,distance)]=(tr,va,reference)
        tr.to_csv(output/'features'/f'{variant}_d{distance}_train_model_features.csv',index=False)
        va.to_csv(output/'features'/f'{variant}_d{distance}_val_model_features.csv',index=False)
        for name,grid in grids().items():
            winner=None
            for index,params in enumerate(grid):
                start=time.perf_counter();model=fit_model(name,estimator(name,params),tr,train)
                pred=predict_model(name,model,va);m=metrics(val['class_name'],pred)
                row={'model':name,'variant':variant,'glcm_distance':distance,'trial':index,'parameters':json.dumps(params,sort_keys=True),
                    'macro_f1':m['macro_f1'],'accuracy':m['accuracy'],'fit_predict_seconds':time.perf_counter()-start}
                trials.append(row)
                if winner is None or (m['macro_f1'],m['accuracy'])>(winner['metrics']['macro_f1'],winner['metrics']['accuracy']):
                    winner={'metrics':m,'model':model,'parameters':params,'predictions':pred,'reference':reference,'variant':variant}
            best[(variant,distance,name)]=winner
        avg_f1=np.mean([best[(variant,distance,n)]['metrics']['macro_f1'] for n in grids()])
        avg_acc=np.mean([best[(variant,distance,n)]['metrics']['accuracy'] for n in grids()])
        pcd_scores.append({'variant':variant,'glcm_distance':distance,'mean_macro_f1':avg_f1,'mean_accuracy':avg_acc,'seconds':time.perf_counter()-t0})
        pd.DataFrame(trials).to_csv(output/'ml_tuning_trials.csv',index=False)
        print(f'ML {variant}, d={distance}: paired validation macro-F1={avg_f1:.3f}',flush=True)
    selected=max(pcd_scores,key=lambda x:(x['mean_macro_f1'],x['mean_accuracy'],-VARIANTS.index(x['variant']),-x['glcm_distance']))
    chosen_variant=selected['variant'];chosen_distance=selected['glcm_distance']
    config={'dataset':identity(root),'class_order':CLASSES,'seed':SEED,'pcd_variant':chosen_variant,
        'glcm_distance':chosen_distance,'features':FEATURE_COLUMNS,'models':{},'test_evaluated':False,'training_samples':len(train)}
    for name in grids():
        w=best[(chosen_variant,chosen_distance,name)];tr,va,reference=cache[(chosen_variant,chosen_distance)]
        filename=f'{name}.joblib';joblib.dump(w['model'],output/'models'/filename)
        config['models'][name]={'parameters':w['parameters'],'reference':reference,'model_path':'models/'+filename,
            'model_sha256':digest(output/'models'/filename),'validation':w['metrics']}
        report_predictions(output,name,'val',val,w['predictions'])
        report_predictions(output,name,'train',train,predict_model(name,w['model'],tr))
    save_json(output/'pcd_validation_selection.json',pcd_scores)
    save_json(output/'ml_selected_config.json',config)
    # A same-split, same-feature untuned reference makes the benefit of tuning auditable.
    baseline_params={'svm':{'kernel':'linear','C':.1},'xgboost':{'n_estimators':80,'max_depth':2,'learning_rate':.05}}
    baseline_distance=1 if 1 in glcm_distances else glcm_distances[0]
    tr,va,_=cache[('center_gray',baseline_distance)];baseline=[]
    for name,params in baseline_params.items():
        model=fit_model(name,estimator(name,params),tr,train);m=metrics(val['class_name'],predict_model(name,model,va))
        baseline.append({'model':name,'variant':'center_gray','glcm_distance':baseline_distance,'parameters':params,'metrics':m})
    save_json(output/'untuned_ml_validation_reference.json',baseline)
    return config

def evaluate_final(root,output):
    root,output=Path(root),Path(output);config=load_json(output/'ml_selected_config.json')
    if config['dataset']!=identity(root):raise ValueError('Dataset fingerprint changed after selection')
    test=read_split(root,'test');raw=feature_table(root,test,config['pcd_variant'],output,'test',config['glcm_distance']);summary=[]
    for name,c in config['models'].items():
        if digest(output/c['model_path'])!=c['model_sha256']:raise ValueError('Model bytes changed')
        x=reference_apply(raw,c['reference']);model=joblib.load(output/c['model_path'])
        summary.append({'model':name,**report_predictions(output,name,'test',test,predict_model(name,model,x))})
    return summary
