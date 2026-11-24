"""One entry point for local, CUDA/ROCm, MPS, and Kaggle retraining."""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,platform,subprocess,sys,time,traceback
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
import pandas as pd,numpy as np
from common import dataset_root,save_json,load_json,identity,read_split,digest,CLASSES

def environment():
    packages={}
    for name in ['numpy','pandas','Pillow','opencv-python-headless','scikit-learn','xgboost','torch','torchvision','matplotlib','joblib']:
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]='not installed'
    import torch
    return {'python':platform.python_version(),'platform':platform.platform(),'logical_cpus_available':os.cpu_count(),
        'cuda_available':torch.cuda.is_available(),'cuda_version':torch.version.cuda,'mps_available':torch.backends.mps.is_available(),
        'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'packages':packages}

def source_fingerprints():
    root=Path(__file__).resolve().parent
    return {p.name:digest(p) for p in sorted(root.glob('*.py'))}

def lock_selection(root,output):
    output=Path(output)
    ml=load_json(output/'ml_selected_config.json');dl=load_json(output/'dl_selected_config.json')
    if ml['dataset']!=identity(root) or dl['dataset']!=identity(root):raise ValueError('Dataset changed before lock')
    lock={'locked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset':identity(root),
        'ml_config_sha256':digest(output/'ml_selected_config.json'),'dl_config_sha256':digest(output/'dl_selected_config.json'),
        'code_sha256':source_fingerprints(),'test_used_for_selection':False}
    save_json(output/'selection_lock.json',lock);return lock

def validate_lock(root,output):
    output=Path(output);lock=load_json(output/'selection_lock.json')
    if lock['dataset']!=identity(root):raise ValueError('Dataset changed after locking')
    for field,path in [('ml_config_sha256','ml_selected_config.json'),('dl_config_sha256','dl_selected_config.json')]:
        if digest(output/path)!=lock[field]:raise ValueError(f'{path} changed after locking')
    if source_fingerprints()!=lock['code_sha256']:raise ValueError('Experiment source changed after locking; create a new run.')
    return lock

def evaluate(root,output,device):
    output=Path(output)
    if (output/'test_evaluation_receipt.json').exists():raise RuntimeError('This run already evaluated test. Use the existing reports; retraining needs a new output directory.')
    validate_lock(root,output)
    import ml,deep_learning
    summaries=ml.evaluate_final(root,output)+deep_learning.evaluate_final(root,output,device)
    save_json(output/'test_evaluation_receipt.json',{'evaluated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'dataset':identity(root),'selection_lock_sha256':digest(output/'selection_lock.json'),'model_count':len(summaries)})
    save_json(output/'test_summary.json',summaries)
    create_summary(output)
    return summaries

def create_summary(output):
    output=Path(output);rows=[]
    for name in ['svm','xgboost','dl_raw','dl_pcd']:
        for split in ['train','val','test']:
            path=output/f'{name}_{split}_metrics.json'
            if path.exists():
                m=load_json(path);rows.append({'model':name,'split':split,'n':m['n'],'macro_f1':m['macro_f1'],'accuracy':m['accuracy']})
    table=pd.DataFrame(rows);table.to_csv(output/'comparison.csv',index=False)
    effect={}
    for split in ['val','test']:
        a=table[(table.model=='dl_raw')&(table.split==split)];b=table[(table.model=='dl_pcd')&(table.split==split)]
        if len(a) and len(b):effect[split]={'delta_macro_f1_pcd_minus_raw':float(b.iloc[0].macro_f1-a.iloc[0].macro_f1)}
    save_json(output/'pcd_effect.json',effect)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    p=table[table.split!='train'].pivot(index='model',columns='split',values='macro_f1')
    p.plot.bar(ax=ax,ylim=(0,1),rot=0);ax.set(title='Progress 3 — macro-F1',ylabel='Macro-F1',xlabel='Model')
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    history=output/'dl_training_history.csv'
    if history.exists():
        frame=pd.read_csv(history);cfg=load_json(output/'dl_selected_config.json');frame=frame[frame.trial==cfg['selected_trial']]
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for branch,group in frame.groupby('branch'):
            axes[0].plot(group.epoch,group.train_loss,label=branch+' train');axes[0].plot(group.epoch,group.val_loss,'--',label=branch+' val')
            axes[1].plot(group.epoch,group.val_macro_f1,label=branch)
        axes[0].set(title='Classifier training loss',xlabel='Epoch',ylabel='Cross-entropy');axes[1].set(title='Validation macro-F1',xlabel='Epoch',ylim=(0,1))
        for ax in axes:ax.legend()
        fig.tight_layout();fig.savefig(output/'selected_dl_learning_curves.png',dpi=160);plt.close(fig)
    if (output/'checkpoint_verification.json').exists():
        from make_report import make_report
        data=load_json(output/'command.json')['resolved_dataset']
        if Path(data).exists():make_report(data,output)
    return table

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['all','ml','dl','explain','lock','test','report'],nargs='?',default='all')
    parser.add_argument('--data',help='Directory containing metadata.csv and splits/')
    parser.add_argument('--output',help='Fresh output directory. Default: runs/<timestamp>')
    parser.add_argument('--device',default='auto',help='auto, cpu, cuda, cuda:0, or mps')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--augmentation-copies',type=int,default=3)
    parser.add_argument('--glcm-distances',type=int,nargs='+',default=[1,2,4],help='Candidate GLCM distances d, in pixels of the prepared ROI. Example: --glcm-distances 1 2 3 4')
    parser.add_argument('--importance-repeats',type=int,default=20,help='Validation permutation-importance repetitions')
    parser.add_argument('--backbone',help='Optional local backbone_imagenet_v2.pt to avoid downloading weights')
    parser.add_argument('--evaluate-test',action='store_true',help='Lock selected configurations and evaluate the reserved test once after tuning.')
    args=parser.parse_args()
    if min(args.epochs,args.patience,args.batch_size)<=0 or args.augmentation_copies<0:parser.error('Epochs, patience and batch size must be positive; augmentation copies cannot be negative.')
    root=dataset_root(args.data)
    output=Path(args.output or Path('runs')/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    if args.stage in {'all','ml'} and output.exists() and (output/'ml_selected_config.json').exists():
        raise SystemExit('Output already contains an ML experiment; choose a fresh --output directory.')
    if args.stage=='dl' and (output/'dl_selected_config.json').exists():
        raise SystemExit('Output already contains a DL experiment; choose a fresh experiment and run ML first.')
    output.mkdir(parents=True,exist_ok=True)
    # One executable log is retained even when a failure stops the run.
    class Tee:
        def __init__(self,stream,file):self.stream,self.file=stream,file
        def write(self,text):self.stream.write(text);self.file.write(text);self.file.flush()
        def flush(self):self.stream.flush();self.file.flush()
    with (output/'run.log').open('a',buffering=1) as logfile:
        original_out,original_err=sys.stdout,sys.stderr;sys.stdout=Tee(sys.stdout,logfile);sys.stderr=Tee(sys.stderr,logfile)
        started=time.perf_counter()
        try:
            print('Dataset:',root,'\nOutput:',output,flush=True)
            env=environment();command={'argv':sys.argv,'options':vars(args),'resolved_dataset':str(root),'resolved_output':str(output)}
            save_json(output/f'environment_{args.stage}.json',env)
            save_json(output/f'command_{args.stage}.json',command)
            if args.stage in {'all','ml'} or not (output/'environment.json').exists():save_json(output/'environment.json',env)
            if args.stage in {'all','ml'} or not (output/'command.json').exists():save_json(output/'command.json',command)
            if args.stage in {'all','ml'}:
                from common import audit
                import ml
                save_json(output/'dataset_audit.json',audit(root));ml.run_tuning(root,output,args.glcm_distances)
            if args.stage in {'all','dl'}:
                import deep_learning
                backbone=args.backbone
                bundled=Path(__file__).resolve().parent/'results/models/backbone_imagenet_v2.pt'
                if not backbone and bundled.is_file():backbone=str(bundled)
                deep_learning.run_tuning(root,output,args.device,args.epochs,args.patience,args.batch_size,args.augmentation_copies,backbone)
                from verify import verify
                from representation_probe import run_probe
                verify(root,output,args.device)
                run_probe(root,output)
            if args.stage in {'all','dl','explain'}:
                from explain import run_explanations
                run_explanations(root,output,args.device,args.importance_repeats)
            if args.stage=='lock' or (args.stage=='all' and args.evaluate_test):lock_selection(root,output)
            if args.stage=='test' or (args.stage=='all' and args.evaluate_test):evaluate(root,output,args.device)
            if args.stage in {'all','dl','explain','report'}:print(create_summary(output).to_string(index=False),flush=True)
            status={'status':'completed','stage':args.stage,'elapsed_seconds':time.perf_counter()-started,
                'test_evaluated':(output/'test_evaluation_receipt.json').exists()}
            save_json(output/'run_status.json',status);save_json(output/f'run_status_{args.stage}.json',status)
        except Exception as exc:
            save_json(output/'run_status.json',{'status':'failed','stage':args.stage,'error':str(exc),'elapsed_seconds':time.perf_counter()-started})
            traceback.print_exc();raise
        finally:sys.stdout,sys.stderr=original_out,original_err

if __name__=='__main__':main()
