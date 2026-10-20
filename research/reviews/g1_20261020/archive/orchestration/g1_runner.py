"""G1 lifecycle: reuse frozen M0 full-pool predictions; zero fit, bounded replay."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,subprocess,sys,time
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from g1_registry import ROOT,BATCH,FLAGS,sha,write,append,sync
from g1_diagnostic import PRIOR,MISSING,top50,groups_before_labels,evaluate_bands,summarize
HERE=Path(__file__).resolve().parent


def verify():
    f=json.loads((BATCH/'FROZEN_IDENTITY.json').read_text())
    for p,h in {**f['code_by_file'],**f['controls']}.items():
        if sha(p)!=h:raise ValueError('frozen identity drift: '+p)
    binding=json.loads((BATCH/'INPUT_BINDING.json').read_text())
    for d in binding['cache_files'].values():
        st=Path(d['path']).stat()
        if st.st_size!=d['bytes'] or st.st_mtime_ns!=d['mtime_ns']:raise ValueError('FAST_BIND cache stat changed')
    return binding


def load_frozen(binding):
    directory=Path(binding['frozen_m0_source_dir']);sys.path.insert(0,str(directory))
    spec=importlib.util.spec_from_file_location('g1_exact_m0',directory/'model_selection.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    m.FROZEN_ROUTE_DEPENDENCY_ROOT=directory
    if m.code_identity()!=binding['m0_code_identity']:raise ValueError('M0 source identity conflict')
    def forbidden_fit(*args,**kwargs):raise RuntimeError('G1_ZERO_FIT_GUARD: fitting prohibited')
    m.HistGradientBoostingRegressor.fit=forbidden_fit
    return m


def lifecycle(execution,stage,operation,truth='NOT_READ'):
    out=BATCH/execution;out.mkdir(exist_ok=False);receipt=out/'RUN_RECEIPT.json'
    write(receipt,dict(status='REGISTER_BEFORE_EXECUTION',fit_calls=0,**FLAGS));append(execution,'RUN_STARTED',stage,receipt,truth=truth,metrics={'fit_calls':0})
    start=time.monotonic()
    try:
        result=operation(out);write(receipt,{'status':'COMPLETED','elapsed_seconds':time.monotonic()-start,'fit_calls':0,**result,**FLAGS})
        append(execution,'TERMINAL',stage,receipt,decision='COMPLETED',truth=truth,metrics={'fit_calls':0,'decision':result.get('decision'),'new_predict_calls':result.get('new_predict_calls',0)})
        return result
    except BaseException as e:
        write(receipt,dict(status='CANNOT_CONFIRM',error=repr(e),elapsed_seconds=time.monotonic()-start,fit_calls=0,**FLAGS))
        append(execution,'TERMINAL',stage,receipt,decision='CANNOT_CONFIRM',truth=truth,metrics={'fit_calls':0});raise
    finally:sync()


def validate_checkpoint(binding,m,fold):
    d=binding['checkpoints'][fold];p=Path(d['path'])
    if sha(p)!=d['sha256'] or sha(str(p)+'.json')!=d['metadata_sha256']:raise ValueError('checkpoint SHA conflict')
    meta=json.loads(Path(str(p)+'.json').read_text())
    if meta['identity']['horizon']!=2 or meta['identity']['arm']!='M0' or meta['identity']['code_sha256']!=binding['m0_code_identity']:raise ValueError('checkpoint target/source conflict')
    payload=joblib.load(p);pred=payload['predictions'].copy();pred.date=pd.to_datetime(pred.date)
    identity=m.key_frame_identity(pred)
    if identity!=meta['identity']['data_identity']['test_keys']:raise ValueError('full-pool stored predictions do not match original test identity')
    if tuple(meta['identity']['data_identity']['feature_columns'])!=m.CORE_FEATURES:raise ValueError('feature order conflict')
    return payload,pred,identity


def replay(binding,m,payload,pred,date,out):
    features=pd.read_parquet(binding['cache_files']['m0']['path'],columns=['date','code',*m.CORE_FEATURES],filters=[('date','==',pd.Timestamp(date))]).sort_values(['date','code']).reset_index(drop=True)
    original=pred.loc[pred.date==pd.Timestamp(date)].reset_index(drop=True)
    if not features[['date','code']].equals(original[['date','code']]):raise ValueError('replay keys mismatch')
    with threadpool_limits(limits=4):scores=payload['model'].predict(features[list(m.CORE_FEATURES)].to_numpy(dtype=float))
    if not np.array_equal(scores,original.prediction.to_numpy()):raise ValueError('sample replay score parity conflict')
    actual=top50(original);reference=pd.read_csv(binding['old_top10']['path'],dtype={'code':str},float_precision='round_trip');reference.date=pd.to_datetime(reference.date)
    reference=reference.loc[(reference.arm=='M0')&(reference.date==pd.Timestamp(date))].sort_values('rank').reset_index(drop=True)
    ours=actual.loc[actual['rank']<=10].reset_index(drop=True)
    if not ours[['date','code','rank']].equals(reference[['date','code','rank']]) or not np.array_equal(ours.prediction,reference.prediction):raise ValueError('saved M0 Top10 parity conflict')
    receipt=dict(date=str(pd.Timestamp(date).date()),rows=len(features),new_predict_calls=1,fit_calls=0,score_array_equal=True,original_Top10_keys_and_scores_exact=True)
    write(out/'REPLAY_RECEIPT.json',receipt);return receipt


def run_canary(binding):
    def op(out):
        m=load_frozen(binding);payload,pred,identity=validate_checkpoint(binding,m,'2024H1');date=pred.date.min()
        rep=replay(binding,m,payload,pred,date,out)
        top=top50(pred.loc[pred.date==date]);prior=pd.read_parquet(binding['cache_files']['s1']['path'],columns=['date','code',PRIOR,MISSING],filters=[('date','==',date)])
        keys,status=groups_before_labels(top,prior);keys.to_parquet(out/'TOP50_GROUP_KEYS_BEFORE_LABELS.parquet',index=False);status.to_csv(out/'BAND_STATUS_BEFORE_LABELS.csv',index=False)
        return dict(fullpool_key_identity=identity,model_loads=1,reused_prediction_rows=len(pred),exported_Top50_rows=len(keys),new_predict_calls=1,canary_date=str(date.date()),target_labels_read=False,replay=rep,group_keys_sha256=sha(out/'TOP50_GROUP_KEYS_BEFORE_LABELS.parquet'))
    return lifecycle('G1_CANARY_01','CANARY',op,'STORED_PREDICTIONS_AND_PRIOR_ONLY_NO_TARGET_LABEL_READ')


def export_fold(binding,m,fold):
    execution='G1_EXPORT_'+fold.fold_id
    out=BATCH/execution;receipt=out/'RUN_RECEIPT.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['status']!='COMPLETED':raise ValueError('failed export requires diagnosis; do not overwrite')
        for p,h in saved['files'].items():
            if sha(p)!=h:raise ValueError('completed export artifact drift')
        return saved
    def op(out):
        payload,pred,identity=validate_checkpoint(binding,m,fold.fold_id)
        calendar=pd.read_parquet(binding['cache_files']['calendar']['path']);dates=calendar.loc[calendar.date.between(fold.start,fold.end)]
        if set(pred.date.unique())!=set(dates.date.unique()):raise ValueError('fullpool predictions lack original validation calendar date')
        rep=None
        if fold.fold_id!='2024H1':rep=replay(binding,m,payload,pred,pred.date.min(),out)
        else:
            canary=json.loads((BATCH/'G1_CANARY_01/RUN_RECEIPT.json').read_text())
            if not canary['replay']['original_Top10_keys_and_scores_exact']:raise ValueError('canary replay prerequisite')
        pred.to_parquet(out/'FULLPOOL_PREDICTIONS.parquet',index=False)
        top=top50(pred);top['fold']=fold.fold_id
        prior=pd.read_parquet(binding['cache_files']['s1']['path'],columns=['date','code',PRIOR,MISSING],filters=[('date','>=',fold.start),('date','<=',fold.end)])
        keys,status=groups_before_labels(top,prior);keys.to_parquet(out/'TOP50_GROUP_KEYS_BEFORE_LABELS.parquet',index=False);status.to_parquet(out/'BAND_STATUS_BEFORE_LABELS.parquet',index=False)
        return dict(fold=fold.fold_id,model_loads=1,reused_prediction_rows=len(pred),new_predict_calls=0 if rep is None else 1,predictions_generated_for_full_pool=0,exported_Top50_rows=len(keys),fullpool_key_identity=identity,validation_dates=len(dates),replay_reused_from_canary=rep is None,replay=rep,files={str(p):sha(p) for p in out.iterdir() if p.suffix in ['.parquet']},target_labels_read=False)
    return lifecycle(execution,'PREDICTION_REUSE_EXPORT',op,'STORED_FULLPOOL_PREDICTIONS_PRIOR_AND_SCORING_KEYS_ONLY')


def run_full(binding):
    canary=json.loads((BATCH/'G1_CANARY_01/RUN_RECEIPT.json').read_text())
    if canary['status']!='COMPLETED':raise ValueError('canary required before full')
    m=load_frozen(binding);exports=[]
    for fold in m.fixed_folds():exports.append(export_fold(binding,m,fold))
    keys=pd.concat([pd.read_parquet(BATCH/('G1_EXPORT_'+f.fold_id)/'TOP50_GROUP_KEYS_BEFORE_LABELS.parquet') for f in m.fixed_folds()],ignore_index=True)
    status=pd.concat([pd.read_parquet(BATCH/('G1_EXPORT_'+f.fold_id)/'BAND_STATUS_BEFORE_LABELS.parquet') for f in m.fixed_folds()],ignore_index=True)
    keyfile=BATCH/'ALL_GROUP_KEYS_BEFORE_LABELS.parquet';keys.to_parquet(keyfile,index=False)
    calendar=pd.read_parquet(binding['cache_files']['calendar']['path']);dateparts=[]
    for fold in m.fixed_folds():
        d=calendar.loc[calendar.date.between(fold.start,fold.end)].copy();d['fold']=fold.fold_id;dateparts.append(d)
    dates=pd.concat(dateparts,ignore_index=True);dates.to_parquet(BATCH/'VALIDATION_DATES_BEFORE_LABELS.parquet',index=False)
    write(BATCH/'KEY_FREEZE_MANIFEST.json',dict(contract_sha256=sha(BATCH/'CONTRACT.json'),group_keys_sha256=sha(keyfile),validation_dates_sha256=sha(BATCH/'VALIDATION_DATES_BEFORE_LABELS.parquet'),fold_exports=exports,target_labels_read=False,fit_calls=0,new_predict_calls=6,model_loads=7,fullpool_prediction_generation_calls=0))
    def evaluation(out):
        # First target-label read occurs after the complete six-fold key freeze above.
        panel=pd.read_parquet(binding['cache_files']['panel']['path'],columns=['date','code','close'])
        labels=m.build_close_return_labels(panel,horizons=(2,),calendar=calendar,signal_keys=keys[['date','code']])[2]
        labeled,bands,daily=evaluate_bands(keys,status,labels,dates)
        result=summarize(daily,bands,minimum_blocks=2)
        labeled.to_parquet(out/'LABELED_TOP50.parquet',index=False);bands.to_csv(out/'BAND_DIAGNOSTICS.csv',index=False);daily.to_csv(out/'DAILY_DIAGNOSTICS.csv',index=False)
        write(out/'METRICS.json',result);write(out/'OUTPUT_MANIFEST.json',dict(files={str(p):sha(p) for p in out.iterdir() if p.is_file()},key_freeze_sha256=sha(BATCH/'KEY_FREEZE_MANIFEST.json'),fit_calls=0,research_only=True))
        return dict(decision=result['decision'],metrics_sha256=sha(out/'METRICS.json'),fit_calls=0,new_predict_calls=0)
    return lifecycle('EVALUATION','DETERMINISTIC_EVALUATION',evaluation,'H2_JOIN_AFTER_ALL_GROUP_KEYS_FROZEN')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['tests','canary','full','review-start']);parser.add_argument('--execution');args=parser.parse_args();binding=verify()
    if args.stage=='tests':
        def op(out):
            p=subprocess.run([sys.executable,'-m','pytest','-q',str(HERE/'tests/test_g1_diagnostic.py')],capture_output=True,text=True);(out/'stdout.txt').write_text(p.stdout);(out/'stderr.txt').write_text(p.stderr)
            if p.returncode:raise RuntimeError(p.stdout+p.stderr)
            return dict(command=p.args,stdout_sha256=sha(out/'stdout.txt'),fit_calls=0)
        result=lifecycle(args.execution or 'G1_SYNTHETIC_TEST_01','SYNTHETIC_TESTS',op)
    elif args.stage=='canary':result=run_canary(binding)
    elif args.stage=='full':result=run_full(binding)
    else:
        out=BATCH/args.execution;out.mkdir(exist_ok=False);r=out/'RUN_RECEIPT.json';write(r,dict(status='AWAITING_INDEPENDENT_REVIEW',fit_calls=0));append(args.execution,'RUN_STARTED','INDEPENDENT_REVIEW',r,metrics={'fit_calls':0});result=sync()
    print(json.dumps(result,ensure_ascii=False,default=str))
if __name__=='__main__':main()
