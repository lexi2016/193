"""S1 serial, four-thread canary/formal runner over the exact frozen M0 code."""
from __future__ import annotations
import argparse,gc,hashlib,importlib.util,io,json,os,resource,subprocess,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from s1_registry import ROOT,BATCH,EXPERIMENT,FLAGS,sha,write,append,sync
from s1_industry_month_prior import PRIOR_COLUMNS,prior_lookup,project_prior,append_prior,acceptance
HERE=Path(__file__).resolve().parent
ORIGINAL=ROOT/'research/a_share_model_selection_v1/batches/pv_environment_r1_20261019'
OLD_RUN=ROOT/'workspace/research/a_share_model_selection_v1/runs/pv_environment_r1_20261019_formal_r01'
PYTHON='/LOCAL_ONLY/private_home/Documents/Codex/2026-08-13/factor-mining-mac/.venv-training-v1/bin/python'


def canonical(d):return hashlib.sha256(json.dumps(d,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify_freeze():
    frozen=json.loads((BATCH/'FROZEN_IDENTITY.json').read_text())
    for p,expected in {**frozen['code_by_file'],**frozen['controls']}.items():
        if sha(p)!=expected:raise ValueError('frozen identity drift: '+p)
    return json.loads((BATCH/'INPUT_BINDING.json').read_text())


def frozen_route():
    directory=HERE/'frozen_m0/research/a_share_model_selection_v1'
    sys.path.insert(0,str(directory))
    spec=importlib.util.spec_from_file_location('s1_exact_frozen_m0',directory/'model_selection.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    m.FROZEN_ROUTE_DEPENDENCY_ROOT=directory
    expected=json.loads((BATCH/'INPUT_BINDING.json').read_text())['m0_code_identity']
    if m.code_identity()!=expected:raise ValueError('M0 component SHA mismatch')
    return m


def resource_guard():
    import re
    config=json.loads((BATCH/'CONFIG.json').read_text())
    rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())],text=True).strip())*1024
    vm=subprocess.check_output(['vm_stat'],text=True)
    page=int(re.search(r'page size of (\d+) bytes',vm).group(1))
    pages={k:int(v) for k,v in re.findall(r'([^\n:]+):\s*(\d+)\.',vm)}
    available=page*sum(pages.get(k,0) for k in ['Pages free','Pages inactive','Pages speculative'])
    rss=max(rss,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if rss>config['max_rss_bytes'] or available<config['min_available_bytes']:
        raise MemoryError(f'frozen resource stop: rss={rss},available={available}')
    return dict(rss_bytes=rss,available_bytes=available)


def bound_read_csv(p,expected,**kwargs):
    if sha(p)!=expected:raise ValueError('bound CSV SHA mismatch: '+str(p))
    return pd.read_csv(p,**kwargs)


def materialize(m,binding,canary):
    old=json.loads(Path(binding['old_binding']['path']).read_text());sources=old['sources']
    price=sources['price_panel']; config=json.loads((BATCH/'CONFIG.json').read_text())
    # Accepted unchanged panel identity FAST_BIND: receipt SHA + stat, no rehash/reproducer.
    stat=Path(price['path']).stat()
    if stat.st_size!=binding['panel_stat']['bytes'] or stat.st_mtime_ns!=binding['panel_stat']['mtime_ns']:
        raise ValueError('panel stat drift invalidates FAST_BIND')
    cols=list(dict.fromkeys(price['column_map'].values()))
    raw=pd.read_parquet(price['path'],columns=cols,
        filters=[('ts_code','in',config['canary_codes'])] if canary else None)
    panel=m.normalize_price_panel(raw,price['column_map'],amount_multiplier=10000.0);del raw
    fp,quarantine=m.derive_price_feature_view(panel)
    if not canary:m._validate_quarantine_binding(quarantine,old['consumer_compatibility'])
    calendar=m.normalize_calendar(pd.read_csv(io.BytesIO(m.read_bound_archive_member(sources['trading_date_index']))),panel)
    index=m.load_bound_index_series(sources['index_series'],panel_dates=set(panel.date))
    industry=m.normalize_industry_columns(pd.read_csv(io.BytesIO(m.read_bound_archive_member(sources['industry_classification']))))
    from formula_adapter import exact_formula_source,formula_feature_frame
    formula={name:exact_formula_source(Path(d['path']),d['sha256'])[0] for name,d in ((name,old['formula_inputs'][name]) for name in ('v192','v193'))}
    ff=formula_feature_frame(fp,v193_text=formula['v193'],v192_text=formula['v192'],index_close_by_date=index,
          include_v192_trigger=True,include_v193_trigger=True,include_auditm=True)
    arms=m.build_nested_arm_frames(fp,ff,calendar,index_close_by_date=index,industry_classification=industry)
    ff,arms,compat=m.apply_ohlc_taint_masks(ff,arms,calendar,quarantine,industry_classification=industry)
    del ff,fp;gc.collect();resource_guard()
    common,columns=m.assemble_common_arm_features(arms);m0=common['M0'];del arms,common;gc.collect()
    stats=binding['statistics']
    annual=bound_read_csv(stats['industry_year']['path'],stats['industry_year']['sha256'])
    monthly=bound_read_csv(stats['month_year']['path'],stats['month_year']['sha256'])
    lookup=prior_lookup(annual,monthly,range(2021,2027))
    roster=bound_read_csv(binding['roster']['path'],binding['roster']['sha256'],dtype={'code':str})
    projection=project_prior(m0[['date','code']],roster,calendar,lookup)
    s1=append_prior(m0,projection)
    if tuple(m0.columns[2:])!=m.CORE_FEATURES or tuple(s1.columns)!=tuple(m0.columns)+PRIOR_COLUMNS:
        raise ValueError('frozen feature order mismatch')
    if not m0.equals(s1[m0.columns]):raise ValueError('S1 modified M0 values or keys')
    report=dict(panel_rows=len(panel),common_rows=len(m0),codes=panel.code.nunique(),
       common_key_identity=m.key_frame_identity(m0),prior_missing_rows=int(projection[PRIOR_COLUMNS[-1]].sum()),
       prior_max_year_by_signal_year=lookup.groupby('signal_year').max_prior_year.max().to_dict(),
       feature_columns=dict(M0=list(m0.columns[2:]),S1=list(s1.columns[2:])),
       canary_sample_scope=bool(canary),global_M0_checkpoint_parity='PENDING_FORMAL_PREPARATION' if canary else 'CHECK_EACH_FOLD')
    return panel,calendar,m0,s1,lookup,report


def run_tests(execution):
    out=BATCH/execution;out.mkdir(exist_ok=False)
    receipt=out/'RUN_RECEIPT.json';write(receipt,dict(status='REGISTER_BEFORE_EXECUTE',execution=execution))
    append(execution,'RUN_STARTED','SYNTHETIC_TESTS',receipt)
    try:
        result=subprocess.run([sys.executable,'-m','pytest','-q',str(HERE/'test_s1_industry_month_prior.py')],capture_output=True,text=True)
        (out/'stdout.txt').write_text(result.stdout);(out/'stderr.txt').write_text(result.stderr)
        write(receipt,dict(status='COMPLETED' if result.returncode==0 else 'FAILED',exit_code=result.returncode,
             command=result.args,stdout_sha256=sha(out/'stdout.txt'),stderr_sha256=sha(out/'stderr.txt'),**FLAGS))
        append(execution,'TERMINAL','SYNTHETIC_TESTS',receipt,decision='COMPLETED' if result.returncode==0 else 'FAILED_NO_FIT')
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    finally:sync()
    return json.loads(receipt.read_text())


def fit_cell(m,train,test,columns,execution,out,identity,arm):
    receipt=out/(execution+'_RUN_RECEIPT.json');checkpoint=out/(execution+'.joblib')
    events=[json.loads(line) for line in (ROOT/'contracts/EXPERIMENT_REGISTRY.jsonl').read_text().splitlines() if line.strip()]
    same=[e for e in events if e.get('execution_id')==execution]
    if checkpoint.exists():
        if len([e for e in same if e.get('event_type')=='TERMINAL' and e.get('decision')=='COMPLETED'])!=1:
            raise ValueError('checkpoint lacks successful registered terminal; diagnose before recovery')
        meta=json.loads(checkpoint.with_suffix('.joblib.json').read_text())
        if meta['identity']!=identity or meta['sha256']!=sha(checkpoint):raise ValueError('checkpoint identity drift')
        return joblib.load(checkpoint)['predictions']
    if receipt.exists() or same:raise ValueError('unfinished cell requires diagnosis, new identity forbidden')
    starts=[e for e in events if e.get('experiment_id')==EXPERIMENT and e.get('event_type')=='RUN_STARTED' and e.get('stage') in ['FORMAL','CANARY_FIT'] and (e.get('deterministic_metrics') or {}).get('planned_real_fit_count')==1]
    if len(starts)>=8:raise ValueError('frozen two-canary plus six-formal fit budget exhausted')
    write(receipt,dict(status='REGISTER_BEFORE_FIT',identity=identity,arm=arm,**FLAGS))
    append(execution,'RUN_STARTED','CANARY_FIT' if identity['canary'] else 'FORMAL',receipt,
           truth='PAST_TRAIN_LABELS_ONLY',metrics={'planned_real_fit_count':1,'real_fit_budget_debit':1})
    start=time.monotonic();entered=False
    try:
        resource_guard();model=m.model_for_arm()
        x=train[columns].to_numpy(dtype=float);xt=test[columns].to_numpy(dtype=float)
        y=train.label_return.to_numpy(dtype=float);w=m.equal_day_weights(train.date)
        entered=True
        with threadpool_limits(limits=4):model.fit(x,y,sample_weight=w);scores=model.predict(xt)
        predictions=test[['date','code']].copy();predictions['prediction']=scores
        tmp=checkpoint.with_suffix('.tmp.joblib');joblib.dump(dict(model=model,predictions=predictions),tmp);os.replace(tmp,checkpoint)
        write(checkpoint.with_suffix('.joblib.json'),dict(identity=identity,sha256=sha(checkpoint)))
        replay=joblib.load(checkpoint)
        with threadpool_limits(limits=4):roundtrip=replay['model'].predict(xt)
        if not np.array_equal(scores,roundtrip):raise ValueError('save/reload prediction mismatch')
        write(receipt,dict(status='COMPLETED',arm=arm,identity=identity,elapsed_seconds=time.monotonic()-start,
              train_rows=len(train),test_rows=len(test),fit_calls=1,roundtrip_exact=True,checkpoint_sha256=sha(checkpoint),
              peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**FLAGS))
        append(execution,'TERMINAL','CANARY_FIT' if identity['canary'] else 'FORMAL',receipt,decision='COMPLETED',
               truth='PAST_TRAIN_LABELS_ONLY',checkpoint=str(checkpoint),
               metrics={'real_fit_returned_count':1,'fit_call_entered':True,'selection_slots':None})
        return predictions
    except BaseException as e:
        write(receipt,dict(status='CANNOT_CONFIRM',error=repr(e),fit_call_entered=entered,**FLAGS))
        append(execution,'TERMINAL','CANARY_FIT' if identity['canary'] else 'FORMAL',receipt,
               decision='CANNOT_CONFIRM',truth='PAST_TRAIN_LABELS_ONLY',metrics={'fit_call_entered':entered})
        raise


def run_canary(execution):
    binding=verify_freeze();out=BATCH/execution;out.mkdir(exist_ok=False)
    receipt=out/'RUN_RECEIPT.json';write(receipt,dict(status='REGISTER_BEFORE_CANARY',**FLAGS))
    append(execution,'RUN_STARTED','CANARY',receipt,truth='BOUND_SMALL_PANEL_AND_ANNUAL_AGGREGATES_NOT_YET_READ')
    start=time.monotonic()
    try:
        m=frozen_route();panel,calendar,m0,s1,lookup,report=materialize(m,binding,True)
        fold=m.fixed_folds()[0]
        labels=m.build_close_return_labels(panel,horizons=(2,),calendar=calendar,
             signal_start=fold.start-pd.DateOffset(years=2),signal_end_exclusive=fold.start)[2]
        train,test=m.split_fold(m0,labels,fold);train,test=m.canary_subset(train,test,max_codes=40,train_dates=80,test_dates=20)
        # Merge just four columns; reuse the exact original training order and labels.
        tr1=train.merge(s1[['date','code',*PRIOR_COLUMNS]],on=['date','code'],how='left',validate='one_to_one')
        te1=test.merge(s1[['date','code',*PRIOR_COLUMNS]],on=['date','code'],how='left',validate='one_to_one')
        if not train.equals(tr1[train.columns]) or not test.equals(te1[test.columns]):raise ValueError('canary key/value parity failed')
        if (train.label_end_date>=fold.start).any():raise ValueError('training label maturation failed')
        oldcols=list(m.CORE_FEATURES);newcols=oldcols+list(PRIOR_COLUMNS)
        report['train_rows']=len(train);report['test_rows']=len(test);report['train_label_max']=str(train.label_end_date.max())
        report['asof_year_limits_pass']=all(int(k)>v for k,v in report['prior_max_year_by_signal_year'].items())
        report['base_train_identity']=m.cell_data_identity(train=train,test=test,feature_columns=oldcols,
             x=train[oldcols].to_numpy(dtype=float),y=train.label_return.to_numpy(dtype=float),
             weights=m.equal_day_weights(train.date),xt=test[oldcols].to_numpy(dtype=float))
        lookup.to_csv(out/'ASOF_PRIOR_LOOKUP.csv',index=False)
        for arm,tr,te,cols in [('M0',train,test,oldcols),('S1',tr1,te1,newcols)]:
            identity=dict(canary=True,arm=arm,fold=fold.fold_id,horizon=2,
               contract_sha256=sha(BATCH/'CONTRACT.json'),code_sha256=json.loads((BATCH/'FROZEN_IDENTITY.json').read_text())['code_composite_sha256'],
               feature_columns=cols,base_train_identity=report['base_train_identity'])
            pred=fit_cell(m,tr,te,cols,execution+'_'+arm,out,identity,arm)
            top=m.stable_top10(pred);top.to_csv(out/(arm+'_TOP10_KEYS_BEFORE_LABELS.csv'),index=False)
        report.update(status='ENGINEERING_CANARY_ONLY',elapsed_seconds=time.monotonic()-start,
            key_and_base_value_parity=True,column_order_exact=True,label_maturity=True,fit_calls=2,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            global_parity='CANNOT_CONFIRM_UNTIL_FORMAL_FULL_POOL_BIND',**FLAGS)
        write(receipt,report)
        append(execution,'TERMINAL','CANARY',receipt,decision='COMPLETED',truth='SMALL_PANEL_PAST_TRAIN_LABELS_ONLY',metrics={'canary_fit_calls':2})
        return report
    except BaseException as e:
        write(receipt,dict(status='CANNOT_CONFIRM',error=repr(e),elapsed_seconds=time.monotonic()-start,**FLAGS))
        append(execution,'TERMINAL','CANARY',receipt,decision='CANNOT_CONFIRM',truth='SMALL_PANEL_PREPARATION_OR_PAST_LABELS_ATTEMPTED')
        raise
    finally:sync()


def verify_formal_activation():
    p=BATCH/'CANARY_REVIEW_ACTIVATION.json'
    activation=json.loads(p.read_text())
    frozen=json.loads((BATCH/'FROZEN_IDENTITY.json').read_text())
    if (activation.get('verdict')!='CANARY_IMPLEMENTATION_PASS' or
        activation.get('code_composite_sha256')!=frozen['code_composite_sha256'] or
        activation.get('contract_sha256')!=sha(BATCH/'CONTRACT.json') or
        activation.get('reviewer_identity')=='/root/s1_prior_writer'):
        raise ValueError('independent canary activation required')


def full_preparation(execution,out,binding,m):
    prep=out/'PREPARATION_RECEIPT.json'
    if prep.exists():
        report=json.loads(prep.read_text())
        if report.get('status')!='COMPLETED':raise ValueError('failed preparation requires diagnosis')
        for p,h in report['cache_files'].items():
            if sha(p)!=h:raise ValueError('prepared cache drift')
        return pd.read_parquet(out/'panel.parquet'),pd.read_parquet(out/'calendar.parquet'),pd.read_parquet(out/'m0.parquet'),pd.read_parquet(out/'s1.parquet')
    write(prep,dict(status='REGISTER_BEFORE_PREPARATION',**FLAGS))
    append(execution+'_PREP','RUN_STARTED','FORMAL_PREPARATION',prep,truth='FULL_FEATURE_INPUTS_NOT_YET_READ')
    try:
        panel,calendar,m0,s1,lookup,report=materialize(m,binding,False)
        # All six old M0 data identities are proved before ANY formal S1 fit.
        oldbinding=json.loads(Path(binding['old_binding']['path']).read_text())
        parities=[]
        for fold in m.fixed_folds():
            labels=m.build_close_return_labels(panel,horizons=(2,),calendar=calendar,
                 signal_start=fold.start-pd.DateOffset(years=2),signal_end_exclusive=fold.start)[2]
            prepared=m.prepare_cell_data(features=m0,labels=labels,fold=fold,canary=False)
            oldmeta=Path(binding['baseline_checkpoints'][fold.fold_id]['path']+'.json')
            if sha(oldmeta)!=binding['baseline_checkpoints'][fold.fold_id]['metadata_sha256']:
                raise ValueError('old M0 checkpoint metadata SHA drift')
            meta=json.loads(oldmeta.read_text())
            if meta['identity']['data_identity']!=prepared['identity']:
                raise ValueError('old M0 data identity parity failed: '+fold.fold_id)
            ident=m.checkpoint_identity(stage='M0_FORMAL',fold=fold,arm='M0',horizon=2,attempt='attempt_01',
                contract_sha256=meta['identity']['contract_sha256'],code_shas=binding['m0_code_identity'],
                binding=oldbinding,feature_names=prepared['feature_columns'],data_identity=prepared['identity'])
            if ident!=meta['identity']:raise ValueError('full old checkpoint identity mismatch: '+fold.fold_id)
            parities.append(dict(fold=fold.fold_id,exact=True,identity=prepared['identity'],old_metadata_sha256=sha(oldmeta)))
            del prepared,labels;gc.collect();resource_guard()
        report['all_six_M0_data_identity_parities']=parities
        for name,frame in [('panel',panel),('calendar',calendar),('m0',m0),('s1',s1)]:frame.to_parquet(out/(name+'.parquet'),index=False)
        lookup.to_csv(out/'ASOF_PRIOR_LOOKUP.csv',index=False)
        report.update(status='COMPLETED',cache_files={str(out/(name+'.parquet')):sha(out/(name+'.parquet')) for name in ['panel','calendar','m0','s1']},**FLAGS)
        write(prep,report);append(execution+'_PREP','TERMINAL','FORMAL_PREPARATION',prep,decision='COMPLETED',truth='PAST_TRAIN_LABELS_FOR_SIX_FOLDS_IDENTITY_ONLY')
        sync();return panel,calendar,m0,s1
    except BaseException as e:
        write(prep,dict(status='CANNOT_CONFIRM',error=repr(e),**FLAGS))
        append(execution+'_PREP','TERMINAL','FORMAL_PREPARATION',prep,decision='CANNOT_CONFIRM',truth='FEATURE_OR_PAST_TRAIN_LABEL_PREPARATION_ATTEMPTED');sync();raise


def evaluate_selections(m,selected,panel,calendar,out):
    # Called only after all selection keys are written and hashed.
    unique=selected[['date','code']].drop_duplicates()
    for h,labels in m.build_close_return_labels(panel,horizons=(1,2,5),calendar=calendar,signal_keys=unique).items():
        labels=labels[['date','code','label_return']].rename(columns={'label_return':f'return_h{h}'})
        selected=selected.merge(labels,on=['date','code'],how='left',validate='many_to_one')
    roster_info=json.loads((BATCH/'INPUT_BINDING.json').read_text())['roster']
    roster=bound_read_csv(roster_info['path'],roster_info['sha256'],dtype={'code':str})
    selected=selected.merge(roster[['code','industry_key']],on='code',how='left',validate='many_to_one')
    selected.industry_key=selected.industry_key.fillna('UNMAPPED')
    # Volatility and amount are diagnostics only, never used for a model input or gate.
    diagnostic=panel[['date','code','amount','close']].sort_values(['code','date']).copy()
    diagnostic['daily_return']=diagnostic.groupby('code').close.pct_change(fill_method=None)
    diagnostic['volatility20']=diagnostic.groupby('code').daily_return.transform(lambda x:x.rolling(20,min_periods=20).std())
    selected=selected.merge(diagnostic[['date','code','amount','volatility20']],on=['date','code'],how='left',validate='many_to_one')
    selected.to_csv(out/'SELECTED_RETURNS_DIAGNOSTICS.csv',index=False)
    a=selected.loc[selected.arm=='M0'];b=selected.loc[selected.arm=='S1']
    result=acceptance(a,b,calendar)
    diagnostics=[]
    for (arm,fold),group in selected.groupby(['arm','fold']):
        counts=group.groupby(['date','industry_key']).size().div(10)
        diagnostics.append(dict(arm=arm,fold=fold,dates=group.date.nunique(),
            missing_h2=int(group.return_h2.isna().sum()),mean_amount=float(group.amount.mean()),
            mean_volatility20=float(group.volatility20.mean()),
            mean_industry_hhi=float(counts.pow(2).groupby('date').sum().mean()),
            mean_max_industry_share=float(counts.groupby('date').max().mean()),
            same_h2_selection_h1_mean=float(group.groupby('date').return_h1.mean().mean()),
            same_h2_selection_h5_mean=float(group.groupby('date').return_h5.mean().mean())))
    result['diagnostics_by_fold']=diagnostics
    result['partial_fold_diagnostic_only']={arm:dict(mean_h2=float(f.groupby('date').return_h2.mean().mean()),
        missing_h2=int(f.return_h2.isna().sum()),dates=int(f.date.nunique()))
        for arm,f in selected.loc[~selected.fold.isin(result['gate_folds'])].groupby('arm')}
    swaps=[]
    for date,g in selected.groupby('date'):
        aa=g.loc[g.arm=='M0'];bb=g.loc[g.arm=='S1'];common=set(aa.code)&set(bb.code)
        swaps.append(dict(date=str(pd.Timestamp(date).date()),retained=len(common),swapped=10-len(common),
            entrant_h2_mean=None if len(common)==10 else float(bb.loc[~bb.code.isin(common),'return_h2'].mean()),
            removed_h2_mean=None if len(common)==10 else float(aa.loc[~aa.code.isin(common),'return_h2'].mean())))
    pd.DataFrame(swaps).to_csv(out/'SWAP_DIAGNOSTICS.csv',index=False)
    write(out/'METRICS.json',result)
    return result


def run_formal(execution):
    if execution!='S1_FORMAL_01':raise ValueError('only the frozen S1_FORMAL_01 execution is authorized')
    binding=verify_freeze();verify_formal_activation();resource_guard();m=frozen_route()
    out=BATCH/execution;out.mkdir(exist_ok=True)
    panel,calendar,m0,s1=full_preparation(execution,out,binding,m)
    selected=[]
    for fold in m.fixed_folds():
        labels=m.build_close_return_labels(panel,horizons=(2,),calendar=calendar,
            signal_start=fold.start-pd.DateOffset(years=2),signal_end_exclusive=fold.start)[2]
        prepared=m.prepare_cell_data(features=m0,labels=labels,fold=fold,canary=False)
        path=Path(binding['baseline_checkpoints'][fold.fold_id]['path'])
        meta=json.loads(path.with_suffix('.joblib.json').read_text())
        if sha(path)!=binding['baseline_checkpoints'][fold.fold_id]['sha256'] or meta['identity']['data_identity']!=prepared['identity']:
            raise ValueError('M0 reuse parity failed after preparation')
        payload=joblib.load(path);oldpred=payload['predictions']
        if not oldpred[['date','code']].equals(prepared['test'][['date','code']]):raise ValueError('baseline prediction keys differ')
        top=m.stable_top10(oldpred);top['arm']='M0';top['fold']=fold.fold_id;selected.append(top)
        del payload
        # Reuse the exact M0 train/test keys+labels; join only the four finite columns.
        train=prepared['train'].merge(s1[['date','code',*PRIOR_COLUMNS]],on=['date','code'],how='left',validate='one_to_one')
        test=prepared['test'].merge(s1[['date','code',*PRIOR_COLUMNS]],on=['date','code'],how='left',validate='one_to_one')
        columns=list(m.CORE_FEATURES)+list(PRIOR_COLUMNS)
        identity=dict(canary=False,arm='S1',fold=fold.fold_id,horizon=2,
             contract_sha256=sha(BATCH/'CONTRACT.json'),input_binding_sha256=sha(BATCH/'INPUT_BINDING.json'),
             code_sha256=json.loads((BATCH/'FROZEN_IDENTITY.json').read_text())['code_composite_sha256'],
             feature_columns=columns,base_data_identity=prepared['identity'],
             s1_train_matrix=m.numeric_matrix_identity(train[columns].to_numpy(dtype=float),columns),
             s1_test_matrix=m.numeric_matrix_identity(test[columns].to_numpy(dtype=float),columns))
        del prepared,labels;gc.collect()
        pred=fit_cell(m,train,test,columns,execution+'_S1_'+fold.fold_id,out,identity,'S1')
        top=m.stable_top10(pred);top['arm']='S1';top['fold']=fold.fold_id;selected.append(top)
        keys=pd.concat(selected,ignore_index=True);keys.to_csv(out/'SELECTION_KEYS_PROGRESS_BEFORE_LABELS.csv',index=False)
        del train,test,pred;gc.collect();sync();resource_guard()
    keys=pd.concat(selected,ignore_index=True);keypath=out/'SELECTION_KEYS_BEFORE_LABELS.csv';keys.to_csv(keypath,index=False)
    receipt=out/'EVALUATION_RUN_RECEIPT.json'
    write(receipt,dict(status='REGISTER_BEFORE_EVALUATION',selection_key_sha256=sha(keypath),**FLAGS))
    append(execution+'_EVAL','RUN_STARTED','FORMAL',receipt,truth='SELECTION_KEYS_FROZEN_FORWARD_OUTCOMES_NOT_YET_READ')
    try:
        result=evaluate_selections(m,keys,panel,calendar,out)
        write(receipt,dict(status='COMPLETED',selection_key_sha256=sha(keypath),metrics_sha256=sha(out/'METRICS.json'),decision=result['decision'],**FLAGS))
        append(execution+'_EVAL','TERMINAL','FORMAL',receipt,decision='COMPLETED',truth='SELECTED_OUTCOMES_READ_AFTER_KEYS_FROZEN',metrics={'selection_slots':len(keys)})
        files={str(p):sha(p) for p in out.iterdir() if p.is_file() and not p.name.endswith('.parquet')}
        write(out/'OUTPUT_MANIFEST.json',dict(status='DIRECTION_ONLY',decision=result['decision'],files=files,fit_calls=6,reused_M0_fits=6,**FLAGS))
        sync();return result
    except BaseException as e:
        write(receipt,dict(status='CANNOT_CONFIRM',error=repr(e),**FLAGS))
        append(execution+'_EVAL','TERMINAL','FORMAL',receipt,decision='CANNOT_CONFIRM',truth='SELECTED_OUTCOMES_READ_OR_ATTEMPTED');sync();raise


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['tests','canary','formal','review-start']);p.add_argument('--execution',required=True)
    args=p.parse_args();verify_freeze()
    if args.stage=='tests':result=run_tests(args.execution)
    elif args.stage=='canary':result=run_canary(args.execution)
    elif args.stage=='formal':result=run_formal(args.execution)
    else:
        out=BATCH/args.execution;out.mkdir(exist_ok=False);receipt=out/'RUN_RECEIPT.json'
        write(receipt,dict(status='AWAITING_INDEPENDENT_REVIEW',reviewer_read_only=True,**FLAGS))
        append(args.execution,'RUN_STARTED','INDEPENDENT_REVIEW',receipt);result=sync()
    print(json.dumps(result,ensure_ascii=False,default=str))
if __name__=='__main__':main()
