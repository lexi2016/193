"""Bound hsjday retrospective descriptive research; never a canonical producer."""
from __future__ import annotations
import argparse, ast, csv, hashlib, importlib.util, io, json, os, subprocess, sys, time, traceback, zipfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path('/LOCAL_ONLY/factor_factory')
WT=ROOT/'workspace/.task_worktrees/a-hsjday-20y-20261019'
sys.path.insert(0,str(WT))
from research.a_share_top100_environment_v1 import hsjday_industry_coverage as ind
from research.a_share_top100_environment_v1 import hsjday_xdxr_coverage_recheck_02 as xd
BATCH=ROOT/'research/a_share_top100_environment_v1/batches/hsjday_20y_20261019'
STUDY=BATCH/'historical_descriptive_01'
CODE=Path(__file__).resolve()
TEST=CODE.parent/'tests/test_hsjday_history_stats.py'
ZIP=BATCH/'inputs/hsjday.zip'
PRIOR= BATCH/'panel_dependency_01/industry_coverage_02/outputs/INDUSTRY_ROSTER.csv'
CAL=BATCH/'panel_dependency_01/outputs/OBSERVED_STOCK_SESSION_CALENDAR.csv'
INVENTORY=ind.INVENTORY_BATCH/'FILE_INVENTORY.csv'
PRODUCER=Path('/LOCAL_ONLY/private_home/Library/Application Support/com.lexi.v35.local/releases/20260910-d1-partial-provenance-v14/app/runtime/v1/data_layer_daily_formula_increment_update_v1.py')
EXPERIMENT='A_SHARE_HSJDAY_20Y_DESCRIPTIVE_20261019'
ROUTE='a-share-hsjday-20y-industry-seasonality-descriptive-20261019'
TRACK='A_SHARE_HSJDAY_20Y_DESCRIPTIVE'
SAFE=ind.SAFETY_FLAGS
DAY_DTYPE=np.dtype([('date','<u4'),('open','<u4'),('high','<u4'),('low','<u4'),('close','<u4'),('amount','<f4'),('volume','<u4'),('reserved','<u4')])

def sha(p): return ind.sha256_file(Path(p))
def read(p): return json.loads(Path(p).read_text())
def write(p,v): ind.write_new(Path(p),ind.canonical_bytes(v))
def now(): return datetime.now(timezone.utc).isoformat()
def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def exact_formula():
    """Compile only the bound original function; no producer main/import side effects."""
    tree=ast.parse(PRODUCER.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_xdxr_pre_close')
    env={'pd':pd,'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(PRODUCER),'exec'),env)
    return env['apply_xdxr_pre_close']

def industry_binding(manifest_path):
    """Consume the final industry owner's derivative, without recomputing it."""
    manifest_path=Path(manifest_path).resolve();manifest=read(manifest_path)
    candidates=[Path(p) for p in manifest['files'] if Path(p).name=='INDUSTRY_ROSTER.csv']
    if len(candidates)!=1:raise RuntimeError('industry manifest must bind exactly one roster')
    roster=candidates[0]
    if sha(roster)!=manifest['files'][str(roster)]:raise RuntimeError('industry derivative drift')
    return {'manifest_path':str(manifest_path),'manifest_sha256':sha(manifest_path),'roster_path':str(roster),'roster_sha256':sha(roster)}

def frozen_industry():
    return read(STUDY/'INPUT_BINDING.json')['industry_derivative']

def normalize_roster_labels(roster):
    """Resolve code-only CAPCO display fallbacks without changing categories."""
    roster=roster.copy();roster['industry_label_original']=roster.industry_label
    for key,g in roster.groupby('industry_key',sort=False):
        if g.industry_label.nunique()<=1:continue
        fallback=(g.classification_system=='CAPCO_CURRENT_RETROSPECTIVE') & (g.industry_label=='CAPCO:'+g.industry_code)
        names=g.loc[~fallback,'industry_label'].unique()
        if len(names)!=1:raise RuntimeError('industry category label ambiguity '+key)
        roster.loc[g.index,'industry_label']=names[0]
    return roster

def load_roster():
    bound=frozen_industry();roster=pd.read_csv(bound['roster_path'],dtype=str,keep_default_na=False)
    required={'code','member_name','industry_key','classification_system','industry_code','industry_label','name','classification_source','as_of'}
    if not required.issubset(roster):raise RuntimeError('industry roster schema missing '+str(sorted(required-set(roster))))
    if roster.empty or roster.code.duplicated().any() or roster.member_name.duplicated().any():raise RuntimeError('industry roster empty/duplicate keys')
    if ((roster.classification_system!='UNKNOWN')&(roster.industry_label=='')).any():raise RuntimeError('industry label empty')
    roster=normalize_roster_labels(roster)
    return roster.sort_values('code',kind='stable').reset_index(drop=True)

def freeze(industry_manifest):
    if STUDY.exists():raise FileExistsError('new freeze requires a new study directory '+str(STUDY))
    industry_input=industry_binding(industry_manifest)
    STUDY.mkdir(parents=True,exist_ok=True)
    paths=[CODE,TEST,ind.SCRIPT_PATH,xd.SCRIPT_PATH,PRODUCER]
    sources=[Path(industry_input['roster_path']),Path(industry_input['manifest_path']),CAL,INVENTORY,xd.CACHE_PATH,
      ind.INVENTORY_MANIFEST_PATH,xd.BATCH/'outputs/OUTPUT_MANIFEST.json']
    binding=read(ind.PRIOR_INPUT_BINDING_PATH)
    binding.update(schema_version='HSJDAY_HISTORY_STATS_BINDING_V2',experiment_id=EXPERIMENT,route_id=ROUTE,track_id=TRACK,
      contract_path=str(STUDY/'CONTRACT.json'),config_path=str(STUDY/'CONFIG.json'),industry_derivative=industry_input,
      source_limits=['PACKAGE_OBSERVED_DESCRIPTIVE_PROXY; current-industry retrospective classification; UNKNOWN retained',
        'No official full-adjustment/action-completeness, all-market, calendar or historical PIT certification',
        'No training, V4, production or trading'],
      baseline_record_path=str(STUDY/'BASELINE.json'),input_paths={str(p):sha(p) for p in sources},
      source_zip={'path':str(ZIP),'sha256':'a56c962378a5d3dd86a90186cc830ee6c8fd968557ce6eddeb310d203c5ff3c0','bytes':549367720,'verification':'REUSE_ACCEPTED_RECEIPT_NO_RESCAN'})
    write(STUDY/'INPUT_BINDING.json',binding)
    config={'start':'2006-01-01','end':'2025-12-31','topn':100,'ties':'proxy_gain descending, code ascending',
      'return_basis':'PACKAGE_OBSERVED_DESCRIPTIVE_PROXY; sequential original apply_xdxr_pre_close across (previous actual traded date,current date]',
      'usable_bar':'finite positive OHLC, volume>0, finite nonnegative amount; keep all in-window bars, unusable bars unrankable',
      'resumed_interval':'at least one intervening date in the existing observed stock-session calendar; initial pre-2006 boundary not calendar-certified',
      'extreme_gain_flag':'abs(proxy_gain)>0.25; no clipping',
      'grids':['industry','month','third','month_third'],'year_grids':'each grid crossed with all 2006..2025 years',
      'expected':'sum over dates daily_slots * industry_rankable / all_rankable; daily_slots=min(100,all_rankable)',
      'companion':'remove resumed intervals, then independently rank Top100 and recompute expected on that universe',
      'stability':'positive expected years only; lift compared to1 with abs tolerance1e-12; no support threshold filtering; longest consecutive calendar years >1; unsupported years break streak',
      'classification':'frozen final industry derivative; current CAPCO primary, sourced supplements keep their own taxonomy; UNKNOWN retained; BJ disposition inherited',
      'industry_label_display':'same-code CAPCO:<code> fallback may use its unique sourced informative label; preserve industry_label_original; genuine conflicts fail closed',
      'industry_derivative':industry_input,'canary_codes':'first8 codes of code-ascending frozen roster; preserve all category grids',
      'canary_receipt_path':str(STUDY/'CANARY_01/RUN_RECEIPT.json'),
      'bounds':'no historical PIT/full-market/calendar certification; missing XDXR cache entry does not prove no action; no pvalues/causality/prediction',**SAFE}
    write(STUDY/'CONFIG.json',config)
    write(STUDY/'BASELINE.json',{'commit':'20610d06','branch':'codex/a-hsjday-20y-20261019','prior_successes_reused':[str(ind.INVENTORY_MANIFEST_PATH),str(xd.BATCH/'outputs/OUTPUT_MANIFEST.json'),industry_input['manifest_path']],'remote':'NOT_SYNCED_NO_REMOTE','requested_role':'sol_high','requested_model':'gpt-6.1-sol','requested_effort':'high','actual_runtime_identity':'CANNOT_CONFIRM'})
    contract={'GOAL':'2006–2025 research derivative, Top100 and all-industry seasonality descriptive study',
      'HYPOTHESIS':'descriptive excess Top100 frequency may vary across current industries and calendar cells',
      'PREDICTION':'no predictive claim; report actual/expected/lift and annual support',
      'REJECTION_CONDITION':'any identity, key, rank, full-grid or count reconciliation fails',
      'BASELINE_ID':'PACKAGE_OBSERVED_DESCRIPTIVE_PROXY_CURRENT_INDUSTRY',
      'SIGNAL_DATE_RANGE':{'start':'2006-01-01','end':'2025-12-31'},'TARGET_DATE_RANGE':None,
      'TARGET_DATE_RANGE_NULL_REASON':'SAME_DAY_DESCRIPTIVE_NO_FORWARD_LABEL',
      'INPUT_PATHS':list(binding['input_paths'])+[str(ZIP)],'INPUT_SHA256':{**binding['input_paths'],str(ZIP):binding['source_zip']['sha256']},
      'CODE_SHA':sha(CODE),'CONFIG_SHA':sha(STUDY/'CONFIG.json'),
      'ONE_VARIABLE':'retrospective descriptive grouping of bound hsjday and cached actions',
      'METRICS':'actual hits, expected hits, lift, valid stock-days, dates, supported years and streak',
      'PASS_THRESHOLD':'all frozen deterministic invariants pass; independent Reviewer acceptance required',
      'SEALED_DATA':'none; all exposed historical descriptive','WRITER':'/root/top100_stats_writer',
      'REVIEWER':'different read-only Reviewer delegated by controller',
      'ALLOWED_PATHS':[str(CODE),str(TEST),str(STUDY),str(ROOT/'contracts/EXPERIMENT_REGISTRY.jsonl')],
      'FORBIDDEN_PATHS':['V4/UI/services/schedules','models/training','BUY/orders','canonical panel or selector/backtest pointers','old frozen outputs'],
      'STOP_RULE':'fail closed on identity drift; preserve completed shards and failure receipt',
      'frozen_identity_path':str(STUDY/'FROZEN_IDENTITY.json'),**SAFE}
    write(STUDY/'CONTRACT.json',contract)
    paths += [Path(dep['path']) for dep in binding['shared_dependencies'].values()]
    write(STUDY/'FROZEN_IDENTITY.json',{'code_files':{str(p):sha(p) for p in paths},'frozen_files':{str(STUDY/n):sha(STUDY/n) for n in ['CONTRACT.json','CONFIG.json','INPUT_BINDING.json','BASELINE.json']}})

def adapter_for(stage,out,claim_new=True):
    identity=read(STUDY/'FROZEN_IDENTITY.json')
    for p,h in {**identity['code_files'],**identity['frozen_files']}.items():
        if sha(p)!=h:raise RuntimeError('frozen identity drift '+p)
    binding=read(STUDY/'INPUT_BINDING.json')
    env=load(binding['shared_dependencies']['official_registry_helper']['path'],'history_registry_env')
    env.EXPERIMENT_ID,env.ROUTE_ID,env.TRACK_ID=EXPERIMENT,ROUTE,TRACK
    a,c=env.load_official_registry(binding,contract_path=STUDY/'CONTRACT.json',input_binding_path=STUDY/'INPUT_BINDING.json',config_path=STUDY/'CONFIG.json',code_path=CODE,test_path=TEST)
    a.event_profile.update(baseline_id='PACKAGE_OBSERVED_DESCRIPTIVE_PROXY_CURRENT_INDUSTRY',one_variable='BOUND_RETROSPECTIVE_DESCRIPTIVE_MEASUREMENT',signal_scope='BOUND_HSJDAY_ACTION_ADJUSTED_DESCRIPTIVE_PROXY',target_scope='NO_FORWARD_TARGET_OR_LABEL',evidence_level='DIRECTION_ONLY')
    ctx={'contract_sha256':sha(STUDY/'CONTRACT.json'),'config_sha256':sha(STUDY/'CONFIG.json'),'input_binding_sha256':sha(STUDY/'INPUT_BINDING.json'),'code_sha256':sha(CODE),
      'input_paths':list(binding['input_paths'])+[str(ZIP)],'artifact_paths':[str(STUDY/n) for n in ['CONTRACT.json','CONFIG.json','INPUT_BINDING.json','FROZEN_IDENTITY.json','BASELINE.json']]+[str(CODE),str(TEST)],
      'input_component_sha256':binding['input_paths'],'code_sha256_by_file':identity['code_files'],
      'signal_date_range':{'start':'2006-01-01','end':'2025-12-31'},'truth_access':'OPENED_EXPOSED_HISTORY' if stage in ['CANARY','FULL','REVIEW_DATA_CHECK'] else 'NOT_READ','sealed_status':'EXPOSED' if stage in ['CANARY','FULL','REVIEW_DATA_CHECK'] else 'NOT_ENTERED',
      'runtime':ind.runtime_metadata(),'evidence_level':'DIRECTION_ONLY','deterministic_metrics':{'execution_kind':'NO_FIT_ENGINEERING_CANARY','planned_real_fit_count':0,'real_fit_budget_debit':0,'real_fit_returned_count':0,'fit_call_entered':False,'truth_positive_denominator':None,'selection_slots':0},
      'null_reasons':{'checkpoint_ref':'NO_MODEL','target_date_range':'NO_FORWARD_LABEL','deterministic_metrics.truth_positive_denominator':'DESCRIPTIVE_NOT_RECALL'}}
    eid=EXPERIMENT+'_'+STUDY.name.upper()+'_'+stage+'_'+out.name
    if claim_new:
        claim=env._claim_stage(a,c,stage=stage,execution_id=eid,contract_sha256=ctx['contract_sha256'],input_binding_sha256=ctx['input_binding_sha256'],context=ctx)
    else:
        # Restore the same official claim only to close an independently run
        # review; never append a second RUN_STARTED or change its identities.
        with a._locked():_,events=a._read_validated()
        starts=[e for e in events if e.get('execution_id')==eid and e.get('event_type')=='RUN_STARTED']
        if len(starts)!=1 or any(e.get('execution_id')==eid and e.get('event_type')=='TERMINAL' for e in events):raise RuntimeError('review claim missing or already terminal')
        if starts[0]['contract_sha256']!=ctx['contract_sha256'] or starts[0]['input_binding_sha256']!=ctx['input_binding_sha256']:raise RuntimeError('review claim identity drift')
        a.set_route_context(eid,ctx)
        claim=c.ExecutionClaim(experiment_id=EXPERIMENT,execution_id=eid,arm=stage,contract_sha256=ctx['contract_sha256'],input_binding_sha256=ctx['input_binding_sha256'],execution_kind='NO_FIT_ENGINEERING_CANARY',real_fit_budget_debit=0)
    return env,a,claim

def progress(a,claim,out,phase,metrics):
    path=out/'PROGRESS.jsonl';ind.append_jsonl(path,{'at':now(),'phase':phase,'metrics':metrics})
    ctx=dict(a._route_context[claim.execution_id]);ctx['artifact_paths']=list(set(ctx['artifact_paths']+[str(path)]));a.set_route_context(claim.execution_id,ctx)
    spec=a._event_spec({'experiment_id':EXPERIMENT,'execution_id':claim.execution_id,'arm':claim.arm,'contract_sha256':claim.contract_sha256,'input_binding_sha256':claim.input_binding_sha256,'execution_kind':'NO_FIT_ENGINEERING_CANARY','planned_real_fit_count':0,'real_fit_budget_debit':0},event_type='RUN_STARTED')
    with a._locked():
        raw,events=a._read_validated();n=sum(e.get('execution_id')==claim.execution_id and e.get('event_type')=='PROGRESS' for e in events)+1
        spec.update(registry_event_id=f'{claim.execution_id}_PROGRESS_{n:04}',event_type='PROGRESS',stage=phase,occurred_at=now(),decision='RUNNING',checkpoint_ref=str(path),deterministic_metrics={**spec['deterministic_metrics'],**metrics})
        spec.get('null_reasons',{}).pop('checkpoint_ref',None)
        a._append_unlocked(raw,events,spec)

def verify_inputs():
    for p,h in read(STUDY/'INPUT_BINDING.json')['input_paths'].items():
        if sha(p)!=h:raise RuntimeError('input drift '+p)
    package=read(STUDY/'INPUT_BINDING.json')['source_zip']
    if ZIP.stat().st_size!=package['bytes'] or sha(ZIP)!=package['sha256']:raise RuntimeError('ZIP identity drift')
    # Identity hash only; reuse the accepted census and do not rescan its dates.

def industry(out):
    model=load(ind.MODEL_SELECTION_PATH,'history_bound_industry_reader')
    source=read(ind.SEASONALITY_BINDING_PATH)['sources']['industry_classification']
    capco,meta=ind._read_bound_capco(source,model)
    prior=pd.read_csv(PRIOR,dtype=str,keep_default_na=False)
    roster=prior.to_dict('records')
    mapped,missing,_=ind._join_capco(roster,capco,meta)
    prior_by={r['code']:r for r in roster}
    for r in mapped:
        old=prior_by[r['code']]
        if r['code'] in missing:r.update(old)
        else:
            r['name']=r['name'] or old['name'];r['as_of']=r['as_of'] or '2026-10-17'
    inventory=pd.read_csv(INVENTORY,dtype=str,keep_default_na=False)
    bj=inventory[(inventory.market=='bj') & (inventory.security_code.str.fullmatch(r'\d{6}'))].copy()
    bj_roster=[{'code':r.security_code+'.BJ','market':'bj','security_code':r.security_code,'member_name':r.member_name,'classification_basis':'BOUND_CAPCO_CURRENT_A_SHARE_MEMBERSHIP'} for r in bj.itertuples()]
    if len({r['code'] for r in bj_roster})!=len(bj_roster):raise RuntimeError('duplicate BJ candidate key')
    bj_mapped,bj_missing,_=ind._join_capco(bj_roster,capco,meta)
    for r in bj_mapped:
        if r['code'] not in bj_missing:
            r['as_of']=r['as_of'] or '2026-10-17';r['mapping_status']='BJ_VERIFIED_BOUND_CURRENT_CAPCO';mapped.append(r)
    pd.DataFrame(bj_mapped).to_csv(out/'BJ_CANDIDATE_DISPOSITIONS.csv',index=False)
    df=pd.DataFrame(mapped).sort_values('code')
    df['industry_key']=df['classification_system']+':'+df['industry_code'].replace('','UNKNOWN')
    df.loc[df.classification_system=='UNKNOWN','industry_key']='UNKNOWN'
    if df.code.duplicated().any():raise RuntimeError('industry roster duplicate code')
    # A category must have one unambiguous label; do not merge TDX and CAPCO.
    labels=df[df.classification_system!='UNKNOWN'].groupby('industry_key').industry_label.nunique()
    if (labels>1).any():raise RuntimeError('ambiguous category labels')
    df.to_csv(out/'INDUSTRY_ROSTER.csv',index=False)
    df[df.classification_system=='UNKNOWN'].to_csv(out/'INDUSTRY_GAPS.csv',index=False)
    df.groupby(['classification_system','industry_code','industry_label'],dropna=False).size().rename('stock_count').reset_index().to_csv(out/'INDUSTRY_COUNTS.csv',index=False)
    metrics={'roster_codes':len(df),'capco_codes':int((df.classification_system=='CAPCO_CURRENT_RETROSPECTIVE').sum()),'tdx_supplements':int((df.classification_system=='TDX_INDUSTRY_CURRENT').sum()),'unknown_codes':int((df.classification_system=='UNKNOWN').sum()),'non_unknown_empty_labels':int(((df.classification_system!='UNKNOWN')&(df.industry_label=='')).sum()),'bj_candidates':len(bj_roster),'bj_verified_included':len(bj_roster)-len(bj_missing),'bj_unverified_excluded':len(bj_missing)}
    write(out/'INDUSTRY_PROVENANCE.json',{'metrics':metrics,'capco_metadata':meta,'prior_selector_provenance':str(PRIOR),'prior_selector_roster_sha256':sha(PRIOR),'taxonomy_rule':'CAPCO and TDX separate; UNKNOWN retained','asof':'2026-10-17 CAPCO; TDX per-row original selector date','current_retrospective_not_historical_PIT':True,**SAFE})
    if metrics['non_unknown_empty_labels']:raise RuntimeError('industry labels remain empty')
    return metrics

def action_adjust(raw,prior_dates,dates,events,formula):
    """Each cached action assigned once to its containing observed-bar interval."""
    den=raw.copy();count=np.zeros(len(dates),dtype='int16');matched=['']*len(dates)
    if len(events):
        events=events.sort_values('_date_int',kind='stable').reset_index(drop=True)
        event_dates=events['_date_int'].to_numpy(dtype='int64');indices=np.searchsorted(dates,event_dates,side='left')
        grouped={}
        for j,i in enumerate(indices):
            if i<len(dates) and prior_dates[i]>0 and prior_dates[i]<event_dates[j]<=dates[i]:grouped.setdefault(int(i),[]).append(j)
        for i,js in grouped.items():
            val=pd.Series([den[i]])
            for j in js:
                event=events.iloc[[j]].reset_index(drop=True)
                val,used=formula(event,val)
                if not bool(used.iloc[0]):raise RuntimeError('cached action not applicable by bound formula')
            den[i]=float(val.iloc[0]);count[i]=len(js);matched[i]=';'.join(str(event_dates[j]) for j in js)
    return den,count,matched

def stock_frame(raw,stock,events,calendar,formula):
    if len(raw)%32:raise RuntimeError('non integral day records '+stock['code'])
    bars=np.frombuffer(raw,dtype=DAY_DTYPE);dates=bars['date'].astype('int64')
    if np.any(np.diff(dates)<=0):raise RuntimeError('nonunique/nonmonotonic stock dates '+stock['code'])
    parsed=pd.to_datetime(dates.astype(str),format='%Y%m%d',errors='coerce')
    if parsed.isna().any():raise RuntimeError('invalid stock date '+stock['code'])
    n=len(bars)
    prices=np.column_stack([bars[k]/100 for k in ['open','high','low','close']])
    usable=(prices>0).all(axis=1)&(bars['volume']>0)&np.isfinite(bars['amount'])&(bars['amount']>=0)
    close=prices[:,3];prior=np.full(n,np.nan);prevdate=np.zeros(n,dtype='int64');last=np.nan;lastdate=0
    for i in range(n):
        prior[i],prevdate[i]=last,lastdate
        if usable[i]:last,lastdate=close[i],dates[i]
    # Zero-volume/invalid placeholders are not actual traded bars. An action on
    # such a date belongs to the next usable bar's interval, not the placeholder.
    den=prior.copy();counts=np.zeros(n,dtype='int16');matched=['']*n
    adjusted,used,action_dates=action_adjust(prior[usable],prevdate[usable],dates[usable],events,formula)
    den[usable],counts[usable]=adjusted,used
    for i,value in zip(np.flatnonzero(usable),action_dates):matched[i]=value
    resumed=np.zeros(n,dtype=bool)
    # Observed calendar only: early pre-2006 gap stays explicitly uncertified.
    positions=np.searchsorted(calendar,dates);previous_positions=np.searchsorted(calendar,prevdate)
    prev_in_main=prevdate>=20060101
    resumed=prev_in_main & ((positions-previous_positions)>1)
    first_main=(dates>=20060101)&(prevdate<20060101)&(prevdate>0)
    resumed[first_main]=positions[first_main]>0
    good=usable&np.isfinite(den)&(den>0)&np.isfinite(prior)
    gain=np.full(n,np.nan);gain[good]=close[good]/den[good]-1
    status=np.where(~np.isfinite(prior),'NO_PRIOR_OBSERVED_CLOSE',np.where(~usable,'UNUSABLE_CURRENT_BAR',np.where(~np.isfinite(den)|(den<=0),'NONPOSITIVE_OR_INVALID_DENOMINATOR','RANKABLE')))
    df=pd.DataFrame({'trade_date':parsed.strftime('%Y-%m-%d'),'code':stock['code'],'open':prices[:,0],'high':prices[:,1],'low':prices[:,2],'close':close,'amount':bars['amount'].astype(float),'volume':bars['volume'].astype('int64'),'previous_observed_close':prior,'prior_date':pd.to_datetime(pd.Series(prevdate).replace(0,np.nan),format='%Y%m%d',errors='coerce').dt.strftime('%Y-%m-%d').fillna(''),'adjusted_pre_close_proxy':den,'proxy_gain':gain,'denominator_status':status,'rankable':good,'resumed_interval':resumed,'boundary_interval_calendar_unverified':first_main,'interval_calendar_days':(parsed-pd.to_datetime(pd.Series(prevdate).replace(0,np.nan),format='%Y%m%d',errors='coerce')).dt.days.to_numpy(),'matched_action_count':counts,'matched_action_dates':matched,'action_cache_absence_not_no_action':counts==0,'extreme_gain':np.abs(gain)>0.25,'member_name':stock['member_name'],'industry_key':stock['industry_key'],'classification_system':stock['classification_system'],'industry_code':stock['industry_code'],'industry_label':stock['industry_label'] or 'UNKNOWN','industry_label_original':stock.get('industry_label_original',stock['industry_label']),'name':stock['name'],'classification_source':stock['classification_source'],'industry_asof':stock['as_of']})
    return df[(dates>=20060101)&(dates<=20251231)].copy()

def read_actions():
    df=pd.read_parquet(xd.CACHE_PATH);text=df['_xdxr_trade_date'].astype(str).str.replace(r'\.0$','',regex=True).str[:10];df['_date_int']=text.str.replace('-','',regex=False).astype('int64')
    return {code:g.sort_values('_date_int').reset_index(drop=True) for code,g in df.groupby('ts_code')}

def make_panel(out,roster,calendar,actions,formula,a,claim,limit=None):
    panel=out/'PANEL';panel.mkdir()
    writers={};buffers=[];total=0;stock_count=0;unrankable=0;resumed=0;action_count=0;extreme=0;member_hashes=[]
    def flush():
        nonlocal buffers
        if not buffers:return
        chunk=pd.concat(buffers,ignore_index=True);buffers=[]
        for year,g in chunk.groupby(chunk.trade_date.str[:4]):
            table=pa.Table.from_pandas(g.reset_index(drop=True),preserve_index=False)
            if year not in writers:writers[year]=pq.ParquetWriter(panel/f'year={year}.parquet',table.schema,compression='zstd')
            writers[year].write_table(table)
    try:
      with zipfile.ZipFile(ZIP) as z:
        for stock in roster.to_dict('records')[:limit]:
            payload=z.read(stock['member_name']);member_hashes.append({'code':stock['code'],'member_name':stock['member_name'],'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
            frame=stock_frame(payload,stock,actions.get(stock['code'],pd.DataFrame()),calendar,formula)
            buffers.append(frame);total+=len(frame);stock_count+=1;unrankable+=int((~frame.rankable).sum());resumed+=int(frame.resumed_interval.sum());action_count+=int(frame.matched_action_count.sum());extreme+=int(frame.extreme_gain.sum())
            if stock_count%200==0:
                flush();progress(a,claim,out,'PANEL_SHARD_PROGRESS',{'processed_codes':stock_count,'panel_rows':total})
      flush()
    finally:
      for w in writers.values():w.close()
    pd.DataFrame(member_hashes).to_csv(out/'DERIVATIVE_MEMBER_INPUTS.csv',index=False)
    return {'panel_rows':total,'processed_codes':stock_count,'unrankable_rows':unrankable,'resumed_rows':resumed,'matched_cached_actions':action_count,'extreme_gain_rows':extreme,'year_files':len(writers)}

GRID={'industry':[],'month':['month'],'third':['third'],'month_third':['month','third']}

def aggregate_cells(daily,industries,out,prefix):
    all_year=[]
    for grid,dimensions in GRID.items():
        periods=[{}]
        if 'month' in dimensions:periods=[{**p,'month':m} for p in periods for m in range(1,13)]
        if 'third' in dimensions:periods=[{**p,'third':t} for p in periods for t in range(1,4)]
        rows=[]
        for year in range(2006,2026):
            yd=daily[daily.year==year]
            for cell in periods:
                d=yd
                for k,v in cell.items():d=d[d[k]==v]
                agg=d.groupby('industry_key',observed=True).agg(actual_hits=('actual_hits','sum'),expected_hits=('expected_hits','sum'),valid_stock_days=('n_industry','sum'),support_dates=('n_industry',lambda s:int((s>0).sum())))
                for key in industries:
                    item=agg.loc[key].to_dict() if key in agg.index else {'actual_hits':0,'expected_hits':0.,'valid_stock_days':0,'support_dates':0}
                    rows.append({'industry_key':key,'year':year,**cell,**item})
        annual=pd.DataFrame(rows);annual['lift']=np.divide(annual.actual_hits,annual.expected_hits,out=np.full(len(annual),np.nan),where=annual.expected_hits>0)
        annual['actual_hits']=annual.actual_hits.astype('int64');annual['valid_stock_days']=annual.valid_stock_days.astype('int64');annual['support_dates']=annual.support_dates.astype('int64')
        annual.to_csv(out/f'{prefix}_{grid.upper()}_YEAR.csv',index=False)
        all_year.append(annual)
        summary=[]
        for key in industries:
            for cell in periods:
                g=annual[annual.industry_key==key]
                for k,v in cell.items():g=g[g[k]==v]
                valid=g[g.expected_hits>0];lifts=valid.lift.to_numpy();above=lifts>1+1e-12;equal=np.abs(lifts-1)<=1e-12;below=lifts<1-1e-12
                streak=longest=0
                for row in g.sort_values('year').itertuples():
                    streak=streak+1 if row.expected_hits>0 and row.lift>1+1e-12 else 0;longest=max(longest,streak)
                actual=int(g.actual_hits.sum());expected=float(g.expected_hits.sum())
                summary.append({'industry_key':key,**cell,'actual_hits':actual,'expected_hits':expected,'lift':actual/expected if expected>0 else np.nan,'valid_stock_days':int(g.valid_stock_days.sum()),'support_dates':int(g.support_dates.sum()),'years_supported':len(valid),'years_above_1':int(above.sum()),'years_equal_1':int(equal.sum()),'years_below_1':int(below.sum()),'years_unsupported':20-len(valid),'longest_consecutive_years_above_1':longest})
        summary=pd.DataFrame(summary)
        if int(summary.actual_hits.sum())!=int(daily.actual_hits.sum()) or not np.isclose(summary.expected_hits.sum(),daily.expected_hits.sum(),rtol=0,atol=1e-7):raise RuntimeError('full grid count reconciliation failed '+grid)
        summary.to_csv(out/f'{prefix}_{grid.upper()}.csv',index=False)
    return {'industry_count':len(industries),'full_grid_rows_per_population':len(industries)*(1+12+3+36),'annual_full_grid_rows_per_population':len(industries)*20*(1+12+3+36)}

def rank_and_stats(out,roster,a,claim,canary=False):
    keys=sorted(roster.industry_key.unique());daily=[];companion=[];tops=[];other_tops=[];coverage=[]
    metadata=roster[['industry_key','classification_system','industry_code','industry_label']].drop_duplicates().sort_values('industry_key')
    # UNKNOWN may have blank labels; one normalized description per category.
    metadata['industry_label']=metadata.industry_label.replace('','UNKNOWN');metadata=metadata.drop_duplicates('industry_key')
    metadata.to_csv(out/'INDUSTRY_DICTIONARY.csv',index=False)
    for path in sorted((out/'PANEL').glob('*.parquet')):
        df=pd.read_parquet(path,columns=['trade_date','code','proxy_gain','rankable','resumed_interval','extreme_gain','industry_key','denominator_status'])
        if df.duplicated(['trade_date','code']).any():raise RuntimeError('duplicate panel keys')
        coverage.append({'year':int(path.stem.split('=')[1]),'panel_rows':len(df),'rankable_rows':int(df.rankable.sum()),'unique_codes':df.code.nunique(),'observed_dates':df.trade_date.nunique(),'resumed_rows':int(df.resumed_interval.sum()),'unknown_rows':int((df.industry_key=='UNKNOWN').sum())})
        for exclude,store,topstore in [(False,daily,tops),(True,companion,other_tops)]:
            valid=df[df.rankable & (~df.resumed_interval if exclude else True)].copy()
            valid=valid.sort_values(['trade_date','proxy_gain','code'],ascending=[True,False,True],kind='stable')
            valid['rank']=valid.groupby('trade_date').cumcount()+1
            top=valid[valid['rank']<=100].copy();topstore.append(top)
            counts=valid.groupby(['trade_date','industry_key']).size().rename('n_industry').reset_index()
            nvalid=valid.groupby('trade_date').size();hits=top.groupby(['trade_date','industry_key']).size().rename('actual_hits')
            counts=counts.join(hits,on=['trade_date','industry_key']).fillna({'actual_hits':0})
            counts['n_valid']=counts.trade_date.map(nvalid);counts['daily_slots']=np.minimum(counts.n_valid,100)
            counts['expected_hits']=counts.daily_slots*counts.n_industry/counts.n_valid
            counts['year']=counts.trade_date.str[:4].astype(int);counts['month']=counts.trade_date.str[5:7].astype(int);day=counts.trade_date.str[8:10].astype(int);counts['third']=np.where(day<=10,1,np.where(day<=20,2,3))
            check=counts.groupby('trade_date').agg(actual=('actual_hits','sum'),expected=('expected_hits','sum'),slots=('daily_slots','first'))
            if not np.array_equal(check.actual.to_numpy(),check.slots.to_numpy()) or not np.allclose(check.expected,check.slots,rtol=0,atol=1e-9):raise RuntimeError('daily hit/expected reconciliation failed')
            if not canary and (check.slots!=100).any():raise RuntimeError('full run has day with fewer than100 rankable')
            store.append(counts)
        progress(a,claim,out,'YEAR_RANKED',{'year':coverage[-1]['year'],'panel_rows':len(df)})
    pd.DataFrame(coverage).to_csv(out/'YEAR_COVERAGE.csv',index=False)
    for prefix,ds,ts in [('MAIN',daily,tops),('EX_RESUMED',companion,other_tops)]:
        d=pd.concat(ds,ignore_index=True);t=pd.concat(ts,ignore_index=True)
        if t.duplicated(['trade_date','code']).any():raise RuntimeError('duplicate Top100 keys')
        t.to_csv(out/('DAILY_TOP100.csv.gz' if prefix=='MAIN' else 'DAILY_TOP100_EX_RESUMED.csv.gz'),index=False,compression={'method':'gzip','mtime':0})
        d.to_csv(out/f'{prefix}_DAILY_INDUSTRY_COUNTS.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        grid=aggregate_cells(d,keys,out,prefix)
    main=pd.concat(tops,ignore_index=True);d=pd.concat(daily,ignore_index=True)
    return {**grid,'valid_days':main.trade_date.nunique(),'selection_slots':len(main),'actual_hits':int(d.actual_hits.sum()),'expected_hits':float(d.expected_hits.sum()),'years_in_top100':sorted(main.trade_date.str[:4].astype(int).unique().tolist()),'daily_top100_min':int(main.groupby('trade_date').size().min()),'daily_top100_max':int(main.groupby('trade_date').size().max())}

def report(out,metrics):
    dictionary=pd.read_csv(out/'INDUSTRY_DICTIONARY.csv',keep_default_na=False).set_index('industry_key')
    lines=['# 2006—2025行业月旬描述统计','',f"观察到{metrics['panel_rows']:,}条个股日线；{metrics['valid_days']:,}个有效观察日，Top100共{metrics['selection_slots']:,}条。",'',
    '本结果是 PACKAGE_OBSERVED_DESCRIPTIVE_PROXY：使用包中价格及既有分类1现金/送配事件缓存顺序调整前收盘，不能证明官方完整复权或所有公司行动完备。缺缓存事件不等于无事件。当前行业回看历史，含存续与分类偏差；UNKNOWN单列，不代表历史全市场或交易可执行结果。2026不混入。', '',
    'lift=实际入选次数/逐日按行业有效股票占比计算的期望次数。上中下旬为1–10、11–20、21–月末。零命中和零支持网格均保留；不做显著性检验，不据此声称预测、因果或突破。', '',
    '主表包含复牌间隔；EX_RESUMED排除中间至少缺一个观察交易日的间隔，再独立排名和计算期望。前2006日历边界另标未认证。极值仅标记，不裁剪。', '',
    '以下完整逐行业展开：月与旬前列只是描述排序，仍需结合全年与跨年度支持判断。完整所有月/旬/月×旬与20年网格见同目录CSV。','']
    overall=pd.read_csv(out/'MAIN_INDUSTRY.csv');months=pd.read_csv(out/'MAIN_MONTH.csv');thirds=pd.read_csv(out/'MAIN_THIRD.csv');cells=pd.read_csv(out/'MAIN_MONTH_THIRD.csv');annual=pd.read_csv(out/'MAIN_INDUSTRY_YEAR.csv')
    for r in overall.sort_values('industry_key').itertuples():
        label=dictionary.loc[r.industry_key,'industry_label'];g=months[(months.industry_key==r.industry_key)&(months.expected_hits>0)].sort_values(['lift','month'],ascending=[False,True]).head(3)
        c=cells[(cells.industry_key==r.industry_key)&(cells.expected_hits>0)].sort_values(['lift','month','third'],ascending=[False,True,True]).head(3)
        best='；'.join(f'{int(x.month)}月 lift={x.lift:.3f}，高于1 {int(x.years_above_1)}/{int(x.years_supported)}年，最长{int(x.longest_consecutive_years_above_1)}年' for x in g.itertuples()) or '无有效支持'
        bestcell='；'.join(f'{int(x.month)}月'+{1:'上旬',2:'中旬',3:'下旬'}[int(x.third)]+f' lift={x.lift:.3f}（{int(x.years_above_1)}/{int(x.years_supported)}年）' for x in c.itertuples()) or '无有效支持'
        yr=annual[annual.industry_key==r.industry_key]
        lines += [f'## {label} ({r.industry_key})', '', f'总体命中{int(r.actual_hits):,}，期望{r.expected_hits:.2f}，lift={r.lift:.3f}；有效股票日{int(r.valid_stock_days):,}，支持{int(r.support_dates):,}日。年度高于/等于/低于1：{int(r.years_above_1)}/{int(r.years_equal_1)}/{int(r.years_below_1)}；未支持{int(r.years_unsupported)}年；最长连续高于1为{int(r.longest_consecutive_years_above_1)}年。', '', '月份前列：'+best+'。','', '月旬前列：'+bestcell+'。','', '年度lift：'+'，'.join(f'{int(x.year)}:{x.lift:.2f}' if pd.notna(x.lift) else f'{int(x.year)}:无支持' for x in yr.itertuples())+'。','']
    lines += ['## 覆盖与限制索引','',f'原始年度覆盖与逐文件缺年：{ind.INVENTORY_BATCH}。观察日缺口候选：{ind.PRIOR_BATCH}/outputs/FILE_YEAR_OBSERVED_GAP_CANDIDATES.csv；缺口不能区分上市前、停牌、退市或缺采。',f"最终行业补漏与BJ处置：{frozen_industry()['manifest_path']}。", '', '安全状态：research_only=true，production_allowed=false，buy_gate_updated=false，orders_sent=false。独立Reviewer验收另行记录。']
    (out/'REPORT.md').write_text('\n'.join(lines))

def execute(stage,attempt):
    out=STUDY/f'{stage}_{attempt}';out.mkdir(parents=True,exist_ok=False)
    env,a,claim=adapter_for(stage,out);started=now();metrics={};err=None
    try:
        verify_inputs()
        if stage=='TEST':
            result=subprocess.run([sys.executable,'-m','unittest','research.a_share_top100_environment_v1.tests.test_hsjday_history_stats'],cwd=WT,text=True,capture_output=True)
            (out/'TEST_OUTPUT.txt').write_text(result.stdout+result.stderr)
            metrics={'returncode':result.returncode}
            if result.returncode:raise RuntimeError('focused regression failed')
        elif stage=='INDUSTRY':metrics=industry(out)
        elif stage in ['CANARY','FULL']:
            roster=load_roster()
            if stage=='FULL':
                canary_path=Path(read(STUDY/'CONFIG.json')['canary_receipt_path']);canary=read(canary_path)
                manifest=read(canary_path.parent/'OUTPUT_MANIFEST.json')
                if manifest['files'][str(canary_path)]!=sha(canary_path):raise RuntimeError('canary receipt drift')
                if canary['status']!='COMPLETED' or canary['code_sha256']!=sha(CODE) or canary['input_binding_sha256']!=sha(STUDY/'INPUT_BINDING.json') or canary['config_sha256']!=sha(STUDY/'CONFIG.json'):raise RuntimeError('matching frozen canary not complete')
            calendar=pd.read_csv(CAL)
            datefield=next(c for c in calendar.columns if c in ['trade_date','date','market_date'])
            dates=pd.to_datetime(calendar[datefield]).dt.strftime('%Y%m%d').astype('int64').sort_values().to_numpy()
            actions=read_actions();formula=exact_formula()
            metrics=make_panel(out,roster,dates,actions,formula,a,claim,limit=8 if stage=='CANARY' else None)
            metrics.update(rank_and_stats(out,roster,a,claim,canary=stage=='CANARY'))
            if stage=='FULL' and metrics['years_in_top100']!=list(range(2006,2026)):raise RuntimeError('not all20years')
            if stage=='FULL':report(out,metrics)
        else:raise ValueError(stage)
    except BaseException as exc:
        err=exc;(out/'TRACEBACK.txt').write_text(traceback.format_exc())
    receipt={'status':'COMPLETED' if err is None else 'CANNOT_CONFIRM','stage':stage,'execution_id':claim.execution_id,'started':started,'finished':now(),'metrics':metrics,'failure':None if err is None else str(err),'code_sha256':sha(CODE),'contract_sha256':sha(STUDY/'CONTRACT.json'),'input_binding_sha256':sha(STUDY/'INPUT_BINDING.json'),'config_sha256':sha(STUDY/'CONFIG.json'),'industry_roster_sha256':frozen_industry()['roster_sha256'],'truth_access':'EXPOSED_HISTORY_DESCRIPTIVE' if stage in ['CANARY','FULL'] else 'NOT_READ',**SAFE}
    write(out/'RUN_RECEIPT.json',receipt)
    ctx=dict(a._route_context[claim.execution_id]);ctx['artifact_paths']=list(set(ctx['artifact_paths']+[str(p) for p in out.rglob('*') if p.is_file()]));ctx['deterministic_metrics']={**ctx['deterministic_metrics'],**metrics};a.set_route_context(claim.execution_id,ctx)
    a.terminalize(claim,{'experiment_id':EXPERIMENT,'execution_id':claim.execution_id,'arm':stage,'contract_sha256':ctx['contract_sha256'],'input_binding_sha256':ctx['input_binding_sha256'],'execution_kind':'NO_FIT_ENGINEERING_CANARY','terminal_status':receipt['status'],'real_fit_returned_count':0,'fit_call_entered':False,'finished_at':now(),'error_type':str(err) if err else None})
    with a._locked():raw,events=a._read_validated()
    route=[e for e in events if e.get('route_id')==ROUTE]
    counts=lambda es:{'routes':len({e.get('route_id') for e in es}),'events':len(es),'executions':len({e.get('execution_id') for e in es}),'corrections':sum(e.get('event_type')=='CORRECTION' for e in es)}
    write(out/'REGISTRY_SNAPSHOT.json',{'registry_sha256':hashlib.sha256(raw).hexdigest(),'global':counts(events),'route':counts(route),'route_unregistered_count':0,'route_contract_only_count':0,'global_unregistered_count':None,'global_contract_only_count':None,'null_reason':'GLOBAL_INVENTORY_NOT_AUDITED','external_or_git_persistence':'CANNOT_CONFIRM','terminal_event_sha256':next(e['event_sha256'] for e in reversed(events) if e.get('execution_id')==claim.execution_id and e.get('event_type')=='TERMINAL')})
    write(out/'OUTPUT_MANIFEST.json',{'status':receipt['status'],'files':{str(p):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},'derived_research_not_authority':True,**SAFE})
    print(json.dumps({'out':str(out),**receipt},ensure_ascii=False),flush=True)
    return 0 if err is None else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','TEST','CANARY','FULL']);p.add_argument('--attempt',default='01');p.add_argument('--study',type=Path,required=True);p.add_argument('--industry-manifest',type=Path);args=p.parse_args()
    STUDY=args.study.resolve()
    if not STUDY.is_relative_to(BATCH.resolve()):p.error('--study must remain inside the approved batch')
    if args.stage=='freeze':
        if args.industry_manifest is None:p.error('freeze requires --industry-manifest')
        freeze(args.industry_manifest)
    else:raise SystemExit(execute(args.stage,args.attempt))
