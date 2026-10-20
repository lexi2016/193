"""S1 lifecycle through the existing locked, fsync official adapter."""
from __future__ import annotations
import hashlib,importlib.util,json,platform,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/LOCAL_ONLY/factor_factory')
BATCH=ROOT/'research/a_share_model_selection_v1/batches/g1_prior_diagnostic_20261020'
ROUTE='a-share-g1-prior-diagnostic-20261020'
EXPERIMENT='A_SHARE_G1_PRIOR_DIAGNOSTIC_20261020'
FLAGS=dict(research_only=True,production_allowed=False,buy_gate_updated=False,orders_sent=False)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,data):
    import os,tempfile
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(data,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()


def load_adapter():
    binding=json.loads((BATCH/'INPUT_BINDING.json').read_text())
    for d in binding['registry_dependencies'].values():
        if sha(d['path'])!=d['sha256']:raise ValueError('registry dependency drift')
    p=Path(binding['registry_dependencies']['adapter']['path']);sys.path.insert(0,str(p.parent))
    from registry_adapter import OfficialRegistryAdapter
    return OfficialRegistryAdapter(contract_path=BATCH/'CONTRACT.json',input_binding_path=BATCH/'INPUT_BINDING.json')


def append(execution,event_type,stage,receipt,*,decision='RUNNING',metrics=None,truth='NOT_READ',checkpoint=None):
    contract=json.loads((BATCH/'CONTRACT.json').read_text())
    identity=json.loads((BATCH/'FROZEN_IDENTITY.json').read_text())
    spec=dict(schema_version='1.0',registry_event_id=execution+'_'+event_type,
       route_id=ROUTE,experiment_id=EXPERIMENT,execution_id=execution,track_id='A_SHARE_PANEL_MODEL_SELECTION_RESEARCH',
       stage=stage,event_type=event_type,registration_mode='LIVE_EXECUTION',arm='S1_SCOPE',
       occurred_at=datetime.now(timezone.utc).isoformat(),occurred_at_reason='LIVE_EXECUTION_UTC',
       contract_path=str(BATCH/'CONTRACT.json'),contract_sha256=sha(BATCH/'CONTRACT.json'),
       input_binding_sha256=sha(BATCH/'INPUT_BINDING.json'),input_sha256=sha(BATCH/'INPUT_BINDING.json'),
       code_sha256=identity['code_composite_sha256'],config_sha256=sha(BATCH/'CONFIG.json'),
       baseline_id='PV_ENVIRONMENT_R1_M0_FORMAL_EXACT_4THREAD',one_variable='FROZEN_PRIOR_RESIDUAL_SPREAD_WITHIN_M0_TOP50_RANK_BANDS',
       signal_date_range=contract['SIGNAL_DATE_RANGE'],target_date_range=contract['TARGET_DATE_RANGE'],
       truth_access=truth,sealed_status='EXPOSED_HISTORY_NO_SEALED_DATA',
       runtime=dict(python=sys.executable,version=platform.python_version(),threads=4),device='CPU_RESEARCH_ONLY',
       artifact_paths=[str(BATCH/'CONTRACT.json'),str(BATCH/'INPUT_BINDING.json'),str(receipt)],checkpoint_ref=checkpoint,
       deterministic_metrics={'truth_positive_denominator':None,'selection_slots':None,**(metrics or {})},
       decision=decision,stop_reason='NONE' if decision=='RUNNING' else decision,
       evidence_level='ENGINEERING_CANARY_ONLY' if stage in ['CANARY','SYNTHETIC_TESTS'] else 'DIRECTION_ONLY',
       input_identity={'code_by_file':identity['code_by_file'],'null_reasons':{
         'truth_positive_denominator':'RETURN_REGRESSION_NOT_CLASSIFICATION',
         'target_date_range':'OUTCOMES_USE_BOUND_OBSERVED_CALENDAR_NO_FUTURE_SOURCE',
         'checkpoint_ref':'NO_CHECKPOINT_YET' if checkpoint is None else 'BOUND_CHECKPOINT'}},**FLAGS)
    load_adapter().append_parity_event(spec,execution_id=execution)


def sync():
    registry=ROOT/'contracts/EXPERIMENT_REGISTRY.jsonl'
    events=[json.loads(l) for l in registry.read_text().splitlines() if l.strip()]
    scoped=[e for e in events if e.get('route_id')==ROUTE]
    def counts(es):return dict(routes=len({e['route_id'] for e in es}),events=len(es),executions=len({e['execution_id'] for e in es}),corrections=sum(e['event_type']=='CORRECTION' for e in es))
    started={e['execution_id'] for e in scoped if e['event_type']=='RUN_STARTED'};terminal={e['execution_id'] for e in scoped if e['event_type']=='TERMINAL'}
    snapshot=dict(registry_path=str(registry),registry_sha256=sha(registry),global_counts=counts(events),scoped_counts=counts(scoped),open_executions=sorted(started-terminal),unregistered_routes=0,contract_only=0,global_unregistered_routes=None,global_contract_only=None,global_inventory_reason='NOT_AUDITED_THIS_SCOPE',external_persistence='CANNOT_CONFIRM',git_sync='NOT_SYNCED_NO_REMOTE',**FLAGS)
    path=BATCH/'REGISTRY_SNAPSHOT.json';write(path,snapshot)
    marker='<!-- G1_PRIOR_DIAGNOSTIC_20261020 -->';endmarker='<!-- END_G1_PRIOR_DIAGNOSTIC_20261020 -->'
    resultpath=BATCH/'EVALUATION/METRICS.json';result=json.loads(resultpath.read_text()) if resultpath.exists() else {}
    paragraph=(f'{marker}\n## 2026-10-20 G1冻结行业月份先验增量诊断\n\n'
     f'- 独立开发诊断，exact industry_month_log_relative_lift，M0 Top50五rank层高低先验组h2残差差；5bp信息资源筛选线，不是Top10收益门。fit=0，无新训练，不改变S1失败，S2未启动。当前机器决策 `{result.get("decision","FROZEN_EXECUTION_PENDING")}`，独立复核/展示状态以本批HANDOFF为准。\n'
     f'- Registry SHA `{snapshot["registry_sha256"]}`；global route/event/execution/correction={list(snapshot["global_counts"].values())}，本route={list(snapshot["scoped_counts"].values())}，open={snapshot["open_executions"]}；scope unregistered/contract-only=0/0，global两项与外部持久化CANNOT_CONFIRM。snapshot `{path}`。\n'
     '- DIRECTION_ONLY；既有PIT/日历/universe/毛收益限制保留。research_only=true、production_allowed=false、buy_gate_updated=false、orders_sent=false；NOT_SYNCED_NO_REMOTE。\n'+endmarker)
    for name in ['CURRENT_STATE.md','HK_OHLCV_RESEARCH_STATE.md','PROGRESS.md']:
        p=ROOT/name;s=p.read_text()
        if marker in s:
            begin=s.index(marker);end=s.index(endmarker,begin)+len(endmarker);s=s[:begin]+paragraph+s[end:]
        else:s=s.rstrip()+'\n\n'+paragraph+'\n'
        p.write_text(s)
    p=ROOT/'research/a_share_model_selection_v1/RESOURCE_INDEX.json';d=json.loads(p.read_text());d['g1_prior_diagnostic_20261020']=dict(batch_path=str(BATCH),registry_snapshot=str(path),decision=result.get('decision','FROZEN_EXECUTION_PENDING'),**snapshot);write(p,d)
    return snapshot
