"""Flags-only diagnosis of the original 24 excluded A/C dates; no source substitution."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import sys

# Reuse the accepted immutable implementation; the isolated checkout is sparse.
SUPPORT_DIR=Path(__file__).resolve().parents[3]/'month-bonus-stability-20261021/research/a_share_model_selection_v1'
sys.path.insert(0,str(SUPPORT_DIR))
import environment_tilt_month_bonus_stability as previous
from environment_tilt_mac_registry import Lifecycle, FLAGS, canonical, sha

PRIMARY=previous.PRIMARY
ARMS=previous.EXP_POLICY['arms']
CAPS=dict(previous.EXP_CAPS,max_exact_rows=128)
EXECUTION='MONTH_BONUS_MISSING24_DIAG_20261023_01'
POLICY=dict(stage='CANONICAL_AND_ORIGINAL_RAW_EXACT_FLAGS',missing_dates=24,missing_signal_code_keys=26,
    successful_primary_dates=461,nominal_primary_dates=485,horizon=2,pair_batch_size=64,
    source_ids=['canonical','original_raw','backtest_retention','selector_retention'],no_returns_output=True,no_source_repair=True,
    no_score_or_model_read=True,no_network=True,market_gate=False)

class Blocked(ValueError):pass


def missing_metadata(ledger,selections,calendar=None):
    """Accepted missing fields and saved selections, never rerank scores."""
    expected=[];primary=[r for f in PRIMARY for r in ledger['rows_by_fold'][f]]
    if len(primary)!=485 or sum(bool(r['valid'])for r in primary)!=461:
        raise Blocked('ACCEPTED_PRIMARY_LEDGER_COUNT_CONFLICT')
    dates=previous.old.calendar_dates(calendar)if calendar is not None else None
    for fold in PRIMARY:
        selected={r['date']:r for r in selections[fold]['records']}
        if len(selected)!=len(selections[fold]['records']):raise Blocked('DUPLICATE_SAVED_SELECTION_DATE')
        for row in ledger['rows_by_fold'][fold]:
            if row['valid']:continue
            record=selected[row['date']]
            if(row['reason']!='MISSING_SELECTED_H2_LABEL_ENTIRE_AC_PAIR_EXCLUDED'
                    or row['fold']!=fold or record['fold']!=fold
                    or row['calendar_index']!=record['calendar_index']
                    or record['reason']is not None or record['selections']is None):
                raise Blocked('ACCEPTED_MISSING_REASON_OR_SELECTION_CONFLICT')
            if set(record['selections'])!=set(ARMS)or any(len(x)!=10 or len(set(x))!=10 for x in record['selections'].values()):
                raise Blocked('FROZEN_TEN_SLOT_SELECTION_REQUIRED')
            if dates is not None:
                pos=dates.index(previous.old._day(row['date']))
                if pos!=row['calendar_index']or dates[pos+2].isoformat()!=record['h2_date']:
                    raise Blocked('EXACT_SHARED_CALENDAR_TPLUS2_CONFLICT')
            for code in row['missing_codes']:
                arms=[arm for arm in ARMS if code in record['selections'][arm]]
                if not arms:raise Blocked('MISSING_CODE_NOT_ORIGINALLY_SELECTED')
                expected.append(dict(fold=fold,signal_date=row['date'],target_date=record['h2_date'],
                    calendar_index=row['calendar_index'],code=code,arms=arms))
    if len(expected)!=26 or len({r['signal_date']for r in expected})!=24 or len({(r['signal_date'],r['code'])for r in expected})!=26:
        raise Blocked('FROZEN_24_DATE_26_KEY_SCOPE_CONFLICT')
    return expected


def source_pairs(keys,kind):
    pairs={(d,r['code'])for r in keys for d in (r['signal_date'],r['target_date'])}
    if kind in ['int64_YYYYMMDD','int32_YYYYMMDD']:return sorted((int(d.replace('-','')),code)for d,code in pairs)
    if kind=='string_YYYYMMDD':return sorted((d.replace('-',''),code)for d,code in pairs)
    if kind!='timestamp':raise Blocked('UNSUPPORTED_FROZEN_DATE_KIND')
    return sorted(pairs)


def decode_exact_table(dataset,predicate,columns):
    import pandas as pd
    # Arrow-backed floats preserve null masks separately from IEEE NaN.
    return dataset.to_table(columns=columns,filter=predicate).to_pandas(types_mapper=pd.ArrowDtype)


def read_exact_source(adapter,c,spec,pairs):
    """Disjoint batches <=64: count every batch and total-cap before close decode."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    a=c.setdefault('_access',{}).setdefault('sources',{}).setdefault(spec['id'],{})
    ref=spec['ref'];dc,cc=spec['date_column'],spec['code_column'];kind=spec['date_kind']
    before=adapter.authenticate(ref);dataset=ds.dataset(ref['path'],format='parquet')
    dt,ct=dataset.schema.field(dc).type,dataset.schema.field(cc).type
    string_code=pa.types.is_string(ct)or pa.types.is_large_string(ct)or(pa.types.is_dictionary(ct)and pa.types.is_string(ct.value_type))
    valid_date=(pa.types.is_timestamp(dt)if kind=='timestamp'else pa.types.is_int64(dt)if kind=='int64_YYYYMMDD'else pa.types.is_int32(dt)if kind=='int32_YYYYMMDD'else (pa.types.is_string(dt)or pa.types.is_large_string(dt))if kind=='string_YYYYMMDD'else False)
    if not string_code or not valid_date:raise Blocked('FROZEN_SOURCE_DATE_CODE_SCHEMA_CONFLICT')
    if pq.ParquetFile(ref['path']).metadata.num_rows!=spec['file_rows']:raise Blocked('FROZEN_SOURCE_ROW_METADATA_CONFLICT')
    if spec.get('expected_bytes')is not None and before['bytes']!=spec['expected_bytes']:raise Blocked('FROZEN_SOURCE_BYTE_IDENTITY_CONFLICT')
    a['arrow_schema']={name:str(dataset.schema.field(name).type)for name in [dc,cc,'close']}
    if spec.get('expected_close_type')and a['arrow_schema']['close']!=spec['expected_close_type']:raise Blocked('FROZEN_CLOSE_PRECISION_SCHEMA_CONFLICT')
    pairs=sorted(set(pairs));batches=[pairs[i:i+64]for i in range(0,len(pairs),64)];counts=[];predicates=[]
    a.update(count_attempted=True,requested_keys=len(pairs),batch_count=len(batches),close_decode_attempted=False)
    for batch in batches:
        predicate=None
        for day,code in batch:
            value=pd.Timestamp(day).to_pydatetime()if kind=='timestamp'else day
            term=(ds.field(dc)==value)&(ds.field(cc)==code)
            predicate=term if predicate is None else predicate|term
        counts.append(dataset.count_rows(filter=predicate));predicates.append(predicate)
    total=sum(counts);a.update(count_returned=True,matched_rows=total)
    if previous.old.support.stat_identity(ref['path'])!=before:raise Blocked('EXACT_COUNT_SOURCE_IDENTITY_DRIFT')
    if total>CAPS['max_exact_rows']:raise Blocked('TOTAL_EXACT_PREDECODE_ROW_CAP')
    parts=[]
    for predicate,count in zip(predicates,counts):
        adapter.authenticate(ref)
        a['close_decode_attempted']=True
        frame=decode_exact_table(dataset,predicate,[dc,cc,'close'])
        if len(frame)!=count:raise Blocked('EXACT_COUNT_DECODE_PARITY_CONFLICT')
        parts.append(frame)
    frame=pd.concat(parts,ignore_index=True)if parts else pd.DataFrame(columns=[dc,cc,'close'])
    requested={(pd.Timestamp(d)if kind=='timestamp'else d,code)for d,code in pairs}
    if len(frame)!=total or not set(frame[[dc,cc]].itertuples(index=False,name=None)).issubset(requested):
        raise Blocked('UNREQUESTED_EXACT_KEY_OR_ROW_COUNT_CONFLICT')
    if previous.old.support.stat_identity(ref['path'])!=before:raise Blocked('EXACT_DECODE_SOURCE_IDENTITY_DRIFT')
    a['close_decode_returned']=True;frame=frame.rename(columns={dc:'date',cc:'code'})
    frame['date']=pd.to_datetime(frame.date.astype(str),format='%Y%m%d')if kind!='timestamp'else pd.to_datetime(frame.date)
    return frame[['date','code','close']]


def flags(frame,day,code):
    import pandas as pd
    values=frame.loc[(frame.date==pd.Timestamp(day))&(frame.code==code),'close'];count=len(values)
    nulls=sum(v is None or v is pd.NA for v in values)
    finite=count==1 and nulls==0 and math.isfinite(float(values.iloc[0]))
    positive=finite and float(values.iloc[0])>0
    issue='DUPLICATE_KEY'if count>1 else 'BAR_ABSENT_STATUS_UNKNOWN'if count==0 else 'CLOSE_NULL'if nulls else 'CLOSE_NONFINITE'if not finite else 'CLOSE_NONPOSITIVE'if not positive else 'VALID_CLOSE'
    return dict(row_count=count,null_count=nulls,present=count>0,close_finite=bool(finite),close_positive=bool(positive),issue=issue)


def diagnostic_rows(keys,canonical_frame,raw_frame,producer_finite=None):
    """Persist flags and exact parity only, never numeric closes or returns."""
    import pandas as pd
    result=[]
    for r in keys:
        cf={s:flags(canonical_frame,r[s+'_date'],r['code'])for s in ['signal','target']}
        rf={s:flags(raw_frame,r[s+'_date'],r['code'])for s in ['signal','target']};parity={}
        for side in ['signal','target']:
            day=pd.Timestamp(r[side+'_date']);code=r['code']
            left=canonical_frame.loc[(canonical_frame.date==day)&(canonical_frame.code==code),'close']
            right=raw_frame.loc[(raw_frame.date==day)&(raw_frame.code==code),'close']
            if len(left)==len(right)==1 and cf[side]['null_count']==rf[side]['null_count']==0:
                lv,rv=float(left.iloc[0]),float(right.iloc[0])
                parity[side]=lv==rv or(math.isnan(lv)and math.isnan(rv))
            else:parity[side]=None
        conflict=any(cf[x]!=rf[x]or parity[x]is False for x in ['signal','target'])
        issue=next((s.upper()+'_'+cf[s]['issue']for s in ['signal','target']if cf[s]['issue']!='VALID_CLOSE'),None)
        current=(producer_finite or {}).get((r['signal_date'],r['code']))
        cause=('SEMANTIC_CONFLICT_CANONICAL_VS_ORIGINAL_RAW_PROJECTION'if conflict else
            'ORIGINAL_RAW_AND_CANONICAL_'+issue if issue else
            'SEMANTIC_CONFLICT_ORIGINAL_MISSING_LABEL_NOW_FINITE'if current is True else
            'PRODUCER_LABEL_NONFINITE_WITH_VALID_CLOSES'if current is False else
            'PRODUCER_LABEL_NOT_EVALUATED_DUPLICATE_OR_UNAVAILABLE')
        result.append(dict(**r,canonical=cf,original_raw=rf,exact_close_parity=parity,
            original_producer_label_finite=current,cause=cause,recovery_status='NOT_REPAIRED_DIAGNOSTIC_ONLY',
            suspension_or_collection_status='STATUS_UNKNOWN_NO_BOUND_STATUS_FIELD'))
    return result


class RecoveryLifecycle(previous.ExpansionLifecycle):
    def event(self,*args):
        e=Lifecycle.event(self,*args);started=e['event_type']=='RUN_STARTED';a={}if started else self.config.get('_access',{})
        e.update(route_id='a-share-month-bonus-missing24-recovery-16',experiment_id='MONTH_BONUS_MISSING_24_16',stage=POLICY['stage'],
            arm='ORIGINAL_FIXED_A_C_24_MISSING_DATES',baseline_id='ACCEPTED_MONTH_BONUS_STABILITY15_DERIVED_LEDGER',
            one_variable='EXPLAIN_ORIGINAL_26_SELECTED_MISSING_LABEL_KEYS_NO_RULE_OR_SOURCE_REPAIR',
            truth_access='NOT_READ'if started else 'EXACT_SELECTED_CLOSE_FLAGS_NO_NUMERIC_RETURNS'if any(x.get('close_decode_attempted')for x in a.get('sources',{}).values())else 'METADATA_OR_EXACT_KEY_COUNTS_ONLY',
            sealed_status='OPENBOOK_HISTORY_NO_SEALED_CLAIM',evidence_level='OPENBOOK_DIAGNOSTIC_ONLY',checkpoint_ref=None)
        e['input_identity'].update(access=a,old_accepted_refs=self.config['old_accepted_refs'])
        e['input_identity']['null_reasons'].update(checkpoint_ref='NO_MODEL_OR_SCORE_CHECKPOINT_ACCESS',selection_slots='EXISTING_SELECTIONS_ONLY_NO_NEW_SELECTION',truth_positive_denominator='FLAGS_ONLY_NO_RETURN_POSITIVITY_CLASSIFICATION')
        returned=a.get('returned_target_dates',[]);requested=a.get('requested_target_dates',[])
        e['target_date_range']=dict(count=len(returned),start=min(returned)if returned else None,end=max(returned)if returned else None,scope='NO_TARGET_CLOSE_ROWS_RETURNED_BEFORE_START'if started else 'EXACT_FROZEN_FOUR_SOURCE_TARGET_FLAGS_ONLY_NO_NUMERIC_RETURNS',requested_count=len(requested),requested_key_count=0 if started else 26)
        e['deterministic_metrics'].update(diagnosed_keys=0 if started else a.get('diagnosed_keys',0),new_fit_count=0,new_predict_count=0,network_calls=0)
        return e


    def checkpoint(self,artifact,source_id):
        self.require();event=self.event('CHECKPOINT','EXACT_SOURCE_FLAGS_CHECKPOINT_SAVED',artifact['path'],dict(source_id=source_id))
        event['registry_event_id']=self.config['execution_id']+'_CHECKPOINT_'+source_id
        event['input_identity']['diagnostic_flags_checkpoint']=artifact
        with self.adapter._locked():
            raw,events=self.adapter._read_validated()
            same=[e for e in events if e.get('execution_id')==self.config['execution_id']]
            starts=[e for e in same if e['event_type']=='RUN_STARTED']
            if len(starts)!=1 or any(e['event_type']=='TERMINAL'or e['registry_event_id']==event['registry_event_id']for e in same):
                raise Blocked('ACTIVE_UNIQUE_DIAGNOSTIC_CHECKPOINT_REQUIRED')
            if any(starts[0][k]!=event[k]for k in ['contract_sha256','input_binding_sha256','code_sha256','config_sha256']):
                raise Blocked('DIAGNOSTIC_CHECKPOINT_START_IDENTITY_DRIFT')
            self.adapter._append_unlocked(raw,events,event)


def reuse_accepted_inputs(adapter,c):
    """Prior post-START file identity+unchanged stat; no full panel audit/hash rerun."""
    refs={s['id']:s['ref']for s in c['source_specs']if s['id']in ['canonical','original_raw']};refs['calendar']=c['inputs']['calendar']
    for name,ref in refs.items():
        receipt=adapter.metadata(c['accepted_binding_receipts'][name]);known=receipt['input_bindings'].get(ref['path'])
        if not known or known['sha256']!=ref['sha256']or known['stat']!=previous.old.support.stat_identity(ref['path']):
            raise Blocked('ACCEPTED_INPUT_FAST_BIND_IDENTITY_DRIFT')
        adapter.bound[ref['path']]=dict(sha256=known['sha256'],stat=dict(known['stat']))


def diagnose(c,life):
    import pandas as pd
    from collections import Counter
    adapter=previous.old.BoundFiles(c,previous.Session(life));a=c.setdefault('_access',{})
    oldrefs=c['old_accepted_refs'];accepted=adapter.metadata(oldrefs['actual_review'])
    if accepted['verdict']!='PASS_INDEPENDENT_ACTUAL_DERIVED_RESULTS_AND_REGISTRY':raise Blocked('PREVIOUS_ACTUAL_ACCEPTANCE_REQUIRED')
    binding=adapter.metadata(c['original_binding_ref']);projection=adapter.metadata(c['original_projection_ref']);prep=adapter.metadata(c['preparation_ref'])
    bound=binding['sources']['price_panel'];raw_ref=c['source_specs'][1]['ref'];can_ref=c['source_specs'][0]['ref']
    if any(bound[k]!=raw_ref[k]for k in ['path','sha256'])or bound['column_map']['date']!='trade_date'or bound['column_map']['code']!='ts_code':
        raise Blocked('ORIGINAL_RAW_PRODUCER_BINDING_CONFLICT')
    if projection['panels']['assembled_qfq']['path']!=raw_ref['path']or prep['status']!='COMPLETED'or prep['cache_files'].get(can_ref['path'])!=can_ref['sha256']:
        raise Blocked('ORIGINAL_PROJECTION_OR_PREPARATION_CONFLICT')
    ledger=adapter.metadata(oldrefs['ledger']);selections={f:adapter.metadata(ref)for f,ref in oldrefs['selection_refs'].items()}
    frozen_keys=adapter.metadata(c['missing_key_metadata_ref'])['keys'];reuse_accepted_inputs(adapter,c)
    calendar=adapter.frame(c['inputs']['calendar'],columns=['date','calendar_index']);calendar.date=pd.to_datetime(calendar.date)
    keys=missing_metadata(ledger,selections,calendar)
    if keys!=frozen_keys:raise Blocked('FROZEN_MISSING_KEY_METADATA_DRIFT')
    a['old_valid_primary_records_logical_sha256']=previous.old.support.digest([r for f in PRIMARY for r in ledger['rows_by_fold'][f]if r['valid']]);a['old_valid_primary_dates']=461
    frames={};target_keys={(r['target_date'],r['code'])for r in keys};a['requested_target_dates']=sorted({d for d,code in target_keys})
    for spec in c['source_specs']:
        frame=read_exact_source(adapter,c,spec,source_pairs(keys,spec['date_kind']));frames[spec['id']]=frame
        a['returned_target_dates']=sorted(set(a.get('returned_target_dates',[]))|{previous.old._day(d).isoformat()for d,code in frame[['date','code']].itertuples(index=False,name=None)if(previous.old._day(d).isoformat(),code)in target_keys})
        saved=Path(c['run_dir'])/(spec['id']+'_FLAGS_CHECKPOINT.json')
        value=dict(source_ref=spec['ref'],keys=[dict(date=r['signal_date'],target_date=r['target_date'],code=r['code'],signal=flags(frame,r['signal_date'],r['code']),target=flags(frame,r['target_date'],r['code']))for r in keys],**FLAGS)
        with saved.open('x')as handle:handle.write(canonical(value)+'\n');handle.flush();os.fsync(handle.fileno())
        life.checkpoint(dict(path=str(saved),sha256=sha(saved)),spec['id'])
    recipe=previous.old.support._load(c['recipe_ref']['path'],'missing24_original_label_recipe');panel=frames['canonical'];finite={}
    if not panel.duplicated(['date','code']).any():
        producer_panel=panel.copy();producer_panel['close']=producer_panel.close.to_numpy(dtype=float,na_value=float('nan'))
        selected=pd.DataFrame([(r['signal_date'],r['code'])for r in keys],columns=['date','code']);selected.date=pd.to_datetime(selected.date)
        labels=recipe.build_close_return_labels(producer_panel,[2],calendar=calendar,signal_keys=selected)[2]
        finite={(previous.old._day(d).isoformat(),code):math.isfinite(float(v))for d,code,v in labels[['date','code','label_return']].itertuples(index=False,name=None)}
    rows=diagnostic_rows(keys,panel,frames['original_raw'],finite);a['diagnosed_keys']=len(rows)
    overlap=adapter.metadata(c['overlap_metadata_ref'])['keys']
    if overlap!=[dict(date=r['signal_date'],code=r['code'])for r in keys]:raise Blocked('FROZEN_SELECTED_SIGNAL_OVERLAP_CONFLICT')
    retention={}
    for name in ['backtest_retention','selector_retention']:
        frame=frames[name];details=[]
        for r in keys:
            value=dict(date=r['signal_date'],target_date=r['target_date'],code=r['code'],signal=flags(frame,r['signal_date'],r['code']),target=flags(frame,r['target_date'],r['code']))
            left=panel.loc[(panel.date==pd.Timestamp(r['signal_date']))&(panel.code==r['code']),'close']
            right=frame.loc[(frame.date==pd.Timestamp(r['signal_date']))&(frame.code==r['code']),'close']
            value['frozen_signal_overlap_exact_close_equal']=(float(left.iloc[0])==float(right.iloc[0]))if flags(panel,r['signal_date'],r['code'])['issue']==value['signal']['issue']=='VALID_CLOSE'else None
            details.append(value)
        retention[name]=dict(status='RETENTION_DIAGNOSTIC_ONLY',label_repair_admission='CANNOT_CONFIRM_PENDING_ORIGINAL_QFQ_BRIDGE',keys=details,precision=a['sources'][name]['arrow_schema']['close'],new_label_values=0)
    return dict(status='MISSING24_EXACT_KEY_DIAGNOSTIC_COMPLETE',rows=rows,cause_counts=dict(Counter(r['cause']for r in rows)),
        retention=retention,restored_dates=0,evaluation_delta=None,old_successful_dates_recomputed=0,source_repair_performed=False,input_bindings=adapter.bound,
        source_status='FOUR_FROZEN_SOURCES_DIAGNOSED_RETENTION_QFQ_LABEL_BRIDGE_NOT_ADMITTED',
        new_fit_count=0,new_predict_count=0,network_calls=0,**FLAGS)


def metadata_config(path):
    c=previous.resources.read_json(path);c.update(config_path=str(Path(path).resolve()),config_sha256=sha(path))
    if c['policy']!=POLICY or c['caps']!=CAPS or c['execution_id']!=EXECUTION or c['network_allowed']is not False or c['fit_count']!=0 or c['predict_count']!=0:
        raise Blocked('FROZEN_DIAGNOSTIC_POLICY_DRIFT')
    root=Path(c['run_dir'])
    if root!=Path(c['scope_root'])/'DIAGNOSTIC_01'or any(p.is_symlink()for p in(root,*root.parents)):
        raise Blocked('EXACT_OWNED_DIAGNOSTIC_OUTPUT_REQUIRED')
    refs=[*c['code_refs'].values(),c['contract_ref'],c['recipe_ref'],c['lineage_review_ref']]
    if any(sha(r['path'])!=r['sha256']for r in refs):raise Blocked('FROZEN_CODE_OR_DOMAIN_CONTRACT_DRIFT')
    domain=previous.resources.read_json(c['lineage_review_ref']['path'])
    if domain.get('gate_verdict')!='ALLOW_EXACTKEY_RETENTION_DIAGNOSTIC_ONLY'or domain.get('label_repair_admission')!='CANNOT_CONFIRM_PENDING_ORIGINAL_QFQ_BRIDGE':raise Blocked('FROZEN_RETENTION_DOMAIN_ADMISSION_REQUIRED')
    contract=previous.resources.read_json(c['contract_ref']['path'])
    if contract['policy']!=POLICY or contract['caps']!=CAPS or contract['source_specs']!=c['source_specs']or contract['lineage_review_ref']!=c['lineage_review_ref']:
        raise Blocked('SOURCE_ADMISSION_CONTRACT_DRIFT')
    if [s['id']for s in c['source_specs']]!=POLICY['source_ids']or c['code_refs']['runner']['path']!=str(Path(__file__).resolve()):
        raise Blocked('UNADMITTED_SOURCE_OR_RUNNER_PATH')
    mapping=[('date','code','timestamp'),('trade_date','ts_code','int64_YYYYMMDD'),('trade_date','ts_code','int32_YYYYMMDD'),('trade_date','ts_code','string_YYYYMMDD')]
    if any((s['date_column'],s['code_column'],s['date_kind'])!=expected for s,expected in zip(c['source_specs'],mapping)):
        raise Blocked('FROZEN_SOURCE_COLUMN_OR_DATE_MAPPING_DRIFT')
    retained={r['source']:r for r in domain['retention_exactkey_read_allowlist_after_START']}
    for name,spec in zip(['backtest','selector'],c['source_specs'][2:]):
        admitted=retained[name]
        if spec['ref']['path']!=admitted['path']or spec['ref']['sha256']!=admitted['sha256']or spec['expected_bytes']!=admitted['bytes']or spec['file_rows']!=admitted['metadata_rows']:
            raise Blocked('RETENTION_EXACT_VERSION_ADMISSION_DRIFT')
    if Path(previous.__file__).resolve()!=SUPPORT_DIR/'environment_tilt_month_bonus_stability.py':raise Blocked('ACCEPTED_SUPPORT_IMPORT_COLLISION')
    c.update(contract_path=c['contract_ref']['path'],contract_sha256=c['contract_ref']['sha256'],code_composite_sha256=previous.old.support.digest(c['code_refs']))
    return c


def run(c,activation,adapter=None):
    return previous.run(c,activation,adapter=adapter,compute_fn=diagnose,lifecycle_type=RecoveryLifecycle,approval='APPROVED_FOR_MISSING24_DIAGNOSTIC')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--activation');p.add_argument('--execute',action='store_true');args=p.parse_args();c=metadata_config(args.config)
    if not args.execute:print(canonical(dict(status='METADATA_ONLY_NOT_FINANCIAL_EXECUTION')));return
    if not args.activation:p.error('--activation required')
    result=run(c,previous.resources.read_json(args.activation));print(canonical({k:result[k]for k in ['status','elapsed_seconds','sampled_peak_rss_bytes','cause_counts']if k in result}))

if __name__=='__main__':main()
