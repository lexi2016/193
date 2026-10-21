"""Registered exact saved-input lookup; no network, price reconstruction or labels."""
import argparse,ast,csv,ctypes,hashlib,json,struct,sys,time
from pathlib import Path
from decimal import localcontext,ROUND_HALF_UP
import pandas as pd
import m33_tdx_source_refresh as source
class Life(source.RefreshLifecycle):
    def event(self,kind,decision,receipt,metrics):
        e=super().event(kind,decision,receipt,metrics)
        e.update(route_id='a-share-environment-tilt-exact-shahe-action-and-first-prefix-row',experiment_id='ENV_TILT_SAVED_INPUT_TWO_KEY_LOOKUP',stage='SAVED_GBBQ_20070711_AND_PREFIX_19991110_LOOKUP',baseline_id='M40_MEMORY_R1_B_TAINT_AND_SHAHE_REQUIRED_CAT1',one_variable='EXACT_SAVED_SOURCE_FIELDS_ONLY_NO_NEW_PRICE_TRANSFORM')
        return e

def lookup(frame):
    rows=frame.loc[(frame.code=='000014')&(frame.datetime==20070711)&(frame.category==1)].to_dict('records')
    result=dict(code='000014.SZ',date=20070711,category=1,record_count=len(rows),records=rows,field_map=dict(fenhong='hongli_panqianliutong',peigujia='peigujia_qianzongguben',songzhuangu='songgu_qianzongguben',peigu='peigu_houzongguben'))
    if len(rows)!=1:result.update(status='CANNOT_CONFIRM_MISSING_OR_AMBIGUOUS');return result
    row=rows[0];fields={k:row[v] for k,v in result['field_map'].items()};expected=struct.unpack('<f',struct.pack('<f',.06))[0]
    result.update(fields=fields,issuer_fenhong_per10_yuan=.06,source_fenhong_float32_expected=expected,status='EXACT_UNIQUE_STANDARD_CASH_ROW_BOUND' if fields==dict(fenhong=expected,peigujia=0.,songzhuangu=0.,peigu=0.) else 'CANNOT_CONFIRM_ISSUER_SOURCE_TERMS_DIFFER',factor_applied=False)
    return result

def authenticate_input(name,ref,root,report,receipt):
    p=Path(ref['path']);before=source.base.file_identity(p.stat());actual=source.registry.sha(p);after=source.base.file_identity(p.stat())
    record=dict(path=ref['path'],expected_sha256=ref['sha256'],actual_sha256=actual,
        expected_bytes=ref['bytes'],stat_before=before,stat_after=after,
        sha_match=actual==ref['sha256'],stat_unchanged=before==after,
        bytes_match=before['bytes']==ref['bytes'],hash_read_after_start=True)
    binding=source.original.new_json(root/('INPUT_AUTH_'+name+'.json'),record)
    report.setdefault('input_identity',{})[name]=dict(**record,evidence=binding)
    source.base.write_json(receipt,report)  # Persist actual hash even on mismatch.
    if not all(record[k] for k in ('sha_match','stat_unchanged','bytes_match')):
        raise source.Stop('INPUT_HASH_OR_STAT_DRIFT_'+name)


def run(c,activation):
    if activation.get('decision')!='INDEPENDENT_REVIEW_PASS_ACTIVATE' or activation.get('execution_id')!=c['execution_id'] or activation.get('config_sha256')!=source.registry.sha(c['config_path']):raise source.Stop('NOT_ACTIVATED')
    if source.registry.sha(c['review_path'])!=activation.get('review_sha256'):raise source.Stop('REVIEW_DRIFT')
    if c['network_allowed'] or c['query_budget']!=0:raise source.Stop('NO_NETWORK_REQUIRED')
    for ref in (c['reader'],c['source_wrapper'],*c['transform_sources'].values()):
        if source.registry.sha(ref['path'])!=ref['sha256']:raise source.Stop('SOURCE_DRIFT')
    if source.registry.sha(__file__)!=c['script_sha256'] or source.registry.sha(c['contract_path'])!=c['contract_sha256']:raise source.Stop('IDENTITY_DRIFT')
    c.update(config_sha256=source.registry.sha(c['config_path']),code_composite_sha256=c['script_sha256']);root=Path(c['run_dir'])
    if root.parent!=Path(__file__).resolve().parent or root.is_symlink() or any(p.is_symlink() for p in root.parents):raise source.Stop('OUTPUT_SCOPE')
    root.mkdir(exist_ok=False);receipt=root/'RECEIPT.json';source.original.new_json(receipt,dict(status='PREPARED_NOT_REGISTERED',**source.FLAGS));life=Life(source.registry.load_adapter(c),c);budget=source.base.Budget(root,c['caps']);report=dict(status='INTENT',network_calls=0,fits=0,factor_applied=False,**source.FLAGS)
    try:
        life.start(receipt)
        with budget:
            life.require()
            if source.base.available_memory()<c['caps']['min_memory_bytes'] or source.shutil.disk_usage(root).free<c['caps']['min_disk_bytes']:raise source.Stop('DEFERRED_RESOURCE')
            for name,ref in c['inputs'].items():
                authenticate_input(name,ref,root,report,receipt)
            ref=c['reader'];tree=ast.parse(Path(ref['path']).read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GbbqReader');scope=dict(struct=struct,pd=pd,sys=sys,c_uint32=ctypes.c_uint32);exec(compile(ast.Module(body=[node],type_ignores=[]),ref['path'],'exec'),scope)
            frame=scope['GbbqReader']().get_df(c['inputs']['gbbq']['path']);report['shahe']=lookup(frame);del frame
            with open(c['inputs']['raw_day']['path'],'rb') as f:raw=struct.unpack('<IIIIIfII',f.read(32))
            if raw[0]!=19991110:raise source.Stop('EXACT_FIRST_RAW_KEY_MISSING')
            report['first_raw_record']=dict(code='600000.SH',date=raw[0],open_fen=raw[1],high_fen=raw[2],low_fen=raw[3],close_fen=raw[4],open_yuan=raw[1]/100,high_yuan=raw[2]/100,low_yuan=raw[3]/100,close_yuan=raw[4]/100,amount_CNY=raw[5],volume_shares=raw[6],source_decoder='BOUND_ORIGINAL_DAY_STRUCT_<IIIIIfII>')
            with open(c['inputs']['prefix']['path'],newline='') as f:row=next(csv.DictReader(f))
            if row.get('trade_date')!='19991110' or row.get('ts_code')!='600000.SH':raise source.Stop('EXACT_FIRST_PREFIX_KEY_MISSING')
            report['first_saved_prefix_record']=row
            events=source.base.read_json(c['inputs']['allcategory']['path'])['events']
            participating=sorted([e for e in events if e['category']==1 and 19991110<e['_date_int']<=20261015],key=lambda e:e['_date_int'])
            with localcontext() as ctx:
                ctx.prec,ctx.rounding=50,ROUND_HALF_UP
                a,b=source.original.affine.compose_events(participating)
                transformed={}
                for field in source.original.affine.OHLC:
                    value=a*source.original.affine.decimal_scalar(report['first_raw_record'][field+'_yuan'])+b
                    transformed[field]=dict(unrounded=str(value),cent=str(value.quantize(source.original.affine.CENT)),saved_prefix=row[field],exact_saved_cent=value.quantize(source.original.affine.CENT)==source.original.affine.decimal_scalar(row[field]))
            report['single_row_chain']=dict(anchor=20261015,source_start_exclusive=19991110,events=participating,event_count=len(participating),a=str(a),b=str(b),formula='T_latest(...T_earliest(raw_price));same_original_category1_only',transformed=transformed)
            report['first_raw_record']['record_offset']=0
            report['first_raw_record']['record_sha256']=hashlib.sha256(struct.pack('<IIIIIfII',*raw)).hexdigest()
            report['transform_lineage']=c['transform_lineage'];report['transform_recalculated']='EXACT_SINGLE_ROW_ONLY_REUSED_ORIGINAL_AFFINE_NO_MATRIX';report['taint_boundary']=dict(date='1999-11-10',code='600000.SH',original_recursive_suffix_keys=6406,source=c['m40_actual_receipt']);report['status']='EXACT_TWO_KEY_SAVED_INPUT_LOOKUP_COMPLETE' if report['shahe']['status']=='EXACT_UNIQUE_STANDARD_CASH_ROW_BOUND' else 'CAPTURED_WITH_SHAHE_TERMS_GAP';budget.check()
    except BaseException as exc:report.update(status='FAILED_WITH_EVIDENCE',exception_class=type(exc).__name__,error=str(exc))
    finally:
        report.update(elapsed_seconds=time.monotonic()-budget.start_time,sampled_peak_rss_bytes=budget.peak);source.base.write_json(receipt,report)
        if life.started and not life.finished:life.finish(receipt,report['status'],dict(network_calls=0,fits=0))
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--activation',required=True);a=p.parse_args();c=source.base.read_json(a.config);c['config_path']=a.config;r=run(c,source.base.read_json(a.activation));print(source.registry.canonical({k:r[k] for k in ('status','network_calls','fits','elapsed_seconds')}))
