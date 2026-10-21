"""One original official TDX archive request; original local GBBQ reader; no factor guess."""
import argparse,ast,ctypes,io,json,struct,sys,time,urllib.request,urllib.error,zipfile
from pathlib import Path
import pandas as pd
import m33_tdx_source_refresh as source
URL='http://www.tdx.com.cn/products/data/data/dbf/gbbq.zip'
REQUEST=dict(url=URL,http_calls=1,redirects=0,body_bytes=32*2**20,member_bytes=32*2**20,timeout_seconds=30)
EVENTS=(('000002',20051201),('600004',20051219),('000014',20070630))
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None
class Life(source.RefreshLifecycle):
    def event(self,kind,decision,receipt,metrics):
        e=super().event(kind,decision,receipt,metrics)
        e.update(route_id='a-share-environment-tilt-original-tdx-gbbq-three-records',experiment_id='ENV_TILT_ORIGINAL_GBBQ',stage='ORIGINAL_OFFICIAL_GBBQ_EXACT_EVENT_LOOKUP',baseline_id='M33_XDXR_THREE_NONSTANDARD_EVENTS_UNRESOLVED',one_variable='ORIGINAL_OFFICIAL_GBBQ_RAW_FIELDS_LOOKUP_NO_FACTOR_INFERENCE')
        return e

def fetch_once(root,reader_ref,opener,report,policy=None):
    # This function is called only after the lifecycle durable START.
    policy=REQUEST if policy is None else policy
    if policy!=REQUEST or report['http_calls']>=1:raise source.Stop('EXACT_ARCHIVE_HTTP_SCOPE')
    report['http_calls']+=1
    try:response=opener.open(policy['url'],timeout=policy['timeout_seconds'])
    except urllib.error.HTTPError as exc:response=exc
    chunks=[];size=0
    try:
        with response:
            report.update(http_status=response.code,response_url=response.geturl(),headers=dict(response.headers))
            while True:
                block=response.read(min(65536,policy['body_bytes']-size+1))
                if not block:break
                chunks.append(block);size+=len(block)
                if size>policy['body_bytes']:raise source.Stop('ARCHIVE_RESPONSE_LIMIT')
    finally:
        payload=b''.join(chunks)
        report['raw_response']=source.original.durable_new(root/'OFFICIAL_HTTP_RESPONSE.bin',payload)
    if report['http_status']!=200:raise source.Stop('OFFICIAL_HTTP_STATUS_'+str(report['http_status']))
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names=z.namelist()
        if names.count('gbbq')!=1:raise source.Stop('EXACT_GBBQ_MEMBER_MISSING_OR_DUPLICATE')
        member=z.getinfo('gbbq')
        if member.file_size>policy['body_bytes'] or member.is_dir() or (member.external_attr>>16)&0o170000==0o120000:raise source.Stop('GBBQ_MEMBER_BOUNDARY')
        raw=z.read('gbbq')
    report['raw_member']=source.original.durable_new(root/'gbbq',raw)
    if len(raw)<4:raise source.Stop('GBBQ_HEADER_SIZE')
    count=struct.unpack('<I',raw[:4])[0]
    if len(raw)!=4+29*count:raise source.Stop('GBBQ_RECORD_LENGTH')
    if source.registry.sha(reader_ref['path'])!=reader_ref['sha256']:raise source.Stop('ORIGINAL_GBBQ_READER_DRIFT')
    tree=ast.parse(Path(reader_ref['path']).read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GbbqReader')
    scope=dict(struct=struct,pd=pd,sys=sys,c_uint32=ctypes.c_uint32)
    exec(compile(ast.Module(body=[node],type_ignores=[]),reader_ref['path'],'exec'),scope)
    frame=scope['GbbqReader']().get_df(str(root/'gbbq'))
    if len(frame)!=count:raise source.Stop('GBBQ_DECODED_COUNT')
    output=[]
    for code,date in EVENTS:
        rows=frame.loc[(frame.code==code)&(frame.datetime==date)].to_dict('records')
        output.append(dict(code=code,date=date,records=rows,record_count=len(rows),lookup_status='EXACT_UNIQUE_RECORD' if len(rows)==1 else ('MISSING_EXACT_RECORD' if not rows else 'MULTIPLE_EXACT_RECORDS_RETAINED'),factor_status='CANNOT_CONFIRM_CATEGORY14_OR4_OR_MISSING_TERMS_NOT_IDENTITY',native_QFQ_certified=False))
    report.update(declared_record_count=count,exact_event_records=output,actual_member_only=True,exact_three_unique_records=all(x['record_count']==1 for x in output),training_ready=False)
    report['lookup']=source.original.new_json(root/'EXACT_THREE_EVENT_RECORDS.json',dict(events=output,source_member=report['raw_member'],reader=reader_ref,latest_events_not_consumed=True,factor_calculated=False))

def run(c,activation):
    if activation.get('decision')!='INDEPENDENT_REVIEW_PASS_ACTIVATE' or activation.get('execution_id')!=c['execution_id'] or activation.get('config_sha256')!=source.registry.sha(c['config_path']):raise source.Stop('NOT_ACTIVATED')
    if source.registry.sha(c['review_path'])!=activation.get('review_sha256'):raise source.Stop('REVIEW_DRIFT')
    if c.get('network_allowed') is not True or c.get('tdx_hq_query_allowed') is not False or c.get('archive_http')!=REQUEST or c.get('member_byte_limit')!=REQUEST['member_bytes']:raise source.Stop('NETWORK_OR_MEMBER_SCOPE')
    source.load_amount_projection_config(c['amount_config_path'])
    if source.registry.sha(__file__)!=c['script_sha256'] or source.registry.sha(c['contract_path'])!=c['contract_sha256']:raise source.Stop('IDENTITY_DRIFT')
    for ref in (c['official_url_source'],c['reader']):
        if source.registry.sha(ref['path'])!=ref['sha256']:raise source.Stop('SOURCE_DRIFT')
    c.update(config_sha256=source.registry.sha(c['config_path']),code_composite_sha256=c['script_sha256'])
    root=Path(c['run_dir'])
    if root.parent!=Path(__file__).resolve().parent or root.is_symlink() or any(p.is_symlink() for p in root.parents):raise source.Stop('OUTPUT_SCOPE')
    root.mkdir(exist_ok=False);receipt=root/'RECEIPT.json';source.original.new_json(receipt,dict(status='PREPARED_NOT_REGISTERED',**source.FLAGS))
    life=Life(source.registry.load_adapter(c),c);budget=source.base.Budget(root,c['caps']);report=dict(status='INTENT',http_calls=0,retries=0,fits=0,factor_calculated=False,training_ready=False,**source.FLAGS)
    try:
        life.start(receipt)
        with budget:
            life.require()
            if source.base.available_memory()<c['caps']['min_memory_bytes'] or source.shutil.disk_usage(root).free<c['caps']['min_disk_bytes']:raise source.Stop('DEFERRED_RESOURCE')
            fetch_once(root,c['reader'],urllib.request.build_opener(NoRedirect()),report,c['archive_http'])
            report['status']='OFFICIAL_GBBQ_EXACT_UNIQUE_THREE_RECORDS_CAPTURED' if report['exact_three_unique_records'] else 'CAPTURED_WITH_EVENT_RECORD_GAP_OR_MULTIPLICITY';budget.check()
    except BaseException as exc:report.update(status='FAILED_WITH_EVIDENCE',exception_class=type(exc).__name__,error=str(exc))
    finally:
        report.update(elapsed_seconds=time.monotonic()-budget.start_time,sampled_peak_rss_bytes=budget.peak);source.base.write_json(receipt,report)
        if life.started and not life.finished:life.finish(receipt,report['status'],dict(http_calls=report['http_calls'],fits=0))
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--activation',required=True);a=p.parse_args();c=source.base.read_json(a.config);c['config_path']=a.config;print(source.registry.canonical(run(c,source.base.read_json(a.activation))))
