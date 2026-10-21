"""Bounded source-derived header/setup diagnostic; original tdxpy parsing is unchanged."""
import argparse,importlib,math,struct,sys,time,traceback,types
from pathlib import Path
import m33_tdx_source_refresh as source
HOSTS=('218.6.170.47','124.71.187.122','122.51.120.217')
MODES=('tdxpy_base','go_header_only','go_setup_only','go_both')
GO_CONNECT=bytes.fromhex('0c0000000001030003000d0001')

def modify_header(packet,mode):
    if mode not in MODES:raise source.Stop('MODE_SCOPE')
    if len(packet)!=38 or packet[6:12]!=bytes.fromhex('1c001c002d05'):raise source.Stop('DAILY_PACKET_LAYOUT')
    return bytes.fromhex('0c0100000001')+packet[6:] if mode in ('go_header_only','go_both') else packet

class Life(source.RefreshLifecycle):
    def event(self,kind,decision,receipt,metrics):
        e=super().event(kind,decision,receipt,metrics);e.update(route_id='a-share-environment-tilt-open-tdxpy-protocol-diagnostic',experiment_id='ENV_TILT_OPEN_TDXPY_PROTOCOL',stage='OPEN_TDXPY_SOURCE_BOUND_HEADER_SETUP_CANARY',baseline_id='M35_COUNT_ONLY_BODY_AND_M39_SOURCE_DIFF',one_variable='SOURCE_DERIVED_HEADER_SETUP_CONTROLLED_DIAGNOSTIC_NO_MODEL');return e

def sdk(c):
    root=Path(c['tdxpy_root']).resolve()
    if any(n=='tdxpy' or n.startswith('tdxpy.') for n in sys.modules):raise source.Stop('TDXPY_PREEXISTING_MODULE')
    sys.path.insert(0,str(root));return importlib.import_module('tdxpy.hq'),importlib.import_module('tdxpy.parser.std.get_security_bars'),importlib.import_module('tdxpy.parser.base')

def attempt(c,life,budget,hq,bars,base,index,host,mode,count):
    life.require();root=Path(c['run_dir'])/('ATTEMPT_%02d'%index);root.mkdir(exist_ok=False)
    r=dict(index=index,host=host,mode=mode,request=dict(category=9,market=1,code='600000',start=0,count=count),query_calls=0,connect_calls=0,status='INTENT');beg=time.monotonic();api=None;old_params=bars.GetSecurityBarsCmd.setParams;old_parse=bars.GetSecurityBarsCmd.parseResponse
    try:
        with source.original.m11.query_wall(budget,15):
            api=hq.TdxHq_API(multithread=False,heartbeat=False,auto_retry=False,raise_exception=True)
            if mode in ('go_setup_only','go_both'):
                def setup(self):
                    class Connect(base.BaseParser):
                        def parseResponse(parser,body):
                            source.original.durable_new(root/'CONNECT_BODY.bin',bytes(body));return body
                    cmd=Connect(self.client);cmd.send_pkg=GO_CONNECT;source.original.durable_new(root/'CONNECT_PACKET.bin',GO_CONNECT);cmd.call_api()
                api.setup=types.MethodType(setup,api)
            def params(parser,*args,**kwargs):
                ret=old_params(parser,*args,**kwargs);parser.send_pkg=modify_header(bytes(parser.send_pkg),mode);source.original.durable_new(root/'DAILY_PACKET.bin',bytes(parser.send_pkg));return ret
            def parse(parser,body):
                r['body']=source.original.durable_new(root/'DAILY_BODY.bin',bytes(body));r['declared_count']=struct.unpack('<H',body[:2])[0] if len(body)>=2 else None
                return old_parse(parser,body)
            bars.GetSecurityBarsCmd.setParams=params;bars.GetSecurityBarsCmd.parseResponse=parse
            r['connect_calls']=1
            if not api.connect(host,7709,time_out=5):raise source.Stop('CONNECT_FAILED')
            api.client.settimeout(5);r['query_calls']=1;rows=api.get_security_bars(**r['request'])
            r['raw_response']=source.original.new_json(root/'RAW_RESPONSE.json',dict(request=r['request'],rows=rows))
            if not isinstance(rows,list) or len(rows)!=count:raise source.Stop('INCOMPLETE_RESPONSE')
            required=('year','month','day','open','high','low','close','vol','amount')
            if not all(all(k in row for k in required) for row in rows):raise source.Stop('DAILY_SCHEMA')
            if not all(math.isfinite(float(x[k])) for x in rows for k in ('open','high','low','close','vol','amount')):raise source.Stop('NONFINITE_BAR_FIELD')
            r.update(status='COMPLETE_BARS',row_count=len(rows),date_min=min(int('%04d%02d%02d'%(x['year'],x['month'],x['day'])) for x in rows),date_max=max(int('%04d%02d%02d'%(x['year'],x['month'],x['day'])) for x in rows),rows_after_cutoff=sum(int('%04d%02d%02d'%(x['year'],x['month'],x['day']))>20261015 for x in rows),unit_binding='RAW_TDXPY_PARSER_OUTPUT; volume/amount physical units CANNOT_CONFIRM in this protocol probe; no normalization/model bridge.')
    except BaseException as exc:
        cause=getattr(exc,'original_exception',None);r.update(status='FAILED',exception_class=type(exc).__name__,error=str(exc),original_exception_class=None if cause is None else type(cause).__name__,original_error=None if cause is None else str(cause));source.original.new_json(root/'EXCEPTION.json',dict(traceback=traceback.format_exc()))
        if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
    finally:
        bars.GetSecurityBarsCmd.setParams=old_params;bars.GetSecurityBarsCmd.parseResponse=old_parse
        if api is not None:
            try:api.disconnect()
            except Exception as exc:r['disconnect_error']=str(exc)
        r['elapsed_seconds']=time.monotonic()-beg;source.original.new_json(root/'RECEIPT.json',r)
    budget.check();return r

def run(c,activation):
    if activation.get('decision')!='INDEPENDENT_REVIEW_PASS_ACTIVATE' or activation.get('execution_id')!=c['execution_id'] or activation.get('config_sha256')!=source.registry.sha(c['config_path']) or activation.get('review_sha256')!=source.registry.sha(c['review_path']):raise source.Stop('NOT_ACTIVATED')
    expected=dict(hosts=list(HOSTS),modes=list(MODES),port=7709,max_daily_requests=8,max_connect_attempts=8,total_wall_sec=180)
    if c.get('network_allowed') is not True or c.get('unscoped_hq_query_allowed') is not False or c.get('hq_request_scope')!=expected:raise source.Stop('EXACT_NETWORK_SCOPE')
    source.load_amount_projection_config(c['amount_config_path'])
    if source.registry.sha(__file__)!=c['script_sha256'] or source.registry.sha(c['contract_path'])!=c['contract_sha256']:raise source.Stop('IDENTITY_DRIFT')
    for ref in c['protocol_sources'].values():
        if source.registry.sha(ref['path'])!=ref['sha256']:raise source.Stop('PROTOCOL_SOURCE_DRIFT')
    c.update(config_sha256=source.registry.sha(c['config_path']),code_composite_sha256=c['script_sha256']);root=Path(c['run_dir'])
    if root.parent!=Path(__file__).resolve().parent or root.is_symlink() or any(p.is_symlink() for p in root.parents):raise source.Stop('OUTPUT_SCOPE')
    root.mkdir(exist_ok=False);receipt=root/'RECEIPT.json';source.original.new_json(receipt,dict(status='PREPARED_NOT_REGISTERED',**source.FLAGS));life=Life(source.registry.load_adapter(c),c);budget=source.base.Budget(root,c['caps']);report=dict(status='INTENT',attempts=[],daily_calls=0,connect_calls=0,fits=0,training_ready=False,**source.FLAGS)
    try:
        life.start(receipt)
        with budget:
            if source.base.available_memory()<c['caps']['min_memory_bytes'] or source.shutil.disk_usage(root).free<c['caps']['min_disk_bytes']:raise source.Stop('DEFERRED_RESOURCE')
            hq,bars,base=sdk(c)
            plans=[(HOSTS[0],m) for m in MODES]+[(h,m) for h in HOSTS[1:] for m in ('go_both','tdxpy_base')]
            for host,mode in plans:
                if len(report['attempts'])>=8:break
                r=attempt(c,life,budget,hq,bars,base,len(report['attempts'])+1,host,mode,10);report['attempts'].append(r);report['daily_calls']+=r['query_calls'];report['connect_calls']+=r['connect_calls'];source.base.write_json(receipt,report)
                if r['status']=='COMPLETE_BARS':
                    if len(report['attempts'])>=8:report['status']='TEN_BARS_ONLY_BUDGET_EXHAUSTED';break
                    r=attempt(c,life,budget,hq,bars,base,len(report['attempts'])+1,host,mode,800);report['attempts'].append(r);report['daily_calls']+=r['query_calls'];report['connect_calls']+=r['connect_calls'];report.update(status='TEN_AND800_COMPLETE' if r['status']=='COMPLETE_BARS' else 'TEN_ONLY_800_FAILED',working_host=host,working_mode=mode);break
            if report['status']=='INTENT':report['status']='BOUNDED_PROTOCOL_NO_COMPLETE_BARS'
    except BaseException as exc:report.update(status='STOPPED_WITH_EVIDENCE',exception_class=type(exc).__name__,error=str(exc))
    finally:
        report.update(elapsed_seconds=time.monotonic()-budget.start_time,sampled_peak_rss_bytes=budget.peak);source.base.write_json(receipt,report)
        if life.started and not life.finished:life.finish(receipt,report['status'],dict(daily_calls=report['daily_calls'],connect_calls=report['connect_calls'],fits=0))
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--activation',required=True);a=p.parse_args();c=source.base.read_json(a.config);c['config_path']=a.config;print(source.registry.canonical(run(c,source.base.read_json(a.activation))))
