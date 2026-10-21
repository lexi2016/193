"""Fixed three-date existing-response lookup; default CLI metadata only."""
from __future__ import annotations
import argparse
from datetime import date
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

TARGETS = [('000002.SZ', 20051201), ('600004.SH', 20051219), ('000014.SZ', 20070630)]
CAPS = dict(threads=4, rss_bytes=2**31, total_wall_sec=120, payload_bytes=10*2**20,
            min_memory_bytes=8*2**30, min_disk_bytes=2**30)
FLAGS = dict(research_only=True, production_allowed=False, buy_gate_updated=False, orders_sent=False)


def load_module(ref, name):
    import hashlib
    path = Path(ref['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('SOURCE_IDENTITY_DRIFT')
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path.resolve():
            raise ValueError('SOURCE_IMPORT_COLLISION')
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return dict(source_nonfinite_literal=repr(value), interpretation='NOT_NORMALIZED')
    if isinstance(value, dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value, list):return [safe(v) for v in value]
    return value


def lookup(snapshot, symbol, date_int):
    """Select by exact date only; retain every same-date category/duplicate."""
    if snapshot.get('symbol') != symbol or snapshot.get('response_type') != 'list' or not isinstance(snapshot.get('response'), list):
        raise ValueError('BOUND_SYMBOL_RESPONSE_SCHEMA_CONFLICT')
    matches=[]
    for i, record in enumerate(snapshot['response']):
        if not isinstance(record, dict):raise ValueError('RECORD_SCHEMA')
        values=[record[k] for k in ('year','month','day')]
        if any(type(v) is not int for v in values):raise ValueError('SOURCE_DATE_TYPE_UNKNOWN')
        d=date(*values)
        if d.year*10000+d.month*100+d.day == date_int:
            matches.append(dict(source_response_index=i,original_record=safe(record),
                original_record_json=json.dumps(record,ensure_ascii=False,sort_keys=True,allow_nan=True)))
    category1=[v for v in matches if v['original_record'].get('category') == 1]
    return dict(symbol=symbol,date=date_int,match_count=len(matches),all_same_date_records=matches,
        category1_count=len(category1),category1_records=category1,
        interpretation='LOOKUP_ONLY_NO_CATEGORY_RECLASSIFICATION_OR_QFQ_FACTOR_INFERENCE',
        missing_means='NO_EXACT_RECORD_IN_THIS_BOUND_PARSER_RESPONSE_NOT_NO_CORPORATE_ACTION')


def load_config(path):
    c=json.loads(Path(path).read_text());c.update(config_path=str(Path(path).resolve()))
    if c['caps']!=CAPS or [(v['symbol'],v['date']) for v in c['targets']]!=TARGETS:
        raise ValueError('FIXED_SCOPE_DRIFT')
    if any(c.get(k)!=v for k,v in FLAGS.items()) or c.get('network_allowed') is not False or c.get('fit_count')!=0:
        raise ValueError('BOUNDARY_DRIFT')
    sys.dont_write_bytecode=True
    registry=load_module(c['sources']['registry'],'environment_tilt_mac_registry')
    base=load_module(c['sources']['base'],'environment_tilt_mac_recovery')
    c['config_sha256']=registry.sha(path)
    if registry.sha(__file__)!=c['script_sha256'] or registry.sha(c['contract_path'])!=c['contract_sha256']:
        raise ValueError('SCRIPT_OR_CONTRACT_DRIFT')
    return c,base,registry


def lifecycle_type(registry):
    class LookupLifecycle(registry.Lifecycle):
        def event(self,kind,decision,receipt,metrics):
            e=super().event(kind,decision,receipt,metrics)
            e.update(route_id='a-share-environment-tilt-three-date-action-lookup-20261021',
                experiment_id='A_SHARE_ENV_TILT_M31_FIXED_ACTION_LOOKUP',stage='M31_EXACT_THREE_DATE_ACTION_LOOKUP',
                arm='FIXED_THREE_DATES_NO_PRICES',baseline_id='M27_UNSUPPORTED_ACTIONS_PRESERVED_CANARY_FALSE',
                one_variable='EXISTING_BOUND_RAW_RESPONSE_EXACT_DATE_ALL_CATEGORY_RECORD_LOOKUP',
                truth_access='NO_PRICES_LABELS_SCORES_OR_MODELS_ACTION_RECORDS_ONLY',
                sealed_status='OPENBOOK_SOURCE_SEMANTICS_ONLY',evidence_level='OPENBOOK_DIAGNOSTIC_ONLY')
            return e
    return LookupLifecycle


def run(c,base,registry,adapter,activation):
    if (c['state']!='FROZEN_READY_FOR_ACTIVATION' or activation.get('decision')!='INDEPENDENT_REVIEW_PASS_ACTIVATE'
            or activation.get('config_sha256')!=c['config_sha256']
            or activation.get('execution_id')!=c['execution_id']
            or activation.get('script_sha256')!=c['script_sha256']):
        raise PermissionError('NOT_ACTIVATED')
    review=c['review_receipt']
    if (registry.sha(review['path'])!=review['sha256']
            or activation.get('review_receipt_sha256')!=review['sha256']):
        raise PermissionError('REVIEW_IDENTITY_DRIFT')
    root=Path(c['run_dir'])
    if root.parent.resolve()!=Path(c['batch_root']).resolve() or root.name!='M31_DATE_LOOKUP_01' or root.is_symlink():
        raise PermissionError('OUTPUT_SCOPE')
    root.mkdir(exist_ok=False)
    with (root/'.owner.lock').open('x') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=root/'RECEIPT.json';base.write_json(receipt,dict(status='PREPARED_NOT_REGISTERED',**FLAGS))
        life=lifecycle_type(registry)(adapter,c);budget=base.Budget(root,CAPS);report={};error=None
        try:
            life.start(receipt)
            with budget:
                before=dict(available_bytes=base.available_memory(),disk_free_bytes=shutil.disk_usage(root).free,rss_bytes=base.rss_bytes())
                if before['available_bytes']<CAPS['min_memory_bytes'] or before['disk_free_bytes']<CAPS['min_disk_bytes']:
                    raise base.DiagnosticError('DEFERRED_RESOURCE')
                rows=[]
                for target in c['targets']:
                    with base.authenticated(life,target['input'],budget) as handle:
                        snapshot=json.load(handle)
                        rows.append(dict(**lookup(snapshot,target['symbol'],target['date']),input_identity=target['input']))
                    # Persist completed targets even if a later input fails. No retry.
                    report=dict(status='LOOKUP_IN_PROGRESS',targets=rows,**FLAGS);base.write_json(receipt,report)
                report.update(status='COMPLETED_FIXED_THREE_DATE_LOOKUP',resource_before=before,
                    factor_metadata=c['factor_metadata'],category_semantics='CANNOT_CONFIRM_UNCHANGED',
                    queries=0,fits=0,prices_or_labels_read=False,training_ready=False)
                budget.check()
        except BaseException as exc:
            error=exc;report.update(status='FAILED_NO_RETRY',error_type=type(exc).__name__,error=str(exc)[:1000],**FLAGS)
        finally:
            report.update(elapsed_seconds=time.monotonic()-budget.start_time,sampled_peak_rss_bytes=budget.peak,**FLAGS)
            base.write_json(receipt,report)
            if life.started and not life.finished:
                try:life.finish(receipt,report['status'],dict(completed_target_count=len(report.get('targets',[])),network_calls=0))
                except BaseException as exc:
                    report.update(terminal_status='CANNOT_CONFIRM',terminal_error=repr(exc));base.write_json(receipt,report);raise
        if error is not None:raise error
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--activation',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    c,base,registry=load_config(a.config)
    if not a.execute:
        print(registry.canonical(dict(status='M31_METADATA_ONLY_NOT_ACTIVATED',targets=TARGETS)));return
    if a.activation is None:p.error('--activation required with --execute')
    activation=json.loads(a.activation.read_text())
    # No registry access occurs on the default metadata CLI.
    print(registry.canonical(run(c,base,registry,registry.load_adapter(c),activation)))


if __name__=='__main__':main()
