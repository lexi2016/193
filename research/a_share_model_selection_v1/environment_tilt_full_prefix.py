"""Mac-only original-universe prefix recovery; CLI is metadata-only by default.

A controller activation and official durable RUN_STARTED precede input reads.
Each process owns one symbol directory/connection. Only the parent owns registry.
Existing successful checkpoints are authenticated and reused, never recomputed.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import io
import importlib.util
import json
import multiprocessing
import os
import re
import shutil
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from decimal import localcontext, ROUND_HALF_UP
from pathlib import Path

import environment_tilt_mac_recovery as base
import environment_tilt_mac_affine as affine
import environment_tilt_mac_prefix_canary as m11
from environment_tilt_mac_registry import Lifecycle, FLAGS, canonical, sha, load_adapter, digest as event_digest

POLICY = dict(cutoff=20261015, needed_start=20200101, needed_end=20231231,
              universe_count=5576, history='ALL_ORIGINAL_AVAILABLE_BEFORE_OLD_FIRST',
              anchor=m11.POLICY['anchor'], precision=50, quantum='0.01',
              rounding='ROUND_HALF_UP', query_retry=False,
              categories=[1, *m11.INFORMATIONAL_CATEGORIES],
              bj_market_hypothesis=2, bj_identity='REQUEST_NO_SYMBOL_ECHO_PLUS_CACHE_AND_OVERLAP')
CANARY = (*base.CODES, '000014.SZ')
DEPENDENCIES = ('environment_tilt_mac_recovery.py', 'environment_tilt_mac_affine.py',
                'environment_tilt_mac_prefix_canary.py', 'environment_tilt_mac_registry.py')
WRITER = '/root/tilt_full_prefix_writer'


class Stop(base.DiagnosticError):
    pass


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def durable_new(path, payload):
    path = Path(path)
    if path.is_symlink() or any(x.is_symlink() for x in path.parents): raise Stop('OUTPUT_SYMLINK')
    with path.open('xb') as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))


def new_json(path, value):
    return durable_new(path, (canonical(value) + '\n').encode())


def bytes_once(path, payload):
    if Path(path).exists():
        binding = dict(path=str(path), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        check_artifact(binding, Path(path).parent)
        return binding
    return durable_new(path, payload)


def content_once(path, value):
    return bytes_once(path, (canonical(value) + '\n').encode())


@contextlib.contextmanager
def owner(root):
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents): raise Stop('OUTPUT_SYMLINK')
    with (root / 'owner.lock').open('a+') as handle:
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise Stop('OWNER_BUSY') from None
        try: yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


def check_artifact(binding, root=None):
    p = Path(binding['path'])
    if p.is_symlink() or any(x.is_symlink() for x in p.parents) or (root is not None and root.resolve() not in p.resolve().parents):
        raise Stop('ARTIFACT_PATH_ESCAPE')
    if ('bytes' in binding and p.stat().st_size != binding['bytes']) or sha(p) != binding['sha256']:
        raise Stop('ARTIFACT_IDENTITY_DRIFT')


def inventory(handle):
    """Same original member fields; retain BJ instead of silently filtering it."""
    result = {}
    for row in csv.DictReader(io.StringIO(handle.read().decode('utf-8-sig'))):
        suffix = {'sh':'SH', 'sz':'SZ', 'bj':'BJ'}.get(row['market'].lower())
        if suffix is None: continue
        code = row['security_code'].zfill(6) + '.' + suffix
        if code in result: raise Stop('AMBIGUOUS_RAW_MEMBER:' + code)
        result[code] = dict(name=row['member_name'], bytes=int(row['uncompressed_bytes']),
                            sha256=row['uncompressed_member_sha256'])
    return result


def coverage_gate(code, dates, old_first, unit_binding):
    """Datewords prove observed member coverage, never IPO or source completeness."""
    if not dates: raise Stop('EMPTY_ORIGINAL_RAW_HISTORY')
    earlier = [date for date in dates if date < old_first]
    early = [date for date in earlier if date <= POLICY['needed_end']]
    if early:
        if not code.endswith(('.SH', '.SZ', '.BJ')):
            raise Stop('CANNOT_CONFIRM_UNKNOWN_SOURCE_MARKET')
        if unit_binding.get('variant') != 'ORIGINAL_STANDARD_A_BJ_DAY' or unit_binding.get('vol') != 'shares' or unit_binding.get('amount') != 'CNY' or unit_binding.get('scale') != 'identity':
            raise Stop('CANNOT_CONFIRM_SOURCE_VARIANT_OR_UNITS')
    return dict(observed_source_first=dates[0], observed_source_last=dates[-1],
                old_first=old_first, needs_prefix=bool(early),
                observed_prefix_rows=len(earlier), observed_early_rows=len(early),
                history_scope=POLICY['history'], complete_member_records_parsed=True,
                historical_source_completeness='CANNOT_CONFIRM', ipo_date='CANNOT_CONFIRM')


def analyze_full(raw, reference, events, code, reuse=None):
    """Frozen M11 operator and exact oracle, over the entire available history."""
    raw, reference = base.normalize(raw, code), base.normalize(reference, code)
    raw = raw.loc[raw.trade_date.le(POLICY['cutoff'])].reset_index(drop=True)
    reference = reference.loc[reference.trade_date.le(POLICY['cutoff'])].reset_index(drop=True)
    common = sorted(set(raw.trade_date) & set(reference.trade_date))
    if not common: raise Stop('NO_COMMON_ORIGINAL_KEYS')
    anchor = common[-1]  # keys only; never move anchor after a numerical failure
    old_first, earliest = int(reference.trade_date.min()), int(raw.trade_date.min())
    if int(reference.trade_date.max()) != anchor: raise Stop('RAW_END_BEFORE_OLD_END')
    source = raw.loc[raw.trade_date.le(anchor)]
    overlap_raw = source.loc[source.trade_date.ge(old_first)]
    keys_equal = overlap_raw.trade_date.tolist() == reference.trade_date.tolist()
    if not keys_equal: raise Stop('WHOLE_OVERLAP_DATE_KEYS_MISMATCH')
    with localcontext() as ctx:
        ctx.prec, ctx.rounding = 50, ROUND_HALF_UP
        identity = all(affine.cent_reference(source.loc[source.trade_date.eq(anchor),f].iloc[0]) ==
                       affine.cent_reference(reference.loc[reference.trade_date.eq(anchor),f].iloc[0])
                       for f in affine.OHLC)
    if reuse is None:
        rebuilt = m11.reconstruct(source, events, anchor)
        old = [row for row in rebuilt if row['trade_date'] >= old_first]
        comparison, mismatches = affine.compare_prices(old, reference)
    else:
        if reuse['anchor'] != anchor or reuse['first_old_bar'] != old_first or reuse['overlap_rows'] != len(reference):
            raise Stop('ACCEPTED_OVERLAP_SCOPE_DRIFT')
        # Keep anchor only to satisfy the unchanged M11 operator; do not rerun old overlap.
        rebuilt = m11.reconstruct(source.loc[source.trade_date.lt(old_first) | source.trade_date.eq(anchor)], events, anchor)
        comparison, mismatches = dict(reused_receipt=reuse['receipt']), 0
    unsupported = [e for e in events if earliest < e['_date_int'] <= anchor
                   and e['category'] not in POLICY['categories']]
    reasons = []
    if not identity: reasons.append('LATEST_COMMON_ANCHOR_NOT_IDENTITY')
    if mismatches: reasons.append('WHOLE_OVERLAP_CENT_MISMATCH')
    if unsupported: reasons.append('UNSUPPORTED_PARTICIPATING_ACTION')
    candidate = [r for r in rebuilt if r['trade_date'] < old_first]
    if not candidate: reasons.append('NO_PREFIX_ROWS')
    return dict(symbol=code, anchor=anchor, anchor_identity=identity,
                first_old_bar=old_first, earliest_raw=earliest,
                overlap_rows=len(reference), OHLC_denominator=4*len(reference),
                overlap_cent_mismatches=mismatches, overlap_comparison=comparison,
                allcategory_counts=dict(Counter(str(e['category']) for e in events)),
                unsupported_actions=unsupported, block_reasons=reasons,
                prefix_rows=len(candidate), history_scope=POLICY['history'], **FLAGS), candidate


def emit_full(root, result, candidate, raw):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['trade_date','ts_code',*affine.OHLC,'vol','amount'])
    writer.writeheader(); lookup = raw.set_index('trade_date')
    for row in candidate:
        writer.writerow(dict(trade_date=row['trade_date'], ts_code=result['symbol'],
            **{field:row[field]['cent'] for field in affine.OHLC},
            vol=lookup.loc[row['trade_date'],'vol'], amount=lookup.loc[row['trade_date'],'amount']))
    return dict(**bytes_once(root / 'historical_prefix_v1.csv', stream.getvalue().encode()),
                rows=len(candidate), vol='shares', amount='CNY', scale='identity')


def normalized_response(path, expected_code=None):
    """Replay durable response only; never requery a consumed attempt."""
    value = base.read_json(path)
    if expected_code is not None and value.get('symbol') != expected_code: raise Stop('RESPONSE_REQUEST_BINDING_MISMATCH')
    response = value['response']
    if value.get('symbol','').endswith('.BJ'):
        if value.get('market') != 2 or not response: raise Stop('BJ_CAPABILITY_CANNOT_CONFIRM_EMPTY_OR_UNBOUND_RESPONSE')
    if not isinstance(response, (list,tuple)): raise Stop('QUERY_RESPONSE_UNKNOWN')
    if value.get('symbol','').endswith('.BJ'):
        for row in response:
            if 'code' in row and str(row['code']) != value['symbol'][:6]: raise Stop('BJ_RESPONSE_SYMBOL_CONFLICT')
            if 'market' in row and int(row['market']) != 2: raise Stop('BJ_RESPONSE_MARKET_CONFLICT')
    records = [dict(event_date=f"{int(e['year']):04d}-{int(e['month']):02d}-{int(e['day']):02d}", record=e)
               for e in response]
    return m11.normalize_allcategory(records, POLICY['cutoff'])


class WorkerGate:
    def __init__(self, start):
        if not start.get('event_sha256'): raise Stop('DURABLE_PARENT_START_REQUIRED')
    def require(self): pass


def bj_query_events(life, config, budget, code, evidence, session_factory=None):
    """One market=2 capability hypothesis; unchanged original API/parser/endpoint."""
    life.require()
    if not code.endswith('.BJ') or evidence['query_calls'] or config.get('bj_capability_probe',{}).get('market') != 2:
        raise Stop('BJ_CAPABILITY_SCOPE_OR_COUNT')
    for item in config['query_sources'].values():
        if sha(item['path']) != item['sha256']: raise Stop('QUERY_SOURCE_DRIFT')
    package_root = None
    if session_factory is None:
        package_root = Path(config['query_package_root']).resolve()
        for name,module in list(sys.modules.items()):
            if name == 'pytdx' or name.startswith('pytdx.'):
                if not getattr(module,'__file__',None) or package_root not in Path(module.__file__).resolve().parents:
                    raise Stop('PYTDX_IMPORT_COLLISION')
        sys.path.insert(0,str(package_root))
        try:
            spec = importlib.util.spec_from_file_location('bound_M19_BJ_DLI',config['query_sources']['interface']['path'])
            module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            session_factory = module.tdx_hq_session
        finally: sys.path.pop(0)
        sys.path.insert(0,str(package_root))
    began = time.monotonic()
    try:
        with m11.query_wall(budget,30):
            evidence['connect_calls'] += 1
            with session_factory(host=config['query_host'],port=config['query_port'],multithread=False) as api:
                if api.auto_retry or api.heartbeat: raise Stop('UNBOUNDED_API_CONFIGURATION')
                api.client.settimeout(5)
                evidence['query_calls'] += 1
                response = api.get_xdxr_info(2,code[:6])
                payload = (json.dumps(dict(symbol=code,market=2,route_status='UNVERIFIED_BJ_XDXR_CAPABILITY_HYPOTHESIS',
                    response_type=type(response).__name__,response=response),sort_keys=True,ensure_ascii=False,allow_nan=True)+'\n').encode()
                evidence['raw_response_snapshot'] = durable_new(Path(config['run_dir'])/'QUERY_RAW_RESPONSE.json',payload)
                evidence['request_identity'] = dict(market=2,code=code[:6],canonical_code=code)
                evidence['response_symbol_echo'] = 'NOT_EXPOSED_BY_BOUND_PARSER_NOT_INDEPENDENTLY_AUTHENTICATED'
                if not isinstance(response,(list,tuple)) or not response:
                    raise Stop('BJ_CAPABILITY_CANNOT_CONFIRM_EMPTY_OR_UNKNOWN_RESPONSE')
                return normalized_response(Path(config['run_dir'])/'QUERY_RAW_RESPONSE.json',code)
    finally:
        evidence['wall_sec'] = time.monotonic()-began
        if package_root is not None: sys.path.pop(0)


def actions_once(root, code, config, budget, start, existing=None, query_fn=None):
    if existing is not None:
        if code.endswith('.BJ') and not existing: raise Stop('BJ_CAPABILITY_CANNOT_CONFIRM_EMPTY_BOUND_ACTIONS')
        return existing, dict(source='BOUND_ALLCATEGORY_ARTIFACT', query_debit=0)
    claim, raw = root/'QUERY_CLAIM.json', root/'QUERY_RAW_RESPONSE.json'
    if claim.exists():
        if base.read_json(claim) != dict(symbol=code, semantic_id=config['semantic_id'], query_debit=1):
            raise Stop('QUERY_CLAIM_IDENTITY_DRIFT')
        if not raw.exists(): raise Stop('CONSUMED_QUERY_WITHOUT_DURABLE_RESPONSE_NO_RETRY')
        evidence_path = root/'QUERY_EVIDENCE.json'
        if not evidence_path.exists(): raise Stop('QUERY_METADATA_MISSING_NO_RETRY')
        evidence = base.read_json(evidence_path)
        snapshot = evidence.get('raw_response_snapshot')
        if not snapshot or Path(snapshot['path']) != raw: raise Stop('QUERY_RAW_BINDING_MISSING')
        check_artifact(snapshot,root)
        return normalized_response(raw,code), dict(source='SINGLE_ORIGINAL_MAC_TDX_QUERY',query_debit=1,**evidence)
    new_json(claim, dict(symbol=code, semantic_id=config['semantic_id'], query_debit=1))
    evidence = dict(query_calls=0, connect_calls=0, auto_retry=False)
    local = dict(config, run_dir=str(root))
    try:
        events = (query_fn or (bj_query_events if code.endswith('.BJ') else m11.query_events))(WorkerGate(start), local, budget, code, evidence)
    finally:
        content_once(root/'QUERY_EVIDENCE.json', evidence)
    return events, dict(source='SINGLE_ORIGINAL_MAC_TDX_QUERY', query_debit=1, **evidence)


class RecoveryBudget(base.Budget):
    """M6 RSS/wall probes; avoid rescanning thousands of outputs every 250 ms."""
    def __init__(self, root, caps):
        super().__init__(root,caps)
        self.last_scan = 0.0
        self.checking = False
    def check(self, *_):
        if self.checking: return
        self.checking = True
        try:
            self.peak = max(self.peak,base.rss_bytes())
            now = time.monotonic()
            if now-self.start_time > self.caps['total_wall_sec']: raise Stop('TOTAL_WALL_LIMIT')
            if self.peak > self.caps['rss_bytes']: raise Stop('OWNER_RSS_LIMIT')
            if now-self.last_scan >= 10:
                size = 0
                for path in self.root.rglob('*'):
                    if path.is_symlink(): raise Stop('OUTPUT_SYMLINK')
                    if path.is_file(): size += path.stat().st_size
                if size > self.caps['payload_bytes']: raise Stop('OUTPUT_PAYLOAD_LIMIT')
                self.last_scan = now
        finally: self.checking = False


def read_member(config, member, budget):
    path = Path(config['archive']['path'])
    if path.is_symlink(): raise Stop('ARCHIVE_SYMLINK')
    with path.open('rb') as handle:
        before = base.file_identity(os.fstat(handle.fileno()))
        if before != config['archive']['frozen_stat']: raise Stop('ARCHIVE_STAT_DRIFT')
        with zipfile.ZipFile(handle) as archive:
            infos = [x for x in archive.infolist() if x.filename == member['name']]
            if len(infos) != 1 or infos[0].file_size != member['bytes']: raise Stop('RAW_MEMBER_MISSING_OR_AMBIGUOUS')
            if member['bytes'] > config['member_byte_limit']: raise Stop('RAW_MEMBER_BYTE_LIMIT')
            budget.check(); payload = archive.read(infos[0])
        if before != base.file_identity(os.fstat(handle.fileno())) or before != base.file_identity(path.stat()):
            raise Stop('ARCHIVE_CHANGED_DURING_READ')
    if hashlib.sha256(payload).hexdigest() != member['sha256']: raise Stop('RAW_MEMBER_SHA_MISMATCH')
    return payload


def verify_cached_actions(cached, events, first_raw, anchor):
    checked = 0
    for old in cached:
        if not first_raw < old['_date_int'] <= min(anchor,20260821): continue
        matches = [e for e in events if e['category'] == 1 and e['_date_int'] == old['_date_int']]
        if len(matches) != 1 or any(affine.decimal_scalar(matches[0][k]) != affine.decimal_scalar(old[k]) for k in base.ACTION_FIELDS):
            raise Stop('CATEGORY1_CACHE_VS_ALLCATEGORY_CONFLICT')
        checked += 1
    return dict(participating_cached_events_checked=checked,
                cache_cutoff=20260821, cache_absence_not_treated_as_no_action=True)


def accepted_overlap(config, code, member):
    proof = config.get('accepted_overlap', {}).get(code)
    if proof is None: return None
    for key in ('config','receipt','review'): check_artifact(proof[key])
    original = base.read_json(proof['config']['path'])
    receipt = base.read_json(proof['receipt']['path'])
    review = base.read_json(proof['review']['path'])
    if original['qfq'] != config['qfq'] or original['archive'] != config['archive'] or original['pure_sources'] != config['pure_sources']:
        raise Stop('ACCEPTED_OVERLAP_INPUT_DRIFT')
    if proof['m11_source_sha256'] != config['code_by_file']['environment_tilt_mac_prefix_canary.py']:
        raise Stop('ACCEPTED_OVERLAP_CODE_DRIFT')
    if review.get('actual_run_receipt_sha256') != proof['receipt']['sha256'] or review.get('reviewer') == WRITER or not review.get('reviewer'):
        raise Stop('ACCEPTED_OVERLAP_REVIEW_DRIFT')
    if review.get('verdict') != 'OPERATIONAL_UNIT_BINDING_SUPPORTED_FOR_BOUND_SH_SZ_VARIANT_AND_OBSERVED_MEMBERS':
        raise Stop('ACCEPTED_OVERLAP_REVIEW_SCOPE')
    expected_events = [b for b in config['allcategory_bindings'] if code in b['symbols']]
    if len(expected_events) != 1 or expected_events[0]['sha256'] != proof['events_sha256']:
        raise Stop('ACCEPTED_OVERLAP_EVENT_DRIFT')
    touched = receipt['input_identity']['touched_candidates']
    if not any(t.get('symbol') == code and all(t.get(k) == member[k] for k in ('name','bytes','sha256')) for t in touched):
        raise Stop('ACCEPTED_OVERLAP_MEMBER_DRIFT')
    if receipt.get('symbol') != code or receipt.get('decision') != 'ONE_SYMBOL_CANDIDATE_PREFIX_ONLY' or receipt.get('overlap_price_mismatches') != 0 or receipt.get('anchor_identity') is not True or receipt.get('overlap_exact_keys') is not True:
        raise Stop('ACCEPTED_OVERLAP_NOT_PASSED')
    return dict(anchor=receipt['anchor'], first_old_bar=receipt['first_old_bar'],
                overlap_rows=receipt['overlap_OHLC_denominator']//4, receipt=proof['receipt'])


def validate_checkpoint(saved, root, config, code):
    if saved.get('semantic_id') != config['semantic_id'] or saved.get('symbol') != code:
        raise Stop('CHECKPOINT_IDENTITY_DRIFT')
    if saved.get('status') not in ('BLOCKED','HISTORICAL_PREFIX_ONLY','NO_OBSERVED_EARLY_ROWS'):
        raise Stop('CHECKPOINT_STATUS_INVALID')
    artifacts = saved.get('artifacts',[])
    for item in artifacts: check_artifact(item,root)
    names = [Path(item['path']).name for item in artifacts]
    if len(names) != len(set(names)): raise Stop('CHECKPOINT_DUPLICATE_ARTIFACT')
    if saved['status'] == 'HISTORICAL_PREFIX_ONLY':
        if not {'historical_prefix_v1.csv','COMPARISON.json','ALLCATEGORY.json'}.issubset(names):
            raise Stop('SUCCESS_CHECKPOINT_MISSING_PREFIX_ARTIFACTS')
        prefix = artifacts[names.index('historical_prefix_v1.csv')]
        if saved.get('prefix_rows',0) <= 0 or prefix.get('rows') != saved['prefix_rows'] or saved.get('block_reasons') != []:
            raise Stop('SUCCESS_CHECKPOINT_INVALID_PREFIX_STATUS')
        detail = base.read_json(root/'COMPARISON.json')
        if detail.get('anchor_identity') is not True or detail.get('overlap_cent_mismatches') != 0 or detail.get('block_reasons') != [] or detail.get('prefix_rows') != saved['prefix_rows']:
            raise Stop('SUCCESS_CHECKPOINT_COMPARISON_MISMATCH')
    if saved['status'] == 'NO_OBSERVED_EARLY_ROWS':
        if saved.get('coverage',{}).get('needs_prefix') is not False or saved['coverage'].get('observed_early_rows') != 0:
            raise Stop('NO_EARLY_CHECKPOINT_COVERAGE_MISSING')
    return saved


def trusted_checkpoint_binding(adapter, config, code, checkpoint):
    # Only the parent consults the official validated registry, never the worker.
    with adapter._locked():
        _,events = adapter._read_validated()
        matches = [e for e in events if e.get('event_type') == 'TERMINAL'
            and e.get('route_id') == 'a-share-environment-tilt-original-universe-full-prefix'
            and e.get('arm') == code and e.get('input_identity',{}).get('recovery_run_dir') == config['run_dir']
            and e.get('input_identity',{}).get('semantic_id') == config['semantic_id']
            and e.get('deterministic_metrics',{}).get('checkpoint_path') == str(checkpoint)
            and e.get('deterministic_metrics',{}).get('checkpoint_sha256')]
    if len(matches) != 1: raise Stop('TRUSTED_CHECKPOINT_TERMINAL_REQUIRED')
    terminal = matches[0]
    return dict(terminal=terminal,checkpoint=dict(path=str(checkpoint),sha256=terminal['deterministic_metrics']['checkpoint_sha256']))


def resume_checkpoint(task, root):
    proof = task.get('checkpoint_binding')
    if not proof: raise Stop('TRUSTED_CHECKPOINT_BINDING_REQUIRED')
    event,config,code = proof['terminal'],task['config'],task['code']
    if event_digest(event) != event.get('event_sha256') or event.get('event_type') != 'TERMINAL' or event.get('arm') != code or event.get('input_identity',{}).get('semantic_id') != config['semantic_id'] or event.get('input_identity',{}).get('recovery_run_dir') != config['run_dir']:
        raise Stop('TRUSTED_CHECKPOINT_EVENT_DRIFT')
    binding = proof['checkpoint']
    if Path(binding['path']) != root/'CHECKPOINT.json' or event['deterministic_metrics'].get('checkpoint_path') != binding['path'] or event['deterministic_metrics'].get('checkpoint_sha256') != binding['sha256']:
        raise Stop('TRUSTED_CHECKPOINT_REFERENCE_DRIFT')
    check_artifact(binding,root)
    saved = validate_checkpoint(base.read_json(binding['path']),root,config,code)
    if event.get('decision') != saved['status']: raise Stop('CHECKPOINT_TERMINAL_STATUS_DRIFT')
    return dict(saved,resumed=True)


def publish_checkpoint(root, result, config, code):
    validate_checkpoint(result,root,config,code)
    prepared = content_once(root/('CHECKPOINT_PREPARED_'+digest(result)+'.json'),result)
    # No replacement: publication exposes an already fsynced complete file.
    os.link(prepared['path'],root/'CHECKPOINT.json')
    descriptor = os.open(root,os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def symbol_job(task):
    """Spawn target; contains no registry adapter or retry loop."""
    config, code = task['config'], task['code']
    root = Path(config['run_dir'])/'symbols'/code
    root.mkdir(parents=True, exist_ok=True)
    with owner(root):
        base.configure_arrow_pools(1)
        checkpoint = root/'CHECKPOINT.json'
        if checkpoint.exists(): return resume_checkpoint(task,root)
        result = dict(symbol=code, semantic_id=config['semantic_id'], status='BLOCKED',
                      artifacts=[], training_ready=False, **FLAGS)
        try:
            budget = RecoveryBudget(root, config['worker_caps'])
            with budget:
                WorkerGate(task['start'])
                if code not in task['members']: raise Stop('ORIGINAL_MEMBER_NOT_BOUND')
                member = task['members'][code]
                payload = read_member(config, member, budget)
                dates = [d for d in m11.raw_dates(payload) if d <= POLICY['cutoff']]
                result.update(observed_source_first=dates[0] if dates else None,
                              observed_source_last=dates[-1] if dates else None,
                              raw_member=member,complete_member_datewords_parsed=True)
                binding = dict(task['unit_binding'], member_sha256=member['sha256'])
                reference = task['reference']
                import pandas as pd
                reference = base.normalize(pd.DataFrame(reference), code)
                need = coverage_gate(code, dates, int(reference.trade_date.min()), binding)
                result.update(coverage=need, raw_member=member)
                if not need['needs_prefix']:
                    result['status'] = 'NO_OBSERVED_EARLY_ROWS'
                else:
                    target = root/'original_raw.day'
                    if target.exists():
                        if sha(target) != member['sha256']: raise Stop('SAVED_RAW_MEMBER_DRIFT')
                    else: durable_new(target, payload)
                    functions = base.pure_functions(config)
                    raw = functions['read_day_tail'](code, target, tail_records=len(payload)//32)
                    events, query = actions_once(root, code, config, budget, task['start'], task['events'])
                    events_ref = content_once(root/'ALLCATEGORY.json', dict(symbol=code, cutoff=POLICY['cutoff'], events=events, query=query))
                    result['artifacts'].append(events_ref)
                    result['category1_cache_parity'] = verify_cached_actions(task.get('cached_events',[]),events,dates[0],int(reference.trade_date.max()))
                    if code.endswith('.BJ') and result['category1_cache_parity']['participating_cached_events_checked'] == 0:
                        raise Stop('BJ_CAPABILITY_CANNOT_CONFIRM_NO_PARTICIPATING_CATEGORY1_BINDING')
                    detail, candidate = analyze_full(raw, reference, events, code, accepted_overlap(config, code, member))
                    comparison = content_once(root/'COMPARISON.json', detail)
                    result['artifacts'].append(comparison)
                    result.update(query=query, block_reasons=detail['block_reasons'],
                                  overlap_rows=detail['overlap_rows'], prefix_rows=detail['prefix_rows'])
                    if not detail['block_reasons']:
                        result['artifacts'].append(emit_full(root, detail, candidate, raw))
                        result['status'] = 'HISTORICAL_PREFIX_ONLY'
                budget.check()
                result.update(wall_sec=time.monotonic()-budget.start_time, sampled_peak_rss_bytes=budget.peak)
        except BaseException as exc:
            result.update(status='BLOCKED', stop_reason=str(exc), error_type=type(exc).__name__)
        if code.endswith('.BJ'):
            result['bj_capability_status'] = 'SUPPORTED_FOR_BOUND_MEMBER_PREFIX_ONLY' if result['status'] == 'HISTORICAL_PREFIX_ONLY' else 'NO_OBSERVED_EARLY_ROWS' if result['status'] == 'NO_OBSERVED_EARLY_ROWS' else 'CAPABILITY_CANNOT_CONFIRM'
            result['response_symbol_identity'] = 'NO_INDEPENDENT_SYMBOL_ECHO_IN_BOUND_PARSER;REQUEST_PLUS_CACHE_AND_OVERLAP_ONLY'
        for name in ('QUERY_CLAIM.json','QUERY_RAW_RESPONSE.json','QUERY_EVIDENCE.json'):
            target = root/name
            if target.exists():
                result['artifacts'].append(dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target)))
        # Failed symbols are sealed too: resume does not retry failed work/query.
        publish_checkpoint(root,result,config,code)
        return result


class FullLifecycle(Lifecycle):
    def __init__(self, adapter, config):
        super().__init__(adapter,config)
        self.append_wall_sec = 0.0
        self.append_calls = 0
    def start(self, receipt):
        began = time.monotonic()
        try: return super().start(receipt)
        finally:
            self.append_wall_sec += time.monotonic()-began
            self.append_calls += 1
    def finish(self, receipt, decision, metrics=None):
        began = time.monotonic()
        try: return super().finish(receipt,decision,metrics)
        finally:
            self.append_wall_sec += time.monotonic()-began
            self.append_calls += 1
    def event(self, kind, decision, receipt, metrics):
        event = super().event(kind, decision, receipt, metrics)
        event.update(route_id='a-share-environment-tilt-original-universe-full-prefix',
                     experiment_id='A_SHARE_ENVIRONMENT_TILT_FULL_PREFIX_M19',
                     stage='MAC_ORIGINAL_UNIVERSE_PREFIX_' + self.config['phase'] + ('_SYMBOL' if self.config.get('scope_symbol') else '_PHASE'),
                     arm=self.config.get('scope_symbol','ORIGINAL_5576_NO_MEMBER_DROPPING'),
                     one_variable='M11_SOURCE_AFFINE_ALL_AVAILABLE_HISTORY',
                     evidence_level='HISTORICAL_PREFIX_ONLY',
                     sealed_status='NO_LABELS_NO_MODELS_NO_SCORES')
        event['input_identity'].update(recovery_run_dir=self.config['run_dir'],semantic_id=self.config.get('semantic_id'))
        return event


def semantic_identity(config):
    return digest({key:config[key] for key in ('policy','code_by_file','qfq','actions','archive','inventory',
        'pure_sources','query_sources','allcategory_bindings','unit_binding','accepted_overlap','bj_capability_probe')})


def validate_config(config, *, activated=False):
    if config['policy'] != POLICY or config.get('fit_count') != 0: raise Stop('FROZEN_POLICY_CHANGED')
    if config['phase'] not in ('CANARY','FULL'): raise Stop('PHASE_INVALID')
    workers = config['workers']
    if type(workers) is not int or not 1 <= workers <= 4: raise Stop('WORKER_LIMIT')
    if config['worker_caps']['threads'] != 1 or config['caps']['threads'] != 4: raise Stop('ONE_THREAD_PER_WORKER_REQUIRED')
    if config['query_host'] != '218.6.170.47' or config['query_port'] != 7709: raise Stop('ORIGINAL_ENDPOINT_CHANGED')
    if config['actions']['sha256'] != base.ACTION_SHA or config['qfq']['sha256'] != base.BASE_SHA or config['archive']['recorded_sha256'] != base.ZIP_SHA:
        raise Stop('ORIGINAL_INPUT_IDENTITY_CHANGED')
    root, output = Path(config['batch_root']).resolve(), Path(config['run_dir']).resolve()
    if output.parent != root or not output.name.startswith('M19_'): raise Stop('OUTPUT_SCOPE')
    if config['canary_symbols'] != list(CANARY): raise Stop('FIXED_CANARY_CHANGED')
    if config['bj_capability_probe'].get('market') != 2 or config['bj_capability_probe'].get('status') != 'UNVERIFIED_CAPABILITY_HYPOTHESIS' or config['bj_capability_probe'].get('source_sha256') != config['pure_sources']['data_layer']['sha256']:
        raise Stop('BJ_ROUTING_HYPOTHESIS_BINDING_DRIFT')
    if type(config['query_budget']) is not int or not 0 <= config['query_budget'] <= POLICY['universe_count']:
        raise Stop('QUERY_BUDGET_INVALID')
    if activated and (config.get('status') != 'FROZEN' or config.get('pending_dependencies')):
        raise Stop('DRAFT_OR_PENDING_DEPENDENCIES')


def load_config(path):
    config = base.read_json(path)
    config.update(config_path=str(Path(path).resolve()), config_sha256=sha(path))
    validate_config(config)
    if config['semantic_id'] != semantic_identity(config): raise Stop('SEMANTIC_IDENTITY_DRIFT')
    expected = config['code_by_file']
    if set(expected) != {*DEPENDENCIES, Path(__file__).name, '../../tests/test_environment_tilt_full_prefix.py'}:
        raise Stop('EXACT_CODE_SET_REQUIRED')
    for name, value in expected.items():
        if sha(Path(__file__).parent/name) != value: raise Stop('CODE_IDENTITY_DRIFT:' + name)
    config['code_composite_sha256'] = digest(expected)
    for item in [*config['pure_sources'].values(),*config['query_sources'].values()]:
        check_artifact(item)
    if config['unit_binding'].get('decoder_sha256') != config['pure_sources']['data_layer']['sha256']:
        raise Stop('UNIT_DECODER_BINDING_DRIFT')
    if sha(config['contract_path']) != config['contract_sha256']: raise Stop('CONTRACT_DRIFT')
    return config


def activation(config):
    if sys.platform != 'darwin': raise Stop('MAC_ONLY')
    validate_config(config, activated=True)
    proof = base.read_json(config['activation_path'])
    for key in ('code_composite_sha256','config_sha256','contract_sha256','semantic_id','phase','workers'):
        if proof.get(key) != config[key]: raise Stop('ACTIVATION_IDENTITY_DRIFT:' + key)
    if proof.get('status') != 'APPROVED_FOR_ONE_MAC_FULL_PREFIX_PHASE' or proof.get('writer') != WRITER or not proof.get('reviewer') or proof['reviewer'] == WRITER:
        raise Stop('INDEPENDENT_REVIEW_AND_CONTROLLER_ACTIVATION_REQUIRED')
    if proof.get('query_budget') != config['query_budget'] or proof.get('fit_count') != 0 or proof.get('network_allowed') is not True:
        raise Stop('ACTIVATION_BUDGET_MISMATCH')
    check_artifact(proof['review_receipt'])
    if not re.fullmatch(r'M19_[A-Za-z0-9_]{5,100}', proof.get('execution_id','')): raise Stop('INVOCATION_ID_REQUIRED')
    return proof


def source_bindings(life, config, budget):
    """One parent authentication per original input, no whole-ZIP rehash."""
    import pandas as pd
    life.require()
    archive_path = Path(config['archive']['path'])
    if archive_path.is_symlink() or base.file_identity(archive_path.stat()) != config['archive']['frozen_stat']:
        raise Stop('ARCHIVE_STAT_DRIFT')
    with base.authenticated(life, config['qfq'], budget) as handle:
        reference = pd.read_parquet(handle, columns=['trade_date','ts_code',*base.FIELDS])
    # Do not drop malformed keys/rows or infer an alternate universe.
    if reference.ts_code.isna().any() or reference[['trade_date','ts_code']].duplicated().any():
        raise Stop('ORIGINAL_UNIVERSE_KEY_ERROR')
    codes = sorted(reference.ts_code.unique())
    if len(codes) != POLICY['universe_count'] or any(not re.fullmatch(r'\d{6}\.(SH|SZ|BJ)', code) for code in codes):
        raise Stop('ORIGINAL_5576_UNIVERSE_MISMATCH')
    with base.authenticated(life, config['inventory'], budget) as handle: members = inventory(handle)
    with base.authenticated(life, config['actions'], budget) as handle: cache = pd.read_parquet(handle)
    cache['_date_int'] = pd.to_datetime(cache._xdxr_trade_date.astype(str).str.replace(r'\.0$','',regex=True).str[:10],errors='raise').dt.strftime('%Y%m%d').astype('int64')
    cached = {code:part.assign(category=1).to_dict('records') for code,part in cache.groupby('ts_code')}
    events = {}
    for binding in config['allcategory_bindings']:
        with base.authenticated(life, binding, budget) as handle:
            if binding['format'] == 'ORIGINAL_JSONL':
                items = [json.loads(line) for line in handle if line.strip()]
                for code in binding['symbols']:
                    events[code] = m11.normalize_allcategory([e for e in items if e['symbol'] == code], POLICY['cutoff'])
            elif binding['format'] == 'M11_NORMALIZED_JSON':
                item = json.load(handle)
                if item['symbol'] != binding['symbols'][0]: raise Stop('EVENT_SYMBOL_BINDING')
                events[item['symbol']] = item['events']
            else: raise Stop('UNKNOWN_EVENT_BINDING_FORMAT')
        if binding['cutoff'] != POLICY['cutoff']: raise Stop('ALLCATEGORY_CUTOFF_BINDING')
    return reference, codes, members, events, cached


def summarize(codes, results):
    rows = [results.get(code, dict(symbol=code,status='NOT_EXECUTED')) for code in codes]
    counts = dict(Counter(row['status'] for row in rows))
    complete = len(results) == len(codes)
    ready = complete and all(r['status'] in ('HISTORICAL_PREFIX_ONLY','NO_OBSERVED_EARLY_ROWS') for r in rows)
    return dict(status='ALL_ORIGINAL_MEMBERS_PREFIX_READY' if ready else 'CANNOT_CONFIRM_FULL_UNIVERSE',
                original_members=len(codes), reported_members=len(rows), completed_members=len(results),
                counts=counts, members=rows, prefix_ready=ready, observed_history_only=True, historical_source_completeness='CANNOT_CONFIRM', training_ready=False, **FLAGS)


def ensure_no_orphan(adapter, config):
    with adapter._locked():
        _,events = adapter._read_validated()
        own = [e for e in events if e.get('route_id') == 'a-share-environment-tilt-original-universe-full-prefix'
               and e.get('input_identity',{}).get('recovery_run_dir') == config['run_dir']]
        terminal = {e['execution_id'] for e in own if e.get('event_type') == 'TERMINAL'}
        if any(e.get('event_type') == 'RUN_STARTED' and e['execution_id'] not in terminal for e in own):
            raise Stop('COMPLETION_PENDING_AUDIT_OWNED_EXECUTION_STILL_OPEN')


def run(config, adapter):
    proof = activation(config)
    config = dict(config, execution_id=proof['execution_id'])
    root = Path(config['run_dir'])
    with owner(root):
        ensure_no_orphan(adapter, config)
        invocation = root/'invocations'/config['execution_id']; invocation.mkdir(parents=True, exist_ok=False)
        receipt = invocation/'RECEIPT.json'
        base.write_json(receipt, dict(status='PREPARED_NOT_REGISTERED', **FLAGS))
        life = FullLifecycle(adapter, config)
        codes, results, pool = [], {}, None
        budget = RecoveryBudget(root, config['caps'])
        try:
            life.start(receipt)
            start = life.require()
            with budget:
                if base.available_memory() < config['caps']['min_memory_bytes'] or shutil.disk_usage(root).free < config['caps']['min_disk_bytes']:
                    raise Stop('DEFERRED_RESOURCE')
                os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
                config['arrow_pools'] = base.configure_arrow_pools(1)
                reference, codes, members, events, cached = source_bindings(life, config, budget)
                groups = reference.groupby('ts_code',sort=False)
                chosen = list(CANARY) if config['phase'] == 'CANARY' else codes
                if any(code not in codes for code in chosen): raise Stop('FIXED_CANARY_NOT_IN_ORIGINAL_UNIVERSE')
                # FULL activation must bind a separately reviewed mechanical canary receipt.
                if config['phase'] == 'FULL':
                    check_artifact(config['canary_receipt'])
                    canary = base.read_json(config['canary_receipt']['path'])
                    if canary.get('semantic_id') != config['semantic_id'] or canary.get('canary_pass') is not True:
                        raise Stop('MECHANICAL_CANARY_NOT_PASSED')
                pending = iter(chosen); active = {}; children = {}
                prior_claims = list((root/'symbols').glob('*/QUERY_CLAIM.json')) if (root/'symbols').exists() else []
                for claim in prior_claims:
                    data = base.read_json(claim)
                    if data.get('semantic_id') != config['semantic_id'] or data.get('query_debit') != 1:
                        raise Stop('PRIOR_QUERY_CLAIM_DRIFT')
                query_reserved = len(prior_claims)
                if query_reserved > config['query_budget']: raise Stop('PRIOR_QUERY_BUDGET_EXCEEDED')
                context = multiprocessing.get_context('spawn')
                pool = ProcessPoolExecutor(max_workers=config['workers'], mp_context=context)
                while True:
                    while len(active) < config['workers']:
                        try: code = next(pending)
                        except StopIteration: break
                        checkpoint = root/'symbols'/code/'CHECKPOINT.json'
                        # Reserve before dispatch: one attempt per missing symbol, including failed attempts.
                        if code not in events and not checkpoint.exists() and not (checkpoint.parent/'QUERY_CLAIM.json').exists() and code.endswith(('.SH','.SZ','.BJ')):
                            query_reserved += 1
                            if query_reserved > config['query_budget']:
                                results[code] = dict(symbol=code,status='BLOCKED',stop_reason='PHASE_QUERY_BUDGET_EXHAUSTED')
                                continue
                        symbol_life, symbol_receipt, token = None, checkpoint, start
                        if not checkpoint.exists():
                            child_config = dict(config, execution_id=config['execution_id']+'_'+code.replace('.','_'), scope_symbol=code)
                            symbol_life = FullLifecycle(adapter, child_config)
                            symbol_receipt = invocation/(code+'_TERMINAL.json')
                            base.write_json(symbol_receipt,dict(status='PREPARED_NOT_REGISTERED',symbol=code))
                            symbol_life.start(symbol_receipt)
                            token = symbol_life.require()
                            children[code] = (symbol_life,symbol_receipt)
                        binding = trusted_checkpoint_binding(adapter,config,code,checkpoint) if checkpoint.exists() else None
                        task = dict(config=config, code=code, start=token, checkpoint_binding=binding, members={code:members[code]} if code in members else {},
                                    unit_binding=config['unit_binding'], events=events.get(code),cached_events=cached.get(code,[]),
                                    reference=groups.get_group(code).sort_values('trade_date',kind='stable').to_dict('records'))
                        active[pool.submit(symbol_job,task)] = code
                    if not active: break
                    done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                    budget.check()
                    for future in done:
                        code = active.pop(future)
                        try: results[code] = future.result()
                        except BaseException as exc: results[code] = dict(symbol=code,status='BLOCKED',stop_reason='WORKER_EXIT:'+str(exc))
                        base.write_json(receipt, dict(status='RUNNING',semantic_id=config['semantic_id'],phase=config['phase'],
                            counts=dict(Counter(r['status'] for r in results.values())),completed_members=len(results),
                            last_checkpoint=code,original_members=len(codes),query_reserved=query_reserved, **FLAGS))
                        if code in children:
                            child, child_receipt = children[code]
                            base.write_json(child_receipt,results[code])
                            checkpoint_file = root/'symbols'/code/'CHECKPOINT.json'
                            if checkpoint_file.exists():
                                expected = (canonical(results[code])+'\n').encode()
                                check_artifact(dict(path=str(checkpoint_file),bytes=len(expected),sha256=hashlib.sha256(expected).hexdigest()),checkpoint_file.parent)
                                validate_checkpoint(results[code],checkpoint_file.parent,config,code)
                            child.finish(child_receipt,results[code]['status'],dict(symbol=code,
                                checkpoint_path=str(root/'symbols'/code/'CHECKPOINT.json'),
                                checkpoint_sha256=sha(root/'symbols'/code/'CHECKPOINT.json') if (root/'symbols'/code/'CHECKPOINT.json').exists() else None))
                    # BJ/source failure closes canary; no replacement market/sample.
                pool.shutdown(wait=True); pool = None
                summary = summarize(codes, results)
                canary_pass = config['phase']=='CANARY' and all(results.get(c,{}).get('status') in ('HISTORICAL_PREFIX_ONLY','NO_OBSERVED_EARLY_ROWS') for c in CANARY)
                actual_query_debits = len(list((root/'symbols').glob('*/QUERY_CLAIM.json')))
                value = dict(summary, actual_query_debits=actual_query_debits, semantic_id=config['semantic_id'], phase=config['phase'],canary_pass=canary_pass,
                             query_reserved=query_reserved, wall_sec=time.monotonic()-budget.start_time,
                             sampled_parent_peak_rss_bytes=budget.peak)
                base.write_json(receipt,value)
            life.finish(receipt,value['status'],dict(completed=len(results),query_reserved=query_reserved))
            append_owners = [life,*[child for child,_ in children.values()]]
            timing = dict(total_wall_sec=time.monotonic()-budget.start_time,
                          official_append_calls=sum(x.append_calls for x in append_owners),
                          official_append_and_start_validation_wall_sec=sum(x.append_wall_sec for x in append_owners),
                          completed_members=len(results),workers=config['workers'],
                          reused_members=sum(bool(x.get('resumed')) for x in results.values()),
                          actual_query_debits=actual_query_debits,
                          scope='Includes official registry append/validation cost; no full-pool throughput extrapolation certified')
            value['operational_timing'] = new_json(invocation/'OPERATIONAL_TIMING.json',timing)
            return value
        except BaseException as exc:
            # Already dispatched workers finish their owned checkpoints before the parent terminal.
            if pool is not None: pool.shutdown(wait=True, cancel_futures=True)
            for code,(child,child_receipt) in locals().get('children',{}).items():
                if child.started and not child.finished:
                    checkpoint = root/'symbols'/code/'CHECKPOINT.json'
                    result = base.read_json(checkpoint) if checkpoint.exists() else dict(symbol=code,status='INTERRUPTED_OR_NOT_DISPATCHED',stop_reason=str(exc))
                    base.write_json(child_receipt,result)
                    if checkpoint.exists(): validate_checkpoint(result,checkpoint.parent,config,code)
                    child.finish(child_receipt,result['status'],dict(symbol=code,checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint) if checkpoint.exists() else None))
                    results[code] = result
            failure = dict(status='FAILED', stop_reason=str(exc), summary=summarize(codes,results), **FLAGS)
            base.write_json(receipt,failure)
            if life.started and not life.finished: life.finish(receipt,'FAILED',dict(completed=len(results)))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path); parser.add_argument('--execute',action='store_true')
    args = parser.parse_args(); config = load_config(args.config)
    if not args.execute:
        print(canonical(dict(status='M19_METADATA_ONLY_NOT_ACTIVATED',phase=config['phase'],pending=config['pending_dependencies'])))
        return
    # Adapter construction is metadata-only; actual adapter reads occur after activation in run.
    activation(config)
    run(config,load_adapter(config))


if __name__ == '__main__': main()
