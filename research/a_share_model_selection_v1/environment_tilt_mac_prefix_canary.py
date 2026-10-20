"""One-symbol identity-anchor source diagnostic; default CLI is metadata only.

Selection, payload authentication and the optional single original TDX query
require a durable registered execution. No model or outcome labels are read.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import struct
import sys
import time
import zipfile
from collections import Counter
from decimal import Decimal, localcontext, ROUND_HALF_UP
from pathlib import Path

import environment_tilt_mac_recovery as base
import environment_tilt_mac_affine as affine
from environment_tilt_mac_registry import Lifecycle, FLAGS, canonical, sha, load_adapter

PREFERRED = ("000001.SZ", "600000.SH")
INFORMATIONAL_CATEGORIES = (2, 3, 5, 6, 7, 8, 9, 10)
BASELINE = "M9 bounded affine240of240 OPENBOOK; original M0 and QFQ unchanged"
POLICY = dict(cutoff=20261015, prefix_start=20200101, precision=50,
    rounding="ROUND_HALF_UP", quantum="0.01", preferred=list(PREFERRED),
    selection="FIRST_NONUNIT_EVENT_AND_EXACT_COMMON_KEYS_NO_RESIDUAL",
    anchor="LATEST_COMMON_RAW_QFQ_KEY_IDENTITY_OHLC_NO_FIT",
    unsupported_prefix_categories="EXCEPT1_AND_INFORMATIONAL_2_3_5_6_7_8_9_10",
    query_count_limit=1, query_wall_sec=30, query_retry=False)
CODE_FILES = (*affine.CODE_FILES, "environment_tilt_mac_prefix_canary.py",
              "tests/test_environment_tilt_mac_prefix_canary.py")


class PrefixStop(base.DiagnosticError):
    pass


class QueryWallStop(BaseException):
    """Bypass legacy SDK exception swallowing/retry wrappers on hard wall."""


class PrefixLifecycle(Lifecycle):
    def event(self, kind, decision, receipt, metrics):
        event = super().event(kind, decision, receipt, metrics)
        event.update(route_id="a-share-environment-tilt-mac-prefix-canary-20261021",
            experiment_id="A_SHARE_ENVIRONMENT_TILT_MAC_PREFIX_CANARY_20261021",
            stage="ONE_SYMBOL_NONUNIT_IDENTITY_ANCHOR_PREFIX_CANARY",
            baseline_id=BASELINE,
            one_variable="IDENTITY_ANCHOR_CATEGORY1_AFFINE_NONUNIT_OVERLAP_AND_SEPARATE_PREFIX",
            sealed_status="OPENBOOK_SOURCE_SEMANTICS_NO_OUTCOME_LABELS",
            evidence_level="OPENBOOK_DIAGNOSTIC_ONLY")
        return event


def nonunit(event):
    try:
        if int(event.get("category", 1)) != 1:
            return False
        bonus = affine.decimal_scalar(event["xdxr_songzhuangu"])
        rights = affine.decimal_scalar(event["xdxr_peigu"])
        with localcontext() as ctx:
            ctx.prec,ctx.rounding=50,ROUND_HALF_UP
            a, _ = affine.event_affine(event)
        return bool((bonus != 0 or rights != 0) and a != 1)
    except (KeyError, ValueError, affine.CannotConfirm):
        return False


def normalize_allcategory(records, cutoff):
    """Preserve original category/fields, never synthesize absent event values."""
    result = []
    for item in records:
        record = item.get("record", item)
        text = str(item.get("event_date", "")).replace("-", "")
        if not re.fullmatch(r"\d{8}", text):
            raise PrefixStop("MALFORMED_ALLCATEGORY_EVENT_DATE")
        import datetime
        datetime.datetime.strptime(text, "%Y%m%d")
        date = int(text)
        if date > cutoff:
            continue
        category = affine.decimal_scalar(record.get("category"))
        if category != category.to_integral_value():
            raise PrefixStop("MALFORMED_ACTION_CATEGORY")
        event = dict(_date_int=date, category=int(category), original_record=record)
        for original, target in zip(("fenhong", "peigujia", "songzhuangu", "peigu"), base.ACTION_FIELDS):
            if original in record:
                event[target] = record[original]
        result.append(event)
    return sorted(result, key=lambda e: (e["_date_int"], e["category"]))


def select_symbol(universe_keys, preferred_events, cached_events, date_loader, touched):
    """Only event fields/date keys enter selection, never OHLC or residuals."""
    def try_code(code, events, source):
        if code not in universe_keys or not code.endswith((".SH", ".SZ")):
            return None
        old = sorted(set(int(d) for d in universe_keys[code] if int(d) <= POLICY["cutoff"]))
        candidates = [e for e in events if nonunit(e) and old and old[0] < e["_date_int"] < old[-1]]
        if not candidates:
            return None
        dates, evidence = date_loader(code)
        touched.append(dict(symbol=code, **evidence))
        common = sorted(set(dates) & set(old))
        for event in sorted(candidates, key=lambda e: e["_date_int"]):
            date = event["_date_int"]
            if any(d < date for d in common) and any(d > date for d in common):
                return dict(symbol=code, selected_event=event, selection_source=source,
                    common_keys=common, anchor=max(common), first_old_bar=min(old))
        return None
    for code in PREFERRED:
        result = try_code(code, preferred_events.get(code, []), "BOUND_ALLCATEGORY_ARTIFACT")
        if result:
            return result
    for code in sorted(cached_events):
        result = try_code(code, cached_events[code], "BOUND_CATEGORY1_CACHE_FALLBACK")
        if result:
            return result
    raise PrefixStop("NO_ELIGIBLE_NONUNIT_SAMPLE_WITHIN_BOUND_INPUTS")


def inventory_bindings(handle):
    result = {}
    for row in csv.DictReader(io.StringIO(handle.read().decode("utf-8-sig"))):
        suffix = {"sh": "SH", "sz": "SZ"}.get(row["market"].lower())
        if suffix is None:
            continue
        code = row["security_code"].zfill(6) + "." + suffix
        # Index includes stocks, indexes and other instruments; original QFQ
        # membership, not this classification, determines the universe later.
        binding = dict(name=row["member_name"], bytes=int(row["uncompressed_bytes"]),
            sha256=row["uncompressed_member_sha256"])
        if code in result:
            raise PrefixStop("AMBIGUOUS_RAW_MEMBER:" + code)
        result[code] = binding
    return result


def raw_dates(payload):
    """Parse dateword only for selection; no price decoding."""
    if len(payload) % 32:
        raise PrefixStop("NONINTEGRAL_DAY_RECORD")
    import datetime
    dates = [struct.unpack_from("<I", payload, offset)[0] for offset in range(0, len(payload), 32)]
    for date in dates:
        datetime.datetime.strptime(str(date), "%Y%m%d")
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise PrefixStop("RAW_DATE_ORDER_OR_DUPLICATE")
    return dates


def choose_from_archive(life, c, budget, bindings, universe, preferred, cached):
    life.require()
    archive_path = Path(c["archive"]["path"])
    if archive_path.is_symlink():
        raise PrefixStop("ARCHIVE_SYMLINK")
    touched, payloads, trace = [], {}, []
    with archive_path.open("rb") as handle:
        before = base.file_identity(os.fstat(handle.fileno()))
        if before != c["archive"]["frozen_stat"]:
            raise PrefixStop("ARCHIVE_STAT_DRIFT")
        with zipfile.ZipFile(handle) as archive:
            def dates(code):
                life.require();budget.check()
                binding = bindings.get(code)
                if binding is None:
                    trace.append(dict(symbol=code,status='NO_BOUND_MEMBER',bytes=0))
                    base.write_json(Path(c['run_dir'])/'SELECTION_TRACE.json',dict(touched=trace))
                    return [], dict(status="NO_BOUND_MEMBER", bytes=0)
                entry=dict(symbol=code,status='READ_STARTED_DATEWORDS_ONLY',**binding);trace.append(entry)
                base.write_json(Path(c['run_dir'])/'SELECTION_TRACE.json',dict(touched=trace))
                if binding["bytes"] > 2**20:
                    raise PrefixStop("MEMBER_BYTE_CAP")
                infos = [i for i in archive.infolist() if i.filename == binding["name"]]
                if len(infos) != 1 or infos[0].file_size != binding["bytes"]:
                    raise PrefixStop("EXACT_BOUND_MEMBER_MISSING")
                payload = archive.read(infos[0])
                if hashlib.sha256(payload).hexdigest() != binding["sha256"]:
                    raise PrefixStop("RAW_MEMBER_SHA_MISMATCH")
                keys = raw_dates(payload)
                entry['status']='SHA_AND_DATEWORDS_AUTHENTICATED'
                base.write_json(Path(c['run_dir'])/'SELECTION_TRACE.json',dict(touched=trace))
                payloads.clear();payloads[code] = payload
                return keys, dict(status="DATEWORDS_ONLY_AUTHENTICATED", **binding)
            chosen = select_symbol(universe, preferred, cached, dates, touched)
            payload = payloads[chosen["symbol"]]
        if before != base.file_identity(os.fstat(handle.fileno())) or before != base.file_identity(archive_path.stat()):
            raise PrefixStop("ARCHIVE_CHANGED_DURING_SELECTION")
    return chosen, payload, dict(touched_candidates=touched,
        authenticated_candidate_bytes=sum(e["bytes"] for e in touched),
        whole_zip_status="REUSED_RECORDED_NOT_REHASHED",
        whole_zip_recorded_sha256=c["archive"]["recorded_sha256"])


def reconstruct(raw, events, anchor):
    """No oracle price or estimated scale/offset enters this operator."""
    dates = raw.trade_date.astype(int).tolist()
    if anchor not in dates or dates != sorted(set(dates)):
        raise PrefixStop("INVALID_IDENTITY_ANCHOR_KEYS")
    participating = sorted([e for e in events if dates[0] < e["_date_int"] <= anchor and e["category"] == 1], key=lambda e:e["_date_int"])
    if len({e["_date_int"] for e in participating}) != len(participating):
        raise PrefixStop("AMBIGUOUS_CATEGORY1_EVENT_DATE")
    by_date = raw.set_index("trade_date")
    with localcontext() as ctx:
        ctx.prec, ctx.rounding = 50, ROUND_HALF_UP
        coefficients = [(e["_date_int"], *affine.event_affine(e)) for e in participating]
        a, b, pos = Decimal(1), Decimal(0), len(coefficients)-1
        result = []
        for date in reversed([d for d in dates if d <= anchor]):
            while pos >= 0 and coefficients[pos][0] > date:
                _, x, y = coefficients[pos]
                a, b = a*x, a*y+b  # already-composed later actions outside this earlier action
                pos -= 1
            row = dict(trade_date=date)
            for field in affine.OHLC:
                value = a*affine.decimal_scalar(by_date.loc[date,field])+b
                row[field] = dict(unrounded=str(value), cent=str(value.quantize(affine.CENT)))
            result.append(row)
    return list(reversed(result))


def analyze(raw, reference, events, chosen):
    code, anchor = chosen["symbol"], chosen["anchor"]
    earliest=int(raw.trade_date.min())
    raw=raw.loc[raw.trade_date.between(min(POLICY['prefix_start'],chosen['first_old_bar']),anchor)]
    reference_keys=reference.trade_date.astype(str)
    if not reference_keys.str.fullmatch(r'\d{8}').all():raise PrefixStop('REFERENCE_DATE_KEY')
    reference=reference.loc[reference_keys.astype('int64').le(anchor)]
    raw, reference = base.normalize(raw, code), base.normalize(reference.sort_values("trade_date"), code)
    original = reference.loc[reference.trade_date.le(anchor)].copy()
    low = int(original.trade_date.min())
    raw_overlap = raw.loc[raw.trade_date.between(low,anchor)].copy()
    exact_keys = raw_overlap.trade_date.tolist() == original.trade_date.tolist()
    anchor_identity = True
    with localcontext() as ctx:
        ctx.prec,ctx.rounding=50,ROUND_HALF_UP
        for field in affine.OHLC:
            x=affine.cent_reference(raw.loc[raw.trade_date.eq(anchor),field].iloc[0])
            y=affine.cent_reference(original.loc[original.trade_date.eq(anchor),field].iloc[0])
            anchor_identity &= x==y
    relevant_raw=raw.loc[raw.trade_date.between(min(POLICY['prefix_start'],chosen['first_old_bar']),anchor)]
    rebuilt = reconstruct(relevant_raw, events, anchor)
    overlap = [row for row in rebuilt if row["trade_date"] in set(original.trade_date)]
    common_reference = original.loc[original.trade_date.isin([row["trade_date"] for row in overlap])]
    comparison,mismatches=affine.compare_prices(overlap,common_reference)
    unsupported=[e for e in events if max(int(relevant_raw.trade_date.min()),POLICY["prefix_start"]) < e["_date_int"] <= anchor and e["category"] not in (1,*INFORMATIONAL_CATEGORIES)]
    # Keep overlap diagnostics even when unsupported economic types block prefix.
    candidate=[row for row in rebuilt if POLICY["prefix_start"] <= row["trade_date"] < chosen["first_old_bar"]]
    reasons=[]
    if not anchor_identity:reasons.append("LATEST_COMMON_ANCHOR_NOT_IDENTITY")
    if not exact_keys:reasons.append("WHOLE_OVERLAP_DATE_KEYS_MISMATCH")
    if mismatches:reasons.append("WHOLE_OVERLAP_CENT_MISMATCH")
    if unsupported:reasons.append("UNSUPPORTED_PARTICIPATING_NON1_CATEGORIES")
    if not candidate:reasons.append("NO_PREFIX_ROWS_IN_FIXED_RANGE")
    ancillary=base.exact_comparison(raw_overlap.loc[raw_overlap.trade_date.isin(common_reference.trade_date)].reset_index(drop=True),common_reference.reset_index(drop=True))
    return dict(decision="PREFIX_BLOCKED" if reasons else "ONE_SYMBOL_CANDIDATE_PREFIX_ONLY",
        symbol=code,anchor=anchor,anchor_identity=bool(anchor_identity),first_old_bar=chosen["first_old_bar"],
        available_earliest_raw_date=earliest,overlap_exact_keys=exact_keys,
        overlap_OHLC_denominator=4*len(overlap),overlap_price_mismatches=mismatches,
        overlap_prices=comparison,allcategory_counts=dict(Counter(str(e["category"]) for e in events)),
        unsupported_participating_categories=dict(Counter(str(e["category"]) for e in unsupported)),
        informational_participating_categories=dict(Counter(str(e["category"]) for e in events if max(int(relevant_raw.trade_date.min()),POLICY['prefix_start']) < e['_date_int'] <= anchor and e['category'] in INFORMATIONAL_CATEGORIES)),
        category_policy='Original producer filters1 for adjustment;2/3/5/6/7/8/9/10 listing/share-capital labels retained/countedas informational for this stock route.4/11/12/13/14/unrecognized price-effective interpretation unresolved and blockprefix; no universal issuer certification.',
        prefix_block_reasons=reasons,prefix_candidate_rows=len(candidate),
        raw_ancillary_comparison={k:ancillary[k] for k in ("vol","amount")},
        ancillary_feature_use="BLOCKED_UNCONFIRMED_PHYSICAL_UNITS_ONLY_TQ_SMALL_DIFFERENCES_NOT_A_GATE",
        action_completeness="CANNOT_CONFIRM_UNIVERSAL_COMPLETENESS_OR_ISSUER_UNITS",
        native_QFQ_certified=False,training_ready=False,**FLAGS),candidate


@contextlib.contextmanager
def query_wall(budget, seconds):
    started=time.monotonic();previous=signal.getsignal(signal.SIGALRM)
    previous_timer=signal.getitimer(signal.ITIMER_REAL)
    def check(*args):
        budget.check()
        remaining=seconds-(time.monotonic()-started)
        if remaining<=0:raise QueryWallStop("SINGLE_QUERY_TOTAL_WALL")
        signal.setitimer(signal.ITIMER_REAL,min(.25,remaining))
    signal.signal(signal.SIGALRM,check)
    signal.setitimer(signal.ITIMER_REAL,min(.25,seconds))
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
        if previous_timer[0] or previous_timer[1]:signal.setitimer(signal.ITIMER_REAL,*previous_timer)


def query_events(life,c,budget,code,evidence,session_factory=None):
    life.require()
    if not code.endswith((".SH",".SZ")) or evidence["query_calls"]:
        raise PrefixStop("QUERY_SCOPE_OR_COUNT")
    for item in c["query_sources"].values():
        if sha(item["path"])!=item["sha256"]:raise PrefixStop("QUERY_SOURCE_DRIFT")
    if session_factory is None:
        root=Path(c["query_package_root"]).resolve()
        for name,mod in list(sys.modules.items()):
            if name=="pytdx" or name.startswith("pytdx."):
                if not getattr(mod,"__file__",None) or root not in Path(mod.__file__).resolve().parents:
                    raise PrefixStop("PYTDX_IMPORT_COLLISION")
        sys.path.insert(0,str(root))
        try:
            path=c["query_sources"]["interface"]["path"]
            spec=importlib.util.spec_from_file_location("bound_prefix_DLI",path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            session_factory=module.tdx_hq_session
        finally:sys.path.pop(0)
        # DLI imports pytdx lazily when session enters; retain the bound lookup
        # only around this one query and restore it afterwards.
        sys.path.insert(0,str(root))
    else:root=None
    started=time.monotonic()
    try:
        with query_wall(budget,POLICY["query_wall_sec"]):
            evidence["connect_calls"]+=1
            with session_factory(host=c["query_host"],port=c["query_port"],multithread=False) as api:
                if api.auto_retry or api.heartbeat:raise PrefixStop("UNBOUNDED_API_CONFIGURATION")
                api.client.settimeout(5)
                evidence["query_calls"]+=1
                response=api.get_xdxr_info(1 if code.endswith('.SH') else 0,code[:6])
                # Forensic source snapshot must survive a malformed date/category.
                # Preserve the parser's raw Python-JSON response (including its
                # nonfinite literals); only subsequent normalization may admit
                # values into the frozen arithmetic/cutoff pipeline.
                raw_path=Path(c['run_dir'])/'QUERY_RAW_RESPONSE.json'
                payload=json.dumps(dict(symbol=code,response_type=type(response).__name__,
                    response=response),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=True).encode()+b'\n'
                with raw_path.open('xb') as handle:
                    handle.write(payload);handle.flush();os.fsync(handle.fileno())
                evidence['raw_response_snapshot']=dict(path=str(raw_path),
                    sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload),
                    serialization='RAW_PARSER_PYTHON_JSON_BEFORE_NORMALIZATION')
                if not isinstance(response,(list,tuple)):raise PrefixStop("QUERY_RESPONSE_UNKNOWN")
                records=[dict(event_date=f"{int(e['year']):04d}-{int(e['month']):02d}-{int(e['day']):02d}",record=e) for e in response]
                return normalize_allcategory(records,POLICY["cutoff"])
    finally:
        evidence['wall_sec']=time.monotonic()-started
        if root is not None:sys.path.pop(0)


def emit_prefix(output,result,candidate,raw,life):
    if result["decision"]!="ONE_SYMBOL_CANDIDATE_PREFIX_ONLY":return None
    life.require()
    output=Path(output)/"candidate_prefix_v1.csv"
    lookup=raw.set_index('trade_date')
    with output.open('x') as handle:
        writer=csv.DictWriter(handle,fieldnames=['trade_date','ts_code',*affine.OHLC,'vol','amount']);writer.writeheader()
        for row in candidate:
            writer.writerow(dict(trade_date=row['trade_date'],ts_code=result['symbol'],
                **{field:row[field]['cent'] for field in affine.OHLC},
                vol=lookup.loc[row['trade_date'],'vol'],amount=lookup.loc[row['trade_date'],'amount']))
    return dict(path=str(output),sha256=sha(output),rows=len(candidate),
        version='ONE_SYMBOL_OBSERVED_CACHE_IDENTITY_ANCHOR_CANDIDATE_V1',ancillary_feature_use='BLOCKED')


def load_config(path):
    path=Path(path).resolve();c=base.read_json(path);c.update(config_path=str(path),config_sha256=sha(path))
    if c['policy']!=POLICY or c['caps']!=base.CAPS or c.get('fit_count')!=0 or c.get('network_allowed') is not True:
        raise PrefixStop('FROZEN_PREFIX_POLICY_CHANGED')
    output,root=Path(c['run_dir']).resolve(),Path(c['batch_root']).resolve()
    if output.parent!=root or not output.name.startswith('M11_'):raise PrefixStop('OUTPUT_OUTSIDE_SCOPE')
    identity=base.read_json(c['identity_path'])
    if sha(c['identity_path'])!=c['identity_sha256'] or set(identity['code_by_file'])!=set(CODE_FILES):raise PrefixStop('IDENTITY_DRIFT')
    for name,digest in identity['code_by_file'].items():
        if sha(Path(__file__).parent/name)!=digest:raise PrefixStop('CODE_DRIFT:'+name)
    c['code_composite_sha256']=hashlib.sha256(canonical(identity['code_by_file']).encode()).hexdigest()
    if sha(c['contract_path'])!=c['contract_sha256']:raise PrefixStop('CONTRACT_DRIFT')
    for item in [*c['pure_sources'].values(),*c['query_sources'].values()]:
        if sha(item['path'])!=item['sha256']:raise PrefixStop('BOUND_SOURCE_DRIFT')
    if c['qfq']['sha256']!=base.BASE_SHA or c['actions']['sha256']!=base.ACTION_SHA or c['archive']['recorded_sha256']!=base.ZIP_SHA:raise PrefixStop('ORIGINAL_INPUT_BINDING_CHANGED')
    return c


def validate_activation(c):
    # Accepted proof schema has network_allowed=False. This isolated optional
    # query needs its own explicit identity-bound review activation instead.
    p=base.read_json(c['activation_path'])
    for key in ['code_composite_sha256','config_sha256','contract_sha256']:
        if p.get(key)!=c[key]:raise PermissionError('PREFIX_REVIEW_IDENTITY_MISMATCH')
    if p.get('status')!='APPROVED_FOR_ONE_MAC_PREFIX_CANARY' or p.get('writer')!='environment_tilt_design_writer' or not p.get('reviewer') or p['reviewer']==p['writer'] or p.get('query_count_limit')!=1 or p.get('network_allowed') is not True or p.get('fit_count')!=0:
        raise PermissionError('PREFIX_INDEPENDENT_REVIEW_REQUIRED')
    if sha(p['receipt_path'])!=p['receipt_sha256']:raise PermissionError('REVIEW_RECEIPT_DRIFT')


def run(c,adapter,*,fixture=False,session_factory=None):
    if fixture:
        base.fixture_guard(c)
        root=Path(c['fixture_root']).resolve()
        for key in ['inventory','allcategory']:
            if root not in Path(c[key]['path']).resolve().parents:raise PermissionError('FIXTURE_INPUT_ESCAPE')
    else:validate_activation(c)
    output=Path(c['run_dir']);output.mkdir(exist_ok=False)
    with (output/'owner.lock').open('x') as handle:
        handle.write(canonical(dict(pid=os.getpid(),execution_id=c['execution_id'])));handle.flush();os.fsync(handle.fileno())
    receipt=output/'RECEIPT.json';base.write_json(receipt,dict(status='PREPARED_NOT_REGISTERED'))
    life,budget=PrefixLifecycle(adapter,c),base.Budget(output,c['caps'])
    query=dict(query_calls=0,connect_calls=0,auto_retry=False,wall_limit=30)
    try:
        life.start(receipt)
        with budget:
            if base.available_memory()<c['caps']['min_memory_bytes'] or shutil.disk_usage(output).free<c['caps']['min_disk_bytes']:raise PrefixStop('DEFERRED_RESOURCE')
            os.environ.update({k:'4' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']});sys.dont_write_bytecode=True
            c['arrow_pools']=base.configure_arrow_pools(4)
            import pandas as pd
            import pyarrow.parquet as pq
            with base.authenticated(life,c['inventory'],budget) as f:bindings=inventory_bindings(f)
            with base.authenticated(life,c['allcategory'],budget) as f:
                artifact=[json.loads(line) for line in f if line.strip()]
            preferred={code:normalize_allcategory([e for e in artifact if e['symbol']==code],POLICY['cutoff']) for code in PREFERRED}
            with base.authenticated(life,c['actions'],budget) as f:cache=pd.read_parquet(f)
            cache['_date_int']=pd.to_datetime(cache._xdxr_trade_date.astype(str).str.replace(r'\.0$','',regex=True).str[:10],errors='raise').dt.strftime('%Y%m%d').astype('int64')
            cached={code:frame.assign(category=1).to_dict('records') for code,frame in cache.groupby('ts_code')}
            eligible_codes=set(PREFERRED)|{code for code,events in cached.items() if code.endswith(('.SH','.SZ')) and any(nonunit(e) for e in events)}
            universe={}
            with base.authenticated(life,c['qfq'],budget) as f:
                for batch in pq.ParquetFile(f).iter_batches(columns=['trade_date','ts_code'],batch_size=65536):
                    budget.check();frame=batch.to_pandas();frame=frame.loc[frame.ts_code.isin(eligible_codes)]
                    for code,part in frame.groupby('ts_code'):universe.setdefault(code,[]).extend(part.trade_date.astype(str).str.replace('-','',regex=False).str[:8].astype(int).tolist())
            chosen,payload,evidence=choose_from_archive(life,c,budget,bindings,universe,preferred,cached)
            base.write_json(output/'SELECTION.json',dict(chosen=chosen,input_identity=evidence))
            code=chosen['symbol'];path=output/'selected_raw.day'
            with path.open('xb') as f:f.write(payload)
            namespace=base.pure_functions(c);raw=namespace['read_day_tail'](code,path,tail_records=len(payload)//32)
            with base.authenticated(life,c['qfq'],budget) as f:reference=pd.read_parquet(f,columns=['trade_date','ts_code',*base.FIELDS],filters=[('ts_code','==',code)])
            if code in c['allcategory_covered_symbols']:
                events=preferred[code];event_source='BOUND_ALLCATEGORY_ARTIFACT'
            else:
                events=query_events(life,c,budget,code,query,session_factory);event_source='SINGLE_ORIGINAL_MAC_TDX_QUERY'
            base.write_json(output/'SELECTED_EVENTS.json',dict(symbol=code,source=event_source,events=events,query=query))
            selected=chosen['selected_event']
            same=[e for e in events if e['_date_int']==selected['_date_int'] and e['category']==1]
            selection_event_parity=(len(same)==1 and all(affine.decimal_scalar(same[0][k])==affine.decimal_scalar(selected[k]) for k in base.ACTION_FIELDS))
            result,candidate=analyze(raw,reference,events,chosen)
            if not selection_event_parity:
                result['prefix_block_reasons'].append('SELECTION_VS_ALLCATEGORY_EVENT_MISMATCH');result['decision']='PREFIX_BLOCKED'
            result.update(selection=chosen,input_identity=evidence,event_source=event_source,query=query,
                selection_event_allcategory_parity=selection_event_parity,
                ancillary_source_binding=c['ancillary_source_binding'])
            result['prefix_artifact']=emit_prefix(output,result,candidate,raw,life)
            budget.check();result.update(status='COMPLETED',wall_sec=time.monotonic()-budget.start_time,sampled_peak_rss_bytes=budget.peak,arrow_pools=c['arrow_pools'])
        base.write_json(receipt,result);life.finish(receipt,result['decision'],dict(symbols=1,query_calls=query['query_calls'],prefix_rows=result['prefix_artifact']['rows'] if result['prefix_artifact'] else 0));return result
    except BaseException as exc:
        decision='INTERRUPTED' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'QUERY_WALL_STOP' if isinstance(exc,QueryWallStop) else 'DEFERRED_RESOURCE' if 'DEFERRED_RESOURCE' in str(exc) else 'FAILED_'+type(exc).__name__
        try:base.write_json(receipt,dict(status=decision,error_type=type(exc).__name__,error=str(exc),query=query,wall_sec=time.monotonic()-budget.start_time,sampled_peak_rss_bytes=budget.peak,arrow_pools=c.get('arrow_pools'),**FLAGS))
        finally:
            if life.started and not life.finished:life.finish(receipt,decision,dict(query_calls=query['query_calls']))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True,type=Path);parser.add_argument('--execute',action='store_true');args=parser.parse_args();c=load_config(args.config)
    if not args.execute:print(canonical(dict(status='SOFTWARE_METADATA_ONLY_NO_SCIENCE',code_sha256=c['code_composite_sha256'])));return
    if c.get('fixture_root'):raise PermissionError('CLI_FIXTURE_MODE_PROHIBITED')
    run(c,load_adapter(c))


if __name__=='__main__':main()
