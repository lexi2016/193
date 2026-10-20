"""Synthetic-only tests. No real payload, root registry, network or SDK access."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import signal
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import environment_tilt_mac_recovery as m
import environment_tilt_mac_registry as registry

META_REL = Path("research/a_share_model_selection_v1/batches/environment_tilt_execution_20261020/M5_SOURCE_HASH_REFS.json")
META = (Path(os.environ["TILT_MAC_SOURCE_REFS"]) if os.environ.get("TILT_MAC_SOURCE_REFS") else
        next((p / META_REL for p in Path(__file__).resolve().parents if (p / META_REL).is_file()), None))
if META is None:
    raise RuntimeError("Bound local original sources required; set TILT_MAC_SOURCE_REFS to the metadata receipt")
PROJECT = META.resolve().parents[4]
GUARD = PROJECT / "workspace/hkconnect_price_semantics_v2_20260919/zero_label_parity"
SOURCE_REFS = json.loads(META.read_text())["sources"]
DATA_SOURCE = Path(next(x["path"] for x in SOURCE_REFS if x["sha256_current_metadata_or_source"] == "901d4802c784decda9d43e757b367496d1171e5e702485a29f4c66fad3c7ae2e"))
HISTORY_SOURCE = Path(next(x["path"] for x in SOURCE_REFS if x["sha256_current_metadata_or_source"] == "e8071ebea55783312d0b1ae5946a9e5f9a5f9c183f19e85710613b9c381ee003"))
SOURCE_IDENTITIES = {
    "data_layer": dict(path=str(DATA_SOURCE), sha256="901d4802c784decda9d43e757b367496d1171e5e702485a29f4c66fad3c7ae2e"),
    "history": dict(path=str(HISTORY_SOURCE), sha256="e8071ebea55783312d0b1ae5946a9e5f9a5f9c183f19e85710613b9c381ee003")}


@pytest.fixture
def namespace():
    return m.pure_functions(dict(pure_sources=SOURCE_IDENTITIES))


def frame(n=35, *, dates=None, code=m.CODES[0]):
    dates = dates if dates is not None else pd.bdate_range("2021-05-06", periods=n).strftime("%Y%m%d").astype(int)
    return pd.DataFrame(dict(trade_date=dates, ts_code=code, open=10., high=12., low=8.,
                             close=10., vol=1000., amount=10000.))


def actions(date=None, **kwargs):
    if date is None:
        return pd.DataFrame(columns=["_date_int", *m.ACTION_FIELDS])
    return pd.DataFrame([dict(_date_int=date, xdxr_fenhong=50., xdxr_peigujia=0.,
                              xdxr_songzhuangu=0., xdxr_peigu=0., **kwargs)])


def fixture_adapter(tmp_path, config):
    # The real append-compat validator requires the immutable full legacy prefix.
    # This fixture validator checks canonical bytes/chain on a temporary ledger only.
    validator = tmp_path / "fixture_validator.py"
    validator.write_text('''import hashlib,json
def validate_registry_bytes(raw,source=None):
    previous='0'*64
    for line in raw.decode().splitlines():
        event=json.loads(line)
        assert event['previous_event_sha256']==previous
        payload={k:v for k,v in event.items() if k!='event_sha256'}
        encoded=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
        assert hashlib.sha256(encoded.encode()).hexdigest()==event['event_sha256']
        assert json.dumps(event,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)==line
        previous=event['event_sha256']
''')
    paths = dict(registry_adapter=GUARD / "registry_adapter.py",
        execution_guard_core=GUARD / "execution_guard_core.py",
        registry_builder=PROJECT / "contracts/build_experiment_registry_backfill_20260821.py",
        registry_validator=validator)
    config["registry_dependencies"] = {k: dict(path=str(p), sha256=registry.sha(p)) for k, p in paths.items()}
    adapter = registry.load_adapter(config)
    seed = registry.Lifecycle(adapter, config).event("RUN_STARTED", "RUNNING", config["config_path"], {})
    seed.update(experiment_id="SYNTHETIC_SEED", execution_id="SYNTHETIC_SEED_0000", registry_event_id="SYNTHETIC_SEED")
    materialized = adapter.builder.materialize([seed])
    Path(config["registry_path"]).write_bytes(adapter.builder._registry_payload(materialized))
    return adapter


@pytest.fixture
def fixture_run(tmp_path, namespace):
    contract = tmp_path / "contract.json"
    contract.write_text("{}\n")
    config_file = tmp_path / "config.json"
    config_file.write_text("{}\n")
    source_zip = tmp_path / "synthetic.zip"
    members, qfqs = {}, []
    with zipfile.ZipFile(source_zip, "w") as archive:
        for code in m.CODES:
            raw = frame(code=code)
            payload = b"".join(struct.pack("<IIIIIfII", int(row.trade_date), 1000, 1200, 800, 1000,
                                             10000., 1000, 0) for row in raw.itertuples())
            name = code.replace(".", "_") + ".day"
            archive.writestr(name, payload)
            members[code] = dict(name=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            qfq = raw.copy()
            qfq[["open", "high", "low", "close"]] *= 2.
            qfqs.append(qfq)
    qfq_path, actions_path = tmp_path / "qfq.parquet", tmp_path / "actions.parquet"
    pd.concat(qfqs, ignore_index=True).to_parquet(qfq_path, index=False)
    pd.DataFrame(columns=["ts_code", "_xdxr_trade_date", *m.ACTION_FIELDS]).to_parquet(actions_path, index=False)
    c = dict(fixture_root=str(tmp_path), batch_root=str(tmp_path), run_dir=str(tmp_path / "M6_SYNTHETIC_01"),
        execution_id="MAC_SYNTHETIC_TEST_01", contract_path=str(contract), contract_sha256=registry.sha(contract),
        config_path=str(config_file), config_sha256=registry.sha(config_file), code_composite_sha256="a" * 64,
        registry_path=str(tmp_path / "registry.jsonl"), diagnostic_dates=dict(start="2020-01-02", end="2026-10-15"),
        caps=m.CAPS, pure_sources=SOURCE_IDENTITIES, members=members,
        archive=dict(path=str(source_zip), frozen_stat=m.file_identity(source_zip.stat()), recorded_sha256="f" * 64),
        qfq=dict(path=str(qfq_path), sha256=registry.sha(qfq_path), bytes=qfq_path.stat().st_size),
        actions=dict(path=str(actions_path), sha256=registry.sha(actions_path), bytes=actions_path.stat().st_size))
    return c, fixture_adapter(tmp_path, c)


def test_ast_selected_original_bodies_and_no_producer_main(namespace):
    assert "main" not in namespace and "load_tdx_xdxr_events" not in namespace
    assert set(["read_day_tail", "action_adjust", "apply_xdxr_pre_close"]).issubset(namespace)
    value, applied = namespace["apply_xdxr_pre_close"](actions(20210601), pd.Series([10.]))
    assert value.tolist() == [5.] and applied.tolist() == [True]


def test_anchor_selection_no_residual_search_and_disjoint_holdout():
    raw, reference = frame(), frame()
    reference[["open", "high", "low", "close"]] *= 9.37
    chosen = m.anchors(raw, reference)
    assert chosen["A"] == int(raw.trade_date.iloc[0])
    assert chosen["B"] == int(raw.trade_date.iloc[20])
    assert chosen["H"] == raw.trade_date.iloc[10:20].tolist()
    assert chosen["A"] not in chosen["H"] and chosen["B"] not in chosen["H"]
    reference.loc[reference.index[-1], "close"] = 999999.
    assert m.anchors(raw, reference) == chosen


def test_anchor_observed_session_distance_not_common_row_distance():
    raw, reference = frame(), frame().drop(index=[2, 4, 6, 8, 10])
    chosen = m.anchors(raw, reference)
    assert chosen["session_distance"] == 20
    assert chosen["B"] == int(raw.trade_date.iloc[20])
    assert len(chosen["H"]) == 10


def test_missing_holdout_does_not_select_alternate_anchor_or_symbol():
    assert m.anchors(frame(15), frame(15)) is None
    raw, reference = frame(), frame().iloc[[0, 20, 21, 22]].copy()
    assert m.anchors(raw, reference) is None


def test_backward_product_excludes_event_on_heldout_bar(namespace):
    raw = frame(4)
    dates = raw.trade_date.tolist()
    events = actions(dates[2])
    rebuilt = m.reconstruct(raw, events, dates[3], 2., namespace, dates[:3])
    assert rebuilt.close.tolist() == [10., 10., 20., 20.]
    assert rebuilt.vol.tolist() == raw.vol.tolist()
    assert rebuilt.amount.tolist() == raw.amount.tolist()


def test_two_actions_across_nontrading_interval_apply_sequentially(namespace):
    raw = frame(dates=[20210506, 20210507, 20210512, 20210513])
    events = pd.concat([actions(20210510), actions(20210511)], ignore_index=True)
    events.loc[:, "xdxr_fenhong"] = 20.  # 10 ->8 ->6, q=.6
    rebuilt = m.reconstruct(raw, events, 20210513, 1., namespace, [20210506, 20210507, 20210512])
    assert rebuilt.cached_action_count.tolist() == [0, 0, 2, 0]
    assert rebuilt.close.tolist() == [6., 6., 10., 10.]


@pytest.mark.parametrize("failure", ["duplicate", "null", "unsupported", "nonpositive"])
def test_invalid_action_event_stops(namespace, failure):
    raw = frame(4)
    dates = raw.trade_date.tolist()
    events = actions(dates[2])
    if failure == "duplicate": events = pd.concat([events, events], ignore_index=True)
    elif failure == "null": events.loc[0, "xdxr_peigu"] = np.nan
    elif failure == "unsupported": events["category"] = 11
    else: events.loc[0, "xdxr_fenhong"] = 1000.
    with pytest.raises(m.DiagnosticError):
        m.reconstruct(raw, events, dates[3], 1., namespace, dates[:3])


def test_outside_holdout_events_do_not_expand_diagnostic(namespace):
    raw = frame(35)
    events = actions(int(raw.trade_date.iloc[0]))
    events["category"] = 11
    rebuilt = m.reconstruct(raw, events, int(raw.trade_date.iloc[20]), 2., namespace,
                            raw.trade_date.iloc[10:20].tolist())
    assert len(rebuilt) == 11 and rebuilt.cached_action_count.sum() == 0


@pytest.mark.parametrize("mutation", ["duplicate", "inf", "malformed", "bad_date", "negative_volume"])
def test_source_structure_fail_closed(mutation):
    raw = frame(3)
    if mutation == "duplicate": raw.loc[1, "trade_date"] = raw.loc[0, "trade_date"]
    elif mutation == "inf": raw.loc[0, "open"] = np.inf
    elif mutation == "malformed": raw["amount"] = raw.amount.astype(object); raw.loc[0, "amount"] = "broken"
    elif mutation == "bad_date": raw.loc[0, "trade_date"] = 20210230
    else: raw.loc[0, "vol"] = -1
    with pytest.raises(m.DiagnosticError): m.normalize(raw, m.CODES[0])


def test_exact_null_mask_and_ulp_discrepancy_no_tolerance():
    first, second = frame(2), frame(2)
    first.loc[0, "amount"] = second.loc[0, "amount"] = np.nan
    assert m.exact_comparison(first, second)["amount"]["mismatches"] == 0
    second.loc[0, "amount"] = 10000.
    assert not m.exact_comparison(first, second)["amount"]["null_mask_exact"]
    second = first.copy()
    second.loc[0, "open"] = np.nextafter(10., np.inf)
    report = m.exact_comparison(first, second)["open"]
    assert report["mismatches"] == 1 and report["max_relative_to_float64_spacing"] == 1.
    assert report["discrepancies"][0]["reconstructed_hex"] != report["discrepancies"][0]["reference_hex"]


def test_diagnostic_events_and_zero_events_have_distinct_evidence(namespace):
    raws, references, events = {}, [], {}
    for i, code in enumerate(m.CODES):
        raw = frame(code=code)
        qfq = raw.copy()
        qfq[["open", "high", "low", "close"]] *= 2.
        if i == 0:
            b = int(raw.trade_date.iloc[20])
            qfq.loc[qfq.trade_date < b, ["open", "high", "low", "close"]] /= 2.
            events[code] = actions(b)
        raws[code] = raw
        references.append(qfq)
    report = m.diagnose(raws, pd.concat(references, ignore_index=True), events, namespace)
    assert report["decision"] == "EARLY_RETURN_UNRESOLVED"  # fixture deliberately starts2021
    assert report["symbols"][m.CODES[0]]["holdout_cached_events"] == 1
    assert report["symbols"][m.CODES[1]]["operator_evidence"] == "SCALE_AND_KEYS_ONLY_ZERO_EVENTS"
    assert report["training_ready"] is False
    assert all(v["pre_A_prefix"] == "NOT_GENERATED_NOT_VALIDATED" for v in report["symbols"].values())


def test_real_guard_fixture_run_registered_before_payload_and_terminal(fixture_run, monkeypatch):
    c, adapter = fixture_run
    monkeypatch.setattr(m, "available_memory", lambda: 2**40)
    before = Path(c["qfq"]["path"]).read_bytes()
    report = m.run(c, adapter, fixture=True)
    assert Path(c["qfq"]["path"]).read_bytes() == before
    assert report["input_identity"]["whole_zip_status"] == "REUSED_RECORDED_NOT_REHASHED"
    events = adapter.list_events(registry.EXPERIMENT)
    assert [e["event_type"] for e in events] == ["RUN_STARTED", "TERMINAL"]
    assert events[0]["truth_access"] == "NOT_READ"
    assert all(e["deterministic_metrics"]["real_fit_budget_debit"] == 0 for e in events)
    assert report["fits"] == 0 and report["network_calls"] == 0
    with pytest.raises(FileExistsError): m.run(c, adapter, fixture=True)


def test_unregistered_actual_input_open_is_denied(fixture_run, monkeypatch):
    c, adapter = fixture_run
    life = registry.Lifecycle(adapter, c)
    class NeverBudget:
        def check(self): raise AssertionError("not admitted")
    with pytest.raises(PermissionError):
        with m.authenticated(life, c["qfq"], NeverBudget()):
            pytest.fail("input opened")


def test_hash_mismatch_before_any_parquet_decode_terminalized(fixture_run, monkeypatch):
    c, adapter = fixture_run
    Path(c["qfq"]["path"]).write_bytes(b"changed fixture only")
    monkeypatch.setattr(m, "available_memory", lambda: 2**40)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("decoded before stream hash"))
    with pytest.raises(m.DiagnosticError, match="INPUT_IDENTITY_MISMATCH"):
        m.run(c, adapter, fixture=True)
    assert adapter.list_events(registry.EXPERIMENT)[-1]["event_type"] == "TERMINAL"


def test_exact_member_mismatch_and_no_substitution(fixture_run, monkeypatch):
    c, adapter = fixture_run
    c["members"][m.CODES[0]]["sha256"] = "0" * 64
    monkeypatch.setattr(m, "available_memory", lambda: 2**40)
    with pytest.raises(m.DiagnosticError, match="MEMBER_IDENTITY_MISMATCH"):
        m.run(c, adapter, fixture=True)
    assert adapter.list_events(registry.EXPERIMENT)[-1]["event_type"] == "TERMINAL"


@pytest.mark.parametrize("kind", ["defer", "interrupt"])
def test_resource_deferral_and_interrupt_have_terminal_receipts(fixture_run, monkeypatch, kind):
    c, adapter = fixture_run
    monkeypatch.setattr(m, "available_memory", lambda: 0 if kind == "defer" else 2**40)
    if kind == "interrupt":
        def interrupt(*args): raise KeyboardInterrupt()
        monkeypatch.setattr(m, "read_raw_members", interrupt)
    with pytest.raises(m.DiagnosticError if kind == "defer" else KeyboardInterrupt):
        m.run(c, adapter, fixture=True)
    terminal = adapter.list_events(registry.EXPERIMENT)[-1]
    assert terminal["decision"] == ("DEFERRED_RESOURCE" if kind == "defer" else "INTERRUPTED")
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_fixture_cannot_point_to_actual_inputs(fixture_run):
    c, adapter = fixture_run
    c["qfq"]["path"] = "/Users/lexi/Desktop/学习/因子工厂/forbidden.parquet"
    with pytest.raises(PermissionError, match="FIXTURE_DATA_ESCAPE"):
        m.run(c, adapter, fixture=True)


def test_duplicate_execution_claim_is_refused(fixture_run):
    c, adapter = fixture_run
    first = registry.Lifecycle(adapter, c)
    first.start(c["config_path"])
    with pytest.raises(Exception, match="already exists"):
        registry.Lifecycle(adapter, c).start(c["config_path"])
    first.finish(c["config_path"], "SYNTHETIC_DONE")


def test_budget_rss_timeout_payload_and_existing_timer(tmp_path, monkeypatch):
    caps = dict(m.CAPS)
    budget = m.Budget(tmp_path, caps)
    monkeypatch.setattr(m, "rss_bytes", lambda: caps["rss_bytes"] + 1)
    with pytest.raises(m.DiagnosticError, match="RSS"): budget.check()
    monkeypatch.setattr(m, "rss_bytes", lambda: 1)
    budget.start_time -= 601
    with pytest.raises(m.DiagnosticError, match="WALL"): budget.check()
    budget.start_time = time.monotonic()
    budget.peak = 0
    (tmp_path / "payload").write_bytes(b"12345")
    budget.caps = dict(caps, payload_bytes=4)
    with pytest.raises(m.DiagnosticError, match="PAYLOAD"): budget.check()


def test_frozen_policy_rejects_change_before_payload_access(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(codes=list(m.CODES), policy=dict(m.POLICY, holdout_count=11), caps=m.CAPS)))
    with pytest.raises(m.DiagnosticError, match="FROZEN_POLICY_CHANGED"): m.load_config(path)


def test_implementation_has_no_network_or_old_remote_dependency():
    tree = ast.parse(Path(m.__file__).read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not imported.intersection({"socket", "requests", "paramiko", "subprocess", "tqcenter",
                                     "environment_tilt_recovery", "environment_tilt_recovery_remote"})


def test_early_probe_does_not_infer_bj_listing_or_change_holdout(namespace):
    raw_by_code, references = {}, []
    for code in m.CODES:
        raw = frame(code=code)
        reference = raw.copy()
        reference[["open", "high", "low", "close"]] *= 2.
        if code in m.CODES[:4]:
            early = frame(dates=[20200102], code=code)
            raw = pd.concat([early, raw], ignore_index=True)
        raw_by_code[code] = raw
        references.append(reference)
    report = m.diagnose(raw_by_code, pd.concat(references, ignore_index=True), {}, namespace)
    assert report["decision"] == "BOUNDED_DIAGNOSTIC_COMPLETE"
    assert all(report["early_return_probes"][c] == "EARLY_STATUS_UNKNOWN_BJ" for c in m.CODES[4:])
    assert report["symbols"][m.CODES[0]]["anchors"]["session_distance"] == 20
    raw_by_code[m.CODES[0]].loc[0, ["open", "high", "low", "close"]] = np.nan
    report = m.diagnose(raw_by_code, pd.concat(references, ignore_index=True), {}, namespace)
    assert report["early_return_probes"][m.CODES[0]] == "EARLY_RETURN_UNRESOLVED"


def test_matching_infinity_is_never_exact_parity():
    left, right = frame(1), frame(1)
    left.loc[0, "close"] = right.loc[0, "close"] = np.inf
    with pytest.raises(m.DiagnosticError, match="NONFINITE"): m.exact_comparison(left, right)


def test_activation_requires_named_independent_writer_and_bound_receipt(tmp_path):
    receipt = tmp_path / "review.json"
    receipt.write_text('{"verdict":"PASS"}')
    path = tmp_path / "activation.json"
    c = dict(activation_path=path, config_sha256="a"*64, contract_sha256="b"*64,
             code_composite_sha256="c"*64)
    proof = dict(config_sha256=c["config_sha256"], contract_sha256=c["contract_sha256"],
        code_composite_sha256=c["code_composite_sha256"], status="APPROVED_FOR_MAC_DIAGNOSTIC_ONLY",
        reviewer="independent_code_reviewer", network_allowed=False, fit_count=0,
        receipt_path=str(receipt), receipt_sha256=registry.sha(receipt))
    path.write_text(json.dumps(proof))
    with pytest.raises(PermissionError): m.validate_activation(c)
    proof["writer"] = "environment_tilt_design_writer"
    proof["reviewer"] = proof["writer"]
    path.write_text(json.dumps(proof))
    with pytest.raises(PermissionError): m.validate_activation(c)
    proof["reviewer"] = "independent_code_reviewer"
    path.write_text(json.dumps(proof))
    m.validate_activation(c)
    receipt.write_text('{"changed":true}')
    with pytest.raises(PermissionError, match="REVIEW_RECEIPT_DRIFT"): m.validate_activation(c)


def test_mac_memory_probe_has_pointer_sized_native_output(monkeypatch):
    class Function:
        def __init__(self, call): self.call = call
        def __call__(self, *args): return self.call(*args)
    def page_size(host, out):
        assert m.ctypes.sizeof(out._obj) == m.ctypes.sizeof(m.ctypes.c_size_t)
        out._obj.value = 16384
        return 0
    def stats(host, flavor, counters, count):
        assert flavor == 2 and count._obj.value == 15
        counters[0], counters[2] = 10, 20
        return 0
    class Library:
        mach_host_self = Function(lambda: 42)
        host_page_size = Function(page_size)
        host_statistics = Function(stats)
    monkeypatch.setattr(m.sys, "platform", "darwin")
    monkeypatch.setattr(m.ctypes, "CDLL", lambda name: Library())
    assert m.available_memory() == 30 * 16384


def test_arrow_cpu_io_limits_precede_reads_and_are_measured_in_receipts(fixture_run, monkeypatch):
    import pyarrow as pa
    c, adapter = fixture_run
    old_cpu, old_io = pa.cpu_count(), pa.io_thread_count()
    pa.set_cpu_count(8)
    pa.set_io_thread_count(8)
    read_parquet = pd.read_parquet
    calls = []
    def guarded_read(*args, **kwargs):
        observed = dict(cpu=pa.cpu_count(), io=pa.io_thread_count())
        calls.append(observed)
        assert observed == dict(cpu=4, io=4)
        return read_parquet(*args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    monkeypatch.setattr(m, "available_memory", lambda: 2**40)
    try:
        result = m.run(c, adapter, fixture=True)
        assert len(calls) == 2
        assert result["arrow_pools"] == dict(cpu=4, io=4)
        terminal = adapter.list_events(registry.EXPERIMENT)[-1]
        assert terminal["runtime"]["arrow_pools"] == result["arrow_pools"]
        assert terminal["runtime"]["active_concurrency_limit"] == 4
        assert "threads" not in terminal["runtime"]
    finally:
        pa.set_cpu_count(old_cpu)
        pa.set_io_thread_count(old_io)
