"""Synthetic IO/lifecycle contracts only; no real registry or market files."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import environment_tilt_early_inputs_runtime as runtime


def doc(path, value):
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=runtime.support.sha(path))


class Session:
    def __init__(self):
        self.reads = 0
        self.checks = 0
    def require_active(self):
        self.reads += 1
    def check_resources(self):
        self.checks += 1


def manifest(tmp_path):
    cells, evidence = {}, {}
    for cell_id in runtime.ORIGINAL_CELLS:
        _, h, fold = cell_id.split(":")
        sidecar = dict(status="COMPLETED", artifact_sha256="NO_MODEL_OPEN",
            identity=dict(arm="M0", fold=fold, horizon=int(h[1:]), data_identity={}))
        ref = doc(tmp_path / (cell_id.replace(":", "_") + ".joblib.json"), sidecar)
        cells[cell_id] = sidecar
        evidence[cell_id] = dict(kind="ORIGINAL_SIDECAR", sidecar_ref=ref)
    fit = dict(status="SINGLE_CELL_SCORES_COMPLETE_NO_PERFORMANCE_EVALUATION",
        policy_sha256="SYNTHETIC", checkpoint_ref=dict(path="MUST_NOT_OPEN.joblib", sha256="model"),
        metrics=dict(fit_calls=1, roundtrip_exact=True), proof=dict(data_identity={}))
    config = dict(policy_sha256="SYNTHETIC", recipe_sources=dict(model_selection=dict(sha256="recipe")))
    fit_ref = doc(tmp_path / "fit.json", fit)
    review = dict(verdict="PASS_SINGLE_CELL_FIT_RECEIPT_AND_SCORE_IDENTITY",
                  fit_receipt_sha256=fit_ref["sha256"])
    review_ref = doc(tmp_path / "review.json", review)
    config_ref = doc(tmp_path / "fit_config.json", config)
    cells["M0:H2:2023H2"] = runtime.normalized_m24_sidecar(fit, config, review)
    evidence["M0:H2:2023H2"] = dict(kind="M24_ACCEPTED_RECEIPT", fit_ref=fit_ref,
                                      config_ref=config_ref, review_ref=review_ref)
    value = dict(status="BOUND_COMPLETE_ACCEPTED_M0_INPUT_IDENTITIES", cells=cells,
        evidence_by_cell=evidence, original_cell_ids=list(runtime.ORIGINAL_CELLS),
        additional_accepted_cell_ids=["M0:H2:2023H2"])
    return doc(tmp_path / "manifest.json", value), value


def test_complete18plusM24_manifest_reads_only_sidecars_and_accepted_receipts(tmp_path, monkeypatch):
    ref, value = manifest(tmp_path)
    read = runtime.checked_json
    opened = []
    def metadata_only(binding):
        opened.append(binding["path"])
        assert binding["path"].endswith(".json")
        return read(binding)
    monkeypatch.setattr(runtime, "checked_json", metadata_only)
    assert runtime.load_model_manifest(ref, list(value["cells"])) == value["cells"]
    assert len(value["cells"]) == 19
    assert not any(Path(p).suffix == ".joblib" for p in opened)


@pytest.mark.parametrize("kind", ["missing", "extra", "omitted_M24", "wrong_sidecar", "wrong_review"])
def test_manifest_omissions_and_metadata_drift_fail_closed(tmp_path, kind):
    ref, value = manifest(tmp_path)
    if kind == "missing":
        value["cells"].pop(runtime.ORIGINAL_CELLS[0])
    elif kind == "extra":
        value["cells"]["M0:H1:2023H2"] = {}
    elif kind == "omitted_M24":
        value["additional_accepted_cell_ids"] = []
    elif kind == "wrong_sidecar":
        value["cells"][runtime.ORIGINAL_CELLS[0]]["identity"]["horizon"] = 5
    else:
        e = value["evidence_by_cell"]["M0:H2:2023H2"]
        e["review_ref"] = doc(tmp_path / "badreview.json", dict(verdict="PASS"))
    ref = doc(Path(ref["path"]), value)
    with pytest.raises(runtime.Blocked):
        runtime.load_model_manifest(ref, list(value["cells"]))


def fast_fixture(tmp_path):
    price = tmp_path / "old.parquet"
    price.write_bytes(b"synthetic-not-real-price-payload")
    price_ref = dict(path=str(price), sha256=runtime.support.sha(price))
    original = dict(sources=dict(price_panel={**price_ref, "column_map": runtime.PRICE_MAP,
        "units": dict(amount_input="10000_CNY", volume="shares")}, industry_classification={"member": "synthetic"}))
    old_ref = doc(tmp_path / "old_binding.json", original)
    stat = runtime.support.stat_identity(price)
    s1_ref = doc(tmp_path / "s1.json", dict(old_binding=old_ref,
        panel_stat=dict(bytes=stat["bytes"], mtime_ns=stat["mtime_ns"], sha256=price_ref["sha256"])))
    return dict(inputs=dict(old_qfq=price_ref), runtime=dict(io=dict(
        original_binding=old_ref, s1_binding=s1_ref))), price


def test_originalS1_fastbind_no_payload_rehash_and_stat_drift_no_fallback(tmp_path, monkeypatch):
    config, price = fast_fixture(tmp_path)
    original_sha = runtime.support.sha
    def no_payload_hash(path):
        assert Path(path) != price
        return original_sha(path)
    monkeypatch.setattr(runtime.support, "sha", no_payload_hash)
    binding = runtime.authenticate_unchanged(config, "old_qfq", Session())
    assert binding["mode"].startswith("ACCEPTED_UNCHANGED")
    price.write_bytes(b"changed")
    with pytest.raises(runtime.Blocked, match="STAT_DRIFT"):
        runtime.authenticate_unchanged(config, "old_qfq", Session())


def test_no_active_session_means_no_unit_metadata_or_payload_admission(monkeypatch):
    class Denied:
        def require_active(self):
            raise PermissionError("NO_START")
    monkeypatch.setattr(runtime, "checked_json", lambda *_: pytest.fail("read before START"))
    with pytest.raises(PermissionError):
        runtime.OriginalLoader().open_once({}, {}, Denied())


def test_unresolved_units_reject_and_failed_loader_cannot_reenter(tmp_path):
    ref = doc(tmp_path / "unit.json", dict(decision="SEMANTIC_CONFLICT_UNRESOLVED"))
    loader = runtime.OriginalLoader()
    with pytest.raises(runtime.Blocked, match="UNIT_BRIDGE"):
        loader.open_once(dict(runtime=dict(io=dict(unit_ruling=ref))), {}, Session())
    with pytest.raises(runtime.Blocked, match="NO_AUTOMATIC_REENTRY"):
        loader.open_once({}, {}, Session())


def loader_fixture(tmp_path):
    cfg, _ = fast_fixture(tmp_path)
    raw = pd.DataFrame(dict(trade_date=[20210104], ts_code=["000001.SZ"], open=[10.], high=[11.],
        low=[9.], close=[10.], vol=[100.], amount=[7.]))
    calendar = pd.DataFrame(dict(date=pd.to_datetime(["2021-01-04"]), calendar_index=[0]))
    m0 = pd.DataFrame(dict(date=pd.to_datetime(["2021-01-04"]), code=["000001.SZ"],
                          **{f: [1.] for f in runtime.producer.FEATURES}))
    cache = {}
    for name, frame in [("m0", m0), ("calendar", calendar), ("old_qfq", raw)]:
        p = tmp_path / (name + ".parquet"); frame.to_parquet(p, index=False)
        cache[name] = dict(path=str(p), sha256=runtime.support.sha(p))
    original = runtime.checked_json(cfg["runtime"]["io"]["original_binding"])
    original["sources"]["price_panel"].update(cache["old_qfq"])
    old_ref = doc(tmp_path / "old_binding.json", original)
    price_stat = runtime.support.stat_identity(cache["old_qfq"]["path"])
    s1 = doc(tmp_path / "s1.json", dict(old_binding=old_ref,
        panel_stat={**price_stat, "sha256": cache["old_qfq"]["sha256"]}))
    prior = doc(tmp_path / "prior.json", dict(cache_files={k: cache[k] for k in ("m0", "calendar")}))
    pre = doc(tmp_path / "pre.json", dict(status="PREFLIGHT_READY_FOR_INDEPENDENT_TIME_USE_REVIEW",
        cache_stats={k: runtime.support.stat_identity(cache[k]["path"]) for k in ("m0", "calendar")}))
    review = doc(tmp_path / "pre_review.json", dict(verdict="PASS_NARROW_NEW_TIME_USE",
        preflight_receipt_sha256=pre["sha256"]))
    prefix = tmp_path / "prefix.csv"; raw.assign(trade_date=20200102).to_csv(prefix, index=False)
    pref = dict(path=str(prefix), sha256=runtime.support.sha(prefix), rows=1)
    parts = doc(tmp_path / "parts.json", dict(schema="M26_FULL_AVAILABLE_PREFIX_PARTS_V1",
        scope="FULL_AVAILABLE_PRE_OLD_HISTORY", parts=[dict(symbol="000001.SZ", artifact=pref)]))
    hist = doc(tmp_path / "hist.json", dict(status="SYNTHETIC_HISTORY"))
    model_ref, model = manifest(tmp_path)
    index = dict(path=str(tmp_path / "index.csv"), sha256="synthetic-index-no-open")
    io = dict(original_binding=old_ref, s1_binding=s1, accepted_cache_preflight=pre,
        accepted_cache_config=prior, accepted_cache_review=review,
        unit_ruling=doc(tmp_path / "units.json", dict(decision="PASS_FOR_FROZEN_M26_UNIT_BRIDGE",
                                                  old_multiplier=10000, prefix_multiplier=1)),
        index_source=index)
    cfg.update(runtime=dict(io=io), inputs=dict(old_qfq=cache["old_qfq"], old_calendar=cache["calendar"],
        old_m0=cache["m0"], prefix=parts, industry=old_ref, index=index, history_proof=hist,
        model_manifest=model_ref), caps=dict(max_input_rows=10), sources=dict(v192=dict(path="v192", sha256="a"),
        v193=dict(path="v193", sha256="b")), expected_cell_ids=list(model["cells"]))
    called = []
    recipe = SimpleNamespace(read_bound_archive_member=lambda s: b"code,industry_code\n000001.SZ,CAPCO:C\n",
        normalize_industry_columns=lambda f: f,
        load_bound_index_series=lambda source, panel_dates: called.append(panel_dates) or pd.Series([1.]))
    modules = dict(recipe=recipe, formula=SimpleNamespace(exact_formula_source=lambda *args: ("original-text", "sha")),
                   engine=object(), fast_builder=object())
    return cfg, modules, called, prefix


def test_loader_reuses_original_reader_maps_and_index_dates_and_limits_arrow_before_reads(tmp_path, monkeypatch):
    cfg, modules, calls, _ = loader_fixture(tmp_path)
    monkeypatch.setattr(runtime, "load_original_modules", lambda _: modules)
    import pyarrow as pa
    pa.set_cpu_count(8); pa.set_io_thread_count(8)
    read = pd.read_parquet
    def guarded_read(*args, **kwargs):
        assert (pa.cpu_count(), pa.io_thread_count()) == (4, 4)
        return read(*args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    loader = runtime.OriginalLoader(); session = Session()
    loaded = loader.open_once(cfg, dict(scientific_identity="identity"), session)
    assert loaded["old_map"] == loaded["prefix_map"] == runtime.PRICE_MAP
    assert loaded["old_raw"].amount.iloc[0] == loaded["prefix_raw"].amount.iloc[0] == 7.
    assert loaded["source_modules"] == modules and len(loader.admissions) == 4
    required = {pd.Timestamp("2020-01-02")}
    loaded["index_provider"](required)
    assert calls == [required]
    with pytest.raises(runtime.Blocked, match="NO_REENTRY"):
        loaded["index_provider"](required)


def test_changed_prefix_hash_mismatch_never_falls_back_or_changes_old_data(tmp_path, monkeypatch):
    cfg, modules, _, prefix = loader_fixture(tmp_path)
    monkeypatch.setattr(runtime, "load_original_modules", lambda _: modules)
    prefix.write_text("changed")
    with pytest.raises(runtime.Blocked, match="PREFIX_PART_SHA"):
        runtime.OriginalLoader().open_once(cfg, dict(scientific_identity="identity"), Session())


def candidate(status="INPUT_VERSION_PARITY_PASS"):
    f = pd.DataFrame(dict(date=pd.to_datetime(["2020-01-02"]), code=["X"], value=[1.]))
    return dict(scientific_identity="synthetic", composition={}, parity=dict(status=status),
        candidate=dict(panel=f, calendar=f, m0=f, early_m0=f, provenance=dict(quarantine=f)))


def test_failed_parity_keeps_only_metadata_and_never_admits_passport(tmp_path):
    cfg = dict(output_root=str(tmp_path), runtime=dict(max_output_bytes=1000000))
    out = runtime.publish_candidate(candidate("INPUT_VERSION_PARITY_FAILED"), cfg)
    assert out["status"] == "INPUT_VERSION_PARITY_FAILED_NO_PUBLISH"
    assert not list(tmp_path.glob("*.parquet")) and not out["active_passport_admitted"]


def test_pass_publishes_only_separate_candidate_and_refuses_overwrite(tmp_path):
    cfg = dict(output_root=str(tmp_path), runtime=dict(max_output_bytes=1000000))
    out = runtime.publish_candidate(candidate(), cfg)
    assert len(list(tmp_path.glob("*.parquet"))) == 4
    assert out["status"] == "CANDIDATE_ONLY_PENDING_INDEPENDENT_REVIEW"
    with pytest.raises(runtime.Blocked, match="NO_OVERWRITE"):
        runtime.publish_candidate(candidate(), cfg)


class Guard:
    def __init__(self, trail, terminal_failure=False):
        self.trail, self.terminal_failure = trail, terminal_failure
        self.experiment_id = "synthetic"
        self.rows = []
        self.backend = SimpleNamespace(list_events=lambda _: self.rows)
    def claim_no_fit(self, **kwargs):
        self.trail.append("START")
        self.rows.append(dict(event_type="RUN_STARTED", **kwargs))
        return SimpleNamespace(**kwargs)
    def append_no_fit_terminal(self, claim, **kwargs):
        self.trail.append("TERMINAL")
        if self.terminal_failure:
            raise RuntimeError("synthetic terminal failure")
        self.rows.append(dict(event_type="TERMINAL", execution_id=claim.execution_id))


def run_fixture(tmp_path, monkeypatch):
    trail = []
    review = doc(tmp_path / "runtime_review.json", dict(status="PASS"))
    contract = doc(tmp_path / "contract.json", dict(status="FROZEN"))
    cfg = dict(output_root=str(tmp_path / "M26_synthetic"), sources=dict(loader=dict(
        path=runtime.__file__, sha256=runtime.support.sha(runtime.__file__))),
        runtime=dict(state="FROZEN", max_output_bytes=1000000, output_wall_seconds=10,
                     review_receipt=review, contract=contract))
    cp = tmp_path / "config.json"; cp.write_text(json.dumps(cfg))
    activation = dict(execution_id="SYNTHETIC", config_sha256=runtime.support.sha(cp),
                      review_receipt_sha256=review["sha256"])
    ap = tmp_path / "activation.json"; ap.write_text(json.dumps(activation))
    monkeypatch.setattr(runtime.producer, "require_activation", lambda *_: "synthetic_identity")
    monkeypatch.setattr(runtime.producer.NativeBudget, "check", lambda *_: {})
    return cp, ap, cfg, trail


@pytest.mark.parametrize("success", [True, False])
def test_official_claim_precedes_any_payload_and_terminal_covers_success_failure(tmp_path, monkeypatch, success):
    cp, ap, cfg, trail = run_fixture(tmp_path, monkeypatch)
    guard = Guard(trail)
    monkeypatch.setattr(runtime, "make_official_guard", lambda *_: guard)
    def prepare(config, activation, session, loader):
        session.require_active(); assert not session.output_exists()
        trail.append("PAYLOAD")
        if not success:
            raise RuntimeError("synthetic IO failure")
        loader.admissions = []; loader.arrow_pools = dict(cpu=4, io=4)
        return candidate()
    monkeypatch.setattr(runtime.producer, "prepare_registered", prepare)
    if success:
        result = runtime.run_registered(cp, ap)
        assert result["status"] == "INPUT_VERSION_PARITY_PASS"
    else:
        with pytest.raises(RuntimeError, match="synthetic IO"):
            runtime.run_registered(cp, ap)
    assert trail == ["START", "PAYLOAD", "TERMINAL"]


def test_wholeconfig_io_drift_rejects_before_output_claim_or_payload(tmp_path, monkeypatch):
    cp, ap, cfg, trail = run_fixture(tmp_path, monkeypatch)
    cfg["runtime"]["io"] = dict(changed=True); cp.write_text(json.dumps(cfg))
    monkeypatch.setattr(runtime, "make_official_guard", lambda *_: pytest.fail("must not claim"))
    with pytest.raises(runtime.Blocked, match="WHOLE_CONFIG"):
        runtime.run_registered(cp, ap)
    assert not Path(cfg["output_root"]).exists()


def test_terminal_failure_is_preserved_without_second_append(tmp_path, monkeypatch):
    cp, ap, cfg, trail = run_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime, "make_official_guard", lambda *_: Guard(trail, terminal_failure=True))
    monkeypatch.setattr(runtime.producer, "prepare_registered", lambda *_: (_ for _ in ()).throw(ValueError("input")))
    with pytest.raises(RuntimeError, match="terminal failure"):
        runtime.run_registered(cp, ap)
    receipt = json.loads((Path(cfg["output_root"]) / "RECEIPT.json").read_text())
    assert trail == ["START", "TERMINAL"] and receipt["terminal_status"] == "CANNOT_CONFIRM"


def test_DRAFT_realentry_rejects_before_owner_registry_or_input(tmp_path, monkeypatch):
    cp = tmp_path / "draft.json"; cp.write_text(json.dumps(dict(state="DRAFT")))
    ap = tmp_path / "activation.json"; ap.write_text("{}")
    monkeypatch.setattr(runtime, "make_official_guard", lambda *_: pytest.fail("no registry"))
    with pytest.raises(runtime.producer.Blocked, match="DRAFT"):
        runtime.run_registered(cp, ap)
