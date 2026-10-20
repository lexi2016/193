"""Bounded Mac-only raw/actions/QFQ diagnostic; default CLI reads metadata only.

No network client, model, old remote adapter, or producer main is imported.
"""
from __future__ import annotations

import __future__
import argparse
import ast
import contextlib
import ctypes
import hashlib
import json
import math
import os
import re
import resource
import shutil
import signal
import struct
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from environment_tilt_mac_registry import FLAGS, Lifecycle, canonical, load_adapter, sha

CODES = ("600000.SH", "600004.SH", "000001.SZ", "000002.SZ", "920000.BJ", "920001.BJ")
FIELDS = ("open", "high", "low", "close", "vol", "amount")
ACTION_FIELDS = ("xdxr_fenhong", "xdxr_peigujia", "xdxr_songzhuangu", "xdxr_peigu")
POLICY = dict(anchor_B_min_sessions=20, holdout_count=10, rtol=0, atol=0,
              formula="ORIGINAL_ACTION_ADJUST_D_OVER_C_BACKWARD_PRODUCT_V1",
              event_boundary="t<event_date<=anchor", no_event_status="CACHE_COVERAGE_UNKNOWN",
              early_probe_start=20200102, early_probe_end=20200115)
CAPS = dict(total_wall_sec=600, rss_bytes=2**31, payload_bytes=2**27,
            min_memory_bytes=2**33, min_disk_bytes=2**31, threads=4)
CODE_FILES = ("environment_tilt_mac_recovery.py", "environment_tilt_mac_registry.py",
              "tests/test_environment_tilt_mac_recovery.py")
BASE_SHA = "c30f847cb0929a398ccfe93b4bc29e910a52e0052fff2c29cbf4bf789cd56c07"
ZIP_SHA = "a56c962378a5d3dd86a90186cc830ee6c8fd968557ce6eddeb310d203c5ff3c0"
ACTION_SHA = "462251f7f12ff876f304d621d771f868df6660254d8bb510ad60e1921482c84f"


class DiagnosticError(RuntimeError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    """Only this new owned run's receipt may be atomically refreshed."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def file_identity(stat):
    return dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def available_memory():
    if sys.platform != "darwin":
        values = Path("/proc/meminfo").read_text()
        return int(re.search(r"MemAvailable:\s+(\d+)", values).group(1)) * 1024
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    lib.mach_host_self.restype = ctypes.c_uint32
    host = lib.mach_host_self()
    # vm_size_t is pointer-sized on64-bit Darwin; never hand native code a4-byte buffer.
    size = ctypes.c_size_t()
    lib.host_page_size.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_size_t)]
    lib.host_page_size.restype = ctypes.c_int
    lib.host_statistics.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_uint32)]
    lib.host_statistics.restype = ctypes.c_int
    counters = (ctypes.c_uint32 * 15)()
    count = ctypes.c_uint32(15)
    if lib.host_page_size(host, ctypes.byref(size)) or lib.host_statistics(host, 2, counters, ctypes.byref(count)):
        raise DiagnosticError("AVAILABLE_MEMORY_UNOBSERVABLE")
    return int(size.value) * (int(counters[0]) + int(counters[2]))


def configure_arrow_pools(limit):
    """Bound Arrow's active CPU and I/O pools before any parquet read."""
    import pyarrow as pa
    pa.set_cpu_count(limit)
    pa.set_io_thread_count(limit)
    measured = dict(cpu=pa.cpu_count(), io=pa.io_thread_count())
    if any(not 1 <= count <= limit for count in measured.values()):
        raise DiagnosticError("ARROW_ACTIVE_POOL_LIMIT")
    return measured


class Budget:
    """One-process sampled guard; native I/O may deliver signals on return."""
    def __init__(self, root, caps):
        self.root, self.caps = Path(root), caps
        self.start_time = time.monotonic()
        self.peak = 0

    def check(self, *_):
        self.peak = max(self.peak, rss_bytes())
        if time.monotonic() - self.start_time > self.caps["total_wall_sec"]:
            raise DiagnosticError("TOTAL_WALL_LIMIT")
        if self.peak > self.caps["rss_bytes"]:
            raise DiagnosticError("OWNER_RSS_LIMIT")
        size = 0
        for p in self.root.rglob("*"):
            if p.is_symlink():
                raise DiagnosticError("OUTPUT_SYMLINK")
            if p.is_file():
                size += p.stat().st_size
        if size > self.caps["payload_bytes"]:
            raise DiagnosticError("OUTPUT_PAYLOAD_LIMIT")

    def __enter__(self):
        if any(signal.getitimer(signal.ITIMER_REAL)):
            raise DiagnosticError("PREEXISTING_PROCESS_TIMER")
        self.check()
        self.previous = signal.signal(signal.SIGALRM, self.check)
        self.previous_timer = signal.setitimer(signal.ITIMER_REAL, .25, .25)
        return self

    def __exit__(self, *_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous)
        if self.previous_timer[0] or self.previous_timer[1]:
            signal.setitimer(signal.ITIMER_REAL, *self.previous_timer)


def load_config(path):
    """Validate only source/JSON identities, without opening scientific inputs."""
    path = Path(path).resolve()
    c = read_json(path)
    c.update(config_path=str(path), config_sha256=sha(path))
    if c["codes"] != list(CODES) or c["policy"] != POLICY or c["caps"] != CAPS:
        raise DiagnosticError("FROZEN_POLICY_CHANGED")
    if c.get("network_allowed") is not False or c.get("fit_count") != 0:
        raise DiagnosticError("DIAGNOSTIC_BOUNDARY_CHANGED")
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,120}", c["execution_id"]):
        raise DiagnosticError("INVALID_EXECUTION_ID")
    root, output = Path(c["batch_root"]).resolve(), Path(c["run_dir"]).resolve()
    if output.parent != root or not output.name.startswith("M6_"):
        raise DiagnosticError("OUTPUT_OUTSIDE_SCOPE")
    identity = read_json(c["identity_path"])
    if sha(c["identity_path"]) != c["identity_sha256"]:
        raise DiagnosticError("IDENTITY_MANIFEST_DRIFT")
    if set(identity["code_by_file"]) != set(CODE_FILES):
        raise DiagnosticError("EXACT_THREE_SOURCE_FILES_REQUIRED")
    for name, expected in identity["code_by_file"].items():
        if sha(Path(__file__).resolve().parent / name) != expected:
            raise DiagnosticError("CODE_DRIFT:" + name)
    c["code_composite_sha256"] = hashlib.sha256(canonical(identity["code_by_file"]).encode()).hexdigest()
    if sha(c["contract_path"]) != c["contract_sha256"]:
        raise DiagnosticError("CONTRACT_DRIFT")
    for item in c["pure_sources"].values():
        if sha(item["path"]) != item["sha256"]:
            raise DiagnosticError("ORIGINAL_PURE_SOURCE_DRIFT")
    if (c["qfq"]["sha256"] != BASE_SHA or c["actions"]["sha256"] != ACTION_SHA
            or c["archive"]["recorded_sha256"] != ZIP_SHA):
        raise DiagnosticError("FROZEN_ORIGINAL_INPUT_CHANGED")
    metadata = read_json(c["member_binding_path"])
    if sha(c["member_binding_path"]) != c["member_binding_sha256"] or set(metadata["members"]) != set(CODES):
        raise DiagnosticError("MEMBER_BINDING_DRIFT")
    c["members"] = metadata["members"]
    return c


def validate_activation(c):
    proof = read_json(c["activation_path"])
    for key in ("config_sha256", "contract_sha256", "code_composite_sha256"):
        if proof.get(key) != c[key]:
            raise PermissionError("REVIEW_ACTIVATION_IDENTITY_MISMATCH")
    if (proof.get("status") != "APPROVED_FOR_MAC_DIAGNOSTIC_ONLY"
            or proof.get("writer") != "environment_tilt_design_writer"
            or not proof.get("reviewer") or proof["reviewer"] == proof["writer"]
            or proof.get("network_allowed") is not False or proof.get("fit_count") != 0):
        raise PermissionError("INDEPENDENT_MAC_DIAGNOSTIC_ACTIVATION_REQUIRED")
    if sha(proof["receipt_path"]) != proof["receipt_sha256"]:
        raise PermissionError("REVIEW_RECEIPT_DRIFT")


def fixture_guard(c):
    root = Path(c["fixture_root"]).resolve()
    if Path(tempfile.gettempdir()).resolve() not in root.parents:
        raise PermissionError("FIXTURE_ROOT_NOT_TEMPORARY")
    for name in ("qfq", "actions", "archive"):
        if root not in Path(c[name]["path"]).resolve().parents:
            raise PermissionError("FIXTURE_DATA_ESCAPE")
    for name in ("run_dir", "registry_path", "contract_path", "config_path"):
        if root not in Path(c[name]).resolve().parents:
            raise PermissionError("FIXTURE_OUTPUT_ESCAPE")


@contextlib.contextmanager
def authenticated(life, binding, budget):
    life.require()
    path = Path(binding["path"])
    if path.is_symlink():
        raise DiagnosticError("INPUT_SYMLINK")
    with path.open("rb") as handle:
        before = file_identity(os.fstat(handle.fileno()))
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            budget.check()
            digest.update(chunk)
        if (digest.hexdigest() != binding["sha256"] or before["bytes"] != binding["bytes"]
                or file_identity(path.stat()) != before or file_identity(os.fstat(handle.fileno())) != before):
            raise DiagnosticError("INPUT_IDENTITY_MISMATCH")
        life.require()
        handle.seek(0)
        yield handle
        if file_identity(path.stat()) != before or file_identity(os.fstat(handle.fileno())) != before:
            raise DiagnosticError("INPUT_CHANGED_DURING_READ")


def pure_functions(c):
    """Bind source SHA, then compile only the original selected pure bodies."""
    import numpy as np
    import pandas as pd
    namespace = dict(np=np, pd=pd, struct=struct, Path=Path)
    selected = []
    wanted = dict(data_layer={"read_day_tail", "is_ext_market_ts", "apply_xdxr_pre_close"},
                  history={"action_adjust"})
    for key, names in wanted.items():
        item = c["pure_sources"][key]
        if sha(item["path"]) != item["sha256"]:
            raise DiagnosticError("PURE_FUNCTION_SOURCE_DRIFT")
        nodes = ast.parse(Path(item["path"]).read_text()).body
        functions = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in functions} != names:
            raise DiagnosticError("ORIGINAL_FUNCTION_MISSING")
        selected.extend(functions)
        if key == "data_layer":
            selected.extend(n for n in nodes if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in {"DAY_STRUCT", "EXT_MARKET_DAY_STRUCT"} for t in n.targets))
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
        "<bound-original-pure-functions>", "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def read_raw_members(life, c, budget, namespace, output):
    """Only six actual members are SHA-authenticated; no whole-ZIP claim."""
    life.require()
    path = Path(c["archive"]["path"])
    if path.is_symlink():
        raise DiagnosticError("ARCHIVE_SYMLINK")
    frames, evidence = {}, {}
    raw_dir = Path(output) / "raw_members"
    raw_dir.mkdir(exist_ok=False)
    with path.open("rb") as handle:
        before = file_identity(os.fstat(handle.fileno()))
        if before != c["archive"]["frozen_stat"]:
            raise DiagnosticError("ARCHIVE_STAT_DRIFT")
        with zipfile.ZipFile(handle) as archive:
            for code in CODES:
                budget.check()
                life.require()
                binding = c["members"][code]
                infos = [info for info in archive.infolist() if info.filename == binding["name"]]
                if len(infos) != 1 or infos[0].file_size != binding["bytes"]:
                    raise DiagnosticError("EXACT_MEMBER_MISSING_OR_AMBIGUOUS:" + code)
                if binding["bytes"] > 2**20:
                    raise DiagnosticError("MEMBER_BYTE_BUDGET")
                payload = archive.read(infos[0])
                actual = hashlib.sha256(payload).hexdigest()
                if actual != binding["sha256"]:
                    raise DiagnosticError("MEMBER_IDENTITY_MISMATCH:" + code)
                target = raw_dir / (code + ".day")
                with target.open("xb") as f:
                    f.write(payload)
                if len(payload) % 32:
                    raise DiagnosticError("NONINTEGRAL_DAY_RECORD")
                frames[code] = namespace["read_day_tail"](code, target, tail_records=len(payload) // 32)
                evidence[code] = dict(name=binding["name"], sha256=actual, bytes=len(payload))
        if before != file_identity(os.fstat(handle.fileno())) or before != file_identity(path.stat()):
            raise DiagnosticError("ARCHIVE_CHANGED_DURING_MEMBER_READ")
    return frames, dict(whole_zip_status="REUSED_RECORDED_NOT_REHASHED",
        whole_zip_recorded_sha256=c["archive"]["recorded_sha256"], members=evidence)


def normalize(frame, code):
    import numpy as np
    import pandas as pd
    required = {"trade_date", "ts_code", *FIELDS}
    if not required.issubset(frame.columns):
        raise DiagnosticError("SOURCE_SCHEMA")
    work = frame[["trade_date", "ts_code", *FIELDS]].copy()
    if not work.ts_code.eq(code).all():
        raise DiagnosticError("SOURCE_SYMBOL")
    dates = work.trade_date.astype(str)
    if not dates.str.fullmatch(r"\d{8}").all() or pd.to_datetime(dates, format="%Y%m%d", errors="coerce").isna().any():
        raise DiagnosticError("SOURCE_DATE_KEY")
    work.trade_date = dates.astype("int64")
    if work.trade_date.duplicated().any() or not work.trade_date.is_monotonic_increasing:
        raise DiagnosticError("NONUNIQUE_NONMONOTONIC_DATE")
    for field in FIELDS:
        values = pd.to_numeric(work[field], errors="coerce")
        if not values.isna().equals(work[field].isna()) or np.isinf(values.to_numpy(dtype=float)).any():
            raise DiagnosticError("NONFINITE_OR_MALFORMED_VALUE")
        work[field] = values.astype(float)
    if work[["vol", "amount"]].lt(0).any().any():
        raise DiagnosticError("NEGATIVE_VOLUME_OR_AMOUNT")
    return work.reset_index(drop=True)


def anchors(raw, oracle):
    """A/B/H selection uses only keys and preregistered bar validity, never residuals."""
    import numpy as np
    raw = raw.copy()
    usable = np.isfinite(raw[list(FIELDS)].to_numpy()).all(axis=1)
    usable &= raw[["open", "high", "low", "close"]].gt(0).all(axis=1).to_numpy()
    usable &= raw.vol.gt(0).to_numpy() & raw.amount.ge(0).to_numpy()
    observed = raw.loc[usable].copy()
    observed["session"] = range(len(observed))
    valid_qfq = np.isfinite(oracle[["open", "high", "low", "close"]].to_numpy()).all(axis=1)
    valid_qfq &= oracle[["open", "high", "low", "close"]].gt(0).all(axis=1).to_numpy()
    common = observed.merge(oracle.loc[valid_qfq, ["trade_date"]], on="trade_date", validate="one_to_one")
    if common.empty:
        return None
    a = common.iloc[0]
    later = common.loc[common.session - int(a.session) >= POLICY["anchor_B_min_sessions"]]
    if later.empty:
        return None
    b = later.iloc[0]
    holdout = common.loc[(common.trade_date < b.trade_date) & (common.trade_date != a.trade_date)].tail(POLICY["holdout_count"])
    if len(holdout) != POLICY["holdout_count"]:
        return None
    return dict(A=int(a.trade_date), B=int(b.trade_date), H=holdout.trade_date.astype(int).tolist(),
                session_distance=int(b.session - a.session))


def reconstruct(raw, events, anchor_date, anchor_scale, namespace, target_dates):
    """No oracle holdout values enter this function; only the fixed anchor scale."""
    import numpy as np
    import pandas as pd
    if not math.isfinite(anchor_scale) or anchor_scale <= 0:
        raise DiagnosticError("INVALID_ANCHOR_SCALE")
    if not target_dates or max(target_dates) >= anchor_date:
        raise DiagnosticError("DISJOINT_HOLDOUT_REQUIRED")
    first = min(target_dates)
    work = raw.loc[raw.trade_date.between(first, anchor_date)].copy()
    usable = np.isfinite(work[list(FIELDS)].to_numpy()).all(axis=1)
    usable &= work[["open", "high", "low", "close"]].gt(0).all(axis=1).to_numpy()
    usable &= work.vol.gt(0).to_numpy() & work.amount.ge(0).to_numpy()
    actual = work.loc[usable].copy().reset_index(drop=True)
    dates = actual.trade_date.to_numpy(dtype="int64")
    if not len(actual) or dates[-1] != anchor_date:
        raise DiagnosticError("ANCHOR_NOT_ACTUAL_OBSERVED_BAR")
    prior = actual.close.shift().to_numpy()
    prior_dates = actual.trade_date.shift(fill_value=0).to_numpy(dtype="int64")
    # The diagnostic never generates pre-A prefix, and does not audit irrelevant history.
    e = events.loc[(events._date_int > first) & (events._date_int <= anchor_date)].copy()
    if len(e):
        if e._date_int.duplicated().any() or not set(ACTION_FIELDS).issubset(e):
            raise DiagnosticError("AMBIGUOUS_ACTION_KEY_OR_FIELDS")
        values = e[list(ACTION_FIELDS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DiagnosticError("INVALID_ACTION_FIELDS")
        if "category" in e and not e.category.eq(1).all():
            raise DiagnosticError("UNSUPPORTED_ACTION_CATEGORY")
    denominator, counts, action_dates = namespace["action_adjust"](
        prior, prior_dates, dates, e, namespace["apply_xdxr_pre_close"])
    q = np.ones(len(actual), dtype=float)
    changed = counts > 0
    if (not np.isfinite(denominator[changed]).all() or (denominator[changed] <= 0).any()
            or not np.isfinite(prior[changed]).all() or (prior[changed] <= 0).any()):
        raise DiagnosticError("INVALID_ACTION_DENOMINATOR")
    q[changed] = denominator[changed] / prior[changed]
    # No cached event means a neutral candidate branch, explicitly UNKNOWN coverage.
    # It is never certified as the economic assertion that no action occurred.
    factors = np.empty(len(actual), dtype=float)
    product = anchor_scale
    for i in range(len(actual) - 1, -1, -1):
        factors[i] = product  # event on this bar does not adjust this bar
        product *= q[i]
        if not math.isfinite(product) or product <= 0:
            raise DiagnosticError("NONFINITE_OR_NONPOSITIVE_PRODUCT")
    for field in ("open", "high", "low", "close"):
        actual[field] *= factors
        if not np.isfinite(actual[field]).all():
            raise DiagnosticError("RECONSTRUCTED_VALUE_NONFINITE")
    actual["cached_action_count"] = counts
    actual["matched_action_dates"] = action_dates
    actual["candidate_factor"] = factors
    return actual


def exact_comparison(rebuilt, oracle):
    import numpy as np
    if rebuilt.trade_date.tolist() != oracle.trade_date.tolist():
        raise DiagnosticError("HOLDOUT_KEY_IDENTITY")
    report = {}
    for field in FIELDS:
        x, y = rebuilt[field].to_numpy(dtype=float), oracle[field].to_numpy(dtype=float)
        if np.isinf(x).any() or np.isinf(y).any():
            raise DiagnosticError("NONFINITE_COMPARISON_VALUE")
        mask = np.isnan(x) == np.isnan(y)
        valid = np.isfinite(x) & np.isfinite(y)
        equal = (x == y) | (np.isnan(x) & np.isnan(y))
        difference = np.abs(x[valid] - y[valid])
        spacing = np.maximum(np.abs(np.spacing(x[valid])), np.abs(np.spacing(y[valid])))
        report[field] = dict(null_mask_exact=bool(mask.all()), mismatches=int((~equal).sum()),
            max_abs_difference=float(difference.max()) if len(difference) else None,
            max_relative_to_float64_spacing=float((difference / spacing).max()) if len(difference) else None,
            discrepancies=[dict(trade_date=int(rebuilt.trade_date.iloc[i]),
                reconstructed_hex=float(x[i]).hex() if math.isfinite(x[i]) else None,
                reference_hex=float(y[i]).hex() if math.isfinite(y[i]) else None)
                for i in range(len(x)) if not equal[i]])
    return report


def diagnose(raw_by_code, qfq, events_by_code, namespace):
    import pandas as pd
    import numpy as np
    results, numeric_mismatch = {}, False
    early_probes = {}
    for code in CODES:
        raw = normalize(raw_by_code[code], code)
        early = raw.loc[raw.trade_date.between(POLICY["early_probe_start"], POLICY["early_probe_end"])]
        valid_early = np.isfinite(early[["open", "high", "low", "close"]].to_numpy()).all(axis=1)
        early_probes[code] = ("RETURNED_VALID_BAR" if valid_early.any() else
            "EARLY_STATUS_UNKNOWN_BJ" if code.endswith(".BJ") else "EARLY_RETURN_UNRESOLVED")
        reference = normalize(qfq.loc[qfq.ts_code.eq(code)].sort_values("trade_date"), code)
        chosen = anchors(raw, reference)
        if chosen is None:
            results[code] = dict(status="CANNOT_CONFIRM_ANCHOR_OR_HOLDOUT", BJ_status="UNKNOWN" if code.endswith(".BJ") else None)
            continue
        a, b, holdout = chosen["A"], chosen["B"], chosen["H"]
        r = raw.set_index("trade_date")
        target = reference.set_index("trade_date")
        scales = {d: float(target.loc[d, "close"] / r.loc[d, "close"]) for d in (a, b)}
        anchor_reports = {}
        for date in (a, b):
            scaled = raw.loc[raw.trade_date.eq(date)].copy()
            for field in ("open", "high", "low", "close"):
                scaled[field] *= scales[date]
            anchor_reports[str(date)] = exact_comparison(scaled, reference.loc[reference.trade_date.eq(date)])
        rebuilt = reconstruct(raw, events_by_code.get(code, pd.DataFrame(columns=["_date_int", *ACTION_FIELDS])),
                              b, scales[b], namespace, holdout)
        actual_holdout = rebuilt.loc[rebuilt.trade_date.isin(holdout)].reset_index(drop=True)
        expected = reference.loc[reference.trade_date.isin(holdout)].reset_index(drop=True)
        comparison = exact_comparison(actual_holdout, expected)
        mismatches = sum(x["mismatches"] for x in comparison.values())
        mismatches += sum(x["mismatches"] for report in anchor_reports.values() for x in report.values())
        numeric_mismatch |= bool(mismatches)
        interval = rebuilt.loc[(rebuilt.trade_date > min(holdout)) & (rebuilt.trade_date <= b)]
        covered_events = int(interval.cached_action_count.sum())
        results[code] = dict(status="NUMERIC_PARITY_NOT_ESTABLISHED" if mismatches else "EXACT_SAMPLED_NUMERIC_PARITY_ONLY",
            anchors=chosen, scale_A=scales[a], scale_B=scales[b], anchors_comparison=anchor_reports,
            holdout_comparison=comparison, holdout_cached_events=covered_events,
            operator_evidence="CACHED_EVENT_OPERATOR_TEST_ONLY" if covered_events else "SCALE_AND_KEYS_ONLY_ZERO_EVENTS",
            action_completeness="CANNOT_CONFIRM", pre_A_prefix="NOT_GENERATED_NOT_VALIDATED")
    incomplete = any(v["status"].startswith("CANNOT_CONFIRM") for v in results.values())
    early_missing = any(early_probes[code] == "EARLY_RETURN_UNRESOLVED" for code in CODES[:4])
    return dict(decision="NUMERIC_PARITY_NOT_ESTABLISHED" if numeric_mismatch else
                "EARLY_RETURN_UNRESOLVED" if early_missing else
                "CANNOT_CONFIRM_ANCHOR_OR_HOLDOUT" if incomplete else "BOUNDED_DIAGNOSTIC_COMPLETE",
                symbols=results, early_return_probes=early_probes,
                numerical_parity_is_native_QFQ_certification=False,
                training_ready=False, fits=0, network_calls=0, **FLAGS)


def run(c, adapter, *, fixture=False):
    if fixture:
        fixture_guard(c)
    else:
        validate_activation(c)
    output = Path(c["run_dir"])
    output.mkdir(exist_ok=False)
    with (output / "owner.lock").open("x") as handle:
        handle.write(canonical(dict(pid=os.getpid(), execution_id=c["execution_id"])))
        handle.flush()
        os.fsync(handle.fileno())
    receipt = output / "RECEIPT.json"
    write_json(receipt, dict(status="PREPARED_NOT_REGISTERED"))
    life = Lifecycle(adapter, c)
    budget = Budget(output, c["caps"])
    try:
        life.start(receipt)
        with budget:
            if available_memory() < c["caps"]["min_memory_bytes"] or shutil.disk_usage(output).free < c["caps"]["min_disk_bytes"]:
                raise DiagnosticError("DEFERRED_RESOURCE")
            os.environ.update({k: "4" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
            sys.dont_write_bytecode = True
            c["arrow_pools"] = configure_arrow_pools(c["caps"]["threads"])
            import pandas as pd
            namespace = pure_functions(c)
            with authenticated(life, c["qfq"], budget) as handle:
                qfq = pd.read_parquet(handle, columns=["trade_date", "ts_code", *FIELDS],
                    filters=[("ts_code", "in", list(CODES))])
            with authenticated(life, c["actions"], budget) as handle:
                events = pd.read_parquet(handle)
            events = events.loc[events.ts_code.isin(CODES)].copy()
            text = events._xdxr_trade_date.astype(str).str.replace(r"\.0$", "", regex=True).str[:10]
            parsed = pd.to_datetime(text, errors="coerce")
            if parsed.isna().any():
                raise DiagnosticError("ACTION_DATE_KEY")
            events["_date_int"] = parsed.dt.strftime("%Y%m%d").astype("int64")
            by_code = {code: frame for code, frame in events.groupby("ts_code")}
            raw, evidence = read_raw_members(life, c, budget, namespace, output)
            life.require()
            result = diagnose(raw, qfq, by_code, namespace)
            budget.check()
            result.update(status="COMPLETED", input_identity=evidence, wall_sec=time.monotonic() - budget.start_time,
                sampled_peak_rss_bytes=budget.peak, footprint_bytes=None, footprint_reason="NOT_OBSERVABLE_BY_RUSAGE",
                arrow_pools=c["arrow_pools"])
        write_json(receipt, result)
        life.finish(receipt, result["decision"], dict(symbols=len(result["symbols"])))
        return result
    except BaseException as exc:
        decision = "INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else (
            "DEFERRED_RESOURCE" if "DEFERRED_RESOURCE" in str(exc) else "FAILED_" + type(exc).__name__)
        try:
            write_json(receipt, dict(status=decision, error_type=type(exc).__name__, error=str(exc),
                wall_sec=time.monotonic() - budget.start_time, sampled_peak_rss_bytes=budget.peak,
                arrow_pools=c.get("arrow_pools"), **FLAGS))
        finally:
            if life.started and not life.finished:
                life.finish(receipt, decision)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    c = load_config(args.config)
    if not args.execute:
        print(canonical(dict(status="SOFTWARE_METADATA_ONLY_NO_SCIENCE", code_sha256=c["code_composite_sha256"])))
        return
    if c.get("fixture_root"):
        raise PermissionError("CLI_FIXTURE_MODE_PROHIBITED")
    run(c, load_adapter(c))


if __name__ == "__main__":
    main()
