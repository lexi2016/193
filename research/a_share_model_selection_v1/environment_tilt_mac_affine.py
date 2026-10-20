"""Mac-only OPENBOOK affine/cent hypothesis; default CLI is metadata only.

No native QFQ, rounding, action completeness or training certification is made.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import time
from decimal import Decimal, localcontext, ROUND_HALF_UP, InvalidOperation
from pathlib import Path

import environment_tilt_mac_recovery as base
from environment_tilt_mac_registry import Lifecycle, FLAGS, canonical, sha, load_adapter

OHLC = ("open", "high", "low", "close")
CENT = Decimal(".01")
POLICY = dict(formula="ORDERED_CATEGORY1_AFFINE_THEN_B_LOW_HIGH_AFFINE_V1",
    anchor_rule="UNCHANGED_M7_A_B_H", precision=50, rounding="ROUND_HALF_UP",
    quantum="0.01", oracle_cent_required=True, event_boundary="t<event_date<=B",
    serialization_status="UNCONFIRMED_FALSIFIABLE_HYPOTHESIS",
    action_completeness="UNKNOWN_OBSERVED_CACHE_ONLY")
CODE_FILES = (*base.CODE_FILES, "environment_tilt_mac_affine.py",
              "tests/test_environment_tilt_mac_affine.py")


class CannotConfirm(base.DiagnosticError):
    pass


def decimal_scalar(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CannotConfirm("INVALID_DECIMAL_SCALAR") from exc
    if not result.is_finite():
        raise CannotConfirm("NONFINITE_DECIMAL_SCALAR")
    return result


def cent_reference(value):
    value = decimal_scalar(value)
    if value != value.quantize(CENT):
        raise CannotConfirm("UNSUPPORTED_REFERENCE_SERIALIZATION")
    return value


def event_affine(event):
    """Original category1 economic-unit candidate, not native QFQ proof."""
    if not set(base.ACTION_FIELDS).issubset(event):
        raise CannotConfirm("MISSING_ACTION_FIELDS")
    if "category" in event and decimal_scalar(event["category"]) != 1:
        raise CannotConfirm("UNSUPPORTED_ACTION_CATEGORY")
    dividend, rights_price, bonus, rights = (
        decimal_scalar(event[name]) for name in base.ACTION_FIELDS)
    denominator = Decimal(1) + bonus / 10 + rights / 10
    if denominator <= 0:
        raise CannotConfirm("NONPOSITIVE_ACTION_DENOMINATOR")
    return Decimal(1) / denominator, (-dividend / 10 + rights_price * rights / 10) / denominator


def compose_events(events):
    """Ascending source date: T_latest(...T_earliest(P)); reject ambiguity."""
    dates = [int(e["_date_int"]) for e in events]
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise CannotConfirm("AMBIGUOUS_ACTION_DATE_ORDER")
    a, b = Decimal(1), Decimal(0)
    for event in events:
        x, y = event_affine(event)
        a, b = x * a, x * b + y
    return a, b


def calibrate(raw, reference):
    """Only the predetermined B low/high identify G; no holdout enters."""
    low, high = decimal_scalar(raw["low"]), decimal_scalar(raw["high"])
    qlow, qhigh = cent_reference(reference["low"]), cent_reference(reference["high"])
    if high <= low or qhigh <= qlow:
        raise CannotConfirm("DEGENERATE_B_RANGE")
    u = (qhigh - qlow) / (high - low)
    return u, qlow - u * low


def reconstruct_holdout(raw, events, chosen, reference_B):
    """No target oracle values accepted; all arithmetic Decimal precision50."""
    b, holdout = chosen["B"], chosen["H"]
    if len(holdout) != 10 or len(set(holdout)) != 10 or chosen["A"] in holdout or max(holdout) >= b:
        raise CannotConfirm("INVALID_DISJOINT_HOLDOUT")
    rows = raw.set_index("trade_date")
    if any(date not in rows.index for date in [b, *holdout]):
        raise CannotConfirm("MISSING_FIXED_KEYS")
    selected = events.loc[(events._date_int > min(holdout)) & (events._date_int <= b)].sort_values("_date_int")
    records = selected.to_dict("records")
    with localcontext() as ctx:
        ctx.prec, ctx.rounding = 50, ROUND_HALF_UP
        u, v = calibrate(rows.loc[b], reference_B)
        # Validate even an event on the first H date if present in the wider cache
        # only when it participates; event on t is excluded by the frozen boundary.
        compose_events(records)
        predicted = []
        for date in holdout:
            relevant = [event for event in records if date < int(event["_date_int"]) <= b]
            a, offset = compose_events(relevant)
            row = {"trade_date": date}
            for field in OHLC:
                value = u * (a * decimal_scalar(rows.loc[date, field]) + offset) + v
                row[field] = dict(unrounded=str(value), cent=str(value.quantize(CENT)))
            predicted.append(row)
    return predicted, dict(u=str(u), v=str(v), calibration_fields=["low", "high"],
        calibration_uncertainty="Rounded B prices may underidentify true u/v; diagnostic matches cannot unlock training.",
        observed_cache_events=len(records), action_completeness="CANNOT_CONFIRM",
        operator_evidence="OBSERVED_CACHE_AFFINE_OPERATOR_ONLY" if records else
                          "AFFINE_SCALE_IDENTIFICATION_ONLY_ZERO_EVENTS")


def compare_prices(predicted, reference):
    if [x["trade_date"] for x in predicted] != reference.trade_date.tolist():
        raise CannotConfirm("HOLDOUT_KEY_IDENTITY")
    report, mismatches = [], 0
    with localcontext() as ctx:
        ctx.prec, ctx.rounding = 50, ROUND_HALF_UP
        for rebuilt, (_, expected) in zip(predicted, reference.iterrows()):
            row = dict(trade_date=rebuilt["trade_date"], fields={})
            for field in OHLC:
                target = cent_reference(expected[field])  # NEVER round reference to accept it
                unrounded = decimal_scalar(rebuilt[field]["unrounded"])
                serialized = decimal_scalar(rebuilt[field]["cent"])
                equal = serialized == target
                mismatches += not equal
                row["fields"][field] = dict(**rebuilt[field], reference=str(target),
                    unrounded_residual=str(unrounded - target), exact_cent_match=equal)
            report.append(row)
    return report, int(mismatches)


def diagnose(raw_by_code, qfq, events_by_code, namespace=None):
    import pandas as pd
    results = {}
    for code in base.CODES:
        try:
            raw = base.normalize(raw_by_code[code], code)
            reference = base.normalize(qfq.loc[qfq.ts_code.eq(code)].sort_values("trade_date"), code)
            chosen = base.anchors(raw, reference)
            if chosen is None:
                raise CannotConfirm("MISSING_FIXED_ANCHOR_OR_HOLDOUT")
            target = reference.set_index("trade_date")
            # Entire compared oracle must already satisfy the frozen serialization.
            with localcontext() as ctx:
                ctx.prec, ctx.rounding = 50, ROUND_HALF_UP
                for date in [chosen["B"], *chosen["H"]]:
                    for field in OHLC:
                        cent_reference(target.loc[date, field])
            events = events_by_code.get(code, pd.DataFrame(columns=["_date_int", *base.ACTION_FIELDS]))
            rebuilt, calibration = reconstruct_holdout(raw, events, chosen, target.loc[chosen["B"]])
            expected = reference.loc[reference.trade_date.isin(chosen["H"])].reset_index(drop=True)
            comparison, mismatches = compare_prices(rebuilt, expected)
            # Ancillary fields remain raw and are never normalized or used for acceptance.
            raw_holdout = raw.loc[raw.trade_date.isin(chosen["H"])].reset_index(drop=True)
            ancillary = base.exact_comparison(raw_holdout, expected)
            results[code] = dict(status="CENT_HYPOTHESIS_MISMATCH" if mismatches else
                "EXACT_CENT_MATCH_UNCONFIRMED_HYPOTHESIS_ONLY", anchors=chosen,
                calibration=calibration, price_mismatches=mismatches, prices=comparison,
                raw_ancillary_comparison={key: ancillary[key] for key in ("vol", "amount")},
                pre_A_prefix="NOT_GENERATED_NOT_VALIDATED")
        except CannotConfirm as exc:
            results[code] = dict(status="CANNOT_CONFIRM", reason=str(exc))
    statuses = [item["status"] for item in results.values()]
    decision = ("CANNOT_CONFIRM" if "CANNOT_CONFIRM" in statuses else
        "AFFINE_CENT_PARITY_NOT_ESTABLISHED" if "CENT_HYPOTHESIS_MISMATCH" in statuses else
        "SAMPLED_AFFINE_CENT_MATCH_UNCONFIRMED_HYPOTHESIS_ONLY")
    return dict(decision=decision, symbols=results, evidence_level="OPENBOOK_INPUT_SEMANTICS_DIAGNOSTIC",
        serialization_status=POLICY["serialization_status"], native_QFQ_certified=False,
        training_ready=False, fits=0, network_calls=0, **FLAGS)


class AffineLifecycle(Lifecycle):
    """Reuse accepted durable guard, changing only the new experiment identity."""
    def event(self, kind, decision, receipt, metrics):
        event = super().event(kind, decision, receipt, metrics)
        event.update(route_id="a-share-environment-tilt-mac-affine-diagnostic-20261021",
            experiment_id="A_SHARE_ENVIRONMENT_TILT_MAC_AFFINE_DIAGNOSTIC_20261021",
            baseline_id="M7 scalar diagnostic terminal preserved NUMERIC_PARITY_NOT_ESTABLISHED",
            stage="MAC_AFFINE_CENT_OPENBOOK_DIAGNOSTIC",
            one_variable="CATEGORY1_AFFINE_PLUS_UNCONFIRMED_CENT_SERIALIZATION_NO_PREFIX",
            sealed_status="OPENBOOK_M7_REVEALED_INPUT_SEMANTICS_ONLY",
            evidence_level="OPENBOOK_DIAGNOSTIC_ONLY")
        return event


def load_config(path):
    """Source/metadata only; no actual payload is hashed or decoded here."""
    path = Path(path).resolve()
    c = base.read_json(path)
    c.update(config_path=str(path), config_sha256=sha(path))
    if c["codes"] != list(base.CODES) or c["policy"] != POLICY or c["caps"] != base.CAPS:
        raise base.DiagnosticError("FROZEN_AFFINE_POLICY_CHANGED")
    if c.get("network_allowed") is not False or c.get("fit_count") != 0:
        raise base.DiagnosticError("DIAGNOSTIC_BOUNDARY_CHANGED")
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,120}", c["execution_id"]):
        raise base.DiagnosticError("INVALID_EXECUTION_ID")
    root, output = Path(c["batch_root"]).resolve(), Path(c["run_dir"]).resolve()
    if output.parent != root or not output.name.startswith("M8_"):
        raise base.DiagnosticError("OUTPUT_OUTSIDE_M8_SCOPE")
    identity = base.read_json(c["identity_path"])
    if sha(c["identity_path"]) != c["identity_sha256"] or set(identity["code_by_file"]) != set(CODE_FILES):
        raise base.DiagnosticError("EXACT_CODE_IDENTITY_REQUIRED")
    for name, expected in identity["code_by_file"].items():
        if sha(Path(__file__).resolve().parent / name) != expected:
            raise base.DiagnosticError("CODE_DRIFT:" + name)
    c["code_composite_sha256"] = hashlib.sha256(canonical(identity["code_by_file"]).encode()).hexdigest()
    if sha(c["contract_path"]) != c["contract_sha256"]:
        raise base.DiagnosticError("CONTRACT_DRIFT")
    for item in c["pure_sources"].values():
        if sha(item["path"]) != item["sha256"]:
            raise base.DiagnosticError("ORIGINAL_PURE_SOURCE_DRIFT")
    if (c["qfq"]["sha256"] != base.BASE_SHA or c["actions"]["sha256"] != base.ACTION_SHA
            or c["archive"]["recorded_sha256"] != base.ZIP_SHA):
        raise base.DiagnosticError("FROZEN_ORIGINAL_INPUT_CHANGED")
    metadata = base.read_json(c["member_binding_path"])
    if sha(c["member_binding_path"]) != c["member_binding_sha256"] or set(metadata["members"]) != set(base.CODES):
        raise base.DiagnosticError("MEMBER_BINDING_DRIFT")
    c["members"] = metadata["members"]
    return c


def run(c, adapter, *, fixture=False):
    if fixture:
        base.fixture_guard(c)
    else:
        base.validate_activation(c)
    output = Path(c["run_dir"])
    output.mkdir(exist_ok=False)
    with (output / "owner.lock").open("x") as handle:
        handle.write(canonical(dict(pid=os.getpid(), execution_id=c["execution_id"])))
        handle.flush()
        os.fsync(handle.fileno())
    receipt = output / "RECEIPT.json"
    base.write_json(receipt, dict(status="PREPARED_NOT_REGISTERED"))
    life, budget = AffineLifecycle(adapter, c), base.Budget(output, c["caps"])
    try:
        life.start(receipt)
        with budget:
            if base.available_memory() < c["caps"]["min_memory_bytes"] or shutil.disk_usage(output).free < c["caps"]["min_disk_bytes"]:
                raise base.DiagnosticError("DEFERRED_RESOURCE")
            os.environ.update({key: "4" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
            sys.dont_write_bytecode = True
            c["arrow_pools"] = base.configure_arrow_pools(c["caps"]["threads"])
            import pandas as pd
            namespace = base.pure_functions(c)
            with base.authenticated(life, c["qfq"], budget) as handle:
                qfq = pd.read_parquet(handle, columns=["trade_date", "ts_code", *base.FIELDS],
                    filters=[("ts_code", "in", list(base.CODES))])
            with base.authenticated(life, c["actions"], budget) as handle:
                events = pd.read_parquet(handle)
            events = events.loc[events.ts_code.isin(base.CODES)].copy()
            text = events._xdxr_trade_date.astype(str).str.replace(r"\.0$", "", regex=True).str[:10]
            parsed = pd.to_datetime(text, errors="coerce")
            if parsed.isna().any():
                raise base.DiagnosticError("ACTION_DATE_KEY")
            events["_date_int"] = parsed.dt.strftime("%Y%m%d").astype("int64")
            by_code = {code: frame for code, frame in events.groupby("ts_code")}
            raw, evidence = base.read_raw_members(life, c, budget, namespace, output)
            life.require()
            result = diagnose(raw, qfq, by_code)
            budget.check()
            result.update(status="COMPLETED", input_identity=evidence, wall_sec=time.monotonic() - budget.start_time,
                sampled_peak_rss_bytes=budget.peak, footprint_bytes=None, footprint_reason="NOT_OBSERVABLE_BY_RUSAGE",
                arrow_pools=c["arrow_pools"])
        base.write_json(receipt, result)
        life.finish(receipt, result["decision"], dict(symbols=len(result["symbols"])))
        return result
    except BaseException as exc:
        decision = "INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else (
            "DEFERRED_RESOURCE" if "DEFERRED_RESOURCE" in str(exc) else "FAILED_" + type(exc).__name__)
        try:
            base.write_json(receipt, dict(status=decision, error_type=type(exc).__name__, error=str(exc),
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
