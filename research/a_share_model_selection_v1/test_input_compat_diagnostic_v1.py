from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import input_compat_diagnostic_v1 as diagnostic


EXPECTED_MAP = {
    "date": "trade_date", "code": "ts_code", "open": "open", "high": "high",
    "low": "low", "close": "close", "volume": "vol", "amount": "amount",
}


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["2024-01-02", "000001.SZ", 10.0, 11.0, 9.0, 10.0, 100.0, 1000.0],
            ["2024-01-03", "000002.SZ", 10.0, 10.0 - 1e-15, 9.0, 10.0, 100.0, 1000.0],
            ["2024-01-04", "000003.SZ", 10.0, 9.5, 10.5, 10.0, 0.0, -1.0],
            ["2024-01-05", "000004.SZ", 0.0, 0.0, 0.0, 0.0, np.inf, 0.0],
            ["2024-01-05", "000004.SZ", 10.0, 11.0, 9.0, np.nan, 100.0, 10.0],
        ],
        columns=diagnostic.EXPECTED_COLUMNS,
    )


def test_price_panel_diagnostic_preserves_raw_values_and_reports_strict_violations():
    frame = _panel()
    before = frame.copy(deep=True)
    result = diagnostic.diagnose_price_panel(frame, EXPECTED_MAP)

    assert frame.equals(before)
    assert result["row_count"] == 5
    assert result["duplicate_date_code_row_count"] == 2
    assert result["numeric_fields"]["close"]["missing_or_unparseable_count"] == 1
    assert result["numeric_fields"]["vol"]["nonfinite_infinity_count"] == 1
    assert result["ohlc"]["nonpositive_row_count_by_field"] == {
        "open": 1, "high": 1, "low": 1, "close": 1,
    }
    assert result["ohlc"]["high_below_max_open_close_count"] == 2
    assert result["ohlc"]["low_above_min_open_close_count"] == 1
    assert result["ohlc"]["strict_violation_epsilon_explainable_count"] == 1
    assert result["ohlc"]["epsilon_is_diagnostic_only"] is True
    assert len(result["raw_violating_rows"]) == 4
    assert result["volume_amount"]["vol"]["zero_count"] == 1
    assert result["volume_amount"]["amount"]["negative_count"] == 1


def test_int64_yyyymmdd_dates_match_frozen_consumer_parsing():
    frame = _panel()
    frame["trade_date"] = pd.Series([20240102, 20240103, 20240104, 20240105, 20240105], dtype="int64")

    result = diagnostic.diagnose_price_panel(frame, EXPECTED_MAP)

    assert frame["trade_date"].dtype == "int64"
    assert result["date"]["parse_failure_count"] == 0
    assert result["date"]["first_valid"] == "2024-01-02"
    assert result["date"]["last_valid"] == "2024-01-05"


def test_zero_volume_and_amount_are_reported_without_strict_raw_violations():
    frame = pd.DataFrame(
        [["2024-01-02", "000001.SZ", 10.0, 11.0, 9.0, 10.0, 0.0, 0.0]],
        columns=diagnostic.EXPECTED_COLUMNS,
    )

    result = diagnostic.diagnose_price_panel(frame, EXPECTED_MAP)

    assert result["volume_amount"]["vol"] == {"zero_count": 1, "negative_count": 0}
    assert result["volume_amount"]["amount"] == {"zero_count": 1, "negative_count": 0}
    assert result["raw_violating_rows"] == []


def test_code_metrics_match_consumer_strip_and_keep_raw_whitespace():
    frame = pd.DataFrame(
        [
            ["2024-01-02", " 000001.SZ ", 10.0, 11.0, 9.0, 10.0, 100.0, 1000.0],
            ["2024-01-02", "000001.SZ", 10.0, 11.0, 9.0, 10.0, 100.0, 1000.0],
        ],
        columns=diagnostic.EXPECTED_COLUMNS,
    )

    result = diagnostic.diagnose_price_panel(frame, EXPECTED_MAP)

    assert result["duplicate_date_code_row_count"] == 0
    assert result["normalized_duplicate_date_code_row_count"] == 2
    assert result["code"]["consumer_normalized_distinct_count"] == 1
    assert result["code"]["raw_whitespace_row_count"] == 1
    assert result["code"]["raw_whitespace_samples"] == [{"row_index": 0, "raw_code": " 000001.SZ "}]
    assert [row["ts_code"] for row in result["raw_violating_rows"]] == [" 000001.SZ ", "000001.SZ"]


def test_calendar_diagnostic_reports_schema_coverage_and_keeps_identity_unconfirmed():
    raw = b"trade_date,calendar_index\n20240102,0\n20240103,1\n20240104,2\n"
    result = diagnostic.diagnose_calendar(
        raw,
        panel_dates={"2024-01-02", "2024-01-04", "2024-01-05"},
        expected={"schema": ["trade_date", "calendar_index"], "rows": 3, "date_range": ["2024-01-02", "2024-01-04"]},
    )
    assert result["calendar_index_contiguous_from_zero"] is True
    assert result["calendar_index_date_order_strictly_increasing"] is True
    assert result["trade_date_dtype"] == "int64"
    assert result["trade_date_parse_failure_count"] == 0
    assert result["panel_dates_missing_from_calendar_count"] == 1
    assert result["panel_dates_missing_from_calendar_sample"] == ["2024-01-05"]
    assert result["official_exchange_calendar_identity"] == "CANNOT_CONFIRM"

    reversed_dates = diagnostic.diagnose_calendar(
        b"trade_date,calendar_index\n20240103,0\n20240102,1\n",
        panel_dates={"2024-01-02", "2024-01-03"},
        expected={"schema": ["trade_date", "calendar_index"], "rows": 2, "date_range": ["2024-01-02", "2024-01-03"]},
    )
    assert reversed_dates["calendar_index_contiguous_from_zero"] is True
    assert reversed_dates["calendar_index_date_order_strictly_increasing"] is False

    duplicate_and_invalid = diagnostic.diagnose_calendar(
        b"trade_date,calendar_index\n20240102,0\n20240102,1\n20241301,2\n",
        panel_dates={"2024-01-02"},
        expected={"schema": ["trade_date", "calendar_index"], "rows": 3, "date_range": ["2024-01-02", "2024-01-02"]},
    )
    assert duplicate_and_invalid["trade_date_dtype"] == "int64"
    assert duplicate_and_invalid["trade_date_parse_failure_count"] == 1
    assert duplicate_and_invalid["duplicate_date_row_count"] == 2
    assert duplicate_and_invalid["calendar_index_date_order_strictly_increasing"] is False


def test_futu_diagnostic_reads_only_date_and_code_metrics():
    csv_raw = b"date,code,open,close\n2024-01-02,SH.000300,1,secret-a\n2024-01-03,SH.000300,2,secret-b\n"
    receipt = {
        "normalized_path": "index_000300_daily.csv",
        "normalized_sha256": hashlib.sha256(csv_raw).hexdigest(),
        "normalized_row_count": 2,
        "index_code": "SH.000300",
    }
    receipt_raw = json.dumps(receipt, sort_keys=True).encode()
    expected = {
        "csv_sha256": hashlib.sha256(csv_raw).hexdigest(),
        "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "csv_name": "index_000300_daily.csv",
        "index_code": "SH.000300",
        "header": ["date", "code", "open", "close"],
    }
    result = diagnostic.diagnose_futu(
        csv_raw, receipt_raw, panel_dates={"2024-01-02", "2024-01-04"}, expected=expected
    )
    assert result["date_and_code_only"] is True
    assert result["panel_date_missing_from_futu_count"] == 1
    assert result["futu_date_outside_panel_count"] == 1
    assert "secret-a" not in json.dumps(result)
    assert "open" not in result


def test_activation_fails_closed_on_any_mismatched_binding_or_permission():
    valid = {
        "schema_version": "A_SHARE_MODEL_SELECTION_PV_INPUT_COMPAT_ACTIVATION_V1",
        "status": "APPROVED_FOR_INPUT_COMPAT_DIAGNOSTIC_ONLY",
        "diagnostic_code": diagnostic.DIAGNOSTIC_CODE,
        "contract_sha256": "c" * 64,
        "input_binding_sha256": "i" * 64,
        "code_sha256": "d" * 64,
        "allow_bound_input_read": True,
        "allow_futu_metadata_read": True,
        "allow_model_fit": False,
        "allow_formula_evaluation": False,
        "allow_label_read": False,
        "allow_network": False,
        "reviewer_identity": "independent-reviewer",
        "reviewer_receipt_path": "receipt.json",
        "reviewer_receipt_sha256": "a" * 64,
    }
    diagnostic.validate_activation(
        valid, contract_sha256="c" * 64, input_binding_sha256="i" * 64, code_sha256="d" * 64
    )
    bad = {**valid, "allow_model_fit": True}
    with pytest.raises(diagnostic.DiagnosticError, match="ACTIVATION_MISMATCH:allow_model_fit"):
        diagnostic.validate_activation(
            bad, contract_sha256="c" * 64, input_binding_sha256="i" * 64, code_sha256="d" * 64
        )


class _FakeGuard:
    def __init__(self, events):
        self.events = events

    def claim_no_fit(self, **kwargs):
        self.events.append(("claim_no_fit", kwargs))
        return object()

    def append_no_fit_terminal(self, claim, *, succeeded, error_type=None):
        self.events.append(("terminal", succeeded, error_type))


class _FakeRoute:
    def __init__(self, events, calendar_bytes):
        self.events = events
        self.calendar_bytes = calendar_bytes

    def make_official_fit_guard(self, **kwargs):
        self.events.append(("guard_factory", kwargs))
        return _FakeGuard(self.events)

    def set_registry_execution_context(self, guard, execution_id, context):
        self.events.append(("set_context", execution_id))

    def read_bound_panel(self, **kwargs):
        self.events.append(("read_panel", kwargs["preparation_claim"], kwargs["activation_path"]))
        return _panel(), {"binding": {}}

    def read_bound_archive_member(self, source):
        self.events.append(("read_calendar", source.get("member")))
        return self.calendar_bytes


def _copy_control_inputs(project: Path) -> tuple[Path, dict]:
    root = Path(__file__).resolve().parents[5]
    source_batch = root / "research/a_share_model_selection_v1/batches/pv_environment_r1_20261019"
    batch = project / "research/a_share_model_selection_v1/batches/pv_environment_r1_20261019"
    batch.mkdir(parents=True)
    for name in ("INPUT_BINDING.json", "CONTRACT.json", diagnostic.FROZEN_ROUTE_READ_ACTIVATION_NAME):
        shutil.copyfile(source_batch / name, batch / name)
    contract = json.loads((source_batch / "INPUT_COMPAT_DIAGNOSTIC_CONTRACT.json").read_text())
    futu_dir = batch / "index_recovery_futu/FUTU_000300_HISTORY_FULL_20261019_R1"
    futu_dir.mkdir(parents=True)
    csv_raw = b"date,code,open,close\n2024-01-02,SH.000300,1,unused-a\n2024-01-03,SH.000300,2,unused-b\n"
    csv_path = futu_dir / "index_000300_daily.csv"
    csv_path.write_bytes(csv_raw)
    futu_receipt = {
        "normalized_path": "index_000300_daily.csv",
        "normalized_sha256": hashlib.sha256(csv_raw).hexdigest(),
        "normalized_row_count": 2,
        "index_code": "SH.000300",
    }
    receipt_raw = json.dumps(futu_receipt, sort_keys=True).encode()
    (futu_dir / "receipt.json").write_bytes(receipt_raw)
    contract["futu"]["csv_sha256"] = hashlib.sha256(csv_raw).hexdigest()
    contract["futu"]["receipt_sha256"] = hashlib.sha256(receipt_raw).hexdigest()
    contract["futu"]["expected_header"] = ["date", "code", "open", "close"]
    diag_path = batch / "INPUT_COMPAT_DIAGNOSTIC_CONTRACT.json"
    diag_path.write_text(json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return batch, contract


def test_run_claims_before_any_bound_read_and_terminalizes_success_once(tmp_path):
    batch, contract = _copy_control_inputs(tmp_path)
    review_raw = b'{"synthetic":true}\n'
    (batch / "synthetic-review.json").write_bytes(review_raw)
    code_sha = diagnostic._sha256_file(Path(diagnostic.__file__).resolve())
    contract_sha = diagnostic._sha256((batch / "INPUT_COMPAT_DIAGNOSTIC_CONTRACT.json").read_bytes())
    binding_sha = diagnostic._sha256((batch / "INPUT_BINDING.json").read_bytes())
    activation = {
        "schema_version": "A_SHARE_MODEL_SELECTION_PV_INPUT_COMPAT_ACTIVATION_V1",
        "status": "APPROVED_FOR_INPUT_COMPAT_DIAGNOSTIC_ONLY",
        "diagnostic_code": diagnostic.DIAGNOSTIC_CODE,
        "contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "code_sha256": code_sha,
        "allow_bound_input_read": True,
        "allow_futu_metadata_read": True,
        "allow_model_fit": False,
        "allow_formula_evaluation": False,
        "allow_label_read": False,
        "allow_network": False,
        "reviewer_identity": "synthetic-reviewer",
        "reviewer_receipt_path": "synthetic-review.json",
        "reviewer_receipt_sha256": hashlib.sha256(review_raw).hexdigest(),
    }
    (batch / diagnostic.ACTIVATION_NAME).write_text(json.dumps(activation), encoding="utf-8")
    events = []
    route = _FakeRoute(events, b"trade_date,calendar_index\n2024-01-02,0\n")

    result = diagnostic.run_diagnostic(
        tmp_path,
        execution_id="SYNTHETIC_NOFIT_ORDERING",
        route_loader=lambda root: route,
    )

    names = [event[0] for event in events]
    assert names.index("claim_no_fit") < names.index("read_panel") < names.index("read_calendar") < names.index("terminal")
    assert names.count("terminal") == 1
    panel_read = next(event for event in events if event[0] == "read_panel")
    assert Path(panel_read[2]).name == diagnostic.FROZEN_ROUTE_READ_ACTIVATION_NAME
    assert hashlib.sha256(Path(panel_read[2]).read_bytes()).hexdigest() == diagnostic.FROZEN_ROUTE_READ_ACTIVATION_SHA256
    assert events[-1][1] is True
    assert result["status"] == "DIAGNOSTIC_COMPLETE_NO_FIT"
    assert result["fit_started"] is False
    assert result["truth_access"] == "NOT_READ"


def test_run_read_error_gets_one_zero_fit_terminal(tmp_path):
    batch, _ = _copy_control_inputs(tmp_path)
    review_raw = b'{"synthetic":true}\n'
    (batch / "synthetic-review.json").write_bytes(review_raw)
    code_sha = diagnostic._sha256_file(Path(diagnostic.__file__).resolve())
    contract_sha = diagnostic._sha256((batch / "INPUT_COMPAT_DIAGNOSTIC_CONTRACT.json").read_bytes())
    binding_sha = diagnostic._sha256((batch / "INPUT_BINDING.json").read_bytes())
    activation = {
        "schema_version": "A_SHARE_MODEL_SELECTION_PV_INPUT_COMPAT_ACTIVATION_V1",
        "status": "APPROVED_FOR_INPUT_COMPAT_DIAGNOSTIC_ONLY",
        "diagnostic_code": diagnostic.DIAGNOSTIC_CODE,
        "contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "code_sha256": code_sha,
        "allow_bound_input_read": True,
        "allow_futu_metadata_read": True,
        "allow_model_fit": False,
        "allow_formula_evaluation": False,
        "allow_label_read": False,
        "allow_network": False,
        "reviewer_identity": "synthetic-reviewer",
        "reviewer_receipt_path": "synthetic-review.json",
        "reviewer_receipt_sha256": hashlib.sha256(review_raw).hexdigest(),
    }
    (batch / diagnostic.ACTIVATION_NAME).write_text(json.dumps(activation), encoding="utf-8")
    events = []

    class ReadFailRoute(_FakeRoute):
        def read_bound_panel(self, **kwargs):
            self.events.append(("read_panel", kwargs["preparation_claim"]))
            raise OSError("synthetic read failure")

    route = ReadFailRoute(events, b"")
    with pytest.raises(OSError, match="synthetic read failure"):
        diagnostic.run_diagnostic(
            tmp_path,
            execution_id="SYNTHETIC_NOFIT_ERROR",
            route_loader=lambda root: route,
        )
    assert [event[0] for event in events].count("claim_no_fit") == 1
    terminals = [event for event in events if event[0] == "terminal"]
    assert terminals == [("terminal", False, "OSError")]


def test_frozen_route_approved_input_loader_accepts_synthetic_reviewed_control_files(tmp_path):
    project_root = Path(diagnostic.__file__).resolve().parents[5]
    route = diagnostic._load_frozen_route(project_root)
    batch = tmp_path / "synthetic-control"
    batch.mkdir()
    receipt_path = batch / "reviewer-receipt.json"
    receipt_raw = b'{"synthetic_review_receipt":true}\n'
    receipt_path.write_bytes(receipt_raw)

    contract_path = batch / "CONTRACT.json"
    contract_path.write_text(
        json.dumps({"status": "FROZEN_AWAITING_PRE_RUN_REVIEW", "writer_identity": "synthetic-writer"}),
        encoding="utf-8",
    )
    binding_path = batch / "INPUT_BINDING.json"
    binding_path.write_text(
        json.dumps({
            "status": "INPUT_BINDING_READY",
            "readiness_by_arm": {"M0": {"status": "READY_WITH_SCOPE_LIMITATIONS"}},
        }),
        encoding="utf-8",
    )
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    binding_sha = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    activation_path = batch / "REVIEW_ACTIVATION_M0_R2.json"
    activation_path.write_text(
        json.dumps({
            "status": "APPROVED_FOR_M0_ONLY",
            "allowed_arms": ["M0"],
            "allowed_stages": ["M0_CANARY"],
            "reviewer_identity": "synthetic-independent-reviewer",
            "contract_sha256": contract_sha,
            "input_binding_sha256": binding_sha,
            "code_sha256": route.route_code_sha256(),
            "reviewer_receipt_path": str(receipt_path),
            "reviewer_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "allow_real_data_read": True,
            "allow_fit": True,
        }),
        encoding="utf-8",
    )

    approved_contract, approved_binding, activation = route._load_approved_inputs(
        contract_path=contract_path,
        binding_path=binding_path,
        activation_path=activation_path,
        stage="M0_CANARY",
        arms=("M0",),
    )

    assert approved_contract["writer_identity"] == "synthetic-writer"
    assert approved_binding["readiness_by_arm"]["M0"]["status"] == "READY_WITH_SCOPE_LIMITATIONS"
    assert activation["reviewer_identity"] == "synthetic-independent-reviewer"
