"""Route-local exact-source formula closure and narrow INDEXC/MONTH adapter.

Formula source is read only after the caller opens the reviewed input gate.  The
selected closure is emitted to a temporary file and removed after evaluation;
formula text and market data are never copied to public-export artifacts.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from tdx_ast_engine_injected import evaluate_tdx_formula_on_df


CORE_OUTPUTS = (
    "DISTW", "DISTB", "C", "YEL0", "LCOUNT", "HIST0", "AMTR", "VOLR",
    "VMA20R", "VSPHHV", "FCOUNT", "SFCOUNT", "RET20P", "GAIN60",
    "RANGE60", "RANGE120", "JVAL", "BKRSI", "CPOS", "UPSH", "WASHSCR",
    "B1PRE", "BKEXIST", "FRESH", "CONT0", "B1SIG", "B2SIG", "B3SIG",
    "BREAKOUT", "BKRES", "RTETYPE", "RISK0", "HEALTH", "LOWEXT", "KQUAL",
    "LIQSTD",
)
FORMULA_FUNCTIONS = frozenset({
    "MA", "EMA", "SMA", "REF", "HHV", "LLV", "SUM", "COUNT", "EXIST",
    "EVERY", "HHVBARS", "BARSLAST", "CROSS", "IF", "MAX", "MIN", "ABS",
    "NOT", "CODELIKE", "NAMELIKE", "DRAWNULL", "AND", "OR",
})
BASE_INPUTS = frozenset({"O", "H", "L", "C", "VOL", "V", "AMOUNT", "INDEXC", "MONTH"})
DEF_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(:=|:)\s*(.*?)\s*;\s*$")
IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")
STRING_LITERAL_RE = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"")


def _formula_identifiers(expression: str) -> list[str]:
    """Return formula names outside quoted string literals."""
    return IDENT_RE.findall(STRING_LITERAL_RE.sub("", expression))


class FormulaBindingError(ValueError):
    """The exact formula source or its selected dependency closure is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_definitions(source_text: str) -> list[tuple[str, str, str]]:
    definitions: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for raw_line in source_text.lstrip("\ufeff").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("{") or stripped.startswith("//"):
            continue
        match = DEF_RE.match(raw_line)
        if not match:
            continue
        name, operator, expression = match.groups()
        key = name.upper()
        if key in seen:
            raise FormulaBindingError(f"duplicate formula definition: {key}")
        seen.add(key)
        definitions.append((key, operator, expression))
    return definitions


def exact_formula_source(path: Path, expected_sha256: str) -> tuple[str, str]:
    if not Path(path).is_file():
        raise FormulaBindingError(f"formula source is unavailable: {path.name}")
    actual = sha256_file(Path(path))
    if actual != expected_sha256:
        raise FormulaBindingError(f"formula source SHA drifted: {path.name}")
    return Path(path).read_text(encoding="utf-8-sig"), actual


def extract_closure(source_text: str, targets: Iterable[str], *, allowed_inputs: frozenset[str] = BASE_INPUTS) -> str:
    """Select original definitions plus their transitive formula dependencies."""
    definitions = parse_definitions(source_text)
    by_name = {name: (operator, expression) for name, operator, expression in definitions}
    closure: set[str] = set()

    def include(name: str) -> None:
        key = name.upper()
        if key in closure or key not in by_name:
            return
        closure.add(key)
        expression = by_name[key][1]
        for token in {value.upper() for value in _formula_identifiers(expression)}:
            if token in by_name:
                include(token)

    requested = tuple(value.upper() for value in targets)
    missing = sorted(set(requested) - set(by_name) - set(allowed_inputs))
    if missing:
        raise FormulaBindingError(f"requested formula outputs are undefined: {missing}")
    for target in requested:
        include(target)
    references: set[str] = set()
    for name in closure:
        references.update(value.upper() for value in _formula_identifiers(by_name[name][1]))
    unresolved = sorted(references - closure - FORMULA_FUNCTIONS - allowed_inputs)
    if unresolved:
        raise FormulaBindingError(f"formula closure has unbound names: {unresolved}")
    return "\n".join(f"{name}{operator}{expression};" for name, operator, expression in definitions if name in closure) + "\n"


def verify_shared_core(v192_text: str, v193_text: str) -> None:
    left = {name: expression for name, _, expression in parse_definitions(v192_text)}
    right = {name: expression for name, _, expression in parse_definitions(v193_text)}
    mismatches = sorted(name for name in CORE_OUTPUTS if name in left and name in right and left[name] != right[name])
    missing = sorted(set(CORE_OUTPUTS) - set(right) - BASE_INPUTS)
    if mismatches or missing:
        raise FormulaBindingError(f"shared formula identity failed: mismatches={mismatches}, missing_v193={missing}")


def formula_feature_frame(
    raw_panel: pd.DataFrame,
    *,
    v193_text: str,
    v192_text: str,
    index_close_by_date: pd.Series | None,
    include_v192_trigger: bool = True,
    include_v193_trigger: bool = False,
    include_auditm: bool = False,
) -> pd.DataFrame:
    """Evaluate core features and optional formula triggers on exact-source rows.

    ``index_close_by_date`` is required only when the requested formula closure
    references INDEXC.  MONTH is a deterministic calendar projection.
    """
    verify_shared_core(v192_text, v193_text)
    required = {"date", "code", "open", "high", "low", "close", "volume", "amount"}
    if not required.issubset(raw_panel.columns):
        raise FormulaBindingError(f"formula input fields missing: {sorted(required - set(raw_panel.columns))}")
    frame = raw_panel.copy()
    frame["trade_date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["ts_code"] = frame["code"].astype(str)
    frame = frame.sort_values(["ts_code", "trade_date"], kind="stable").reset_index(drop=True)
    frame["MONTH"] = frame["trade_date"].dt.month.astype(np.float64)
    if index_close_by_date is not None:
        index_map = pd.Series(index_close_by_date.to_numpy(dtype=np.float64),
                              index=pd.to_datetime(index_close_by_date.index).normalize())
        if index_map.index.duplicated().any():
            raise FormulaBindingError("index close has duplicate dates")
        frame["INDEXC"] = frame["trade_date"].map(index_map).astype(np.float64)
    targets = list(CORE_OUTPUTS)
    if include_v192_trigger:
        targets.append("XG")
    if include_v193_trigger:
        targets.append("XG")
    v193_targets = list(CORE_OUTPUTS) + (["XG"] if include_v193_trigger else []) + (["AUDITM"] if include_auditm else [])
    v192_targets = ["XG"] if include_v192_trigger else []
    input_fields = {
        "INDEXC": frame["INDEXC"].to_numpy(dtype=np.float64) if "INDEXC" in frame else None,
        "MONTH": frame["MONTH"].to_numpy(dtype=np.float64),
    }
    outputs: dict[str, pd.Series] = {}
    if v193_targets:
        text = extract_closure(v193_text, v193_targets)
        outputs.update(_evaluate_closure(text, frame, input_fields, "V193"))
    if v192_targets:
        text = extract_closure(v192_text, v192_targets)
        v192 = _evaluate_closure(text, frame, input_fields, "V192")
        outputs["V192_XG"] = v192["XG"]
    if include_v193_trigger:
        outputs["V193_XG"] = outputs.pop("XG")
    result = frame[["trade_date", "ts_code"]].rename(columns={"trade_date": "date", "ts_code": "code"}).copy()
    for name in CORE_OUTPUTS + (("AUDITM",) if include_auditm else ()):
        if name not in outputs:
            raise FormulaBindingError(f"formula engine did not produce {name}")
        result[name] = outputs[name].to_numpy(dtype=np.float64)
    for name in ("V192_XG", "V193_XG"):
        if name in outputs:
            result[name] = outputs[name].to_numpy(dtype=np.float64)
    return result


def _evaluate_closure(text: str, frame: pd.DataFrame, inputs: Mapping[str, np.ndarray | None], label: str) -> dict[str, pd.Series]:
    injected: dict[str, np.ndarray] = {}
    for name, values in inputs.items():
        if values is None:
            continue
        injected[name] = np.asarray(values, dtype=np.float64)
    if "INDEXC" not in injected and "INDEXC" in {value.upper() for value in IDENT_RE.findall(text)}:
        raise FormulaBindingError(f"{label} closure requires unavailable INDEXC input")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=True) as handle:
        handle.write(text)
        handle.flush()
        raw = frame.rename(columns={"trade_date": "trade_date", "ts_code": "ts_code", "volume": "volume"})
        return evaluate_tdx_formula_on_df(Path(handle.name), raw, injected_inputs=injected)

