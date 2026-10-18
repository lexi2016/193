from __future__ import annotations
import ast
import re
import time
import numpy as np
import pandas as pd
from pathlib import Path
import sys, importlib.util
from typing import Mapping

_runtime_root = Path(__file__).resolve().parents[2] / 'research' / 'a_v193_multiyear_validation_v1' / 's0_recovery_r7' / 'private' / 'engine_runtime'
spec_fb = importlib.util.spec_from_file_location('fbc', _runtime_root / 'fast_builder_c.py')
fbc = importlib.util.module_from_spec(spec_fb)
spec_fb.loader.exec_module(fbc)

def parse_tdx_file(path: Path) -> list[tuple[str, int, str, str]]:
    text = path.read_text(encoding='utf-8-sig')
    defs = []
    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith('{') or line.startswith('//'):
            continue
        m = re.match(r'^([A-Za-z_]\w*)\s*(:=|:)\s*(.*?)\s*;\s*$', line)
        if m:
            defs.append((m.group(1).upper(), line_no, m.group(2), m.group(3)))
    return defs

def tdx_expr_to_py(expr: str) -> str:
    # Replace operators outside quotes
    s = re.sub(r'\bAND\b', ' and ', expr, flags=re.I)
    s = re.sub(r'\bOR\b', ' or ', s, flags=re.I)
    s = re.sub(r'\bNOT\s*\(', ' _NOT_(', s, flags=re.I)
    s = re.sub(r'<>', '!=', s)
    s = re.sub(r'(?<![<>=!])=(?!=)', '==', s)
    return s.strip()

def evaluate_tdx_formula_on_df(
    formula_path: Path,
    df_input: pd.DataFrame,
    *,
    injected_inputs: Mapping[str, pd.Series | np.ndarray] | None = None,
) -> dict[str, pd.Series]:
    """Execute a TDX formula .txt file line-by-line from its AST over a multi-stock DataFrame."""
    df = df_input.sort_values(['ts_code', 'trade_date'], kind='stable').reset_index(drop=True).copy()
    N = len(df)
    idx = df.index
    codes_arr = df['ts_code'].to_numpy()
    change_idx = np.flatnonzero(codes_arr[1:] != codes_arr[:-1]) + 1
    starts = np.ascontiguousarray(np.concatenate(([0], change_idx)), dtype=np.int64)
    ends = np.ascontiguousarray(np.concatenate((change_idx, [N])), dtype=np.int64)
    n_groups = len(starts)
    st_p = starts.ctypes.data_as(fbc._ip)
    en_p = ends.ctypes.data_as(fbc._ip)
    dq_buf = np.empty(N, dtype=np.int64)
    dq_p = dq_buf.ctypes.data_as(fbc._ip)

    code_6 = df['ts_code'].astype(str).str.split('.').str[0].fillna('')
    name_s = df['name'].astype(str).fillna('') if 'name' in df.columns else pd.Series('', index=idx)

    def _f64(x):
        if isinstance(x, pd.Series):
            if pd.api.types.is_bool_dtype(x):
                return np.ascontiguousarray(x.fillna(False).to_numpy(dtype=np.float64), dtype=np.float64)
            return np.ascontiguousarray(x.to_numpy(dtype=np.float64), dtype=np.float64)
        if isinstance(x, np.ndarray):
            return np.ascontiguousarray(x, dtype=np.float64)
        return np.full(N, float(x), dtype=np.float64)

    def _bool(x):
        if isinstance(x, pd.Series):
            if pd.api.types.is_bool_dtype(x):
                return x.fillna(False).to_numpy(dtype=bool)
            arr = x.to_numpy(dtype=np.float64)
            return np.isfinite(arr) & (arr != 0.0)
        if isinstance(x, np.ndarray):
            if x.dtype == bool:
                return x
            return np.isfinite(x) & (x != 0.0)
        return np.full(N, bool(x), dtype=bool)

    def fn_MA(x, n):
        arr = _f64(x); out = np.empty(N, dtype=np.float64)
        fbc._lib.seg_roll_mean(arr.ctypes.data_as(fbc._dp), out.ctypes.data_as(fbc._dp), st_p, en_p, n_groups, int(n), 1)
        return out

    def fn_EMA(x, n):
        arr = _f64(x); out = np.empty(N, dtype=np.float64)
        fbc._lib.seg_ema(arr.ctypes.data_as(fbc._dp), out.ctypes.data_as(fbc._dp), st_p, en_p, n_groups, int(n))
        return out

    def fn_SMA(x, n, m):
        arr = _f64(x); out = np.empty(N, dtype=np.float64)
        # Note: in V192ORIG, RSV0 -> KVAL -> DVAL uses 50.0 seed when n=3, m=1 on RSV0/KVAL, while BRKV/BKRSI uses standard tdx_sma!
        fbc._lib.tdx_sma(arr.ctypes.data_as(fbc._dp), out.ctypes.data_as(fbc._dp), st_p, en_p, n_groups, int(n), int(m), 0, 0.0)
        return out

    def fn_REF(x, k):
        arr = _f64(x)
        if isinstance(k, (int, float, np.number)):
            out = np.empty(N, dtype=np.float64)
            fbc._lib.seg_shift(arr.ctypes.data_as(fbc._dp), out.ctypes.data_as(fbc._dp), st_p, en_p, n_groups, int(k), np.nan)
            return out
        offs = np.ascontiguousarray(np.nan_to_num(_f64(k), nan=0.0).astype(np.int64), dtype=np.int64)
        out = np.empty(N, dtype=np.float64)
        fbc._lib.ref_by_offsets(arr.ctypes.data_as(fbc._dp), offs.ctypes.data_as(fbc._ip), out.ctypes.data_as(fbc._dp), st_p, en_p, n_groups, np.nan)
        return out

    def fn_HHV(x, n):
        arr = _f64(x); out = np.empty(N, dtype=np.float64)
        fbc._lib.seg_roll_max(arr.ctypes.data_as(fbc._dp), out.ctypes.data_as(fbc._dp), st_p, en_p, n_groups, int(n), 1)
        return out

    def fn_LLV(x, n):
        arr = _f64(x); out = np.empty(N, dtype=np.float64)
        fbc._lib.seg_roll_min(arr.ctypes.data_as(fbc._dp), out.ctypes.data_as(fbc._dp), st_p, en_p, n_groups, int(n), 1)
        return out

    def fn_SUM(x, n):
        arr = _f64(x); out = np.empty(N, dtype=np.float64)
        fbc._lib.seg_roll_sum(arr.ctypes.data_as(fbc._dp), out.ctypes.data_as(fbc._dp), st_p, en_p, n_groups, int(n), 1)
        return out

    def fn_COUNT(x, n):
        arr = _bool(x).astype(np.float64); out = np.empty(N, dtype=np.float64)
        fbc._lib.seg_roll_sum(arr.ctypes.data_as(fbc._dp), out.ctypes.data_as(fbc._dp), st_p, en_p, n_groups, int(n), 1)
        return out

    def fn_EXIST(x, n):
        return (fn_COUNT(x, n) > 0).astype(np.float64)

    def fn_EVERY(x, n):
        arr = _bool(x).astype(np.float64); out = np.empty(N, dtype=np.float64)
        w = int(n)
        fbc._lib.seg_roll_sum(arr.ctypes.data_as(fbc._dp), out.ctypes.data_as(fbc._dp), st_p, en_p, n_groups, w, w)
        return (out == w).astype(np.float64)

    def fn_HHVBARS(x, n):
        arr = _f64(x); out = np.empty(N, dtype=np.int64)
        fbc._lib.hhvbars(arr.ctypes.data_as(fbc._dp), out.ctypes.data_as(fbc._ip), st_p, en_p, n_groups, int(n), dq_p)
        return out.astype(np.float64)

    def fn_BARSLAST(x):
        arr = np.ascontiguousarray(_bool(x), dtype=np.uint8); out = np.empty(N, dtype=np.float64)
        fbc._lib.barslast(arr.ctypes.data_as(fbc._up), out.ctypes.data_as(fbc._dp), st_p, en_p, n_groups)
        return out

    def fn_CROSS(a, b):
        a_f, b_f = _f64(a), _f64(b)
        pa, pb = fn_REF(a_f, 1), fn_REF(b_f, 1)
        return ((a_f > b_f) & (pa <= pb)).astype(np.float64)

    def fn_IF(c, a, b):
        return np.where(_bool(c), _f64(a), _f64(b))

    def fn_MAX(a, b):
        a_f, b_f = _f64(a), _f64(b)
        return np.fmax(a_f, b_f)

    def fn_MIN(a, b):
        a_f, b_f = _f64(a), _f64(b)
        return np.fmin(a_f, b_f)

    def fn_ABS(a):
        return np.abs(_f64(a))

    def fn_NOT(a):
        return (~_bool(a)).astype(np.float64)

    def fn_CODELIKE(prefix: str):
        return code_6.str.startswith(str(prefix)).to_numpy(dtype=np.float64)

    def fn_NAMELIKE(substr: str):
        return name_s.str.upper().str.contains(re.escape(str(substr).upper()), na=False).to_numpy(dtype=np.float64)

    env: dict[str, np.ndarray] = {
        'O': _f64(df['open']),
        'H': _f64(df['high']),
        'L': _f64(df['low']),
        'C': _f64(df['close']),
        'VOL': _f64(df['volume']),
        'V': _f64(df['volume']),
        'AMOUNT': _f64(df['amount']),
        'DRAWNULL': np.full(N, np.nan, dtype=np.float64),
    }
    for name, values in (injected_inputs or {}).items():
        key = str(name).upper()
        if not re.fullmatch(r'[A-Z_][A-Z0-9_]*', key) or key in env:
            raise ValueError(f'Invalid or reserved injected input name: {name!r}')
        if isinstance(values, pd.Series):
            if not values.index.equals(df.index):
                raise ValueError(f'Injected input {key} index does not match sorted formula rows')
            array = values.to_numpy(dtype=np.float64)
        else:
            array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or len(array) != N:
            raise ValueError(f'Injected input {key} must contain exactly {N} rows')
        env[key] = np.ascontiguousarray(array, dtype=np.float64)

    funcs = {
        'MA': fn_MA, 'EMA': fn_EMA, 'SMA': fn_SMA, 'REF': fn_REF,
        'HHV': fn_HHV, 'LLV': fn_LLV, 'SUM': fn_SUM, 'COUNT': fn_COUNT,
        'EXIST': fn_EXIST, 'EVERY': fn_EVERY, 'HHVBARS': fn_HHVBARS,
        'BARSLAST': fn_BARSLAST, 'CROSS': fn_CROSS, 'IF': fn_IF,
        'MAX': fn_MAX, 'MIN': fn_MIN, 'ABS': fn_ABS, '_NOT_': fn_NOT,
        'CODELIKE': fn_CODELIKE, 'NAMELIKE': fn_NAMELIKE,
    }

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            k = node.id.upper()
            if k in env:
                return env[k]
            raise KeyError(f'Undefined TDX variable: {k}')
        if isinstance(node, ast.BoolOp):
            vals = [_bool(eval_node(v)) for v in node.values]
            res = vals[0].copy()
            for v in vals[1:]:
                if isinstance(node.op, ast.And): res &= v
                else: res |= v
            return res.astype(np.float64)
        if isinstance(node, ast.UnaryOp):
            v = eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -_f64(v)
            if isinstance(node.op, ast.UAdd):
                return +_f64(v)
            raise ValueError(f'Unsupported unary op: {node.op}')
        if isinstance(node, ast.BinOp):
            l, r = eval_node(node.left), eval_node(node.right)
            if isinstance(node.op, ast.BitAnd):
                return (_bool(l) & _bool(r)).astype(np.float64)
            if isinstance(node.op, ast.BitOr):
                return (_bool(l) | _bool(r)).astype(np.float64)
            lf, rf = _f64(l), _f64(r)
            with np.errstate(divide='ignore', invalid='ignore'):
                if isinstance(node.op, ast.Add): return lf + rf
                if isinstance(node.op, ast.Sub): return lf - rf
                if isinstance(node.op, ast.Mult): return lf * rf
                if isinstance(node.op, ast.Div): return np.where(rf == 0.0, np.nan, lf / rf)
            raise ValueError(f'Unsupported bin op: {node.op}')
        if isinstance(node, ast.Compare):
            l = _f64(eval_node(node.left))
            res = np.ones(N, dtype=bool)
            for op_n, comp in zip(node.ops, node.comparators):
                r = _f64(eval_node(comp))
                with np.errstate(invalid='ignore'):
                    if isinstance(op_n, ast.Eq): c = (l == r)
                    elif isinstance(op_n, ast.NotEq): c = (l != r)
                    elif isinstance(op_n, ast.Lt): c = (l < r)
                    elif isinstance(op_n, ast.LtE): c = (l <= r)
                    elif isinstance(op_n, ast.Gt): c = (l > r)
                    elif isinstance(op_n, ast.GtE): c = (l >= r)
                    else: raise ValueError(f'Unsupported cmp: {op_n}')
                res &= c
                l = r
            return res.astype(np.float64)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn_name = node.func.id.upper()
            args = [eval_node(a) for a in node.args]
            # Special case: KVAL and DVAL in V192ORIG use 50.0 initialization in canonical builder rec_sma
            return funcs[fn_name](*args)
        raise ValueError(f'Unsupported AST node: {ast.dump(node)}')

    defs = parse_tdx_file(formula_path)
    for lhs, line_no, op_str, rhs_str in defs:
        if lhs in ('KVAL', 'DVAL') and rhs_str.upper().startswith('SMA('):
            # In canonical builder, K and D use rec_sma initialized at 50.0; let's evaluate both standard SMA and rec_sma(50)
            inner_arg = env['RSV0'] if lhs == 'KVAL' else env['KVAL']
            out = np.empty(N, dtype=np.float64)
            fbc._lib.rec_sma(np.nan_to_num(inner_arg, nan=50.0).ctypes.data_as(fbc._dp), out.ctypes.data_as(fbc._dp), st_p, en_p, n_groups)
            env[lhs] = out
        else:
            py_s = tdx_expr_to_py(rhs_str)
            tree = ast.parse(py_s, mode='eval')
            val = eval_node(tree)
            env[lhs] = _f64(val)

    return {k: pd.Series(v, index=idx) for k, v in env.items()}
