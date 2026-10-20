"""Registered, saved-input-only G1 descriptive audit lifecycle."""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[_key] = '4'
from pathlib import Path
import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time

import pandas as pd
from g1_attribution_core import audit, COMPLETE, PARTIAL
from g1_attribution_registry import ROOT, BATCH, FLAGS, sha, write, append, sync

HERE = Path(__file__).resolve().parent


def freeze():
    """Metadata operation only: no input numeric values or evaluator execution."""
    sources = [HERE/'g1_attribution_core.py', HERE/'g1_attribution_registry.py',
               HERE/'g1_attribution_runner.py', HERE/'tests/test_g1_attribution.py']
    hashes = {str(p): sha(p) for p in sources}
    composite = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    contract = json.loads((BATCH/'CONTRACT.json').read_text())
    if (BATCH/'FROZEN_IDENTITY.json').exists():
        raise ValueError('freeze already exists; do not overwrite execution identity')
    proposal = BATCH/'DESIGN_FREEZE_ACCEPTED'
    proposal.mkdir(exist_ok=False)
    for name in ['CONTRACT.json', 'CONFIG.json', 'INPUT_BINDING.json', 'FREEZE_READY.json']:
        (proposal/name).write_bytes((BATCH/name).read_bytes())
    contract.update(status='DESIGN_ACCEPTED_SOURCE_FROZEN_EXECUTION_NOT_STARTED',
                    CODE_SHA=composite, CODE_SHA_REASON='SOURCE_COMPOSITE_CANONICAL_PATH_SHA256_MAP',
                    DESIGN_ACCEPTANCE_FREEZE_READY_SHA='e85c1a840ef33939ab07b8cdc031ba748d2e9f0f4f150d90f49d62ded024ca89',
                    ADDITIONAL_STAGE_GATE='Independent factor Reviewer canary PASS required before one formal audit')
    write(BATCH/'CONTRACT.json', contract)
    controls = {str(BATCH/n): sha(BATCH/n) for n in
                ['CONTRACT.json', 'CONFIG.json', 'INPUT_BINDING.json', 'DESIGN_REVIEW_TRANSCRIPTION.json']}
    identity = dict(code_by_file=hashes, code_composite_sha256=composite, controls=controls,
                    baseline_commit='c1be26aead7b1a210d2f48fac5cdb011699fe63e',
                    frozen_before_any_numerical_execution=True, **FLAGS)
    write(BATCH/'FROZEN_IDENTITY.json', identity)
    return identity


def verify_code():
    identity = json.loads((BATCH/'FROZEN_IDENTITY.json').read_text())
    for path, expected in {**identity['code_by_file'], **identity['controls']}.items():
        if sha(path) != expected:
            raise ValueError('source/control drift: '+path)


def input_binding():
    return json.loads((BATCH/'INPUT_BINDING.json').read_text())


def verify_inputs(binding):
    for key, item in binding['files'].items():
        path = Path(item['path'])
        stat = path.stat()
        if stat.st_size != item['bytes'] or stat.st_mtime_ns != item['mtime_ns']:
            raise ValueError('frozen narrow input stat drift: '+key)
        if sha(path) != item['sha256']:
            raise ValueError('frozen narrow input SHA drift: '+key)


def resources():
    result = {'process_maxrss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    for name, command in [('load', ['sysctl', '-n', 'vm.loadavg']),
                          ('swap', ['sysctl', '-n', 'vm.swapusage']),
                          ('memory_pressure', ['memory_pressure', '-Q'])]:
        p = subprocess.run(command, capture_output=True, text=True)
        result[name] = p.stdout.strip() if p.returncode == 0 else 'UNAVAILABLE'
    return result


def lifecycle(execution, stage, operation, truth):
    verify_code()
    out = BATCH/execution
    out.mkdir(exist_ok=False)
    receipt = out/'RUN_RECEIPT.json'
    write(receipt, dict(status='REGISTER_BEFORE_EXECUTION', **FLAGS))
    append(execution, 'RUN_STARTED', stage, receipt, truth=truth,
           metrics={'fit_calls': 0, 'predict_calls': 0, 'model_loads': 0})
    sync()
    start = time.monotonic()
    before = resources()
    try:
        result = operation(out)
        after = resources()
        if after['process_maxrss_bytes'] > 4*1024**3:
            raise RuntimeError('RESOURCE_RSS_LIMIT_STOP')
        write(receipt, dict(status='COMPLETED', elapsed_seconds=time.monotonic()-start,
            resources_before=before, resources_after=after, fit_calls=0, predict_calls=0,
            model_loads=0, ci_calls=0, **result, **FLAGS))
        append(execution, 'TERMINAL', stage, receipt, decision=result.get('decision', 'COMPLETED'),
               truth=truth, metrics=result)
        sync()
        return result
    except BaseException as error:
        write(receipt, dict(status='CANNOT_CONFIRM', error=repr(error),
              elapsed_seconds=time.monotonic()-start, fit_calls=0, predict_calls=0,
              model_loads=0, **FLAGS))
        append(execution, 'TERMINAL', stage, receipt, decision='CANNOT_CONFIRM', truth=truth,
               metrics={'fit_calls': 0, 'predict_calls': 0, 'model_loads': 0, 'error': repr(error)})
        sync()
        raise


def tests(execution):
    def operation(out):
        command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                   str(HERE/'tests/test_g1_attribution.py')]
        p = subprocess.run(command, capture_output=True, text=True,
                           env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        (out/'stdout.txt').write_text(p.stdout)
        (out/'stderr.txt').write_text(p.stderr)
        if p.returncode:
            raise RuntimeError(p.stdout+p.stderr)
        return dict(command=command, stdout_sha256=sha(out/'stdout.txt'),
                    decision='FIXTURES_READY_FOR_REVIEW', real_rows_read=False)
    return lifecycle(execution, 'SYNTHETIC_TESTS', operation, 'SYNTHETIC_FIXTURES_ONLY')


def read_inputs(binding, canary):
    files = binding['files']
    parquet = lambda key: pd.read_parquet(files[key]['path'])
    csv = lambda key: pd.read_csv(files[key]['path'], dtype={'code': str},
                                  float_precision='round_trip')
    dates = parquet('validation_dates')
    dates['date'] = pd.to_datetime(dates.date)
    chosen = None
    if canary:
        chosen = dates.sort_values('calendar_index').groupby('fold', sort=False).head(1).date
        dates = dates.loc[dates.date.isin(chosen)].copy()
    def narrowed(frame):
        frame['date'] = pd.to_datetime(frame.date)
        return frame if chosen is None else frame.loc[frame.date.isin(chosen)].copy()
    # Arrow filters keep the tiny canary's actual row reads bounded to its six dates.
    def narrow_parquet(key):
        if chosen is None:
            return parquet(key)
        filters = [('date', 'in', list(chosen))]
        return pd.read_parquet(files[key]['path'], filters=filters)
    statuses = pd.concat([narrow_parquet('band_status_'+fold)
                          for fold in [*COMPLETE, PARTIAL]], ignore_index=True)
    return dict(rows=narrowed(narrow_parquet('labeled_top50')),
        keys=narrowed(narrow_parquet('frozen_keys')), dates=dates,
        statuses=narrowed(statuses), old_bands=narrowed(csv('old_band')),
        old_daily=narrowed(csv('old_daily')), calendar=parquet('calendar'),
        roster=csv('industry_roster'), lookup=csv('prior_lookup'))


def save_outputs(out, tables, summary):
    for name, table in tables.items():
        table.to_csv(out/(name.upper()+'.csv'), index=False, float_format='%.17g')
    write(out/'SUMMARY.json', summary)
    write(out/'OUTPUT_MANIFEST.json', dict(files={str(p): sha(p) for p in out.iterdir()
        if p.is_file() and p.name != 'RUN_RECEIPT.json'}, input_binding_sha256=sha(BATCH/'INPUT_BINDING.json'),
        source_identity_sha256=sha(BATCH/'FROZEN_IDENTITY.json'), **FLAGS))


def run(canary, execution):
    if not canary:
        approval = json.loads((BATCH/'CANARY_INDEPENDENT_REVIEW.json').read_text())
        if approval.get('verdict') != 'PASS_CANARY_FOR_FORMAL':
            raise ValueError('independent canary PASS gate unavailable')
        if approval.get('frozen_identity_sha256') != sha(BATCH/'FROZEN_IDENTITY.json'):
            raise ValueError('canary approval identity drift')
        if approval.get('canary_manifest_sha256') != sha(BATCH/'G1_ATTRIBUTION_CANARY_01/OUTPUT_MANIFEST.json'):
            raise ValueError('canary approval output drift')
    def operation(out):
        binding = input_binding()
        verify_inputs(binding)
        data = read_inputs(binding, canary)
        tables, summary = audit(**data, expected_primary=None if canary else (601, 547, 54),
                               expected_point=None if canary else .002975649093957072)
        summary['canary_not_scientific_estimate'] = canary
        summary['rows_read_scope'] = len(data['rows'])
        summary['calendar_selection'] = 'FIRST_ORIGINAL_DATE_PER_FOLD' if canary else 'ALL_ORIGINAL_DATES'
        save_outputs(out, tables, summary)
        return dict(decision='CANARY_READY_FOR_INDEPENDENT_REVIEW' if canary else 'FORMAL_READY_FOR_INDEPENDENT_REVIEW',
                    summary_sha256=sha(out/'SUMMARY.json'), output_manifest_sha256=sha(out/'OUTPUT_MANIFEST.json'),
                    rows=len(data['rows']), **{k: summary[k] for k in
                    ['primary_calendar_dates', 'primary_valid_dates', 'primary_invalid_dates']})
    return lifecycle(execution, 'CANARY' if canary else 'FORMAL_SAVED_G1_ATTRIBUTION', operation,
                     'EXISTING_SAVED_H2_LABELS_AND_PREDICTIONS_NO_NEW_TRUTH_DECODE')


def review_start(execution, stage):
    verify_code()
    out = BATCH/execution
    out.mkdir(exist_ok=False)
    receipt = out/'RUN_RECEIPT.json'
    write(receipt, dict(status='AWAITING_INDEPENDENT_REVIEW', fit_calls=0,
                        predict_calls=0, model_loads=0, **FLAGS))
    append(execution, 'RUN_STARTED', stage, receipt,
           truth='INDEPENDENT_FROZEN_OUTPUT_AND_SAVED_INPUT_READBACK', metrics={'fit_calls': 0})
    return sync()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'tests', 'canary', 'formal', 'review-start'])
    parser.add_argument('--execution')
    parser.add_argument('--review-stage', default='INDEPENDENT_CANARY_REVIEW')
    args = parser.parse_args()
    if args.stage == 'freeze': result = freeze()
    elif args.stage == 'tests': result = tests(args.execution or 'G1_ATTRIBUTION_FIXTURES_01')
    elif args.stage == 'canary': result = run(True, args.execution or 'G1_ATTRIBUTION_CANARY_01')
    elif args.stage == 'formal': result = run(False, args.execution or 'G1_ATTRIBUTION_FORMAL_01')
    else: result = review_start(args.execution, args.review_stage)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
