"""Mask-only calibration -> immutable q lock -> validation -> one real CI."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='4'
from pathlib import Path
import argparse,fcntl,json,resource,subprocess,sys,time,re
import numpy as np
import pandas as pd
from g1_qstar_core import *
from g1_qstar_state import save_checkpoint,load_checkpoint,sha
from g1_qstar_registry import BATCH,ROOT,FLAGS,append,sync,write
HERE=Path(__file__).resolve().parent


def resources():
    p=subprocess.run(['sysctl','-n','vm.swapusage'],capture_output=True,text=True);m=re.search(r'used = ([0-9.]+)M',p.stdout)
    return dict(peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),swap_used_mib=None if m is None else float(m.group(1)))


def resource_guard():
    r=resources()
    if r['peak_rss_bytes']>8*1024**3:raise RuntimeError('RESOURCE_RSS_LIMIT_STOP')
    return r


def verify():
    i=json.loads((BATCH/'FROZEN_IDENTITY.json').read_text())
    for p,h in {**i['code_by_file'],**i['controls']}.items():
        if sha(p)!=h:raise ValueError('frozen identity drift: '+p)
    return json.loads((BATCH/'INPUT_BINDING.json').read_text())


def basic(execution,stage,op,truth,start_metrics=None):
    out=BATCH/execution
    if out.exists():raise RuntimeError('execution exists; preserve completed/failure evidence')
    out.mkdir();receipt=out/'RUN_RECEIPT.json';write(receipt,dict(status='REGISTER_BEFORE_EXECUTION'));append(execution,'RUN_STARTED',stage,receipt,truth=truth,metrics=start_metrics);before=resources();start=time.monotonic()
    try:
        result=op(out);after=resource_guard();write(receipt,dict(status='COMPLETED',elapsed_seconds=time.monotonic()-start,resources_before=before,resources_after=after,swap_delta_mib=None if before['swap_used_mib'] is None else after['swap_used_mib']-before['swap_used_mib'],fit_calls=0,predict_calls=0,model_loads=0,**result,**FLAGS));append(execution,'TERMINAL',stage,receipt,decision=result.get('decision','COMPLETED'),truth=truth,metrics=result);sync();return result
    except Exception as e:
        write(receipt,dict(status='CANNOT_CONFIRM',error=str(e),fit_calls=0,predict_calls=0,model_loads=0,**FLAGS));append(execution,'TERMINAL',stage,receipt,decision='CANNOT_CONFIRM',truth=truth);sync();raise


def tests():
    def op(out):
        p=subprocess.run([sys.executable,'-m','pytest','-q',str(HERE/'test_g1_qstar.py')],capture_output=True,text=True);(out/'stdout.txt').write_text(p.stdout);(out/'stderr.txt').write_text(p.stderr)
        if p.returncode:raise RuntimeError(p.stdout+p.stderr)
        return dict(command=p.args,stdout_sha256=sha(out/'stdout.txt'),real_x_read=False)
    return basic('G1_QSTAR_TEST_01','SYNTHETIC_TESTS',op,'NEW_QSTAR_FIXTURES_ONLY_OLD_HAC_PASS_REUSED')


def mask_stage(binding):
    def op(out):
        for key in ['daily','original_metrics','accepted_manifest','accepted_review']:
            if sha(binding[key]['path'])!=binding[key]['sha256']:raise ValueError('accepted input SHA drift '+key)
        frame=pd.read_csv(binding['daily']['path'],usecols=['date','calendar_index','fold','valid']);primary,selected,m=build_mask(frame);primary.to_csv(out/'PRIMARY_MASK.csv',index=False);selected.to_csv(out/'SELECTED_MASK.csv',index=False)
        report=dict(N=m['N'],full_calendar_length=m['full_length'],original_calendar=[648,1248],n=list(COUNTS),D=m['D'],c=m['c'],partial_excluded=True,gaps_preserved=True,real_x_read=False);write(out/'MASK_DIAGNOSTICS.json',report);return dict(decision='MASK_READY',**report)
    return basic('G1_QSTAR_MASK_01','MASK_ONLY_CHECK',op,'ONLY_DATE_CALENDAR_INDEX_FOLD_VALID_RAW_SHA_ALLOWED_NO_REAL_X')


def load_mask():return build_mask(pd.read_csv(BATCH/'G1_QSTAR_MASK_01/PRIMARY_MASK.csv',usecols=['date','calendar_index','fold','valid']))[2]


def canary(chunk):
    def op(out):
        m=load_mask();start=time.monotonic()
        for scenario in [1,6,8]:Y,se=simulate_segment(rng_for(0,scenario),128,m,scenario,chunk);assert len(Y)==128 and np.isfinite(Y).all()
        seconds=time.monotonic()-start;report=dict(scenarios=[1,6,8],draws_each=128,draws_per_second=384/seconds,seconds=seconds,chunk=chunk,canonical_math_tile=32,workers=1,worker2='NOT_ACTIVATED_WITHOUT_MEASURED_GAIN; bounded serial throughput sufficient',formal_streams_reset=True,real_x_read=False);write(out/'THROUGHPUT.json',report);return dict(decision='ENGINEERING_CANARY_ONLY',report_sha256=sha(out/'THROUGHPUT.json'),**report)
    return basic('G1_QSTAR_CANARY_01','CANARY',op,'SYNTHETIC_ONLY_ON_MASK_NO_REAL_X')


def science_identity(phase,scenario,total,lock_sha):
    i=json.loads((BATCH/'FROZEN_IDENTITY.json').read_text());return dict(phase=phase,scenario_id=scenario,total=total,entropy=19320261021,spawn_key=[phase,scenario],config_sha256=sha(BATCH/'CONFIG.json'),stat_core_sha256=i['code_by_file'][str(HERE/'g1_qstar_core.py')],state_code_sha256=i['code_by_file'][str(HERE/'g1_qstar_state.py')],frozen_hac_sha256=i['code_by_file'][str(HAC_SOURCE)],mask_sha256=sha(BATCH/'G1_QSTAR_MASK_01/PRIMARY_MASK.csv'),qstar_lock_sha256=lock_sha)


def current_lock():
    lockpath=BATCH/'QSTAR_LOCK.json'
    if not lockpath.exists():raise ValueError('qstar lock absent')
    expected=None
    with (ROOT/'contracts/EXPERIMENT_REGISTRY.jsonl').open() as f:
        for line in f:
            e=json.loads(line)
            if e.get('execution_id')=='G1_QSTAR_LOCK_01' and e.get('event_type')=='TERMINAL':expected=e['deterministic_metrics'].get('qstar_lock_sha256')
    if expected is None or sha(lockpath)!=expected:raise ValueError('qstar lock SHA/registry identity mismatch')
    lock=json.loads(lockpath.read_text())
    if lock['config_sha256']!=sha(BATCH/'CONFIG.json') or lock['core_sha256']!=sha(HERE/'g1_qstar_core.py') or lock['qstar']<Z95:raise ValueError('locked q/core/config drift')
    for item in lock['calibration_bindings'].values():
        for path,h in item.items():
            if sha(path)!=h:raise ValueError('locked calibration artifact drift')
    return lock,expected


def close_interrupted(attempt):
    for cp in sorted(attempt.glob('CAPTURE_*/RUN_RECEIPT.json')):
        d=json.loads(cp.read_text())
        if d.get('status') not in ['COMPLETED','CANNOT_CONFIRM','INTERRUPTED']:
            execution=d['execution'];write(cp,dict(status='INTERRUPTED',reason='Owner file lock released after process interruption; checkpoint not claimed successful',**FLAGS));append(execution,'TERMINAL','CHECKPOINT_CAPTURE',cp,decision='INTERRUPTED_CHECKPOINT_CAPTURE',truth='RECOVERY_METADATA_ONLY')
    receipt=attempt/'RUN_RECEIPT.json';d=json.loads(receipt.read_text())
    if d.get('status') not in ['COMPLETED','CANNOT_CONFIRM','INTERRUPTED']:
        write(receipt,{**d,'status':'INTERRUPTED','reason':'Previous process no longer holds exclusive case lock; resume from hashed checkpoint only'});append(d['execution'],'TERMINAL','SYNTHETIC_CASE',receipt,decision='INTERRUPTED_CASE_RESUMABLE',truth='RECOVERY_METADATA_ONLY')


def run_case(phase,scenario,chunk,mask,lock=None,lock_sha=None):
    total=50000 if phase==0 else 10000;step=10000 if phase==0 else 5000;case=BATCH/f'{"CAL" if phase==0 else "VAL"}_CASE_{scenario:02d}';case.mkdir(exist_ok=True);identity=science_identity(phase,scenario,total,lock_sha)
    owner=(case/'OWNER.lock').open('a+');fcntl.flock(owner.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        completed=case/'COMPLETED.json'
        if completed.exists():
            done=json.loads(completed.read_text())
            if done['identity']!=identity or sha(done['result_path'])!=done['result_sha256'] or sha(done['receipt_path'])!=done['receipt_sha256']:raise ValueError('successful case reuse identity/hash conflict')
            return json.loads(Path(done['result_path']).read_text())
        attempts=sorted(case.glob('ATTEMPT_*'))
        for old in attempts:close_interrupted(old)
        attempt=case/f'ATTEMPT_{len(attempts)+1:02d}';attempt.mkdir();execution=f'G1_QSTAR_{"CAL" if phase==0 else "VAL"}_{scenario:02d}_A{len(attempts)+1:02d}';receipt=attempt/'RUN_RECEIPT.json';write(receipt,dict(status='REGISTER_BEFORE_EXECUTION',execution=execution));start_metrics=dict(phase=phase,scenario_id=scenario,qstar_lock_sha256=lock_sha,qstar=None if lock is None else lock['qstar']);append(execution,'RUN_STARTED','SYNTHETIC_CASE',receipt,truth='SIMULATED_ONLY_NO_REAL_X',metrics=start_metrics);write(receipt,dict(status='RUNNING',execution=execution,identity=identity,owner_pid=os.getpid()));before=resources();start=time.monotonic()
        try:
            state=load_checkpoint(case,identity);rng=rng_for(phase,scenario)
            if state is None:Y=np.empty(0);se=np.empty(0);restored=0
            else:
                st,Y,se=state;rng.bit_generator.state=st['rng_state'];restored=len(Y)
            if len(Y)>total or len(Y)%step:raise ValueError('checkpoint completed count not at frozen segment boundary')
            while len(Y)<total:
                if phase==1 and sha(BATCH/'QSTAR_LOCK.json')!=lock_sha:raise ValueError('q changed during validation')
                y,s=simulate_segment(rng,min(step,total-len(Y)),mask,scenario,chunk);Y=np.r_[Y,y];se=np.r_[se,s];done=len(Y);capture=execution+'_CP'+str(done);cap=attempt/('CAPTURE_'+str(done));cap.mkdir();cr=cap/'RUN_RECEIPT.json';write(cr,dict(status='REGISTER_BEFORE_EXECUTION',execution=capture));append(capture,'RUN_STARTED','CHECKPOINT_CAPTURE',cr,truth='PERSIST_CURRENT_SYNTHETIC_RNG_AND_ARRAYS_ONLY')
                try:
                    pointer=save_checkpoint(case,Y,se,rng,identity,capture);write(cr,dict(status='COMPLETED',execution=capture,completed=done,pointer=pointer,**FLAGS));append(capture,'TERMINAL','CHECKPOINT_CAPTURE',cr,decision='DURABLE_CHECKPOINT_SAVED',truth='PERSIST_CURRENT_SYNTHETIC_RNG_AND_ARRAYS_ONLY',metrics=dict(completed=done,state_sha256=pointer['state_sha256']),checkpoint=pointer['state_path'])
                except Exception as e:
                    write(cr,dict(status='CANNOT_CONFIRM',execution=capture,error=str(e),**FLAGS));append(capture,'TERMINAL','CHECKPOINT_CAPTURE',cr,decision='CANNOT_CONFIRM_CHECKPOINT',truth='SYNTHETIC_CHECKPOINT_FAILURE');raise
                resource_guard()
            if phase==0:
                rank=critical_index();u=order_bound(Y,rank['k']);result=dict(phase=0,scenario_id=scenario,scenario=SCENARIOS[scenario],reps=total,U=u,order_index=rank,stream=dict(entropy=19320261021,spawn_key=[0,scenario]),real_x_read=False)
            else:
                if sha(BATCH/'QSTAR_LOCK.json')!=lock_sha:raise ValueError('q changed before validation result')
                result=validation_summary(Y,se,lock['qstar'],scenario);result.update(qstar_lock_sha256=lock_sha,stream=dict(entropy=19320261021,spawn_key=[1,scenario]),real_x_read=False)
            resultpath=case/'RESULT.json'
            if resultpath.exists() and json.loads(resultpath.read_text())!=result:raise ValueError('existing finalized result differs; fail closed')
            if not resultpath.exists():write(resultpath,result)
            after=resource_guard();write(receipt,dict(status='COMPLETED',execution=execution,identity=identity,restored_completed=restored,new_draws=total-restored,elapsed_seconds=time.monotonic()-start,resources_before=before,resources_after=after,swap_delta_mib=None if before['swap_used_mib'] is None else after['swap_used_mib']-before['swap_used_mib'],result_path=str(resultpath),result_sha256=sha(resultpath),fit_calls=0,predict_calls=0,model_loads=0,real_x_read=False,**FLAGS));append(execution,'TERMINAL','SYNTHETIC_CASE',receipt,decision='CALIBRATION_CASE_COMPLETED' if phase==0 else 'VALIDATION_CASE_PASS' if result['pass_validation'] else 'VALIDATION_CASE_FAIL',truth='SIMULATED_ONLY_NO_REAL_X',metrics=dict(phase=phase,scenario_id=scenario,result_sha256=sha(resultpath),qstar_lock_sha256=lock_sha));write(completed,dict(identity=identity,result_path=str(resultpath),result_sha256=sha(resultpath),receipt_path=str(receipt),receipt_sha256=sha(receipt)));sync();print(json.dumps(dict(phase=phase,scenario=scenario,status='COMPLETED',result=result)),flush=True);return result
        except Exception as e:
            write(receipt,dict(status='CANNOT_CONFIRM',execution=execution,error=str(e),identity=identity,real_x_read=False,**FLAGS));append(execution,'TERMINAL','SYNTHETIC_CASE',receipt,decision='CANNOT_CONFIRM_CASE_RESUMABLE',truth='SIMULATED_CASE_FAILURE_NO_REAL_X');sync();raise
    finally:fcntl.flock(owner.fileno(),fcntl.LOCK_UN);owner.close()


def calibrate_phase(chunk):
    if (BATCH/'QSTAR_LOCK.json').exists():raise RuntimeError('q already locked; recalibration prohibited')
    can=json.loads((BATCH/'G1_QSTAR_CANARY_01/RUN_RECEIPT.json').read_text())
    if can['status']!='COMPLETED' or can['decision']!='ENGINEERING_CANARY_ONLY':raise ValueError('successful canary required')
    mask=load_mask();results={s:run_case(0,s,chunk,mask) for s in SCENARIOS};write(BATCH/'CALIBRATION_RESULT.json',dict(phase=0,reps_each=50000,scenarios=results,real_x_read=False));return results


def lock_phase():
    def op(out):
        if (BATCH/'QSTAR_LOCK.json').exists():raise ValueError('q lock exists; never replace/recalibrate')
        data=json.loads((BATCH/'CALIBRATION_RESULT.json').read_text());results={int(s):v for s,v in data['scenarios'].items()}
        if set(results)!=set(SCENARIOS) or any(r['reps']!=50000 or r['phase']!=0 or not np.isfinite(r['U']) for r in results.values()):raise ValueError('incomplete/nonfinite calibration')
        bindings={}
        for s in SCENARIOS:
            case=BATCH/f'CAL_CASE_{s:02d}';done=json.loads((case/'COMPLETED.json').read_text());bindings[str(s)]={str(case/'COMPLETED.json'):sha(case/'COMPLETED.json'),done['result_path']:done['result_sha256'],done['receipt_path']:done['receipt_sha256']}
        q=max(Z95,*[r['U'] for r in results.values()]);lock=dict(schema='G1_QSTAR_LOCK_V1',qstar=float(q),floor=Z95,order_index=critical_index(),U_by_scenario={s:r['U'] for s,r in results.items()},calibration_bindings=bindings,calibration_result_sha256=sha(BATCH/'CALIBRATION_RESULT.json'),config_sha256=sha(BATCH/'CONFIG.json'),core_sha256=sha(HERE/'g1_qstar_core.py'),frozen_hac_sha256=sha(HAC_SOURCE),mask_sha256=sha(BATCH/'G1_QSTAR_MASK_01/PRIMARY_MASK.csv'),family_confidence_scope='99% simultaneous MonteCarlo upper bounds on eight specified95%absT quantiles only; not real-data95%coverage certification')
        path=BATCH/'QSTAR_LOCK.json';write(path,lock);os.chmod(path,0o444);return dict(decision='QSTAR_LOCKED',qstar=q,qstar_lock_sha256=sha(path),real_x_read=False)
    return basic('G1_QSTAR_LOCK_01','QSTAR_LOCK',op,'CALIBRATION_ONLY_CRITICAL_LOCK_NO_REAL_X')


def validate_phase(chunk):
    lock,h=current_lock();mask=load_mask();results={s:run_case(1,s,chunk,mask,lock,h) for s in SCENARIOS}
    def op(out):
        current,current_sha=current_lock();passed=validation_gate(results,current_sha,h,lock['qstar']);report=dict(validation_pass=passed,decision='VALIDATION_PASS' if passed else 'CANNOT_CONFIRM_QSTAR_VALIDATION',qstar=lock['qstar'],qstar_lock_sha256=h,scenarios=results,real_x_read=False);write(BATCH/'VALIDATION_RESULT.json',report)
        if not passed:write(BATCH/'FINAL_RESULT.json',dict(accepted_status='INDEPENDENT_REVIEW_PENDING',decision=report['decision'],qstar=lock['qstar'],ci95=None,real_x_read=False,real_ci_executed=False,**FLAGS))
        return dict(decision=report['decision'],validation_result_sha256=sha(BATCH/'VALIDATION_RESULT.json'),qstar_lock_sha256=h,real_x_read=False)
    return basic('G1_QSTAR_VALIDATION_GATE_01','INDEPENDENT_VALIDATION_GATE',op,'VALIDATION_PHASE1_ONLY_NO_REAL_X',start_metrics=dict(qstar_lock_sha256=h,qstar=lock['qstar']))


def real_phase(binding):
    lock,h=current_lock();val=json.loads((BATCH/'VALIDATION_RESULT.json').read_text());results={int(s):v for s,v in val['scenarios'].items()}
    if not val['validation_pass'] or not validation_gate(results,sha(BATCH/'QSTAR_LOCK.json'),h,lock['qstar']):raise RuntimeError('VALIDATION_NOT_PASSED_REAL_X_PROHIBITED')
    def op(out):
        for k in ['daily','original_metrics']:
            if sha(binding[k]['path'])!=binding[k]['sha256']:raise ValueError('real input identity drift')
        original=json.loads(Path(binding['original_metrics']['path']).read_text());frame=pd.read_csv(binding['daily']['path'],usecols=['date','calendar_index','fold','valid','residual_spread'],float_precision='round_trip');selected=frame.loc[frame.fold.isin(FOLDS)&frame.valid].sort_values('calendar_index');prior=pd.read_csv(BATCH/'G1_QSTAR_MASK_01/SELECTED_MASK.csv')
        if not np.array_equal(selected[['date','calendar_index','fold']].to_numpy(),prior[['date','calendar_index','fold']].to_numpy()):raise ValueError('real selected-key identity drift')
        m=load_mask();result=fixed_real_ci(selected.residual_spread.to_numpy(),[original['fold_residual_spreads'][f] for f in FOLDS],original['primary_point'],m,lock['qstar']);decision=HAC.information_gate(result,original);write(out/'CI_RESULT.json',result);write(BATCH/'FINAL_RESULT.json',dict(accepted_status='INDEPENDENT_REVIEW_PENDING',decision=decision,qstar=lock['qstar'],ci95=result['ci95'],hac=result,point=original['primary_point'],fold_means=original['fold_residual_spreads'],coverage=original['overall_coverage'],coverage_by_fold=original['coverage_by_fold'],positive_complete_folds=original['complete_fold_positive_count'],qstar_lock_sha256=h,validation_result_sha256=sha(BATCH/'VALIDATION_RESULT.json'),real_x_read=True,real_ci_executed=True,original_point_groups_labels_coverage_recomputed=False,**FLAGS));return dict(decision=decision,result_sha256=sha(out/'CI_RESULT.json'),qstar_lock_sha256=h,real_ci_executed=True)
    return basic('G1_QSTAR_REAL_CI_01','ONE_REAL_CI',op,'ACCEPTED_DAILY_X_ONLY_AFTER_LOCK_AND_ALL_VALIDATION_PASS',start_metrics=dict(qstar_lock_sha256=h))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['tests','mask','canary','calibrate','lock','validate','real','review-start']);p.add_argument('--chunk',type=int,default=256);p.add_argument('--workers',type=int,choices=[1,2],default=1);p.add_argument('--execution');a=p.parse_args();binding=verify()
    if a.workers!=1:raise RuntimeError('WORKER2_NOT_ADMITTED_WITHOUT_MEASURED_THROUGHPUT_GAIN; frozen default serial')
    if a.stage=='tests':r=tests()
    elif a.stage=='mask':r=mask_stage(binding)
    elif a.stage=='canary':r=canary(a.chunk)
    elif a.stage=='calibrate':r=calibrate_phase(a.chunk)
    elif a.stage=='lock':r=lock_phase()
    elif a.stage=='validate':r=validate_phase(a.chunk)
    elif a.stage=='real':r=real_phase(binding)
    else:
        out=BATCH/a.execution;out.mkdir();receipt=out/'RUN_RECEIPT.json';write(receipt,dict(status='AWAITING_REVIEW'));append(a.execution,'RUN_STARTED','INDEPENDENT_REVIEW',receipt,truth='REGISTER_BEFORE_READ_ONLY_REVIEW');r=sync()
    print(json.dumps(r,ensure_ascii=False,default=str),flush=True)
if __name__=='__main__':main()
