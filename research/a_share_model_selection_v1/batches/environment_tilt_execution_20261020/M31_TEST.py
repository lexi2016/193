"""Synthetic fixed-date lookup and registration ordering, no actual payload."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location('m31_lookup',Path(__file__).with_name('M31_RUN.py'))
x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)


def test_all_same_day_categories_duplicates_and_nonfinite_retained():
    cat1=dict(year=2005,month=12,day=19,category=1,songzhuangu=1.9,fenhong=0.)
    cat14=dict(year=2005,month=12,day=19,category=14,fenshu=6.,peigu=None)
    other=dict(year=2005,month=12,day=20,category=1)
    data=dict(symbol='600004.SH',response_type='list',response=[cat14,cat1,copy.deepcopy(cat1),other])
    out=x.lookup(data,'600004.SH',20051219)
    assert out['match_count']==3 and out['category1_count']==2
    assert out['all_same_date_records'][1]['original_record']==cat1
    assert [r['source_response_index'] for r in out['all_same_date_records']]==[0,1,2]
    assert x.safe(dict(a=float('nan')))['a']['interpretation']=='NOT_NORMALIZED'


def test_no_exact_row_does_not_infer_absence_or_substitute_date():
    out=x.lookup(dict(symbol='000014.SZ',response_type='list',response=[]),'000014.SZ',20070630)
    assert out['match_count']==0 and 'NOT_NO_CORPORATE_ACTION' in out['missing_means']


@pytest.mark.parametrize('change',['symbol','type','month','date_type'])
def test_malformed_keys_fail_without_normalization(change):
    d=dict(symbol='000002.SZ',response_type='list',response=[dict(year=2005,month=12,day=1,category=14)])
    if change=='symbol':d['symbol']='OTHER'
    if change=='type':d['response_type']='dict'
    if change=='month':d['response'][0]['month']=13
    if change=='date_type':d['response'][0]['year']='2005'
    with pytest.raises(ValueError):x.lookup(d,'000002.SZ',20051201)


def fixture(tmp_path,monkeypatch,fail=False):
    # Existing audited local JSON/guard helpers; synthetic adapter only.
    config=json.loads(Path(__file__).with_name('M31_CONFIG_DRAFT.json').read_text())
    c,base,registry=x.load_config(Path(__file__).with_name('M31_CONFIG_DRAFT.json'))
    root=tmp_path/'M31_DATE_LOOKUP_01';c.update(state='FROZEN_READY_FOR_ACTIVATION',batch_root=str(tmp_path),run_dir=str(root))
    review=tmp_path/'review.json';review.write_text('{}');c['review_receipt']=dict(path=str(review),sha256=registry.sha(review))
    for i,t in enumerate(c['targets']):
        p=tmp_path/f'input{i}.json';y,m,d=int(str(t['date'])[:4]),int(str(t['date'])[4:6]),int(str(t['date'])[6:])
        p.write_text(json.dumps(dict(symbol=t['symbol'],response_type='list',response=[dict(year=y,month=13 if fail and i==1 else m,day=d,category=14)])))
        t['input']=dict(path=str(p),sha256=registry.sha(p),bytes=p.stat().st_size)
    activation=dict(decision='INDEPENDENT_REVIEW_PASS_ACTIVATE',config_sha256=c['config_sha256'],execution_id=c['execution_id'],script_sha256=c['script_sha256'],review_receipt_sha256=c['review_receipt']['sha256'])
    events=[]
    class Life:
        def __init__(self,*args):self.started=False;self.finished=False
        def start(self,receipt):self.started=True;events.append('START')
        def require(self):assert self.started and not self.finished
        def finish(self,receipt,decision,metrics):self.require();events.append('TERMINAL');self.finished=True
    monkeypatch.setattr(x,'lifecycle_type',lambda r:Life)
    auth=base.authenticated
    def registered_read(life,ref,budget):
        assert events==['START'];return auth(life,ref,budget)
    monkeypatch.setattr(base,'authenticated',registered_read)
    monkeypatch.setattr(base,'available_memory',lambda:2**36)
    return c,base,registry,activation,events


@pytest.mark.parametrize('fail',[False,True])
def test_start_before_input_and_one_terminal_keeps_partial_results(tmp_path,monkeypatch,fail):
    c,base,registry,a,events=fixture(tmp_path,monkeypatch,fail)
    if fail:
        with pytest.raises(ValueError):x.run(c,base,registry,object(),a)
    else:
        assert x.run(c,base,registry,object(),a)['status']=='COMPLETED_FIXED_THREE_DATE_LOOKUP'
    assert events==['START','TERMINAL']
    receipt=json.loads((Path(c['run_dir'])/'RECEIPT.json').read_text())
    assert len(receipt['targets'])==(1 if fail else 3)
    if fail:assert receipt['status']=='FAILED_NO_RETRY'
    with pytest.raises(FileExistsError):x.run(c,base,registry,object(),a)


def test_draft_rejects_before_output_and_start(tmp_path,monkeypatch):
    c,base,registry,a,events=fixture(tmp_path,monkeypatch)
    c['state']='DRAFT'
    with pytest.raises(PermissionError):x.run(c,base,registry,object(),a)
    assert events==[] and not Path(c['run_dir']).exists()
