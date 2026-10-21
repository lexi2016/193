import importlib.util,io,hashlib,zipfile,struct
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('gbbq_capture',Path(__file__).with_name('run.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_original_reader_called_after_durable_raw_one_request(tmp_path):
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as z:z.writestr('gbbq',struct.pack('<I',4)+b'x'*(29*4))
    class Response(io.BytesIO):
        code=200;headers={}
        def geturl(self):return m.URL
    class Opener:
        calls=0
        def open(self,url,timeout):
            self.calls+=1;assert url==m.URL and timeout==30;return Response(data.getvalue())
    reader=tmp_path/'reader.py';reader.write_text('''class GbbqReader:
    def get_df(self,fname):
        assert PathNotNeeded if False else __import__('pathlib').Path(fname).exists()
        assert __import__('pathlib').Path(fname).with_name('OFFICIAL_HTTP_RESPONSE.bin').exists()
        return pd.DataFrame([dict(code='000002',datetime=20051201,category=14),dict(code='600004',datetime=20051219,category=1),dict(code='000014',datetime=20070630,category=4),dict(code='000002',datetime=20270101,category=1)])
''')
    op=Opener();report=dict(http_calls=0);m.fetch_once(tmp_path,dict(path=str(reader),sha256=hashlib.sha256(reader.read_bytes()).hexdigest()),op,report)
    assert op.calls==report['http_calls']==1
    assert [r['record_count'] for r in report['exact_event_records']]==[1,1,1]
    assert report['exact_event_records'][0]['records'][0]['category']==14
    assert report['training_ready'] is False
    # A real non-200 body is also durable, with no parsing/retry.
    other=tmp_path/'failure';other.mkdir();Response.code=403
    with pytest.raises(m.source.Stop,match='OFFICIAL_HTTP_STATUS_403'):m.fetch_once(other,{},op,dict(http_calls=0))
    assert (other/'OFFICIAL_HTTP_RESPONSE.bin').read_bytes()==data.getvalue() and op.calls==2

def test_missing_multiple_event_records_are_gaps_and_member_bound(tmp_path,monkeypatch):
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as z:z.writestr('gbbq',struct.pack('<I',3)+b'x'*87)
    class Response(io.BytesIO):
        code=200;headers={}
        def geturl(self):return m.URL
    class Opener:
        calls=0
        def open(self,*a,**k):self.calls+=1;return Response(data.getvalue())
    reader=tmp_path/'reader.py';reader.write_text("class GbbqReader:\n def get_df(self,fname):\n  return pd.DataFrame([dict(code='000002',datetime=20051201,category=14),dict(code='000002',datetime=20051201,category=1),dict(code='000014',datetime=20070630,category=4)])\n")
    ref=dict(path=str(reader),sha256=hashlib.sha256(reader.read_bytes()).hexdigest());op=Opener();r=dict(http_calls=0)
    m.fetch_once(tmp_path,ref,op,r,m.REQUEST)
    assert [x['record_count'] for x in r['exact_event_records']]==[2,0,1]
    assert [x['lookup_status'] for x in r['exact_event_records']]==['MULTIPLE_EXACT_RECORDS_RETAINED','MISSING_EXACT_RECORD','EXACT_UNIQUE_RECORD']
    assert r['exact_three_unique_records'] is False and op.calls==1
    class OversizedZip:
        def __init__(self,*a):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def namelist(self):return ['gbbq']
        def getinfo(self,*a):return __import__('types').SimpleNamespace(file_size=m.REQUEST['member_bytes']+1,is_dir=lambda:False,external_attr=0)
        def read(self,*a):raise AssertionError('oversize member must not be read')
    monkeypatch.setattr(m.zipfile,'ZipFile',OversizedZip);other=tmp_path/'oversized';other.mkdir()
    with pytest.raises(m.source.Stop,match='GBBQ_MEMBER_BOUNDARY'):m.fetch_once(other,ref,op,dict(http_calls=0),m.REQUEST)
    assert (other/'OFFICIAL_HTTP_RESPONSE.bin').exists()
    with pytest.raises(m.source.Stop,match='EXACT_ARCHIVE_HTTP_SCOPE'):m.fetch_once(tmp_path,ref,op,dict(http_calls=0),dict(m.REQUEST,member_bytes=1))
    assert op.calls==2
