import importlib.util,sys,types
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('protocol_canary',Path(__file__).with_name('run.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_source_packet_body_preserved_across_header_modes():
    root=Path(__file__).parents[3]/'M39_OPEN_TDX_SOURCE_REVIEW/tdxpy-0.2.7/tdxpy-0.2.7'
    sys.path.insert(0,str(root))
    try:
        from tdxpy.parser.std.get_security_bars import GetSecurityBarsCmd
        cmd=GetSecurityBarsCmd(None);cmd.setParams(9,1,'600000',0,10);packet=bytes(cmd.send_pkg)
        assert packet[:6].hex()=='0c0108640101'
        assert m.modify_header(packet,'tdxpy_base')==m.modify_header(packet,'go_setup_only')==packet
        changed=m.modify_header(packet,'go_header_only');assert changed==m.modify_header(packet,'go_both')
        assert changed[:6].hex()=='0c0100000001' and changed[6:]==packet[6:]
        assert m.GO_CONNECT.hex()=='0c0000000001030003000d0001'
        with pytest.raises(m.source.Stop):m.modify_header(b'bad','go_both')
    finally:sys.path.pop(0)

def test_parser_failure_raw_body_durable_and_original_forwarded_once(tmp_path):
    counters=dict(connect=0,query=0,parse=0)
    packet=bytes.fromhex('0c01086401011c001c002d05')+b'x'*26
    class Parser:
        def setParams(self,*a,**k):self.send_pkg=packet
        def parseResponse(self,body):
            counters['parse']+=1;assert body is response;raise ValueError('original-parser-failure')
    response=bytearray(b'\x20\x03')
    class API:
        def __init__(self,**kwargs):
            assert kwargs==dict(multithread=False,heartbeat=False,auto_retry=False,raise_exception=True)
            self.client=types.SimpleNamespace(settimeout=lambda _:None)
        def connect(self,*args,**kw):counters['connect']+=1;return self
        def get_security_bars(self,**kw):
            counters['query']+=1;p=Parser();p.setParams(**kw);return p.parseResponse(response)
        def disconnect(self):pass
    class Life:
        def require(self):pass
    class Budget:
        def check(self):pass
    c=dict(run_dir=str(tmp_path))
    result=m.attempt(c,Life(),Budget(),types.SimpleNamespace(TdxHq_API=API),types.SimpleNamespace(GetSecurityBarsCmd=Parser),None,1,m.HOSTS[0],'go_header_only',10)
    assert counters==dict(connect=1,query=1,parse=1)
    assert result['status']=='FAILED' and result['error']=='original-parser-failure'
    assert (tmp_path/'ATTEMPT_01/DAILY_BODY.bin').read_bytes()==response
    assert (tmp_path/'ATTEMPT_01/DAILY_PACKET.bin').read_bytes()[6:]==packet[6:]
