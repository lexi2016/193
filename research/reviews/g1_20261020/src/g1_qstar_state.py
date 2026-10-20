"""Atomic, hashed simulation state; checkpoint arrays are runtime artifacts."""
from pathlib import Path
import hashlib,json,os,tempfile
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):
    path=Path(path);fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(data,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()


def save_checkpoint(directory,Y,se,rng,identity,capture_execution):
    directory=Path(directory);done=len(Y);arrays=directory/f'ARRAYS_{done}_{capture_execution}.npz';statepath=directory/f'STATE_{done}_{capture_execution}.json'
    if arrays.exists() or statepath.exists():raise ValueError('immutable checkpoint already exists')
    fd,tmp=tempfile.mkstemp(prefix='.arrays',dir=directory)
    try:
        with os.fdopen(fd,'wb') as f:np.savez(f,absT=Y,se=se);f.flush();os.fsync(f.fileno())
        os.replace(tmp,arrays)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()
    state=dict(identity=identity,completed=done,rng_state=rng.bit_generator.state,arrays_path=str(arrays),arrays_sha256=sha(arrays),capture_execution=capture_execution)
    write_json(statepath,state);pointer=dict(state_path=str(statepath),state_sha256=sha(statepath),capture_execution=capture_execution);write_json(directory/'LAST_STATE.json',pointer);return pointer


def load_checkpoint(directory,expected_identity):
    pointerpath=Path(directory)/'LAST_STATE.json'
    if not pointerpath.exists():return None
    pointer=json.loads(pointerpath.read_text());statepath=Path(pointer['state_path'])
    if sha(statepath)!=pointer['state_sha256']:raise ValueError('checkpoint state hash drift')
    state=json.loads(statepath.read_text())
    if state['identity']!=expected_identity or sha(state['arrays_path'])!=state['arrays_sha256']:raise ValueError('checkpoint recovery identity/array hash drift')
    with np.load(state['arrays_path'],allow_pickle=False) as d:Y=d['absT'].copy();se=d['se'].copy()
    if len(Y)!=state['completed'] or len(se)!=len(Y) or not np.isfinite(Y).all() or not np.isfinite(se).all() or (se<=0).any():raise ValueError('invalid checkpoint array shape/values')
    return state,Y,se
