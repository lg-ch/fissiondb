"""Native-width int8 representatives must preserve the padded reference exactly."""
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pytest
from fission_layout_fixture import read_layout

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def padded_center_library(tmp_path_factory):
    target=tmp_path_factory.mktemp('padded_center_reference')/'libfissiondb_padded.so'
    arch='armv8.2-a+dotprod+fp16' if platform.machine()=='aarch64' else 'x86-64'
    cmd=['gcc','-O3','-std=c11','-fPIC','-shared','-fopenmp',f'-march={arch}',
        '-DFISSIONDB_TESTING','-DFISSIONDB_TEST_PADDED_CENTERS',str(ROOT/'src/anchor.c'),str(ROOT/'src/anchor_live.c'),
        '-lm','-luring','-lroaring','-lxxhash','-lpthread','-lcurl','-o',str(target)]
    if 'asan' in os.environ.get('LD_PRELOAD',''):cmd+=['-fsanitize=address,undefined']
    subprocess.run(cmd,check=True,capture_output=True,timeout=120)
    return target


@pytest.mark.parametrize('dim',[1,3,31,63,96,129,384,512,768,1000,1024])
def test_packed_centers_preserve_build_codes_routes_and_reopen(tmp_path,padded_center_library,dim):
    script='''
import json,sys,numpy as np
from pathlib import Path
from fissiondb import AnchorIndex
root=Path(sys.argv[1]);dim=int(sys.argv[2]);rng=np.random.default_rng(981)
x=rng.normal(size=(640,dim)).astype(np.float32);queries=rng.normal(size=(8,dim)).astype(np.float32)
with AnchorIndex.create(root,dim,cell_capacity=64,auto_pack_bytes=0) as index:
 for start in range(0,len(x),64):
  index.insert_batch(x[start:start+64],[{'group':start//64}]*64,group_commit=True)
  index.flush_fission()
 index.update(7,x[11],{'group':99});index.delete(8);index.flush_fission();index.pack_live()
 with index.context(nprobe=7,rerank=400) as ctx:
  answers=[ctx.search(q) for q in queries]
 (root/'memory.json').write_text(json.dumps(index.fission_stats))
with AnchorIndex(root,root/'base.f16bin',residual_dir=root/'residual',live_dir=root/'live',auto_pack_bytes=0) as index:
 with index.context(nprobe=7,rerank=400) as ctx:
  again=[ctx.search(q) for q in queries]
 for a,b in zip(answers,again):
  np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1]);assert a[2]['entries']==b[2]['entries']
 index.insert(x[3]);index.flush_fission();index.pack_live()
 with index.context(nprobe=7,rerank=400) as ctx:
  answers=[ctx.search(q) for q in queries]
np.savez(root/'answers.npz',ids=np.stack([a[0] for a in answers]),scores=np.stack([a[1] for a in answers]),entries=np.array([a[2]['entries'] for a in answers]))
'''
    for mode in ('padded','native'):
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','FISSIONDB_INGEST_THREADS':'1'}
        if mode=='padded':env['FISSIONDB_ANCHOR_LIBRARY']=str(padded_center_library)
        subprocess.run([sys.executable,'-c',script,str(tmp_path/mode),str(dim)],env=env,check=True,timeout=30)
    padded=read_layout(tmp_path/'padded');native=read_layout(tmp_path/'native')
    assert native[0][8:]==padded[0][8:]
    assert native[1:]==padded[1:] # exact assignments, centers and encoded bytes
    a=np.load(tmp_path/'padded/answers.npz');b=np.load(tmp_path/'native/answers.npz')
    for field in ('ids','scores','entries'):np.testing.assert_array_equal(a[field],b[field])
    old=json.loads((tmp_path/'padded/memory.json').read_text());new=json.loads((tmp_path/'native/memory.json').read_text())
    padded_dim=max(8,1<<(dim-1).bit_length())
    assert old['center_dim']==padded_dim and new['center_dim']==dim
    assert new['center_bytes']==new['cells']*dim
    assert old['center_allocated_bytes']*dim==new['center_allocated_bytes']*padded_dim
    assert old['owned_bytes']-new['owned_bytes']==old['center_allocated_bytes']-new['center_allocated_bytes']
