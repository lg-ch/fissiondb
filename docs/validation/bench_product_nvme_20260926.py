"""Paired product-library test on the unchanged full 200k-anchor snapshot."""
import ctypes as C
import argparse
import json
import os
from pathlib import Path
import random
import time
import numpy as np

root=Path('/root/mangrove-product-20260926')
parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['both','before','integrated'],default='both')
args=parser.parse_args()
data=Path('/root/mangrove-datasets/msmarco-e78737fe')
z=np.load('/root/mangrove-robust-calibration/workload.npz')
selected=np.flatnonzero(z['partitions']=='validation').tolist();assert len(selected)==600
cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
assert 999000000<=int((cg/'memory.max').read_text())<=1000000000
assert len(os.sched_getaffinity(0))==1
class Stats(C.Structure):
    _fields_=[(k,C.c_double) for k in ['anchor_ms','io_ms','score_ms','rerank_ms','total_ms']]+[(k,C.c_uint64) for k in ['entries','bytes']]
fds=[os.open(path,os.O_RDONLY) for path in [data/'base.f16bin',data/'residual-200k/res512.bin']]
expected={};rows=[]
if args.variant=='integrated':
    for row in json.loads((root/'nvme-product-rows.json').read_text()):
        if row['variant']=='before':expected[row['query']]=(np.array(row['ids'],np.uint32),np.array(row['scores'],np.float32))
variants=[('before','/root/mangrove-product-before-20260926/libmangrove_anchor.so'),('integrated',str(root/'libmangrove_anchor.so'))]
if args.variant!='both':variants=[v for v in variants if v[0]==args.variant]
for variant,path in variants:
    lib=C.CDLL(path)
    lib.anchor_index_open.argtypes=[C.c_char_p,C.c_char_p,C.c_int];lib.anchor_index_open.restype=C.c_void_p
    lib.anchor_index_enable_residual.argtypes=[C.c_void_p,C.c_char_p]
    lib.anchor_query_create.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_uint64,C.c_char_p,C.c_int];lib.anchor_query_create.restype=C.c_void_p
    lib.anchor_query_search.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p]
    lib.anchor_query_close.argtypes=[C.c_void_p];lib.anchor_index_close.argtypes=[C.c_void_p]
    index=lib.anchor_index_open(os.fsencode(data/'index-200k'),os.fsencode(data/'base.f16bin'),1);assert index
    assert lib.anchor_index_enable_residual(index,os.fsencode(data/'residual-200k'))==0
    ctx=lib.anchor_query_create(index,1536,400,1,900000000,None,0);assert ctx
    order=selected.copy();random.Random(62926).shuffle(order)
    for qi in order:
        for fd in fds:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
        q=np.ascontiguousarray(z['queries'][qi],np.float32);ids=np.zeros(10,np.uint32);scores=np.zeros(10,np.float32);st=Stats()
        t=time.perf_counter();count=lib.anchor_query_search(ctx,q.ctypes.data,10,ids.ctypes.data,scores.ctypes.data,C.byref(st));elapsed=(time.perf_counter()-t)*1000
        assert count==10
        if variant=='before':expected[qi]=(ids.copy(),scores.copy())
        else:
            np.testing.assert_array_equal(ids,expected[qi][0]);np.testing.assert_array_equal(scores,expected[qi][1])
        rows.append(dict(variant=variant,query=qi,ms=elapsed,recall=len(set(ids)&set(z['ids'][qi]))/10,
                         ids=ids.tolist(),scores=scores.tolist(),stats={k:getattr(st,k) for k,_ in Stats._fields_}))
        if len(rows)%100==0:print(json.dumps(dict(variant=variant,completed=len(rows))),flush=True)
    lib.anchor_query_close(ctx);lib.anchor_index_close(index)
report={'index':str(data/'index-200k'),'n':113520750,'dim':1024,'anchors':200000,'probes':1536,'rerank':400,'queries':600,
        'protocol':'Reused 600 real validation queries; one CPU5, cgroup1GB/swap0. fadvise before each query. Before buffered serial waves; integrated direct overlapped waves. Full originals on NVMe. Every final ID and score compared exactly. No S3 or ingestion.',
        'resources':{name:(cg/name).read_text().strip() for name in ['memory.max','memory.peak','memory.events','memory.swap.max','cpuset.cpus.effective']},'summary':[]}
for variant,_ in variants:
    group=[r for r in rows if r['variant']==variant]
    report['summary'].append(dict(variant=variant,recall=float(np.mean([r['recall'] for r in group])),
        p50_ms=float(np.median([r['ms'] for r in group])),p95_ms=float(np.percentile([r['ms'] for r in group],95)),
        mean_stages={k:float(np.mean([r['stats'][k] for r in group])) for k,_ in Stats._fields_}))
prefix='nvme-product' if args.variant=='both' else 'nvme-product-'+args.variant
(root/(prefix+'-comparison.json')).write_text(json.dumps(report,indent=2)+'\n')
(root/(prefix+'-rows.json')).write_text(json.dumps(rows)+'\n')
print(json.dumps(report),flush=True)
