"""Same original 600-query MS MARCO panel and frozen files, before/after live fix."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--variant',choices=['baseline','candidate'],required=True)
args=p.parse_args();root=Path('/root/fission-frozen-controlled-20260926')
code='/root/FissionDB-live-perf-'+('base' if args.variant=='baseline' else 'dev')
library=Path(code)/'libfissiondb_anchor.so';lib=C.CDLL(str(library))
data=Path('/root/mangrove-datasets/msmarco-e78737fe')
z=np.load('/root/mangrove-robust-calibration/workload.npz')
selected=np.flatnonzero(z['partitions']=='validation').tolist();assert len(selected)==600
cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
assert 999000000<=int((cg/'memory.max').read_text())<=1000000000
assert os.sched_getaffinity(0)=={5}
class Stats(C.Structure):
    _fields_=[(k,C.c_double) for k in ['anchor_ms','io_ms','score_ms','rerank_ms','total_ms']]+[(k,C.c_uint64) for k in ['entries','bytes']]
lib.anchor_index_open.argtypes=[C.c_char_p,C.c_char_p,C.c_int];lib.anchor_index_open.restype=C.c_void_p
lib.anchor_index_enable_residual.argtypes=[C.c_void_p,C.c_char_p]
lib.anchor_query_create.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_uint64,C.c_char_p,C.c_int];lib.anchor_query_create.restype=C.c_void_p
lib.anchor_query_search.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p]
lib.anchor_query_close.argtypes=[C.c_void_p];lib.anchor_index_close.argtypes=[C.c_void_p]
index=lib.anchor_index_open(os.fsencode(data/'index-200k'),os.fsencode(data/'base.f16bin'),1);assert index
assert lib.anchor_index_enable_residual(index,os.fsencode(data/'residual-200k'))==0
ctx=lib.anchor_query_create(index,1536,400,1,900000000,None,0);assert ctx
fds=[os.open(path,os.O_RDONLY) for path in [data/'base.f16bin',data/'residual-200k/res512.bin']]
expected={r['query']:r for r in json.loads((root/'baseline-rows.json').read_text())} if args.variant=='candidate' else {}
rows=[];order=selected.copy();random.Random(62926).shuffle(order)
for qi in order:
    for fd in fds:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
    q=np.ascontiguousarray(z['queries'][qi],np.float32);ids=np.zeros(10,np.uint32);scores=np.zeros(10,np.float32);stats=Stats()
    t=time.perf_counter();n=lib.anchor_query_search(ctx,q.ctypes.data,10,ids.ctypes.data,scores.ctypes.data,C.byref(stats));ms=(time.perf_counter()-t)*1000
    assert n==10
    if expected:
        np.testing.assert_array_equal(ids,expected[qi]['ids'])
        np.testing.assert_array_equal(scores,np.asarray(expected[qi]['scores'],np.float32))
    rows.append(dict(query=qi,ms=ms,ids=ids.tolist(),scores=scores.tolist(),recall=len(set(ids)&set(z['ids'][qi]))/10,
                     stats={k:getattr(stats,k) for k,_ in Stats._fields_}))
    if len(rows)%100==0:print(args.variant,len(rows),flush=True)
lib.anchor_query_close(ctx);lib.anchor_index_close(index)
for fd in fds:os.close(fd)
report=dict(variant=args.variant,library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
    protocol='Original reused 600 MS MARCO validation queries, unchanged 113520750 x 1024 frozen index, 200k representatives, 1536 probes/400 reranks. CPU5, 1GB cgroup/swap0, fadvise before each query, direct overlapped codes, no ingestion. Hardware caches not flushed; separate transfer remains active.',
    recall_at_10=float(np.mean([r['recall'] for r in rows])),p50_ms=float(np.median([r['ms'] for r in rows])),
    p95_ms=float(np.percentile([r['ms'] for r in rows],95)),
    mean_stages={k:float(np.mean([r['stats'][k] for r in rows])) for k,_ in Stats._fields_},
    resources={k:(cg/k).read_text().strip() for k in ['memory.max','memory.peak','memory.events','memory.swap.max']})
if expected:
    baseline=json.loads((root/'baseline.json').read_text())
    report['identical_ids_and_scores']=True
    report['regression_gate']=report['p50_ms']<=1.1*baseline['p50_ms'] and report['p95_ms']<=1.2*baseline['p95_ms']
(root/(args.variant+'.json')).write_text(json.dumps(report,indent=2))
(root/(args.variant+'-rows.json')).write_text(json.dumps(rows))
print(json.dumps(report),flush=True)
if expected and not report['regression_gate']:raise SystemExit('Frozen latency regression')
