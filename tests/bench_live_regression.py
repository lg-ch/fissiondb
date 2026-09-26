"""Rebuild a real-data live index and compare recall/latency against a baseline.

Run both revisions on the same idle host, CPU pair and external memory cgroup.
Inputs: f16bin header(N,D), query.npy, exact GT .npz with `ids` for exactly --rows.
Every invocation creates a NEW collection; it refuses to reuse an old one.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse
from collections import deque
import json
from pathlib import Path
import struct
import threading
import time
import numpy as np
from fissiondb import AnchorIndex

p=argparse.ArgumentParser(description=__doc__)
for flag in ('source','queries','truth','output'):p.add_argument('--'+flag,required=True)
p.add_argument('--rows',type=int,default=1_000_000)
p.add_argument('--query-count',type=int,default=128)
p.add_argument('--ingest-cpu',type=int,default=7)
p.add_argument('--search-cpu',type=int,default=8)
p.add_argument('--baseline')
p.add_argument('--max-p50-ratio',type=float,default=1.10)
p.add_argument('--max-p95-ratio',type=float,default=1.20)
p.add_argument('--max-recall-drop',type=float,default=.005)
args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
os.sched_setaffinity(0,{args.ingest_cpu})
queries=np.load(args.queries)[:args.query_count].astype(np.float32)
truth=np.load(args.truth)['ids'][:len(queries),:10]
with open(args.source,'rb') as f:n,dim=struct.unpack('<II',f.read(8))
assert n>=args.rows and queries.shape[1]==dim and truth.shape==(len(queries),10)
protocol=dict(rows=args.rows,dim=dim,queries=len(queries),nprobe=1536,rerank=400,cell_capacity=2048,
              batch=256,ingest_cpu=args.ingest_cpu,search_cpu=args.search_cpu,
              source=str(Path(args.source).resolve()),query_file=str(Path(args.queries).resolve()),
              truth_file=str(Path(args.truth).resolve()),cache='Buffered code path baseline; optimized default may use direct code IO. No eviction.',
              metadata='none',rebuild=True)
stop=threading.Event();measurements=[];errors=[];progress=0
def summary(times):
    if not times:return dict(n=0)
    return dict(n=len(times),**{f'p{v}_ms':float(np.percentile(times,v)) for v in (50,95,99)},max_ms=float(max(times)))
with AnchorIndex.create(out/'collection',dim,cell_capacity=2048,auto_pack_bytes=256*1024*1024) as index:
    def reader():
        os.sched_setaffinity(0,{args.search_cpu})
        try:
            with index.context(nprobe=1536,rerank=400) as context:
                i=0
                while not stop.is_set():
                    t=time.perf_counter();ids,scores,stats=context.search(queries[i%len(queries)])
                    measurements.append(dict(ms=(time.perf_counter()-t)*1000,rows=progress,stats=stats))
                    i+=1;stop.wait(.001)
        except BaseException as exc:errors.append(repr(exc));stop.set()
    task=None;started=time.perf_counter();tail_start=None;last_report=started
    with open(args.source,'rb',buffering=0) as f:
        f.seek(8)
        for first in range(0,args.rows,256):
            if errors:raise RuntimeError(errors)
            length=min(256,args.rows-first)
            x=np.frombuffer(f.read(length*dim*2),'<f2').reshape(length,dim).astype(np.float32)
            ids=index.insert_batch(x,group_commit=True)
            assert np.array_equal(ids,np.arange(first,first+length,dtype=np.uint32))
            progress=first+length
            if task is None and progress>=args.rows*3//4:
                tail_start=time.perf_counter();task=threading.Thread(target=reader);task.start()
            if time.perf_counter()-last_report>15:
                print(json.dumps(dict(rows=progress,elapsed=time.perf_counter()-started)),flush=True);last_report=time.perf_counter()
    index.flush_fission();ingestion_seconds=time.perf_counter()-started
    tail_seconds=time.perf_counter()-tail_start if tail_start else 0
    stop.set()
    if task:task.join()
    if errors:raise RuntimeError(errors)
    index.pack_live();os.sched_setaffinity(0,{args.search_cpu})
    rows=[]
    with index.context(nprobe=1536,rerank=400) as context:
        for q in queries:
            t=time.perf_counter();ids,scores,stats=context.search(q)
            rows.append(dict(ms=(time.perf_counter()-t)*1000,ids=ids.tolist(),scores=scores.tolist(),stats=stats))
    recall=float(np.mean([len(set(r['ids'])&set(gt))/10 for r,gt in zip(rows,truth)]))
    state=index.stats()
cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::')[-1].lstrip('/')
resources={name:(cg/name).read_text().strip() for name in ('memory.peak','memory.max','memory.events','memory.swap.max') if (cg/name).exists()}
report=dict(protocol=protocol,ingestion_seconds=ingestion_seconds,tail_ingestion_seconds=tail_seconds,
            recall_at_10=recall,quiescent=summary([r['ms'] for r in rows]),
            concurrent=summary([r['ms'] for r in measurements]),state=state,resources=resources)
failures=[]
if args.baseline:
    baseline=json.loads(Path(args.baseline).read_text());assert baseline['protocol']==protocol,'Different benchmark protocol'
    if recall<baseline['recall_at_10']-args.max_recall_drop:failures.append('recall')
    for mode in ('quiescent','concurrent'):
        for percentile,limit in [('p50_ms',args.max_p50_ratio),('p95_ms',args.max_p95_ratio)]:
            if report[mode].get(percentile,float('inf'))>baseline[mode][percentile]*limit:failures.append(mode+'_'+percentile)
    report['regression_gate']=dict(passed=not failures,failures=failures,baseline=str(args.baseline),
        max_p50_ratio=args.max_p50_ratio,max_p95_ratio=args.max_p95_ratio,max_recall_drop=args.max_recall_drop)
(out/'report.json').write_text(json.dumps(report,indent=2))
(out/'queries.json').write_text(json.dumps(rows))
(out/'concurrent.json').write_text(json.dumps(measurements))
print(json.dumps(report),flush=True)
if failures:raise SystemExit('Regression gate failed: '+', '.join(failures))
