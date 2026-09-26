"""Automatic fixed-index calibration with an independently reserved audit.

Run with python -m fissiondb.calibration. Ground truth must describe exactly the
indexed snapshot. Statistical interpretation assumes representative independent
query groups; confidence does not protect against distribution drift.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from .adaptive import fingerprint


def recall_lower_bound(values, delta=.05):
    """One-sided empirical Bernstein bound, Maurer & Pontil (2009), Thm 4.

    Applied to per-query recall in [0,1], not to k independent neighbor trials.
    Training bounds are selection heuristics. Only the separately reserved audit
    supports a confidence interpretation for the selected fixed configurations.
    """
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=1 or len(x)<2 or not 0<delta<1 or not np.isfinite(x).all() or np.any((x<0)|(x>1)):
        raise ValueError('Invalid confidence sample')
    n=len(x);log=math.log(2/delta)
    return max(0.,float(x.mean()-math.sqrt(2*x.var(ddof=1)*log/n)-7*log/(3*(n-1))))


def _summary(rows,delta):
    r=[x['recall'] for x in rows];ms=[x['ms'] for x in rows]
    return dict(queries=len(rows),recall=float(np.mean(r)),recall_lower=recall_lower_bound(r,delta),
                mean_ms=float(np.mean(ms)),p50_ms=float(np.median(ms)),p95_ms=float(np.percentile(ms,95)))


def calibrate_snapshot(index,queries,ground_truth,partitions,*,groups=None,
        target_recall=.95,confidence=.95,latency_target_ms=100.,memory_bytes=0,
        initial_probes=32,maximum_probes=2048,initial_rerank=1000,maximum_rerank=16000,
        min_samples=200,before_query=None,protocol_id="natural-cache-v1",checkpoint=None,progress=None):
    """Tune on calibration only; audit at most two preselected profiles once.

    Returns a strict profile and measured compromises. Never retries tuning after
    reading the audit. Unfiltered, fixed-index, single-thread search; no codec or
    structure selection. Explicitly records failure rather than relaxing targets.
    Checkpoints resume completed measurements only on the identical run identity.
    """
    if getattr(index,"live",False):raise ValueError("Calibration requires an immutable frozen snapshot")
    if before_query is not None and protocol_id=="natural-cache-v1":raise ValueError("Specify protocol_id for a custom cache/measurement hook")
    q=np.asarray(queries,dtype=np.float32);gt=np.asarray(ground_truth);labels=list(partitions)
    if q.ndim!=2 or q.shape[1]!=index.dim or gt.ndim!=2 or len(gt)!=len(q) or not 1<=gt.shape[1]<=64:
        raise ValueError('Query/GT shape mismatch')
    if not np.isfinite(q).all() or np.any(np.linalg.norm(q,axis=1)<=0):raise ValueError('Invalid query vectors')
    if not np.issubdtype(gt.dtype,np.integer) or np.any(gt<0) or np.any(gt>=index.count) or any(len(set(row))!=gt.shape[1] for row in gt):
        raise ValueError('GT must contain distinct existing document IDs')
    if len(labels)!=len(q) or any(x not in ('calibration','validation') for x in labels):raise ValueError('Invalid partitions')
    if groups is None:groups=[hashlib.sha256(row.tobytes()).hexdigest() for row in q]
    if len(groups)!=len(q):raise ValueError('Invalid query groups')
    if len(set(map(str,groups)))!=len(groups):raise ValueError('Provide one representative per independent query group')
    train=[i for i,x in enumerate(labels) if x=='calibration'];audit=[i for i,x in enumerate(labels) if x=='validation']
    if min_samples<2 or min(len(train),len(audit))<min_samples:raise ValueError('Insufficient independent queries per partition')
    if not 0<target_recall<1 or not 0<confidence<1 or not np.isfinite(latency_target_ms) or latency_target_ms<=0:
        raise ValueError('Invalid quality/confidence/latency target')
    if not 1<=initial_probes<=maximum_probes or not gt.shape[1]<=initial_rerank<=maximum_rerank or memory_bytes<0:
        raise ValueError('Invalid search budgets')
    delta=(1-confidence)/2
    if recall_lower_bound(np.ones(min(len(train),len(audit))),delta)<target_recall:
        required=math.ceil(1+7*math.log(2/delta)/(3*(1-target_recall)))
        raise ValueError(f"Insufficient samples to certify this target even at perfect recall; need at least {required} per partition")
    settings=dict(target_recall=target_recall,confidence=confidence,latency_target_ms=latency_target_ms,
        initial_probes=initial_probes,maximum_probes=maximum_probes,initial_rerank=initial_rerank,
        maximum_rerank=maximum_rerank,memory_bytes=memory_bytes,min_samples=min_samples,protocol_id=protocol_id)
    h=hashlib.sha256(q.tobytes()+gt.astype('<u4').tobytes()+json.dumps([labels,list(map(str,groups)),settings],sort_keys=True).encode())
    fp=fingerprint(index);h.update(fp.encode());identity=h.hexdigest()
    state={'run_identity':identity,'measurements':{},'stage':'calibration'}
    cp=Path(checkpoint) if checkpoint else None
    if cp and cp.exists():
        state=json.loads(cp.read_text())
        if state['run_identity']!=identity:raise ValueError('Checkpoint belongs to another snapshot/workload/settings')
    def save():
        if cp:
            cp.parent.mkdir(parents=True,exist_ok=True);tmp=cp.with_suffix('.tmp')
            with tmp.open('w') as f:json.dump(state,f);f.flush();os.fsync(f.fileno())
            os.replace(tmp,cp)
    # Each of at most two final configurations receives half of the audit error
    # probability. This remains conservative if both choices are identical.
    delta=(1-confidence)/2
    def measure(config,role):
        key=f"{role}:{config['maximum']}:{config['rerank']}"
        if key in state['measurements']:return state['measurements'][key]
        indices=train if role=='calibration' else audit
        rows=[]
        with index.context(nprobe=config['maximum'],rerank=config['rerank'],threads=1,memory_bytes=memory_bytes) as ctx:
            ctx.adapt(minimum=config['minimum'],gap=0)
            for i in indices:
                if before_query:before_query()
                start=time.perf_counter();ids,_,stats=ctx.search(q[i],top_k=gt.shape[1])
                rows.append(dict(query=i,recall=len(set(ids)&set(gt[i]))/gt.shape[1],ms=1000*(time.perf_counter()-start),stats=stats))
        value={'configuration':dict(config),'role':role,'rows':rows,'summary':_summary(rows,delta)}
        state['measurements'][key]=value;save()
        if progress:progress({'stage':role,'configuration':config,'summary':value['summary']})
        return value
    history=[];probes=initial_probes;rerank=initial_rerank;quality=None
    if state['stage']=='calibration':
        while True:
            config=dict(minimum=probes,maximum=probes,filtered_minimum=probes,rerank=rerank,gap=0)
            result=measure(config,'calibration');history.append(result)
            if result['summary']['recall_lower']>=target_recall:
                quality=config;break
            # Diagnostic coverage guides only which budget to expand, never
            # whether a configuration is accepted. Trace a bounded train sample.
            routing_loss=[];rerank_loss=[]
            with index.context(nprobe=probes,rerank=rerank,threads=1,memory_bytes=memory_bytes) as ctx:
                ctx.adapt(minimum=probes,gap=0)
                for i in train[::max(1,len(train)//64)][:64]:
                    ids,_,stats=ctx.search(q[i],top_k=gt.shape[1],diagnostic_ids=gt[i])
                    reached=stats['diagnostic']['routed']/gt.shape[1]
                    served=len(set(ids)&set(gt[i]))/gt.shape[1]
                    routing_loss.append(1-reached);rerank_loss.append(max(0.,reached-served))
            # Expand the stage losing most true neighbors. A fixed coverage
            # threshold can waste all rerank trials while routing is the bottleneck.
            if np.mean(routing_loss)>=np.mean(rerank_loss) and probes<maximum_probes:probes=min(maximum_probes,probes*2)
            elif rerank<maximum_rerank:rerank=min(maximum_rerank,rerank*2)
            elif probes<maximum_probes:probes=min(maximum_probes,probes*2)
            else:break
        fast=[v for v in history if v['summary']['p95_ms']<=latency_target_ms]
        alternative=max(fast,key=lambda v:(v['summary']['recall_lower'],-v['summary']['mean_ms'])) if fast else min(history,key=lambda v:v['summary']['p95_ms'])
        if quality is None:quality=max(history,key=lambda v:v['summary']['recall_lower'])['configuration']
        # Freeze all choices BEFORE touching validation. Crash/resume retains them.
        state['choices']={'quality':quality,'latency':alternative['configuration']};state['stage']='validation';save()
    audits={name:measure(config,'validation') for name,config in state['choices'].items()}
    good={name:v for name,v in audits.items() if v['summary']['recall_lower']>=target_recall and v['summary']['p95_ms']<=latency_target_ms}
    best=min(good.values(),key=lambda v:v['summary']['mean_ms']) if good else audits['quality']
    quality_ok=audits['quality']['summary']['recall_lower']>=target_recall
    status='validated' if good else ('latency_tradeoff' if quality_ok else 'quality_not_validated')
    report=dict(version=1,fingerprint=fp,run_identity=identity,validated=bool(good),status=status,
        configuration=best['configuration'],top_k=gt.shape[1],target_recall=target_recall,confidence=confidence,
        latency_target_ms=latency_target_ms,calibration_count=int(index.count),settings=settings,
        calibration_queries=len(train),validation_queries=len(audit),audit=audits,
        measurements=[v for v in state['measurements'].values() if v['role']=='calibration'],
        limitations='Fixed-index unfiltered calibration; confidence assumes representative independent query groups; no drift or per-query guarantee; p95 is empirical, not a confidence bound; owned-allocation budget is not total RAM; not a global optimum over all structures/configurations')
    state['stage']='complete';save();return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('index');p.add_argument('base');p.add_argument('workload',help='NPZ: queries, ids, partitions, optional groups')
    p.add_argument('--residual');p.add_argument('--output',required=True);p.add_argument('--target-recall',type=float,default=.95)
    p.add_argument('--confidence',type=float,default=.95);p.add_argument('--p95-ms',type=float,default=100)
    p.add_argument('--memory-bytes',type=int,default=800000000);p.add_argument('--max-probes',type=int,default=2048)
    p.add_argument("--initial-rerank",type=int,default=1000)
    p.add_argument('--max-rerank',type=int,default=16000);p.add_argument('--cold',action='store_true')
    a=p.parse_args();z=np.load(a.workload,allow_pickle=False);output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    from .anchors import AnchorIndex
    paths=[a.base,str(Path(a.residual)/'res512.bin') if a.residual else str(Path(a.index)/'blocks.bin')]
    fds=[os.open(path,os.O_RDONLY) for path in paths] if a.cold else []
    def cold():
        for fd in fds:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
    try:
        with AnchorIndex(a.index,a.base,**({'residual_dir':a.residual} if a.residual else {})) as index:
            k=int((Path(a.index)/'meta.txt').read_text().split()[0]);maximum=min(k,a.max_probes)
            result=calibrate_snapshot(index,z['queries'],z['ids'],z['partitions'].tolist(),groups=z['groups'].tolist() if 'groups' in z else None,
                target_recall=a.target_recall,confidence=a.confidence,latency_target_ms=a.p95_ms,memory_bytes=a.memory_bytes,
                initial_probes=min(32,maximum),maximum_probes=maximum,initial_rerank=a.initial_rerank,maximum_rerank=a.max_rerank,
                before_query=cold if a.cold else None,protocol_id="fadvise-dontneed-v1" if a.cold else "natural-cache-v1",checkpoint=str(output)+'.checkpoint',progress=lambda x:print(json.dumps(x),flush=True))
        result['cache_protocol']='POSIX_FADV_DONTNEED request before timed queries' if a.cold else 'natural cache'
        temp=output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));os.replace(temp,output)
        quality=result['audit']['quality']
        # Explicit separate profile: meets statistical quality target only, not
        # the requested latency contract. Never silently substitute it in service.
        if quality['summary']['recall_lower']>=a.target_recall:
            relaxed={**result,'validated':True,'configuration':quality['configuration'],'latency_target_ms':None,'status':'quality_only_latency_not_promised'}
            output.with_name(output.stem+'-quality-only.json').write_text(json.dumps(relaxed,indent=2))
        print(json.dumps({'status':result['status'],'configuration':result['configuration'],'audit':{k:v['summary'] for k,v in result['audit'].items()}}),flush=True)
        return 0 if result['validated'] else 2
    finally:
        for fd in fds:os.close(fd)

if __name__=='__main__':raise SystemExit(main())
