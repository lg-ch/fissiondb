"""Explicit offline calibration with a disjoint validation partition.

Ground truth must be exact within each query's filter and exclude self where
appropriate. A profile records an empirical target, never a per-query guarantee.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def fingerprint(index):
    directory=index.directory
    digest=hashlib.sha256(str(index.int8_mode).encode())
    if index.residual_dir is not None:
        digest.update(b"residual-query-projection-v2")
        digest.update((index.residual_dir/"residual.meta").read_bytes())
    for name in ('meta.txt','anchors.bin','offs.bin','scale.bin'):
        path=Path(directory)/name
        digest.update(name.encode())
        if not path.exists():
            digest.update(b'absent');continue
        with path.open('rb') as stream:
            for data in iter(lambda:stream.read(1048576),b''):digest.update(data)
    return digest.hexdigest()


def calibrated_context(index, profile, *, memory_bytes=0, latency_budget_ms=0, code_bytes=0):
    if profile.get('version')!=1 or not profile.get('validated'):
        raise ValueError('Profile has not passed validation')
    if fingerprint(index)!=profile['fingerprint']:
        raise ValueError('Calibration belongs to a different frozen index')
    config=profile['configuration']
    context=index.context(nprobe=config['maximum'],rerank=config['rerank'],threads=1,memory_bytes=memory_bytes)
    try:
        context.adapt(minimum=config['minimum'],filtered_minimum=config['filtered_minimum'],
                      gap=config['gap'],code_bytes=code_bytes,latency_budget_ms=latency_budget_ms)
    except Exception:
        context.close();raise
    context.calibration_top_k=profile["top_k"]
    return context


def calibrate(index, queries, ground_truth, configurations, *, where=None,
              target_recall=.98, top_k=10, memory_bytes=0, before_query=None,
              partitions=None, latency_target_ms=None, query_kind='unspecified'):
    """Select on even occurrences of each predicate, validate on odd occurrences.

    `before_query` can enforce a cold-cache protocol. No automatic cache drop.
    At least four queries per predicate are required. Validation never chooses
    an alternative configuration after the selected one fails.
    """
    if not 0<target_recall<=1 or top_k<1:raise ValueError('Invalid quality target')
    if latency_target_ms is not None and (not np.isfinite(latency_target_ms) or latency_target_ms<=0):
        raise ValueError('Invalid p95 latency target')
    q=np.asarray(queries,dtype=np.float32);gt=np.asarray(ground_truth)
    if q.ndim!=2 or q.shape[1]!=index.dim or gt.shape!=(len(q),top_k):
        raise ValueError('Query/GT shape mismatch')
    if any(len(set(row))!=top_k for row in gt):raise ValueError('GT contains duplicate IDs')
    predicates=[None]*len(q) if where is None else list(where)
    if len(predicates)!=len(q):raise ValueError('Predicate count mismatch')
    keys=[json.dumps(p,sort_keys=True) for p in predicates];groups={}
    for i,key in enumerate(keys):groups.setdefault(key,[]).append(i)
    if any(len(ids)<4 for ids in groups.values()):raise ValueError('Need four queries per predicate')
    if partitions is None:
        train={i for ids in groups.values() for i in ids[::2]}
    else:
        partitions=list(partitions)
        if len(partitions)!=len(q) or any(p not in ('calibration','validation') for p in partitions):
            raise ValueError('Invalid query partitions')
        train={i for i,p in enumerate(partitions) if p=='calibration'}
        if any(sum(i in train for i in ids)<2 or sum(i not in train for i in ids)<2 for ids in groups.values()):
            raise ValueError('Each predicate needs two calibration and two validation queries')
    measured=[]
    for config in configurations:
        rows=[]
        with index.context(nprobe=config['maximum'],rerank=config['rerank'],threads=1,memory_bytes=memory_bytes) as ctx:
            ctx.adapt(minimum=config['minimum'],filtered_minimum=config['filtered_minimum'],gap=config['gap'])
            for i,vector in enumerate(q):
                if before_query:before_query()
                started=time.perf_counter();ids,_,stats=ctx.search(vector,top_k=top_k,where=predicates[i])
                rows.append(dict(recall=len(set(ids)&set(gt[i]))/top_k,ms=1000*(time.perf_counter()-started),stats=stats))
        summary={}
        for partition,wanted in [('train',True),('validation',False)]:
            summary[partition]={key:{'queries':len(sample),'recall':float(np.mean([rows[i]['recall'] for i in sample])),
                'p95_ms':float(np.percentile([rows[i]['ms'] for i in sample],95))}
                for key,indices in groups.items() if (sample:=[i for i in indices if (i in train)==wanted])}
        measured.append(dict(configuration=dict(config),summary=summary,rows=rows))
    def acceptable(group):
        return group['recall']+1e-12>=target_recall and (latency_target_ms is None or group['p95_ms']<=latency_target_ms)
    candidates=[r for r in measured if all(acceptable(g) for g in r['summary']['train'].values())]
    chosen=min(candidates,key=lambda r:float(np.mean([r['rows'][i]['ms'] for i in train]))) if candidates else None
    passed=chosen is not None and all(acceptable(g) for g in chosen['summary']['validation'].values())
    return dict(version=1,fingerprint=fingerprint(index),target_recall=target_recall,top_k=top_k,
                validated=passed,configuration=chosen['configuration'] if chosen else None,
                calibration_count=int(index.count),measurements=measured,
                query_kind=query_kind,latency_target_ms=latency_target_ms,
                partitions=['calibration' if i in train else 'validation' for i in range(len(q))],
                limitations='Empirical mean recall, not a confidence bound or per-query guarantee; small groups are fragile; memory_bytes excludes total process/cache RAM; recalibrate after distribution drift.')


def diagnose(index, queries, ground_truth, configuration, *, allowed_ids=None, memory_bytes=0):
    """Separate untimed diagnostic pass. Frozen/local/single-thread only.

    GT must be exhaustive for the supplied predicates. Candidate loss includes
    prefix preselection and full-code selection. Live overlays are unsupported.
    """
    q=np.asarray(queries,dtype=np.float32);gt=np.asarray(ground_truth)
    if q.ndim!=2 or q.shape[1]!=index.dim or gt.ndim!=2 or len(gt)!=len(q) or not 1<=gt.shape[1]<=64:
        raise ValueError('Invalid diagnostic query/GT shapes')
    allowed=[None]*len(q) if allowed_ids is None else list(allowed_ids)
    if len(allowed)!=len(q):raise ValueError('Predicate count mismatch')
    rows=[]
    with index.context(nprobe=configuration['maximum'],rerank=configuration['rerank'],threads=1,memory_bytes=memory_bytes) as ctx:
        ctx.adapt(minimum=configuration['minimum'],filtered_minimum=configuration['filtered_minimum'],gap=configuration['gap'])
        for i,vector in enumerate(q):
            ids,_,stats=ctx.search(vector,top_k=gt.shape[1],allowed_ids=allowed[i],diagnostic_ids=gt[i])
            trace=stats['diagnostic'];k=trace['watched'];routed=trace['routed'];candidates=trace['candidates']
            rows.append(dict(routing_recall=routed/k,candidate_recall=candidates/k,
                final_recall=len(set(ids)&set(gt[i]))/k,routing_misses=k-routed,
                compression_misses=routed-candidates,probes=stats['probes']))
    return {'rows':rows,'timings_excluded':True,'configuration':dict(configuration)}


def autocalibrate(index, queries, ground_truth, *, partitions, target_recall=.95,
                  top_k=10, initial_probes=32, maximum_probes=1024,
                  initial_rerank=1000, maximum_rerank=16000, memory_bytes=0,
                  latency_target_ms=None, before_query=None):
    """Diagnostic-guided frozen, unfiltered budget tuning; validation once.

    Expand routing if cells miss GT, otherwise expand candidate precision budget.
    Only calibration queries drive expansion. No structure/codec changes occur.
    A failed final validation never starts another tuning attempt. Empirical
    mean recall is not a confidence bound. Use fresh questions after manual tuning.
    """
    q=np.asarray(queries,dtype=np.float32);gt=np.asarray(ground_truth)
    labels=list(partitions)
    if q.ndim!=2 or q.shape[1]!=index.dim or gt.shape!=(len(q),top_k) or not 1<=top_k<=64:
        raise ValueError('Invalid query/GT shapes')
    if len(labels)!=len(q) or any(p not in ('calibration','validation') for p in labels):
        raise ValueError('Invalid partitions')
    train=[i for i,p in enumerate(labels) if p=='calibration'];valid=[i for i,p in enumerate(labels) if p=='validation']
    if min(len(train),len(valid))<2:raise ValueError('Need both query partitions')
    if not 0<target_recall<=1 or not 1<=initial_probes<=maximum_probes or not top_k<=initial_rerank<=maximum_rerank:
        raise ValueError('Invalid search budgets')
    if latency_target_ms is not None and (not np.isfinite(latency_target_ms) or latency_target_ms<=0):
        raise ValueError('Invalid latency target')
    if not np.issubdtype(gt.dtype,np.integer) or any(len(set(row))!=top_k for row in gt) or np.any(gt<0) or np.any(gt>=index.count):
        raise ValueError('GT must contain unique existing IDs')
    if not np.isfinite(q).all():raise ValueError('Invalid queries')
    def measure(config,indices):
        rows=[]
        with index.context(nprobe=config['maximum'],rerank=config['rerank'],threads=1,memory_bytes=memory_bytes) as ctx:
            ctx.adapt(minimum=config['minimum'],gap=0)
            for i in indices:
                if before_query:before_query()
                started=time.perf_counter();ids,_,stats=ctx.search(q[i],top_k=top_k)
                rows.append(dict(query=i,recall=len(set(ids)&set(gt[i]))/top_k,ms=1000*(time.perf_counter()-started),stats=stats))
        return rows,{'queries':len(rows),'recall':float(np.mean([r['recall'] for r in rows])),
                     'p95_ms':float(np.percentile([r['ms'] for r in rows],95))}
    probes=initial_probes;rerank=initial_rerank;history=[];selected=None;reason=None
    while True:
        config=dict(minimum=probes,maximum=probes,filtered_minimum=probes,rerank=rerank,gap=0)
        rows,summary=measure(config,train)
        record={'configuration':config,'summary':{'train':{'null':summary}},'rows':rows}
        history.append(record)
        if summary['recall']+1e-12>=target_recall:
            if latency_target_ms is None or summary['p95_ms']<=latency_target_ms:selected=config
            else:reason='Recall reached, but this configuration exceeds p95 target; no claim that other structures cannot succeed'
            break
        trace=diagnose(index,q[train],gt[train],config,memory_bytes=memory_bytes)
        record['diagnostic']=trace
        routing=float(np.mean([r['routing_recall'] for r in trace['rows']]))
        if routing+1e-12<target_recall:
            if probes==maximum_probes:reason='Routing budget exhausted';break
            probes=min(maximum_probes,probes*2)
        else:
            if rerank==maximum_rerank:reason='Candidate budget exhausted';break
            rerank=min(maximum_rerank,rerank*2)
    passed=False
    if selected is not None:
        rows,summary=measure(selected,valid)
        history[-1]['validation_rows']=rows;history[-1]['summary']['validation']={'null':summary}
        passed=summary['recall']+1e-12>=target_recall and (latency_target_ms is None or summary['p95_ms']<=latency_target_ms)
        if not passed:reason='Independent validation failed; no automatic retuning on validation'
    return dict(version=1,fingerprint=fingerprint(index),target_recall=target_recall,top_k=top_k,
        validated=passed,configuration=selected,calibration_count=int(index.count),measurements=history,
        partitions=labels,latency_target_ms=latency_target_ms,reason=reason,
        limitations='Frozen unfiltered fixed-index empirical calibration; first qualifying budget, not global latency optimum; no confidence-bound guarantee; total RAM requires external cgroup')
