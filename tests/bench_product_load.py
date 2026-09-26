"""Explicit load test: one query caller, durable writer and background packing.

Run under an external cgroup. This measures latency/ingestion, not full-corpus
ANN recall. Filter GT is exact over the newly inserted documents only.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fissiondb.anchors import AnchorIndex


def main():
    p=argparse.ArgumentParser()
    for name in ('index','base','queries','live-dir','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--residual')
    p.add_argument('--calibration', help='Validated profile for adaptive queries')
    p.add_argument('--existing-live', action='store_true')
    p.add_argument('--metadata-file')
    p.add_argument('--where-file')
    p.add_argument('--vectors', help='Optional f32bin real vectors for ingestion')
    p.add_argument('--inserts',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--nprobe',type=int,default=1024)
    p.add_argument('--rerank',type=int,default=1000)
    args=p.parse_args()
    live=Path(args.live_dir)
    if live.exists() and not args.existing_live:raise ValueError('Use a new live directory for a reproducible load test')
    dim=int(np.fromfile(args.base,np.uint32,count=2)[1])
    q=np.fromfile(args.queries,np.float32,offset=8).reshape(-1,dim)
    if args.inserts < 1 or not 1 <= args.batch_size <= 256:raise ValueError('Invalid insertion/batch count')
    rng=np.random.default_rng(712)
    vectors=(np.fromfile(args.vectors,np.float32,offset=8).reshape(-1,dim)[:args.inserts].copy()
             if args.vectors else rng.normal(size=(args.inserts,dim)).astype(np.float32))
    if len(vectors)!=args.inserts:raise ValueError('Insufficient ingestion vectors')
    vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
    real_metadata=json.loads(Path(args.metadata_file).read_text()) if args.metadata_file else [{} for _ in vectors]
    if len(real_metadata)!=len(vectors):raise ValueError('Metadata count mismatch')
    concurrent_filter=json.loads(Path(args.where_file).read_text()) if args.where_file else {'bucket':5}
    report={'query_threads':1,'writer_threads':1,'packing_enabled':bool(args.residual),'vector_source':args.vectors or 'synthetic seed 712','cache':'natural cache, no forced drops during concurrent work',
            'seed':712,'batch_size':args.batch_size,'insertion_fsync':'journal and derived rows fsynced before acknowledgement; grouped when batch-size > 1',
            'baseline':[],'concurrent':[],'post_pack':[],'inserts':[],'memory':[],'errors':[]}
    end=threading.Event();stop_monitor=threading.Event();started=time.perf_counter()
    cg=None
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):cg=Path('/sys/fs/cgroup')/line[3:].lstrip('/')
    def monitor():
        while not stop_monitor.wait(.1):
            row={'seconds':time.perf_counter()-started}
            status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines())
            row['rss_bytes']=int(status['VmRSS'].split()[0])*1024
            if cg:
                row['cgroup_bytes']=int((cg/'memory.current').read_text())
                stat=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
                row.update(anon_bytes=int(stat['anon']),file_bytes=int(stat['file']))
            report['memory'].append(row)
    watcher=threading.Thread(target=monitor);watcher.start()
    try:
        with AnchorIndex(args.index,args.base,residual_dir=args.residual,live_dir=live,
                         auto_pack_bytes=1048576 if args.residual else 0,auto_pack_interval=.5) as idx:
            context=(idx.calibrated_context(json.loads(Path(args.calibration).read_text()),memory_bytes=850000000)
                     if args.calibration else idx.context(nprobe=args.nprobe,rerank=args.rerank,memory_bytes=850000000))
            with context as query:
                def search(i,where=None):
                    t=time.perf_counter();ids,scores,stats=query.search(q[i%len(q)],where=where)
                    return {'ms':1000*(time.perf_counter()-t),'filtered':where is not None,'count':len(ids),'stats':stats}
                for i in range(min(40,len(q))):report['baseline'].append(search(i))
                def write():
                    try:
                        for begin in range(0,len(vectors),args.batch_size):
                            stop=min(begin+args.batch_size,len(vectors))
                            metadata=[{**real_metadata[i],'bucket':i%10,'sequence':i} for i in range(begin,stop)]
                            t=time.perf_counter()
                            if args.batch_size==1:
                                ids=[idx.insert(vectors[begin],metadata[0])]
                            else:
                                ids=idx.insert_batch(vectors[begin:stop],metadata,group_commit=True)
                            ms=1000*(time.perf_counter()-t)
                            report['inserts'].extend({'id':int(doc_id),'ms':ms/len(ids),'batch_ms':ms} for doc_id in ids)
                            print(json.dumps({'inserted':stop}),flush=True)
                    except Exception:report['errors'].append(traceback.format_exc())
                    finally:end.set()
                t=time.perf_counter();writer=threading.Thread(target=write);writer.start();i=0
                while not end.is_set():
                    report['concurrent'].append(search(i,concurrent_filter if i%5==4 else None));i+=1
                writer.join();report['ingestion_wall_seconds']=time.perf_counter()-t
                if report['errors']:raise RuntimeError(report['errors'])
                if args.residual:idx.pack_live()
                for i in range(min(40,len(q))):report['post_pack'].append(search(i))
                for bucket in (0,5,9):
                    eligible=np.flatnonzero(np.arange(len(vectors))%10==bucket)
                    if not len(eligible):continue
                    query_vector=vectors[bucket%len(vectors)]
                    ids,scores,_=query.search(query_vector,where={'bucket':bucket})
                    exact=vectors[eligible].astype(np.float64)
                    exact/=np.linalg.norm(exact,axis=1,keepdims=True)
                    qunit=query_vector.astype(np.float64);qunit/=np.linalg.norm(qunit)
                    order=np.argsort(-(exact@qunit))[:10]
                    expected=[report['inserts'][int(eligible[j])]['id'] for j in order]
                    eligible_ids={report['inserts'][int(j)]['id'] for j in eligible}
                    assert len(ids)==len(expected) and set(ids)<=eligible_ids
                    delta=float(max(0,np.mean((exact@qunit)[order])-np.mean(scores)))
                    recall=len(set(ids)&set(expected))/len(expected)
                    report.setdefault('filtered_quality',[]).append(dict(bucket=bucket,recall=recall,delta_cos=delta))
                    if args.vectors:assert delta < 2e-6
                    else:assert list(ids)==expected
                report['filtered_exact_checks']=3
                report['live_stats']=idx.stats()
                report['pack_runs']=idx._packer.runs if idx._packer else 0
                report['pack_error']=idx._packer.last_error if idx._packer else None
                if report['pack_error']:raise RuntimeError(report['pack_error'])
        with AnchorIndex(args.index,args.base,residual_dir=args.residual,live_dir=live) as idx:
            assert idx.count==report['live_stats']['allocated_count']
            report['reopen_count']=idx.count
    finally:
        stop_monitor.set();watcher.join()
        report['peak_rss']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if cg:report['cgroup_events']=(cg/'memory.events').read_text()
        for name in ('baseline','concurrent','post_pack'):
            for filtered in (False,True):
                values=[r['ms'] for r in report[name] if r['filtered']==filtered]
                if values:report[name+('_filtered' if filtered else '_unfiltered')+'_summary']={
                    'queries':len(values),**{f'p{k}_ms':float(np.percentile(values,k)) for k in (50,95,99)}}
        if report['inserts']:
            report['inserts_per_second']=len(report['inserts'])/report.get('ingestion_wall_seconds',1)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k.endswith('_summary') or k in ('inserts_per_second','peak_rss','pack_runs','pack_error')}))


if __name__=='__main__':main()
