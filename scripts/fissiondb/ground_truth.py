"""Exhaustive normalized float64 reference over the served float16 base.

Sequential reads bound resident data memory; no ANN candidate set is used.
Ties are resolved by ascending document ID. Preparation cost is not query latency.
"""
import struct
from pathlib import Path
import numpy as np


def exact_top_k(base, queries, *, top_k=10, count=None, block_rows=4096,
                query_batch=32, allowed_ids=None):
    q=np.asarray(queries,dtype=np.float64)
    if q.ndim!=2 or not np.isfinite(q).all() or top_k<1 or block_rows<1 or query_batch<1:
        raise ValueError('Invalid reference parameters')
    norms=np.linalg.norm(q,axis=1)
    if np.any(norms<=0):raise ValueError('Zero query')
    q=q/norms[:,None]
    path=Path(base)
    with path.open('rb') as stream:
        header=stream.read(8)
        if len(header)!=8:raise ValueError('Truncated header')
        n,dim=struct.unpack('<II',header)
        if dim!=q.shape[1] or path.stat().st_size!=8+n*dim*2:raise ValueError('Base shape/size mismatch')
        count=n if count is None else count
        if not top_k<=count<=n:raise ValueError('Invalid indexed prefix')
        predicates=[None]*len(q) if allowed_ids is None else list(allowed_ids)
        if len(predicates)!=len(q):raise ValueError('Predicate count mismatch')
        for i,ids in enumerate(predicates):
            if ids is None:continue
            ids=np.asarray(list(ids))
            if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or np.any(ids>=count):
                raise ValueError('Invalid allowed IDs')
            ids=np.unique(ids.astype(np.int64))
            if len(ids)<top_k:raise ValueError('Fewer eligible rows than top_k')
            predicates[i]=ids
        best_ids=np.empty((len(q),0),dtype=np.int64)
        best_scores=np.empty((len(q),0),dtype=np.float64)
        # Fixed-width state uses sentinels until each predicate has enough rows.
        best_ids=np.full((len(q),top_k),np.iinfo(np.int64).max,dtype=np.int64)
        best_scores=np.full((len(q),top_k),-np.inf)
        for first in range(0,count,block_rows):
            size=min(block_rows,count-first)
            raw=stream.read(size*dim*2)
            if len(raw)!=size*dim*2:raise ValueError('Truncated vectors')
            x=np.frombuffer(raw,dtype='<f2').reshape(size,dim).astype(np.float64)
            norm=np.linalg.norm(x,axis=1)
            if not np.isfinite(x).all() or np.any(norm<=0):raise ValueError('Invalid base vectors')
            x/=norm[:,None];doc_ids=np.arange(first,first+size,dtype=np.int64)
            for start in range(0,len(q),query_batch):
                scores=q[start:start+query_batch]@x.T
                for local,score in enumerate(scores):
                    i=start+local
                    if predicates[i] is not None:
                        ids=predicates[i];lo=np.searchsorted(ids,first);hi=np.searchsorted(ids,first+size)
                        eligible=ids[lo:hi]-first
                    else:eligible=np.arange(size)
                    merged_scores=np.concatenate((best_scores[i],score[eligible]))
                    merged_ids=np.concatenate((best_ids[i],doc_ids[eligible]))
                    order=np.lexsort((merged_ids,-merged_scores))[:top_k]
                    best_scores[i]=merged_scores[order];best_ids[i]=merged_ids[order]
        if not np.isfinite(best_scores).all():raise ValueError('Insufficient eligible results')
        return best_ids.astype(np.uint32),best_scores
