"""Reproduce the fresh MS MARCO audit reference inside the NVIDIA PyTorch image.
Mount the prepared msmarco-e78737fe dataset read-only at /data and a writable
output directory at /output. Required inputs: base.f16bin, queries_jsonl query
file and the three historical GT files whose chosen IDs must be excluded.
Use nvcr.io/nvidia/pytorch:25.09-py3 with --gpus all --network none.
This is benchmark preparation, not an ingestion or query-serving dependency.
"""
import gzip,json,os,struct,time
from pathlib import Path
import numpy as np
import torch

root=Path('/data');out=Path('/output');out.mkdir(exist_ok=True)
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
rows=[json.loads(x) for x in gzip.open(root/'queries_jsonl/queries.jsonl.gz','rt')]
used=set()
for p in [root/'pilot/gt.npz',root/'pilot/gt-expanded.npz',root/'gt-full.npz']:used.update(np.load(p)['chosen'].tolist())
old_texts={' '.join(rows[i]['text'].casefold().split()) for i in used};fresh={}
for i,r in enumerate(rows):
    text=' '.join(r['text'].casefold().split())
    if text not in old_texts:fresh.setdefault(text,i)
chosen=np.random.default_rng(8341996).permutation(list(fresh.values()))[:1200]
assert len(chosen)==1200
queries=np.asarray([rows[i]['emb'] for i in chosen],np.float32)
partitions=np.array(['calibration']*600+['validation']*600)
(out/'split.json').write_text(json.dumps({'chosen':chosen.tolist(),'partitions':partitions.tolist(),'excluded_used':len(used),'seed':8341996}))

def select(scores,ids,k):
    # Sort only the compact merged top-k, deterministically by score then ID.
    order=torch.argsort(ids,dim=1,stable=True);ids=ids.gather(1,order);scores=scores.gather(1,order)
    order=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:k]
    return scores.gather(1,order),ids.gather(1,order)

def scan(path,q,count=None,checkpoint=None):
    q=torch.as_tensor(q,dtype=torch.float64,device='cuda');q=q/torch.linalg.vector_norm(q,dim=1,keepdim=True)
    k=10;best=torch.full((len(q),k),-torch.inf,dtype=torch.float64,device='cuda')
    ids=torch.full((len(q),k),2**62,dtype=torch.int64,device='cuda');offset=0
    with open(path,'rb') as f:
        n,d=struct.unpack('<II',f.read(8));n=n if count is None else min(n,count)
        if checkpoint and checkpoint.exists():
            z=np.load(checkpoint);assert np.array_equal(z['chosen'],chosen)
            offset=int(z['offset']);best=torch.tensor(z['scores'],device='cuda');ids=torch.tensor(z['ids'],device='cuda')
        f.seek(8+offset*d*2);started=time.time();initial=offset
        while offset<n:
            m=min(4096,n-offset);raw=f.read(m*d*2);assert len(raw)==m*d*2
            x=torch.tensor(np.frombuffer(raw,dtype='<f2').reshape(m,d).copy(),device='cuda',dtype=torch.float64)
            x=x/torch.linalg.vector_norm(x,dim=1,keepdim=True);scores=q@x.T
            vals,ind=torch.topk(scores,min(k,m),dim=1,sorted=True)
            # Repair ties at the cutoff, including more than k identical rows.
            tied=torch.nonzero((scores==vals[:,-1:]).sum(dim=1)>1).flatten().tolist()
            for j in tied:
                mask=vals[j]==vals[j,-1];positions=torch.nonzero(scores[j]==vals[j,-1]).flatten()
                ind[j,mask]=positions[:int(mask.sum())]
            best,ids=select(torch.cat((best,vals),dim=1),torch.cat((ids,ind+offset),dim=1),k)
            offset+=m
            if checkpoint and (offset%1048576==0 or offset==n):
                torch.cuda.synchronize();elapsed=time.time()-started
                temp=checkpoint.with_suffix('.tmp.npz');np.savez(temp,offset=offset,scores=best.cpu().numpy(),ids=ids.cpu().numpy(),chosen=chosen);os.replace(temp,checkpoint)
                (out/'progress.json').write_text(json.dumps({'rows':offset,'total':n,'elapsed_seconds':elapsed,'vectors_per_second':(offset-initial)/max(elapsed,1e-9)}))
                print((out/'progress.json').read_text(),flush=True)
    return ids.cpu().numpy(),best.cpu().numpy()

# Verify normalization, ties, and block merging against an exhaustive CPU oracle.
rng=np.random.default_rng(52);x=rng.normal(size=(8201,1024)).astype(np.float16);x[1:20]=x[0]
testq=x[:12].astype(np.float64);p=out/'gpu-check.f16bin';p.write_bytes(struct.pack('<II',*x.shape)+x.tobytes())
actual,actual_scores=scan(p,testq)
norm=x.astype(np.float64);norm/=np.linalg.norm(norm,axis=1,keepdims=True);testq/=np.linalg.norm(testq,axis=1,keepdims=True)
full=testq@norm.T;expected=np.array([np.lexsort((np.arange(len(x)),-s))[:10] for s in full])
assert np.array_equal(actual,expected)
np.testing.assert_allclose(actual_scores,np.take_along_axis(full,expected,axis=1),atol=1e-12,rtol=0)
print('GPU float64 exhaustive oracle check passed',torch.cuda.get_device_name(),flush=True)
ids,scores=scan(root/'base.f16bin',queries,checkpoint=out/'checkpoint.npz')
np.savez(out/'workload.npz',queries=queries,ids=ids,scores=scores,partitions=partitions,chosen=chosen,groups=chosen)
print('GT complete',flush=True)
