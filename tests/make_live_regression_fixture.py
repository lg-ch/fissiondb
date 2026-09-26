"""Seeded clustered vectors, independent queries and exhaustive cosine GT."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse
import json
from pathlib import Path
import struct
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--rows',type=int,default=65536)
p.add_argument('--dim',type=int,default=768);args=p.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
rng=np.random.default_rng(959);centers=rng.normal(size=(512,args.dim)).astype(np.float32)
x=(centers[rng.integers(0,len(centers),size=args.rows)]+rng.normal(0,.15,(args.rows,args.dim))).astype('<f2')
q=(centers[rng.integers(0,len(centers),size=64)]+rng.normal(0,.15,(64,args.dim))).astype(np.float32)
with (out/'base.f16bin').open('wb') as f:f.write(struct.pack('<II',len(x),args.dim));f.write(x.tobytes())
np.save(out/'queries.npy',q)
unit=x.astype(np.float64);unit/=np.linalg.norm(unit,axis=1,keepdims=True)
uq=q.astype(np.float64);uq/=np.linalg.norm(uq,axis=1,keepdims=True)
scores=uq@unit.T;ids=np.argsort(-scores,axis=1,kind='stable')[:,:10]
np.savez(out/'truth.npz',ids=ids,scores=np.take_along_axis(scores,ids,axis=1))
(out/'truth.json').write_text(json.dumps(dict(rows=len(x),dim=args.dim,queries=len(q),seed=959,
    protocol='Synthetic clustered corpus, independent queries; exhaustive FP64 cosine after float16 storage conversion')))
