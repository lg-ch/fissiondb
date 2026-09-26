"""Exhaustive input-width smoke check. Run separately from the quick suite.

Creates tiny native indices for all dimensions 1..1024, converts residuals,
and checks exact reranking against independent float64 cosine distances.
"""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
from test_residual_dimensions import build,AnchorIndex


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    for dim in range(1,1025):
        with tempfile.TemporaryDirectory(prefix='mangrove-dimension-') as tmp:
            directory,base,out,x=build(Path(tmp),dim,n=24)
            unit=x.astype(np.float64);unit/=np.linalg.norm(unit,axis=1,keepdims=True)
            q=np.random.default_rng(dim).normal(size=dim)
            q/=np.linalg.norm(q)
            with AnchorIndex(directory,base,residual_dir=out,live=True) as idx:
                with idx.context(nprobe=8,rerank=24) as ctx:
                    ids,scores,_=ctx.search(q)
                    np.testing.assert_allclose(scores,np.sort(unit@q)[-10:][::-1],atol=3e-6)
                    added=idx.insert(q)
                    idx.pack_live()
                    assert added in ctx.search(q,allowed_ids=[added])[0]
                    idx.delete(added)
                    assert len(ctx.search(q,allowed_ids=[added])[0])==0
        if dim%64==0:print(json.dumps({'dimensions_verified':dim}),flush=True)
    Path(args.output).write_text(json.dumps({'dimensions_verified':1024,'first':1,'last':1024,
        'checks':['native_build','residual_conversion','exact_cosine','live_pack','delete'],
        'recall_claim':'exhaustive small-corpus correctness, not large-corpus recall'})+'\n')


if __name__=='__main__':main()
