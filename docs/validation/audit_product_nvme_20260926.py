"""Audit completed measurements without rerunning timed work."""
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path('/root/mangrove-product-20260926')
gt=np.load('/root/mangrove-robust-calibration/workload.npz')
queries=set(np.flatnonzero(gt['partitions']=='validation').tolist())
rows={};reports={}
for name in ['before','integrated']:
    report=json.loads((root/f'nvme-product-{name}-comparison.json').read_text())
    raw=json.loads((root/f'nvme-product-{name}-rows.json').read_text())
    assert len(raw)==600 and {r['query'] for r in raw}==queries
    assert report['rerank']==400 and report['probes']==1536
    assert int(report['resources']['memory.max'])<=1000000000
    assert int(report['resources']['memory.peak'])<=1000000000
    assert int(report['resources']['memory.swap.max'])==0
    events=dict(line.split() for line in report['resources']['memory.events'].splitlines())
    assert events['oom']=='0' and events['oom_kill']=='0'
    for row in raw:
        assert len(row['ids'])==10 and len(set(row['ids']))==10
        assert row['recall']==len(set(row['ids'])&set(gt['ids'][row['query']]))/10
        assert np.isfinite(row['scores']).all() and row['ms']>0
    summary=report['summary'][0]
    assert np.isclose(summary['recall'],np.mean([r['recall'] for r in raw]))
    assert np.isclose(summary['p50_ms'],np.median([r['ms'] for r in raw]))
    assert np.isclose(summary['p95_ms'],np.percentile([r['ms'] for r in raw],95))
    rows[name]={r['query']:r for r in raw};reports[name]=report
for qi in queries:
    a,b=rows['before'][qi],rows['integrated'][qi]
    assert a['ids']==b['ids'] and a['scores']==b['scores']
    assert a['stats']['entries']==b['stats']['entries']
files=list((root/'src').glob('*.c'))+list((root/'src').glob('*.inc'))+list((root/'src').glob('*.h'))
files += [root/'libmangrove_anchor.so',root/'nvme-product-before-rows.json',root/'nvme-product-integrated-rows.json']
result={'status':'passed','paired_queries':600,'exact_ids_scores_entries':True,
        'reference_git_commit':'0194375','validation_set':'reused 600 real queries, not newly held out',
        'source_and_evidence_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        'bench_script_sha256':hashlib.sha256(Path('/root/bench_product_nvme.py').read_bytes()).hexdigest()}
(root/'nvme-product-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'passed','paired_queries':600,'exact_ids_scores_entries':True}),flush=True)
