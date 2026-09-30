from pathlib import Path
import json
import subprocess
import time

status=Path('/root/fission-avx512-validation.json')
while True:
    data=json.loads(status.read_text()) if status.exists() else {}
    if data.get('phase')=='failed':raise RuntimeError(data)
    if data.get('phase')=='passed':break
    time.sleep(5)
root=Path('/root/fissiondb-bench/msmarco/results-avx512-20260930')
assert not root.exists(),'Do not overwrite completed benchmark passes'
root.mkdir()
for pass_id,mode in enumerate(['avx2','avx512bw','avx512bw','avx2'],1):
    (root/'progress.json').write_text(json.dumps(dict(phase='running',pass_id=pass_id,backend=mode)))
    with (root/f'pass{pass_id}.log').open('w') as log:
        subprocess.run(['systemd-run','--quiet','--wait','--pipe','--collect',
            '--unit=fission-avx512-msmarco-pass'+str(pass_id),
            '--property=MemoryMax=1000000000','--property=MemorySwapMax=0',
            '--property=AllowedCPUs=0','--property=CPUAffinity=0',
            '--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1',
            '--setenv=FISSIONDB_DISABLE_AVX512='+('1' if mode=='avx2' else '0'),
            '/usr/bin/python3','/root/bench_avx512_msmarco.py','--pass-id',str(pass_id),'--mode',mode],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    report=json.loads((root/f'report-pass{pass_id}.json').read_text())
    print(json.dumps({k:report[k] for k in ['backend','pass_id','p50_ms','p95_ms','recall','mean_stages']}),flush=True)
baseline=[json.loads(x) for x in (root/'rows-pass1.jsonl').read_text().splitlines()]
for pass_id in (2,3,4):
    other=[json.loads(x) for x in (root/f'rows-pass{pass_id}.jsonl').read_text().splitlines()]
    assert len(other)==len(baseline)==600
    for a,b in zip(baseline,other):
        assert a['query']==b['query'] and a['ids']==b['ids'] and a['scores']==b['scores']
        assert a['stats']['entries']==b['stats']['entries'] and a['stats']['bytes']==b['stats']['bytes']
(root/'progress.json').write_text(json.dumps(dict(phase='complete',passes=4,exact_results=True)))
print('All four passes: exactly identical IDs, scores, candidate counts and bytes.',flush=True)
