"""Full MS MARCO retrieval, same 600 validation queries as the GB10 reference."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import struct
import subprocess
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pass-id', type=int, required=True)
parser.add_argument('--mode', choices=['avx2','avx512bw'], required=True)
args = parser.parse_args()
root = Path('/root/fissiondb-bench/msmarco')
data = root / 'data'
repo = Path('/root/FissionDB-avx512-dev')
output = root / 'results-avx512-20260930'
output.mkdir(exist_ok=True)
cg = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
assert 999000000 <= int((cg / 'memory.max').read_text()) <= 1000000000
assert (cg / 'memory.swap.max').read_text().strip() == '0'
assert os.sched_getaffinity(0) == {0}
expected_sizes = {'base.f16bin': 232490496008, 'index-200k/anchors.bin': 819200000,
                  'index-200k/offs.bin': 1600008, 'index-200k/blocks.bin': 29954363208,
                  'index-200k/scale.bin': 4096, 'residual-200k/res512.bin': 16338743568}
for path, size in expected_sizes.items():
    assert (data / path).stat().st_size == size, path
with (data / 'base.f16bin').open('rb') as f:
    assert struct.unpack('<II', f.read(8)) == (113520750, 1024)
metadata = (data / 'index-200k/meta.txt').read_text().split()
assert metadata[:4] == ['200000', '1024', '2', '1'] and metadata[5] == '113520750'
reference = {r['query']: r for r in json.loads((root / 'arm-reference-rows.json').read_text())}
z = np.load(root / 'workload.npz')
queries = z['queries']; ground_truth = z['ids']
selected = np.flatnonzero(z['partitions'] == 'validation').tolist()
assert len(selected) == len(reference) == 600 and set(selected) == set(reference)
random.Random(62926).shuffle(selected)

class Stats(C.Structure):
    _fields_ = [(k, C.c_double) for k in ('anchor_ms', 'io_ms', 'score_ms', 'rerank_ms', 'total_ms')] + [(k, C.c_uint64) for k in ('entries', 'bytes')]

lib = C.CDLL(str(repo / 'libfissiondb_anchor.so'))
lib.anchor_index_open.argtypes = [C.c_char_p, C.c_char_p, C.c_int]
lib.anchor_index_open.restype = C.c_void_p
lib.anchor_index_enable_residual.argtypes = [C.c_void_p, C.c_char_p]
lib.anchor_query_create.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_uint64, C.c_char_p, C.c_int]
lib.anchor_query_create.restype = C.c_void_p
lib.anchor_query_search.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p]
lib.anchor_query_close.argtypes = [C.c_void_p]
lib.anchor_index_close.argtypes = [C.c_void_p]
index = lib.anchor_index_open(os.fsencode(data / 'index-200k'), os.fsencode(data / 'base.f16bin'), 1)
assert index
assert lib.anchor_index_enable_residual(index, os.fsencode(data / 'residual-200k')) == 0
context = lib.anchor_query_create(index, 1536, 400, 1, 900000000, None, 0)
assert context
lib.anchor_integer_backend.restype=C.c_char_p
assert lib.anchor_integer_backend().decode()==args.mode
lib.anchor_query_residual_io.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int]
assert lib.anchor_query_residual_io(context,64,1,1)==0
fds = [os.open(data / path, os.O_RDONLY) for path in ('base.f16bin', 'residual-200k/res512.bin')]

def cpu0():
    return [int(v) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('cpu0 ') for v in line.split()[1:]]

def resources():
    return {name: ((cg / name).read_text().strip() if (cg / name).exists() else None) for name in (
        'memory.max', 'memory.peak', 'memory.events', 'memory.swap.max',
        'cpuset.cpus.effective', 'cpu.stat', 'cpu.pressure', 'io.stat')}

started = time.time()
resources_before = resources()
cpu_before = cpu0()
rows = []
with (output / f'rows-pass{args.pass_id}.jsonl').open('w') as stream:
    for qi in selected:
        for fd in fds:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        query = np.ascontiguousarray(queries[qi], np.float32)
        ids = np.zeros(10, np.uint32); scores = np.zeros(10, np.float32); stats = Stats()
        cpu_start = time.process_time()
        start = time.perf_counter()
        count = lib.anchor_query_search(context, query.ctypes.data, 10, ids.ctypes.data, scores.ctypes.data, C.byref(stats))
        ms = (time.perf_counter() - start) * 1000
        cpu_ms = (time.process_time() - cpu_start) * 1000
        assert count == 10 and len(set(ids.tolist())) == 10 and np.isfinite(scores).all()
        row = dict(query=qi, ms=ms, cpu_ms=cpu_ms,
                   recall=len(set(ids.tolist()) & set(ground_truth[qi].tolist())) / 10,
                   ids=ids.tolist(), scores=scores.tolist(),
                   stats={key: getattr(stats, key) for key, _ in Stats._fields_})
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if len(rows) % 100 == 0:
            stream.flush()
            print(json.dumps({'pass': args.pass_id, 'completed': len(rows),
                              'recall_so_far': float(np.mean([r['recall'] for r in rows])),
                              'median_ms_so_far': float(np.median([r['ms'] for r in rows]))}), flush=True)
cpu_after = cpu0()
resources_after = resources()
for fd in fds:
    os.close(fd)
lib.anchor_query_close(context)
lib.anchor_index_close(index)
events = dict(line.split() for line in resources_after['memory.events'].splitlines())
assert events['oom'] == events['oom_kill'] == '0'
matching = [r for r in rows if r['ids'] == reference[r['query']]['ids']]
score_deltas = [max(abs(a-b) for a,b in zip(r['scores'], reference[r['query']]['scores'])) for r in matching]
cpu_delta = [a-b for a,b in zip(cpu_after, cpu_before)]
total_ticks = sum(cpu_delta[:8])
report = dict(
    platform='Kamatera 4A shared x86 VM, virtual non-rotational disk', pass_id=args.pass_id,
    source_base='5e2a506c21f4062a785f4dc160b36eda912e3ffd', backend=args.mode,
    library_sha256=hashlib.sha256((repo/'libfissiondb_anchor.so').read_bytes()).hexdigest(),
    n=113520750, dim=1024, anchors=200000, probes=1536, rerank=400, threads=1, queries=600,
    protocol='Full float16 originals and residual payload on guest disk. Same reused query-to-document validation panel and order as GB10. fadvise DONTNEED before every query; direct residual IO. CPU0 pinned, decimal1GB cgroup, swap0. No ingestion or concurrent queries. Physical device cache is not flushed. Native call latency excludes network and Python HTTP serving.',
    file_sizes=expected_sizes, workload_sha256=hashlib.sha256((root / 'workload.npz').read_bytes()).hexdigest(),
    started_unix=started, completed_unix=time.time(), resources=resources_after, resources_before=resources_before,
    cpu0_steal_percent=100*cpu_delta[7]/total_ticks if total_ticks else None,
    recall=float(np.mean([r['recall'] for r in rows])),
    p50_ms=float(np.median([r['ms'] for r in rows])),
    p95_ms=float(np.percentile([r['ms'] for r in rows], 95)),
    p99_ms=float(np.percentile([r['ms'] for r in rows], 99)),
    mean_ms=float(np.mean([r['ms'] for r in rows])),
    mean_cpu_ms=float(np.mean([r['cpu_ms'] for r in rows])),
    mean_stages={k: float(np.mean([r['stats'][k] for r in rows])) for k, _ in Stats._fields_},
    arm_comparison=dict(reference_p50_ms=float(np.median([r['ms'] for r in reference.values()])),
                        reference_recall=float(np.mean([r['recall'] for r in reference.values()])),
                        identical_top10_queries=len(matching),
                        different_top10_queries=[r['query'] for r in rows if r['ids'] != reference[r['query']]['ids']],
                        equal_entries_queries=sum(r['stats']['entries'] == reference[r['query']]['stats']['entries'] for r in rows),
                        max_score_delta_for_identical_ids=max(score_deltas, default=None)))
(output / f'report-pass{args.pass_id}.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
