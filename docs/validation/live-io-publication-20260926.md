# Live reader and insertion regression checks

Historical results for the September 26 implementation. The live layout and
default IO policy below were superseded by [contiguous cells and direct IO](contiguous-direct-20260927.md)
after the 10M-vector comparison on September 27.

The native adaptive reader now batches up to 64 chunk reads, overlaps the next
batch with scoring, and batches exact rerank reads. It reuses each query's two
bounded buffers. Mutable chunks keep buffered IO by default; direct IO remains
an explicit option. The contiguous frozen reader retains direct IO by default.

Durable insert batches now route, encode, write and fsync outside the exclusive
live lock. Short reservation/publication sections remain. Queries continue to
hold a shared lock, so this is not a lock-free engine. Checkpointing, updates,
deletes and metadata mutations can still block readers.

Engine changes: `01dcdeb` and `68a6bc3`, compared with `a62389b`.

## Fresh adaptive builds on real 768d vectors

Both runs rebuilt a **1,000,000-vector** prefix of Qdrant/FineWeb-10B from scratch
on GB10. They used the same 128 real query embeddings and exhaustive GT for that
prefix, 1,536 probes, 400 reranks, cell capacity 2,048 and batches of 256 inserts.
Ingestion/worker used CPU7; one query thread used CPU8. Queries ran continuously
with a 1 ms pause during the final quarter of ingestion, then a quiescent pass
measured recall on the complete prefix. No metadata or cache eviction was used.

| Measurement | Before | After |
|---|---:|---:|
| Recall@10, complete prefix | 99.766% | 99.844% |
| Quiescent p50 | 85.41 ms | 76.29 ms |
| Quiescent p95 | 104.84 ms | 78.48 ms |
| Concurrent p50 | 107.81 ms | 75.10 ms |
| Concurrent p95 | 122.58 ms | 104.83 ms |
| Total ingestion including final split drain | 227.79 s | 236.25 s |
| Concurrent queries completed | 1,251 | 2,011 |

Both processes stayed within the effective 1,999,998,976-byte cgroup limit,
including page cache, with no swap or OOM. Both reached that limit. The final
quarter took 136.42 s before and 162.37 s after: lower query latency also produces
more query load in this closed-loop workload. Total ingestion was 3.7% slower.

The latency/recall gate passed (at most +10% p50, +20% p95 and 0.005 recall loss).
Rebuilds produced different topologies: 1,398 and 1,358 cells. At this size all
cells fit within the probe budget. These results do not establish recall or
latency for the full 50M campaign, nor for cold reads. Concurrent prefix queries
were not scored against full-prefix GT.

The unrelated 50M ingestion was frozen during both runs and resumed afterward.
The separate MS MARCO network transfer remained active. No sanitizer or other
benchmark ran on GB10 during this pair. This is one sequential paired run.

Reports: [before](live-io-20260926/live-before.json),
[after](live-io-20260926/live-after.json). Raw query records remain in
`/root/fission-live-controlled-20260926` on the test host. The reproducible runner
is `tests/bench_live_regression.py`; reports include input fingerprints.

## Existing frozen MS MARCO snapshot

The **113,520,750 × 1024d**, 200,000-representative snapshot was not rebuilt.
Both libraries used the original 600 validation queries, 1,536 probes and 400
reranks on CPU5 with a 1 GB cgroup and no swap. `POSIX_FADV_DONTNEED` was requested
before every query; codes used direct overlapping reads. Hardware caches were
not flushed. No ingestion was running, but the separate transfer remained active.

| Measurement | Before | After |
|---|---:|---:|
| Recall@10 | 96.25% | 96.25% |
| p50 | 54.83 ms | 54.82 ms |
| p95 | 64.54 ms | 64.91 ms |

All 600 ID lists and scores match exactly. This confirms preservation of the
frozen reader, not a new 52.69 ms measurement or an independent recall panel.
The before cgroup reached its 1 GB cap; the after peak was 346.65 MB. File-cache
accounting differs, so this pair does not establish a memory reduction.

Reports: [before](live-io-20260926/frozen-before.json),
[after](live-io-20260926/frozen-after.json),
[benchmark script](live-io-20260926/bench_frozen.py).

## Correctness and future guards

- 169 tests passed on ARM64 and x86-64; the x86 package/wheel also built. The ARM
  full-suite pass preceded the default buffered-live policy change; its four
  IO tests passed again afterward. The x86 suite used the final IO policy.
- 32 fission/IO/concurrency checks passed under ASan and UBSan on ARM64, with
  Python-host leak detection disabled. This sanitizer pass preceded the default
  buffered-live policy change; final buffered/direct IO checks passed separately.
- IO schedules preserve exact IDs/scores on the same index at dimensions 128,
  768 and 1024, including filtered reads, tail chunks and corrupt in-flight IO.
- Deterministic barriers verify searches during prepared and durable-but-not-yet-
  published appends, checkpoint interaction, failure and crash recovery.
- Engine pull requests get a fresh synthetic rebuild comparison with exhaustive
  GT under 2 GB. Shared-runner thresholds are +25% p50, +50% p95 and 0.005 recall
  loss. The same workflow accepts a manually selected baseline revision.

The first non-isolated IO-only experiment failed its quiescent latency gate;
its timings are not used above. An identical-index check also showed that forcing
direct reads can hurt cache-resident mutable chunks, motivating their buffered
default. Large adaptive indexes still need their own direct/buffered evaluation.
