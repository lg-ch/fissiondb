# Current-engine benchmarks

This page reports measurements of the current cell/residual retrieval engine.
Each result is tied to its corpus, search parameters and hardware.

## MS MARCO v2: GB10, one CPU core

| Parameter | Value |
|---|---|
| Corpus | 113,520,750 passage embeddings, 1024 dimensions |
| Representatives | 200,000, int8 at query time |
| Cell assignments | Two per vector |
| Query budget | 1,536 cells, 400 exact reranks, top 10 |
| Queries | 600 real query-to-document validation queries |
| Recall@10 | 0.9625 |
| Median / p95 | 52.693 / 62.591 ms |
| Peak cgroup memory | 261,742,592 bytes |
| Memory cap | 999,997,440 bytes, swap disabled, no OOM |

The full 232.49 GB original float16 file and the residual index were stored on
local NVMe. The process was pinned to GB10 CPU5 and used one native query thread.
The reference source is included in snapshot `7b342ac`; subsequent public-name
changes do not change its retrieval algorithm.

The harness requested `POSIX_FADV_DONTNEED` on originals and residuals before
every query. Residual reads used direct IO with 64-cell overlapping batches.
SSD hardware caches were not flushed. Timings cover the native retrieval call;
index loading, HTTP, network transit, ingestion and simultaneous queries are
outside the timed section. The 600-query panel was reused during development,
so this is empirical performance, not certification on unseen workloads.

Mean stage timings were 7.51 ms for representative routing, 8.03 ms of visible
code IO, 34.90 ms for candidate scoring and 2.11 ms for exact reranking. IO and
scoring overlap. Mean aligned code traffic was 138.39 MB per query.

[Machine-readable result](benchmarks/msmarco-arm-20260926.json)

## Comparing another machine

Use the same immutable index, complete original vectors, query/GT files, query
order, memory cap, cache protocol and search budgets. Report recall again on the
target machine; floating-point differences across architectures can change ties.

Report median and tail latency, peak total cgroup memory, OOM events and the
routing/IO/scoring/reranking breakdown. Cloud VM comparisons also reflect CPU
sharing and storage performance; they do not isolate the instruction set alone.
