# Contiguous live cells and direct IO: 10M-vector check

On the GB10 NVMe, direct IO reduced median query latency from **153.44 ms to
77.20 ms**, with identical results on the same live index. This corrects the
buffered default used by the previous live reader. It does not establish the
same latency during concurrent ingestion or on other hardware.

## Measurement

The index was rebuilt from the first 10,000,000 vectors of the downloaded
50M-row subset of `Qdrant/FineWeb-10B` (768d), dataset revision
`3aea0b8d5f98741e343d07f067ce0f49dca1ccd2`. This measures FissionDB, not the
Qdrant engine. It contains 13,725 adaptive cells, with
a capacity target of 2,048 and two assignments per vector. This test retains
the existing 768-to-1024 internal padding and 512-bit residual codec; it does
not measure a new native-dimension codec.

There were 512 real MS MARCO queries, embedded with
`Alibaba-NLP/gte-multilingual-base`, with exact top-10 ground truth for this prefix.
Each query ran twice in each mode, for 1,024 measurements per mode. Query order
was shuffled, and buffered/direct order was balanced within each repetition.
Both modes used one query thread on CPU5, 1,536 probes, 400 reranks and batches
of up to 64 overlapping reads. No ingestion ran during retrieval.

The cgroup limit was 2,000,000,000 bytes with swap disabled. Kernel rounding
made the actual limit 1,999,998,976 bytes; the cgroup reached this limit without
OOM. This includes file cache and is not a measurement of process-owned RAM.
`POSIX_FADV_DONTNEED` was requested on the code, journal and row files before
every query. Device caches were not flushed. Collection open time (109.2 s),
query embedding and HTTP transport are outside the query timer.

| Code IO mode | Median | p95 | Recall@10 | Code requests/query |
|---|---:|---:|---:|---:|
| Buffered | 153.44 ms | 165.61 ms | 0.971484375 | 1,536 |
| Direct | 77.20 ms | 95.38 ms | 0.971484375 | 1,536 |

Every returned ID, score and candidate count matched exactly across modes and
the preceding layout comparison. Mean candidate entries were 2,272,084.50.
The direct-read counter was 1,536 in direct mode and zero in buffered mode.
The 400 rerank reads remain buffered. Median speedup was 1.987x.

Mean live IO submit/wait time fell from 85.76 to 15.69 ms. Mean live scoring
time was 67.55 versus 61.99 ms. These instrumented timers overlap and must not
be added into a wall-time breakdown.

The preceding buffered ABBA experiment kept all cells, members, centers and
encoded bytes identical. Contiguity reduced code requests from a mean of
5,258.42 to 1,536, but did **not** improve buffered latency: medians were
149.63/148.15 ms fragmented and 153.28/153.66 ms contiguous. Fewer requests
alone therefore did not explain the earlier latency gap. Engine requests are
not physical NVMe operations.

The 10M build took 2,853.17 s (47m33s), with 20 routing/encoding workers enabled
and a 2 GB cgroup. All 13,725 cells were within capacity when measured. The live
code arena occupied 8.13 GB of logical file space for 1.68 GB of used records;
reserved slots and retired extents account for the excess. Slot reuse does
not yet make this a compact on-disk representation. Journal and original-vector
storage are additional.

## Integration and guards

Contiguous live code reads now request direct IO by default. If opening a direct
descriptor is unsupported, the reader falls back to buffered IO; actual use is
reported in `stats['live']['direct_reads']`. Explicit buffered mode remains
available through `query.residual_io(direct=False)`.

Correctness tests check the default at dimensions 128, 768 and 1024, and compare
buffered/direct results, filtering, tail reads and corruption handling. Layout
tests cover growth, fission, reopen, legacy migration and crash boundaries.
The rebuilt performance gate's `--require-contiguous --require-direct` options
require one ordinary-cell request per selected cell and actual direct reads
during both concurrent ingestion and quiescent retrieval. The same gate also
checks recall and latency against a freshly rebuilt baseline.

The final default-policy change passed all 183 tests on both ARM64 and x86-64.
A new 65,536-vector 768d rebuild passed the stricter local gate (10% median,
20% p95 tolerance, maximum 0.005 recall loss): recall was 1.0 for both revisions,
and every code read was direct in the concurrent and quiescent passes. Its
quiescent median was 10.18 ms versus the baseline's 9.77 ms; on this small
cache-resident fixture direct IO is slightly slower, within the gate. This is
not evidence of a small-index speedup.

The [rebuilt gate report](contiguous-direct-20260927/rebuilt-gate.json) records
the exact counters. A final pass over all 512 real queries on the unchanged
10M collection used the rebuilt engine with **no explicit IO configuration**:
median **77.38 ms**, p95 **79.37 ms**, recall **0.971484375**, and exactly the
same IDs, scores and candidate counts. All 1,536 code requests per query were
direct. This separate pass confirms the new default; its p95 is not substituted
for the paired experiment above. See the [default verification report](contiguous-direct-20260927/default-direct-verification.json).

## Provenance

The paired experiment used engine commit `7b5d7434aeca0b273b02fb0cae572356aea3b6ec`
with explicit `residual_io(batch_cells=64, overlap=True, direct=mode)` on the
same query context. Library SHA-256:
`a529150fb0f04032ef026e4d86242bd69ff98adcf0aa703e97b7b7aa041d87b7`.
The machine-readable [paired report](contiguous-direct-20260927/direct-comparison.json)
contains the exact library hash, protocol, memory counters and per-stage means.
Full query records and the preserved layouts remain in
`/root/fission-qdrant10m-contiguous-20260927` on the benchmark host.
