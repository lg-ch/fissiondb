# Live ingestion, cell capacity and resident representative width

These are sequential experiments on the same GB10 NVMe collection, using the
`Qdrant/FineWeb-10B` subset and 768d embeddings described in the
[10M direct-IO report](contiguous-direct-20260927.md). They measure FissionDB,
not the Qdrant engine. The residual transform remains 1024-wide with 512-bit
codes throughout. Representative RAM is a separate setting from that transform.

## 50,000 durable insertions while querying

The existing 10M collection received source rows 10,000,000 through 10,049,999,
ending at 10,050,000 vectors. The cgroup had a 2,000,000,000-byte memory limit,
no swap and CPUs 5–7 only. These are GB10 performance cores (maximum 3.9 GHz).
The ingest thread was pinned to CPU7, the native split worker to CPU6, and one
continuous query thread to CPU5. Kernel io_uring helpers also ran within this
three-core allocation; there were not just three OS threads in total.

Inserts used durable group commits of 256 vectors. There was no metadata or
HTTP transport. Queries used 1,536 probes, 400 reranks and default direct code
reads, with no forced page-cache eviction. Every query verified that all 1,536
code reads were direct. Journal/rerank reads remain buffered.

Periodic checkpoint scheduling was disabled; the final checkpoint was timed
separately. The appended journal bytes (208.8 MB) were below the default 256 MiB
checkpoint threshold. This run therefore does not measure repeated checkpoints
over sustained ingestion. The searcher remained active through split drainage
and the final checkpoint; the table's concurrent sample contains only queries
fully inside the insertion interval.

| Phase | Queries | Median | p95 | Recall@10 |
|---|---:|---:|---:|---:|
| Before ingestion, 10M | 512 | 75.86 ms | 77.35 ms | 97.1484% |
| During the 50k insertions | 721 | 77.94 ms | 83.00 ms | Not measured against a changing prefix |
| After drainage, 10.05M | 512 | 74.09 ms | 77.16 ms | 97.1680% |

All 50,000 insertions were acknowledged in **56.77 s: 880.69 vectors/s**.
Including completion of every queued split took **82.12 s: 608.84 vectors/s**.
The remaining drain was 25.35 s; the final checkpoint took 318.41 ms.
There were 59 splits, bringing the cell count from 13,725 to 13,784, with no
pending splits and a largest cell of 2,048 at completion. This short run is not
evidence that 881 vectors/s can be sustained with a bounded backlog indefinitely.

Split preparation/publication averaged about 1.28 s per split under this load;
the maximum was 2.01 s. The maximum final publication lock hold was 0.178 ms.
Mean query lock wait was 0.055 ms. The preparation and query IO timers overlap;
they are not additive wall-time components.

Exact ground truth for 10.05M was formed by merging the existing exhaustive FP32
top100 for 10M with an exhaustive normalized FP32 cosine scan of all 50k new
vectors for all 512 queries. No neighbors from the larger published corpus were
substituted. The cgroup reached its rounded 1,999,998,976-byte limit without OOM.

## Lower split threshold, fixed corpus and query budgets

The native `set_fission_capacity(1536)` API queued existing oversized cells on
the same 10.05M collection. No assignments were edited manually. The change
persisted in 5.38 ms; 5,420 ordinary background splits took 880.07 s to finish
with no concurrent ingestion or search. The cell count rose to **19,204**.

The principal comparison retained **1,536 probes and 400 reranks**. Every result
before the threshold change matched the preceding 10.05M run exactly. There
were 512 real queries per pass, one search core, direct code IO, no eviction and
the same 2 GB cgroup. An additional pass reduced probes to 1,024 explicitly.

| Cell capacity | Probes | Mean candidate entries | Median | p95 | Recall@10 |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 1,536 | 2,273,435 | 73.13 ms | 74.26 ms | 97.1680% |
| 1,536 | 1,536 | 1,614,036 | 61.31 ms | 68.30 ms | 96.5234% |
| 1,536 | 1,024 | 1,075,942 | 38.49 ms | 50.28 ms | 95.2930% |

At the fixed probe budget, candidate entries fell by 29.0%, and median latency
by 16.2%. Recall also fell by 0.645 percentage point. This is a useful measured
tradeoff above 95% recall on this corpus, not an equal-recall speedup or a
guarantee for all 768d data. The threshold change used the previous padded
representatives so that its effect could be separated from the RAM correction.

## Native-width resident representatives

Adaptive int8 representatives now retain only the input components in RAM:
**768 bytes per representative instead of 1,024** for this corpus. Integer
routing scans those 768 components. Residual encoding and scoring reconstruct
the zero tail in bounded scratch before the existing 1024-wide transform.
Cells, assignments, quantization and the 512-bit residual format do not change.
The existing checkpoint wire format remains compatible; reading packs its
centers, and writing reconstructs zero tails in a bounded 64 KiB IO buffer.

Tests compare native centers with a padded reference on dimensions 1, 3, 31,
63, 96, 129, 384, 512, 768, 1000 and 1024. They check exact encoded bytes,
representatives, returned IDs/scores, candidate counts, reopen, updates, deletes
and subsequent inserts, as well as the allocated representative bytes.

The native-width build also reopened the unchanged 10.05M collection and ran
all 512 real queries with 1,536 probes, 400 reranks and the same 2 GB limit.
**Every returned ID, FP32 score and candidate count matched the padded-center
reference exactly.** Recall remained **96.5234%**. The observed median was
**54.27 ms**, with **54.86 ms p95**; opening took 102.98 seconds and is excluded
from retrieval timings. There was no concurrent ingestion or forced eviction.

This last timing is not an isolated estimate of the packing speedup. Mean routing
time changed from about 0.88 to 0.70 ms, while the overlapping IO timer changed
from 14.77 to 7.73 ms between passes. IO/cache variability contributes to the
61.31-to-54.27 ms difference. Exact result preservation and the center allocation
reduction are the directly verified effects of packing.

The 25% RAM saving is for a fixed number of 768d centers. More cells and spare
array capacity affect total memory. Here, 19,204 native centers use **14,748,672
bytes**, versus **14,114,816 bytes** for the original 13,784 padded centers:
about 4.5% more used center payload after both changes. The representative array
also crosses from 16,384 to 32,768 allocated slots. Thus the new allocated
center payload is 25,165,824 bytes versus the original 16,777,216; total RAM
is not unchanged. The engine exposes used and allocated center byte counters.

The per-dimension capacity formula is not a global default change in this
revision. Existing collections keep their persisted threshold, and the tested
10.05M collection now uses 1,536. Recall and allocation behavior still need to
be checked for other corpus geometries before generalizing that policy.

## Evidence and validation

The full suite passed **202 tests on ARM and 202 on x86**. A further **48 focused
tests passed with AddressSanitizer and UndefinedBehaviorSanitizer on x86**,
covering native center widths, capacity changes, direct IO, contiguous layouts,
concurrent appends and background splits. Capacity tests include reopen,
concurrent insertion and crashes on either side of the durable configuration
change. These checks supplement, rather than replace, the large-corpus results.
The setuptools build also passed an incremental rebuild check after touching
the new center include, followed by a native-width query and capacity-API smoke
test. All native headers and includes are now explicit extension dependencies.

Machine-readable reports:

- [Live 50k insertion report](live50k-native-centers-20260927/live50k.json), built
  from commit `32a98c306387ce51ba5b20aaf390af6360daa6a8`.
- [Capacity 1,536 comparison](live50k-native-centers-20260927/capacity1536.json),
  with the new capacity API and padded reference centers.
- [Native center comparison](live50k-native-centers-20260927/native-centers.json).

Each report records its tested library SHA-256 and protocol. The intermediate
capacity build and the final native-center build are intentionally distinguished.
Per-thread `/proc` dumps are omitted from the published live report; counters
and measured results are preserved. All three runs completed without OOM.
