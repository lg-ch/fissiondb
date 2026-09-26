# NVMe product integration — 26 September 2026

The product branch now defaults to 400 reranks, overlaps native residual reads
with scoring, dispatches exact AVX2/F16C kernels on supported x86 CPUs, and
supports input dimensions 1 through 1024. Durable deletion remains shared by
the TQ, residual and packed-live search paths.

## Full MS MARCO product comparison

The existing **113,520,750 × 1024d** snapshot with **200,000 fixed anchors** was
opened without rebuilding it or its legacy residual files. Both libraries used
1,536 probes, 400 reranks and the same 600 real query-to-document validation
queries. The pre-change library was built from commit `0194375`.

| Product library | Recall@10 | Median | p95 | Peak cgroup RAM |
|---|---:|---:|---:|---:|
| Before integration | 96.25% | 88.65 ms | 107.27 ms | 659.31 MB |
| Integrated | 96.25% | **52.69 ms** | **62.59 ms** | **261.74 MB** |

Median latency decreased by 40.6%; measured peak memory by 60.3%. All 600 final
ID lists, scores and scored-entry counts match. Mean residual traffic increases
from 132.85 to 138.39 MB/query because direct reads round cell boundaries to
4 KiB. There is no additional persistent slot copy in the product: its existing
dense residual file already keeps each cell contiguous.

Each confirmation ran in a separate process and cgroup, pinned to GB10 CPU5,
with a 999,997,440-byte effective cap, no swap, no OOM and one query thread.
The full 232.49 GB float16 file was on NVMe. `POSIX_FADV_DONTNEED` was requested
before every query; the integrated code reader also used `O_DIRECT`. SSD hardware
caches were not flushed. No ingestion, competing test job or concurrent query
was running during the isolated confirmation. The 600-query panel was reused;
these results are not an independent certification of future recall.

An initial paired trial preceded the isolated confirmation; sanitizer tests ran
alongside part of that initial baseline. Its timings are therefore not the
figures above. Confirmation reversed library order and ran after those tests.

Evidence: [before](nvme-product-before-20260926.json),
[integrated](nvme-product-integrated-20260926.json),
[independent result audit](nvme-product-audit-20260926.json).
Full rows and binaries remain in `/root/mangrove-product-20260926` on GB10.
The benchmark and audit scripts are preserved beside these reports.

## Separate adaptive-index rerank check

The earlier experimental snapshot has 153,009 adaptive anchors and a different
160-query panel. With its optimized reader, 400 reranks give 96.625% recall,
61.20 ms median, 67.28 ms p95 and 419.34 MB peak. There were three passes per
variant and identical results against its reference reader. This is a separate
index/measurement from the product comparison above.

[Adaptive rerank-400 results](adaptive-r400-20260926.json).

## Format and correctness validation

- 136 product tests pass on ARM64; 67 residual/quantization/dimension tests pass
  under ASan and UBSan, including native builds, failed reads and live packing.
  Leak checking is disabled for the Python host.
- Every integer input width from **1 to 1024** was built, converted, searched
  against independent float64 cosine results, live-packed and deleted on a tiny
  corpus. [Exhaustive width report](dimensions-verified-20260926.json).
- Tests also limit candidate reranking to four of 512 vectors to exercise the
  approximate codec, rather than validating only exhaustive reranking.
- Serial/overlapped and buffered/direct reads give identical results; concurrent
  contexts use separate rings and buffers. This checks correctness, not a large
  corpus throughput claim.
- 102,400 standalone exact x86 kernel comparisons pass, including all 65,536
  float16 bit patterns against the existing x86 conversion semantics.
- 47 focused dimension, IO and quantization tests also pass on x86-64. The full
  x86 suite was stopped during the unrelated 5,120 individually fsynced metadata
  writes on the shared VM; the full suite result above is ARM64.

The codec uses four bits through padded dimension 128, two through 256, and one
through 512. At padded dimension 1024 it encodes 512 rotated coordinates. Public
vectors and original storage keep their original width; only internal rotation
buffers and anchors are padded. V1 1024d residuals stay compatible. Other widths
use `MGRRESV2`, while retaining 72-byte records and the `res512.bin` filename.

Each query normally reserves two 8 MiB read buffers instead of the previous
128 MiB wave. A lower allocation budget reduces these buffers; an unusually
large cell can enlarge them within the same total budget. The index is shared
between contexts. Additional concurrent contexts still consume private scratch
space and require load testing for a global RAM/latency target.

Deleting an ID records a durable tombstone, filters it before candidate admission
and survives packing, updates, compaction and restart. IDs are never reused.
Compaction reclaims obsolete live records, **not immutable frozen files**.
Physical reclamation of frozen vectors/codes remains future work. Likewise,
support for a dimension does not guarantee 95% recall for every geometry.
