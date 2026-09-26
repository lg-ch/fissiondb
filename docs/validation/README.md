# Validation evidence

- `247m-1gb.json`: 200 cold NVMe doc-to-doc queries on 247,154,006 frozen 1024d
  vectors, normalized cosine exact GT, one native query thread. Source state
  recorded by product commit 2f65cfd, before the later bulk metadata API.
  nprobe=1024, rerank=1001; request 11 results, then remove the source document
  and keep ten. This differs by one rerank slot from the original prototype.
- `final-cgroup.txt`: associated 999,997,440-byte total-memory cgroup, swap disabled.
  The maximum includes page cache; RSS is not the total memory consumption.
- `load-batch64-247m.json`: earlier integrated source f9b7cf7, 1,000 synthetic
  inserted vectors, 64-vector durable groups, one query caller, automatic packing.
  Natural page cache throughout concurrent work. Only 120 concurrent unfiltered
  and 30 filtered queries; do not treat these tails as a long-running load test.
- `test-postings-final.log`: 65 tests on isolated GB10 product tree, including
  native bulk metadata filtering, old journal migration and interrupted-tail retry.
- `sanitizer-postings.log`: 18 residual/live tests with AddressSanitizer and
  UndefinedBehaviorSanitizer; leak detection disabled for the Python host.
- `docker-query-uring.log`: actual ARM64 container search with an explicit
  Moby-derived seccomp profile allowing io_uring. Default Docker seccomp rejected
  context creation, which is why README documents the required profile.

GT and reports are kept on the GB10. On 2026-09-13 the user explicitly approved removal of old Wikipedia/DEEP bases, code blocks and f100 forest data to make room for FineWeb2; rerunning those old benchmarks now requires recovering their raw datasets. reproducible campaign and 247M runner
scripts are archived in research/archive-20260912 under
benchmarks/datasets_20260912. No 1B or S3 residual performance is claimed here.

## Common Crawl TQ4 correction and tuning (2026-09-13)

`commoncrawl-tq4.json` contains all aggregate unfiltered and filtered results of
three configurations after the four-accumulator lookup optimization.
`evaluate-tq4-unroll-cgroup.txt` records total memory including page cache;
no OOM or swap. `test-lut-unroll.log` records 73 passing tests.
The original serial lookup at 64 probes / rerank 2000 measured 0.978 recall,
39.8 ms median, 68.6 ms P95; four independent sums retain that recall at
33.0 ms median, 57.5 ms P95. Most of the gain from the earlier 250 ms result
comes from reducing probe count, not this arithmetic change.
Filtering occurs before candidate admission. Rare filters require more probes;
the unfiltered setting is not a universal filtered recall guarantee. There is
no ingestion during these cold measurements. Raw rows and launch scripts are
preserved in the research archive. The frozen live journal is kept separately
from the journal used for concurrent insertion.

`live-results-tq4.json` records the 10,000-vector concurrent load run on a
separate snapshot of frozen metadata, nprobe=64 / rerank=4000. Natural cache,
64-vector durable groups, one caller and one writer. Full frozen-corpus recall
is not measured while the corpus changes; three exact filtered comparisons
cover only the inserted vectors. See `live-tq4-load-cgroup.txt` for memory.
`sanitizer-l2.log` records 26 passing sanitizer tests for the corrected lookup
score before the four-accumulator arithmetic optimization.


`fineweb-passages.json`: original passage embeddings from the same 380,000-parent
FineWeb2 shard, no averaging. 826,988 frozen passages; 2,420 passages belonging
to the reserved 1,000 parents excluded. 100 passage queries and 100 mean-document
queries are evaluated against that same passage database. Cold, single thread,
1 GB total-memory limit. The additional 12,000-candidate setting evaluates the
100 passage queries only. Parent IDs and source Parquet are preserved on GB10.
`test-projection-final.log`: 79 tests; `sanitizer-projection.log`: 24 residual and
adaptive tests under ASAN/UBSAN (Python leak detection disabled).
