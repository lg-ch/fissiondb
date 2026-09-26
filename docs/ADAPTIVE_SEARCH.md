# Calibrated adaptive search

The shipped TQ and residual engines share adaptive anchor routing for one query
thread. A context allocates for the maximum probe and rerank budgets. For each
query it ranks anchors once, selects a calibrated minimum (larger for filters),
and doubles exploration while the relative anchor-score drop is below the
configured threshold, up to the maximum. Code blocks are streamed in waves of
at most 32 cells on the TQ path; the residual reader defaults to overlapped
64-cell batches. Survivors persist across waves and exact reranking runs once.
There is no repeated routing, cell reread, or repeated exact rerank during this
progression. This is anchor-driven selection, not a feedback loop observing
successive exact top-k results. Rerank is chosen independently during calibration
and stays fixed within a context. Highly selective filters of <=4096 IDs retain
the exact-filter shortcut.

```python
from mangrove.adaptive import calibrate

# Exact GT must match each query's predicate and self-exclusion protocol.
configs = [dict(minimum=32, maximum=256, filtered_minimum=256,
                rerank=4000, gap=g) for g in (0, .01, .02)]
profile = calibrate(index, queries, exact_ids, configs, where=predicates,
                    target_recall=.98, memory_bytes=800_000_000)
with index.calibrated_context(profile, memory_bytes=800_000_000) as query:
    ids, scores, stats = query.search(vector, top_k=10, where={'lang':'fr'})
```

Calibration uses alternating occurrences of each predicate for training and
validation by default. Pass `partitions`, one `calibration` or `validation`
label per query, to use an externally frozen split. Group duplicate question
texts before splitting. Each predicate must occur at least twice per partition;
this API minimum is not evidence of statistical reliability. `query_kind`
records query provenance. `latency_target_ms` additionally requires each group's
measured p95 to meet that target, on both calibration and validation.
Select the lowest mean training latency among configurations meeting
the target recall in every training group, then validate only that selection.
A failed profile cannot be opened; validation never selects a replacement.
Provide representative independent queries, adequate samples per filter, and
optionally a `before_query` hook to impose a cold-cache protocol. Ground truth
is supplied explicitly: the API does not silently substitute ANN results for GT.
The calibration records the frozen index fingerprint, anchor representation,
residual metadata, sample counts, raw observations and empirical recall target.
The fingerprint prevents applying a profile to another encoding/index. Recalibrate
after distribution drift or substantial mutations. New, untested predicates have
no calibrated recall guarantee. This is offline explicit calibration, not implicit
training during every index build.

## Query-to-document diagnostics and reference

`mangrove.ground_truth.exact_top_k(base, queries, ...)` scans the served float16
base sequentially with normalized float64 scoring and deterministic ID tie breaks.
It supports an indexed prefix and separate allowed-ID sets per query. It refuses
fewer eligible rows than top-k. Memory depends on block size and query batch size,
not corpus size (except caller-provided filter IDs). The implementation is a CPU
correctness reference, not a GPU-scale preparation throughput claim. Published
GT must be checked for metric, normalization, precision and ID-order agreement;
full-corpus GT is not valid for a subset or a filter by simply removing other IDs.

`mangrove.adaptive.diagnose(index, queries, exact_ids, configuration)` runs a
separate diagnostic pass. Its `routing_recall` counts GT document IDs present in
eligible entries of cells actually read. `candidate_recall` counts those selected
for exact rerank; `final_recall` counts returned IDs. Multiple assignments count
once. Routing misses and compression/selection misses are reported separately.
The diagnostic deliberately excludes latency results because watch-ID checks add
work. The native `diagnostic_ids` search option supports up to 64 unique frozen
IDs, local storage and one query thread. Live overlays are currently rejected;
explicit allowed-ID filters, including the exact shortcut, are supported.

For an offline coverage study, `ctx.search(q, diagnostic_ids=gt,
diagnostic_unique=True)` additionally reports `eligible_entries` and
`unique_vectors` in `stats['diagnostic']`. The first counts eligible assignments
encountered; the second deduplicates their IDs with a temporary Roaring bitmap.
Counts reset between searches, and normal searches release the optional bitmap.
This counts coverage before candidate caps, not the final rerank pool. Exact
filter shortcuts count each eligible vector once. The extra bitmap allocation
is not covered by the context's owned-allocation budget; use an external cgroup.
The count is known only after reading IDs; it is not a free pre-routing signal.
Time its work separately and do not publish traced timings as normal query
latency. This optional diagnostic does not implement an adaptive stopping rule.
88 tests pass, including 6 diagnostic tests under ASAN/UBSAN (Python leak
detection disabled).

MS MARCO campaign source: CohereLabs/msmarco-v2.1-embed-english-v3, pinned revision
e78737fe92ac1b783211b705c12207ca75fcc9b7. It provides 113,520,750 passages and
1,677 real TREC queries with embeddings and published flat top-1000. First-shard
preparation verifies 1,760,180 text/vector rows and 30,624 published GT IDs at their
expected offsets. Full download and pilot evaluation are separate stages; this
does not establish a recall or latency result at 113M. Native and reference changes
pass 86 tests on the GB10. The calibrator still chooses query budgets over a fixed
index: automatic codec/structure selection and confidence-bound acceptance remain
future work.

`autocalibrate(..., partitions=..., initial_probes=32, maximum_probes=1024,
initial_rerank=1000, maximum_rerank=16000)` provides a first diagnostic-guided
budget controller for frozen unfiltered indexes. It measures only calibration
queries while tuning. If their routed GT coverage falls short, it doubles the
probe budget; otherwise it doubles rerank candidates. It stops at the first
quality-qualified budget, then evaluates validation queries once. A failed p95
target, exhausted budget or failed validation produces an unvalidated profile;
it never silently relaxes the quality target. This monotone exploration is not
a globally optimal cost search and does not yet support live/metadata workloads.

On the 1,760,180-row MS MARCO pilot, this controller starts at 32 probes / 1,000
candidates and reaches 512 probes / 1,000 candidates in five attempts. It measures
mean recall .9765625 on 64 calibration questions, then .975 on 64 validation
questions at p95 31.44 ms, one query thread and 1,000,000,000-byte cgroup limit.
These questions replay the preceding expanded-grid experiment, so this is a
controller reproduction rather than a new external holdout. The queued 113M
campaign reserves previously unused questions. See
`validation/msmarco-pilot-autocalibration.json` for the protocol and limitations.

## Build memory

Assignment IDs and distances, including the hierarchical coarse assignment
copy, use shared mappings of preallocated temporary files in the output directory.
The files are unlinked immediately and disappear when the process releases the
mappings, including on process termination. These pages are reclaimable file
cache, not corpus-sized anonymous allocations. The OS/cgroup still accounts for
resident mapped pages; mapping a file is not an exemption from the memory limit.
Input and output staging uses 8,192 rows instead of 200,000 (16.8 MB input at
1024 dimensions). Anchor storage still depends on K and dimension. Temporary
disk space must accommodate two N*M*4-byte arrays, plus the coarse ID copy when
using hierarchical construction; insufficient preallocation fails explicitly.
Assignment-cache and published index formats are unchanged. Direct TQ1 and
hierarchical TQ4 builds on 25,001 vectors, including cache reloads, match the
previous binary bit-for-bit across all six index/data files. All 86 tests pass.

`latency_budget_ms` is a soft deadline checked between IO waves. It excludes no
already-started operation; filter construction, the first wave and final exact/live
reranking can exceed it. `code_bytes` limits frozen code reads, excluding exact
vectors/live IO, and allows one oversized cell. `stats.probes` reports the visited
cell count; `budget_limited` indicates early truncation and `deadline_exceeded`
reports total native time beyond the deadline. Neither a target nor stable results
constitute a per-query recall guarantee. Defaults impose no deadline/byte cutoff.
The service accepts `--calibration profile.json`, `--latency-budget-ms` and
`--code-bytes`; calibrated HTTP contexts currently use local storage.

## Common Crawl validation, 2026-09-13

52,903,544 frozen 128d vectors; TQ4, 200 held-out document queries plus ten per
six predicates, one thread, cold NVMe, total memory limit 999,997,440 bytes.
The final profile chooses minimum=64, maximum=256, filtered minimum=256,
rerank=4000, gap=0. The positive gap alternatives were slower here: adaptive
calibration is allowed to select a fixed learned floor. Unfiltered recall over
all 200 queries is .992 at p50 45.3 ms / p95 70.2 ms; the validation half alone
has .994 recall and p95 66.5 ms. Each filtered validation group contains just five
queries and achieves 1.0 recall, with p95 as high as 193.3 ms. These queries
were already used in earlier experiments; the split is disjoint within this
calibration procedure, not a newly collected unseen external benchmark.
79 tests pass; four adaptive tests also pass under ASAN/UBSAN, leak checks disabled
for Python. Raw observations and launchers live in the research archive.


## Concentrated residual geometry

Residual scoring projects the query away from the canonical anchor before its
512-bit dot product. Exact residuals are orthogonal to that anchor; quantized
sketches need not be. Scoring with the full query otherwise magnifies sketch noise
along the shared mean direction of concentrated embeddings. The same projected
scorer serves frozen and packed live codes. No format change or index rebuild is
required. Residual calibration fingerprints include this scorer revision so an
old profile cannot silently claim its measured recall with the new scorer.
The converter also accepts an immutable base containing a held-out tail beyond
the indexed prefix; it still verifies the full file length and source identity.

On the first FineWeb2 shard, 380,000 documents originally contain 829,408 passage
embeddings. The initial experiment used 379,000 normalized document means and
held out 1,000 parents. This is a different representation from original passages.
After query projection, 100 cold queries measure .963 recall at p50 56.2 ms with
64 probes / rerank 8000, or .985 at 116.2 ms with 128 probes / rerank 16000.
The latter passes the .98 calibration/validation target across the tested
predicates. Five validation queries per predicate are only a small diagnostic.
A follow-up keeps all original passage embeddings, withholds all passages from
the same 1,000 parents, and compares both passage and mean-document queries.
No 400M-vector performance claim follows from either pilot.
