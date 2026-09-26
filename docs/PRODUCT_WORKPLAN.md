# Product engine work plan

Branch: `product/engine`. Preserve research source and benchmark evidence before
removing legacy files from the product tree. Do not delete source during dataset
cleanup. No publication or claim of production readiness before validation.

## Implemented and validated

- 2026-09-26: 400 reranks by default; bounded per-context NVMe pipeline,
  runtime AVX2/F16C dispatch, input dimensions 1..1024 and durable deletion
  regression coverage. 136 tests, 67 ASan/UBSan tests and all 1024 widths pass.
  Existing MS MARCO 200k-anchor product index: 96.25% recall, 52.69 ms p50,
  62.59 ms p95, 261.74 MB peak, one core/1 GB cgroup, 1536 probes/400 reranks.
  [Protocol and evidence](validation/nvme-integration-20260926.md).

- Versioned residual reader/converter, stable-ID updates and durable group commit.
- Native filters before candidate admission; compressed live snapshot with exact journal tail.
- Compaction removes obsolete/deleted vector records; portable backup/restore.
- Independent C build, platform wheel, bounded HTTP service/client and maintenance counters.
- 65 product tests passed on GB10; 18 residual tests passed under ASAN/UBSAN.
- Final 247M cold NVMe: recall@10 0.9595, p50 91.65 ms, p95 130.76 ms,
  p99 174.02 ms, one query thread, strict 999997440-byte cgroup, zero OOM.
- Concurrent batches of 64: 78.84 vectors/s, query p50 99.63 ms / p95 145.29 ms;
  1000 inserted vectors, 120 concurrent unfiltered queries. Raw evidence in validation/.
- Historical code and experiments preserved in research/archive-20260912 (85b854e).

## Remaining validation and limitations

- Large live populations and realistic metadata selectivity remain to be validated.
- The compressed live snapshot is a cache: current vectors remain in the float32 journal.
  Tiered authoritative segments/WAL truncation are not implemented.
- Native regex metadata filtering is not full-text search; design text predicates against
  dataset schemas. High-cardinality postings RAM requires measurement.
- Docker build and a real query passed on ARM64 with the documented io_uring seccomp profile.
  GitHub CI has not run remotely. Packaging remains a development release.
- The new dataset campaign is not complete; do not infer its recall from the 247M result.

## Current campaign status

Common Crawl revision cc1b20615e04067e135824c80f07e88597d7bb33:
14 shards / 13,148,629,591 repository bytes, all shard SHA-256 values verified;
52,913,544 vectors, 128 dimensions. Conversion finished in 729 seconds.
The initial frozen index reserves the final 10,000 real rows for live ingestion.
Native TQ2 build uses one thread and a 1,999,994,880-byte cgroup. Evaluation
runs with a 999,997,440-byte cgroup: native metadata postings, exhaustive
float64 GT, six filter cases, then simultaneous ingestion/query. Corrected TQ4 now measures 0.992 recall@10 at 42.6 ms median / 65.6 ms P95
on 200 cold queries, one thread, 1 GB total memory; nprobe=64, rerank=4000.
Rare filters need more probes. Live TQ4: 10,000 real vectors, 1,292 vectors/s in durable groups of 64;
127 concurrent unfiltered queries, p50 48.2 / p95 64.1 ms, natural cache.
Reopen count and filtered exact checks pass. This is a short load test, not
a full-corpus concurrent recall measurement. No old dataset was removed because this campaign fits.

Remote campaign: /root/mangrove-datasets/commoncrawl-cc1b2061;
status file: campaign-status.json. Runner stops on a failed dependency.

## Subsequent dataset campaigns (user-authorized order)

Only after the product work above. Inspect actual schemas, embedding formats,
row counts, licenses and storage requirements before selecting columns/shards.

1. https://huggingface.co/datasets/commoncrawl/web-graph-embeddings
2. Marqo skipped at the user's request (2026-09-13).
3. https://huggingface.co/datasets/epfml/FineWeb2-embedded — cap at 400 million
   rows; do not attempt to download the entire dataset.

Before each campaign inventory NVMe usage and remove only explicitly identified
obsolete large dataset/index artifacts. Preserve all source, scripts, seeds,
configuration, exact GT and compact result reports. Account for raw data, index,
temporary build files and live growth together; 2 TB is a user estimate, not a
verified size. Prefer reproducible streaming/subsetting over duplicate full
downloads. Do not upload these datasets to S3 without a new scope decision.

For each corpus, in sequence:

- Build and test unfiltered recall against exact GT with explicit self handling.
- Test filtered recall against exact GT *within the eligible document set*.
- Select available date/range, categorical and text predicates after schema
  inspection. Keep filtering native, before candidate admission and reranking;
  do not introduce an external filtering database.
- Distinguish literal/full-text predicates from existing regex metadata filters.
- Measure live insertion throughput and durability settings.
- Run queries during ingestion, recording throughput, p50/p95/p99 latency,
  process RSS, cgroup memory/page cache, IO and errors over time.
- Preserve measurements before reclaiming space for the next corpus.

Adaptive routing/calibration implemented and validated; see ADAPTIVE_SEARCH.md. FineWeb2 conversion preflight started, pinned revision 4631ff28026f717d33016acebea7874f71872c15. One normalized mean per document, 768 dimensions padded to 1024; at most 400M documents. Full campaign is not yet complete.


The passage comparison is complete on the first shard: 826,988 frozen original
passages, 100 held-out-parent passage queries, .953 recall at p50 96.4 / p95
102.9 ms with 64 probes and rerank 12000; .969 at p50 131.6 ms with 128/16000.
Original passage vectors remain highly concentrated (median exact-neighbour
cosine .99958), so averaging was not the sole cause. The massive 400M preparation
has not started; document versus passage counts and disk requirements need to
follow the chosen representation. No 400M build or search result is claimed.


## Automatic calibration delivery (2026-09-14)

The product now provides `mangrove-calibrate` for immutable unfiltered TQ/residual
snapshots: automatic routing/rerank expansion, uncertainty-aware training stop,
two choices frozen before an independent audit, strict versus quality-only
profiles, and checkpoint resume. It uses supplied exact GT; GT preparation is
separate from insertion. It does not automatically select the index structure
or calibrate changing live/filtered distributions. See AUTOMATIC_CALIBRATION.md.

The prior 64+64 MS MARCO experiment was not a successful calibration (training
recall 0.9531, validation 0.8922). A fresh 600+600 question reference was computed on the full
113,520,750-passage snapshot. At the user's request the production default is
now the simple dimension/2 rule, with no GT or calibration work at ingestion.
The heavy controller remains optional offline. Larger-budget exploration was
stopped before completed validation measurements; the chosen 512/1000 setting
passed the empirical target on 600 reserved questions: recall 0.95883, median
152.4 ms, p95 188.3 ms, peak cgroup RAM 599,916,544 bytes. This does not
certify a statistical 0.95 lower bound (conservative bound 0.9304), nor the
original 100 ms target. See validation/msmarco-pragmatic.json.
