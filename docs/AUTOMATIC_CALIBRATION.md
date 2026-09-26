# Optional offline snapshot calibration

For production startup, the default is simply half the indexed dimension, capped
at the available cells. It does not run this workflow or prepare GT. Use this
offline tool when the cost of a measured quality/latency audit is justified.

`mangrove-calibrate` selects search budgets for an existing immutable local index,
then audits the frozen choices on a separate set of queries. It supports the TQ
and residual backends. It does not rebuild the partition tree, select a codec,
calibrate metadata filters or certify a changing live overlay.

## Run

Prepare an NPZ with `queries` (float32, Q by dimension), `ids` (exact top-k row IDs,
Q by k), and `partitions` (strings `calibration` or `validation`). Optional `groups`
identifies independent query groups; supply only one representative per group.
Otherwise duplicate query bytes are rejected. Exact GT must use the indexed
snapshot and the same cosine normalization. Prefer representative real user
questions with embeddings from the document encoder's compatible query mode.

```sh
mangrove-calibrate /data/index /data/base.f16bin /data/workload.npz \
  --residual /data/residual --output /data/profile.json \
  --target-recall 0.95 --confidence 0.95 --p95-ms 100 \
  --memory-bytes 800000000 --cold
```

Omit `--residual` for TQ. The same entry point is available as
`python -m mangrove.calibration`. At least 200 independent questions per partition
are required by default, and an additional check rejects counts that cannot
certify the requested target even at perfect recall (206 at the default target
and confidence). 600 per partition provides a more useful confidence
margin near recall 0.95. More samples can be necessary for a variable workload.
The caller prepares GT explicitly; its compute and IO are not hidden in ingestion.

The search process uses one native thread. `--memory-bytes` limits owned context
allocations; use an external cgroup to bound total RAM and page cache. `--cold`
requests `POSIX_FADV_DONTNEED` on the base and code files before each timed query;
this is a cache eviction request, not proof of physical NVMe reads. Run without
competing ingestion or GPU preparation if measuring isolated query latency.

## Selection and validation

The controller starts at 32 cells and rerank 1000 (bounded by the index size),
and expands routing or reranking using exact-GT coverage on at most 64 training
questions. The stage losing more true neighbors (unrouted versus routed but
not returned) receives the next expansion. Defaults cap the search at 2048 cells and rerank 16000. Every candidate
is measured on the calibration partition. A conservative training lower bound
serves as a stopping heuristic, rather than stopping at a mean barely above the
target. It does not carry a statistical guarantee after adaptive selection.

Before accessing validation, the controller freezes at most two configurations:
the quality choice, and the strongest measured choice within the training p95
budget (or the fastest measured choice when none fits). It evaluates those
choices on validation once, and does not retune from those results.

For independent representative query groups, an empirical Bernstein lower bound
on mean per-query recall is computed using the unbiased sample variance
([Maurer and Pontil, 2009, Theorem 4](https://arxiv.org/pdf/0907.3740)). It treats a
query's top-k neighbors as a single observation. The audit error probability is
split over the two preselected choices, including when they coincide. This
controls the family of two quality claims, not distribution drift, correlated
questions, per-query recall, or latency tails. p95 latency remains an empirical
sample percentile without a confidence bound.

## Results and serving

The JSON contains all configurations, timings, per-query recall, confidence
bounds, audit results and the frozen index fingerprint.

* `validated`: at least one preselected profile passes both audited recall lower
  bound and empirical p95. Exit code 0; load the selected profile normally.
* `latency_tradeoff`: quality passes its audited bound, but no choice passes both
  targets. Exit code 2; the strict profile is rejected by the serving API.
* `quality_not_validated`: quality did not pass its audited bound. Exit code 2;
  no validated result is claimed. This does not prove the architecture cannot
  reach the target with other structures or a larger budget.

When quality passes, an explicit `profile-quality-only.json` is also written. It
can be loaded if the operator accepts the measured latency; it promises no p95
threshold. This file is never substituted automatically for the strict profile.

```python
import json
from mangrove import AnchorIndex

profile = json.load(open('/data/profile.json'))
with AnchorIndex('/data/index', '/data/base.f16bin',
                 residual_dir='/data/residual') as index:
    with index.calibrated_context(profile, memory_bytes=800000000) as query:
        ids, scores, stats = query.search(vector, top_k=10)
```

The profile is also accepted by the existing server's `--calibration` option.
The result applies to the frozen unfiltered workload; it does not validate the
same budgets for metadata filters or after live corpus growth.

A sibling `.checkpoint` file saves completed measurements and freezes audit
choices before validation starts. Rerunning the same command resumes it; a
completed run reuses measurements without running queries again. Changing the
snapshot fingerprint, queries, GT, groups, search settings or cache protocol
rejects the checkpoint. Interrupted individual measurements are repeated.
Keep the workload and snapshot immutable, do not run two writers against the
same checkpoint, and use a new output path for a new experiment. Python callers
with a custom `before_query` hook must specify its `protocol_id`.
