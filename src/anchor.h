#ifndef MANGROVE_ANCHOR_H
#define MANGROVE_ANCHOR_H
#include <stdint.h>
#include <stddef.h>
typedef struct AnchorIndex AnchorIndex;
typedef struct AnchorQuery AnchorQuery;
typedef struct {
    double anchor_ms, io_ms, score_ms, rerank_ms, total_ms;
    uint64_t entries, bytes;
} AnchorStats;
/* Frozen index plus optional synchronized live overlay. Must outlive contexts.
   Open/enable/close require quiescence; insert, set_tags and queries may overlap. */
AnchorIndex* anchor_index_open(const char* dir, const char* base_path, int int8_mode);
/* Enable residual blocks before contexts: local NVMe, int8 anchors, input
   dimensions 1..1024, one query thread. Legacy v1 blocks remain readable. */
int anchor_index_enable_residual(AnchorIndex* index, const char* residual_dir);
/* Resumable conversion of a frozen TQ index. Published output is immutable. */
int anchor_index_build_residual(AnchorIndex* index, const char* output_dir);
/* Publish a rebuildable compressed snapshot; ingestion can overlap the build.
   A concurrent journal compaction may invalidate the attempt (returns -1). */
int anchor_index_pack_live(AnchorIndex* index);
int anchor_index_snapshot_live(AnchorIndex* index,const char* output_dir);
uint64_t anchor_index_unpacked_bytes(AnchorIndex* index);
void anchor_index_close(AnchorIndex* index);
int anchor_index_dim(const AnchorIndex* index);
uint64_t anchor_index_bytes(const AnchorIndex* index);
/* Enable a local durable overlay before creating contexts. One open writer
   handle per live directory; multiple query contexts share its read lock. */
int anchor_index_enable_live(AnchorIndex* index, const char* live_dir);
uint64_t anchor_index_count(AnchorIndex* index);
int anchor_index_insert(AnchorIndex* index, const float* vector,
    const char* const* keys, int nkeys, uint32_t* id);
/* Replace metadata for an existing frozen or live document. */
/* token and digest each address exactly 32 bytes. -3 denotes a key conflict. */
int anchor_index_insert_once(AnchorIndex*,const float*,const char* const*,int,const uint8_t*,const uint8_t*,uint32_t*);
/* Group commit, 1..256 vectors. Counts partition the flat key array. Routing
   precedes the write lock; journal and rows are fsynced before acknowledgement.
   A failed batch may leave a durable prefix; committed is the known prefix. */
int anchor_index_insert_batch(AnchorIndex*,int,const float*,const char* const*,const int*,const uint8_t* const*,const uint8_t* const*,uint32_t*,int*);
/* Atomically replace vector and metadata at an existing, non-deleted ID. */
int anchor_index_update(AnchorIndex*,uint32_t,const float*,const char* const*,int);
int anchor_index_delete(AnchorIndex*,uint32_t);
/* One locked snapshot: allocated, deleted, maintenance bytes, journal bytes, request count. */
int anchor_index_live_stats(AnchorIndex*,uint64_t values[5]);
uint64_t anchor_index_maintenance_bytes(AnchorIndex*);
uint64_t anchor_index_deleted_count(AnchorIndex*);
/* Add one metadata posting, up to 8192 IDs; idempotent set union. */
int anchor_index_add_tag(AnchorIndex*,const uint32_t*,int,const char* const*,int);
int anchor_index_set_tags(AnchorIndex* index, uint32_t id,
    const char* const* keys, int nkeys);
/* Blocking maintenance: atomically replace live history with a checkpoint. */
int anchor_index_compact(AnchorIndex* index, uint64_t* before, uint64_t* after);
int anchor_index_tag_keys(AnchorIndex* index, char* output, int capacity);
/* One context per concurrent caller. threads=1 is the default recommended mode.
   memory_bytes bounds owned index + context allocations, not kernel page cache,
   curl buffers, OpenMP stacks, allocator overhead, live metadata or filter bitmaps. 0 means no explicit bound. */
AnchorQuery* anchor_query_create(const AnchorIndex* index, int nprobe, int rerank,
    int threads, uint64_t memory_bytes, const char* s3_url, int hedge_ms);
void anchor_query_close(AnchorQuery* query);
/* Quiescent residual context only. 1..256 cells per batch; overlap and direct
   are booleans. Defaults: 64 cells, overlap on, direct if available. */
int anchor_query_residual_io(AnchorQuery*,int batch_cells,int overlap,int direct);
/* Adaptive routing for a single-thread context. Gap is relative to best anchor
   score; min/max budgets must be calibrated for this index. Filtered requests
   use filtered_minimum. 0 gap selects the floor. Soft byte limit covers frozen
   code reads only, excludes exact/live reads, permits one oversized cell.
   Soft deadline is checked between waves, before final rerank/live merge.
   No per-query recall guarantee. Caller serializes configuration/search/stats. */
int anchor_query_adapt(AnchorQuery*,int minimum,int filtered_minimum,float gap,
                       uint64_t code_bytes,double milliseconds);
int anchor_query_adapt_stats(AnchorQuery*,double out[3]);
/* Offline frozen/local/single-thread diagnostic; sorted unique IDs, up to 64.
   n=0 disables. Counters reset on search: watched, routed, selected for exact
   rerank. Filters count eligible IDs only. Traced latency is not a benchmark. */
int anchor_query_trace(AnchorQuery*,const uint32_t*,int);
int anchor_query_trace_stats(AnchorQuery*,uint64_t out[3]);
/* Optional exact eligible-entry/unique-ID counts for a traced search. Extra
   bitmap memory/CPU are diagnostic costs; traced timings include this work. */
int anchor_query_unique(AnchorQuery*,int enable);
int anchor_query_unique_stats(AnchorQuery*,uint64_t out[2]);
/* Returns result count, -1 on invalid input/IO, -2 on memory limit.
   On failure outputs are invalid and the context must be closed. */
int anchor_query_search(AnchorQuery* query, const float* vector, int top_k,
    uint32_t* ids, float* scores, AnchorStats* stats);
/* nallowed=-1 means unrestricted; 0 means empty. Metadata is AND of OR groups.
   Both predicates intersect. At <=4096 allowed documents, exact filtered
   reranking bypasses cell routing. Otherwise filtering precedes candidate caps. */
int anchor_query_search_filtered(AnchorQuery* query, const float* vector, int top_k,
    const uint32_t* allowed_ids, int nallowed, const char* const* keys,
    const int* group_lengths, int ngroups,
    uint32_t* ids, float* scores, AnchorStats* stats);
#endif
