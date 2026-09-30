#ifndef ANCHOR_LIVE_INTERNAL_H
#define ANCHOR_LIVE_INTERNAL_H
#include <stdint.h>
#include <stddef.h>
#include <roaring/roaring.h>
typedef struct AnchorLive AnchorLive;
typedef int (*AnchorLiveEncode)(void*,uint32_t,const float*,uint8_t*);
typedef float (*AnchorLiveScore)(void*,uint32_t,const uint8_t*);
/* Adaptive live cells are derived from the durable journal. All callbacks run
   under its read/write lock; they must not acquire that lock again. */
typedef struct {
    int (*dot)(const int8_t*,const int8_t*,int);
    int (*encode)(void*,const float*,const int8_t*,uint8_t*);
    void (*sketch)(void*,const float*,float*);
    int center_dim; /* native input width; journal/residual vectors retain dim */
} AnchorFissionOps;
typedef float (*AnchorFissionScore)(void*,const int8_t*,const uint8_t*);
/* Reuse the query's ring and bounded aligned buffers for mutable cells too. */
struct io_uring;
typedef struct {
    struct io_uring* ring;
    int* ring_ok;
    uint8_t* buffer[2];
    size_t capacity;
    int width,overlap,direct;
    uint64_t reads,code_reads,rerank_reads,submits,overlaps,max_pending,direct_reads;
    double route_ms,io_ms,score_ms,rerank_ms;
} AnchorLiveIO;
int anchor_live_enable_fission(AnchorLive*,uint32_t,uint32_t,const AnchorFissionOps*,void*);
int anchor_live_fission_config(AnchorLive*,uint32_t*,uint32_t*);
int anchor_live_fission_stats(AnchorLive*,uint64_t values[8],double timings[3]);
int anchor_live_fission_checkpoint(AnchorLive*);
int anchor_live_fission_flush(AnchorLive*);
int anchor_live_fission_set_capacity(AnchorLive*,uint32_t);
uint32_t anchor_live_fission_capacity(AnchorLive*);
int anchor_live_fission_center_info(AnchorLive*,uint64_t values[3]);
int anchor_live_fission_progress(AnchorLive*,uint64_t values[8],double timings[3]);
int anchor_live_search_fission(const AnchorLive*,int,const float*,const roaring_bitmap_t*,uint32_t*,float*,int,int,uint64_t*,uint64_t*,AnchorFissionScore,void*,AnchorLiveIO*);
int anchor_live_has_fission(const AnchorLive*);
int anchor_live_pack(AnchorLive*,AnchorLiveEncode,void*);
int anchor_live_snapshot(AnchorLive*,const char*);
uint64_t anchor_live_unpacked_bytes(AnchorLive*);
int anchor_live_search_packed(const AnchorLive*,const uint32_t*,int,const float*,const roaring_bitmap_t*,uint32_t*,float*,int,int,uint64_t*,uint64_t*,AnchorLiveScore,void*);
AnchorLive* anchor_live_open(const char* dir, uint64_t fingerprint,
                            uint64_t base_n, int dim, int cells, int copies);
void anchor_live_close(AnchorLive* live);
int anchor_live_read_lock(AnchorLive* live);
void anchor_live_read_unlock(AnchorLive* live);
uint64_t anchor_live_count(const AnchorLive* live); /* caller holds read lock */
int anchor_live_rows_fd(const AnchorLive* live);
int anchor_live_append_batch(AnchorLive*,int,const float*,const uint32_t*,const char* const*,const int*,const uint8_t* const*,const uint8_t* const*,uint32_t*,int*);
int anchor_live_append(AnchorLive* live, const float* unit_vector,
                       const uint32_t* cells, const char* const* keys, int nkeys,
                       uint32_t* id);
int anchor_live_add_tag(AnchorLive*,const uint32_t*,int,const char* const*,int);
int anchor_live_set_tags(AnchorLive* live, uint32_t id, const char* const* keys, int nkeys);
int anchor_live_compact(AnchorLive* live, uint64_t* before, uint64_t* after);
int anchor_live_append_once(AnchorLive*,const float*,const uint32_t*,const char* const*,int,const uint8_t*,const uint8_t*,uint32_t*);
void anchor_live_exclude_deleted(const AnchorLive*,roaring_bitmap_t*);
int anchor_live_lookup(AnchorLive*,const uint8_t*,const uint8_t*,uint32_t*);
int anchor_live_update(AnchorLive*,uint32_t,const float*,const uint32_t*,const char* const*,int);
int anchor_live_is_overridden(const AnchorLive*,uint32_t);
/* Under read lock: 1 if replacement read, 0 if none, -1 on IO failure. */
int anchor_live_override_vector(const AnchorLive*,uint32_t,float*);
int anchor_live_delete(AnchorLive*,uint32_t);
int anchor_live_is_deleted(const AnchorLive*,uint32_t);
uint64_t anchor_live_journal_bytes(const AnchorLive*);
uint64_t anchor_live_request_count(const AnchorLive*);
uint64_t anchor_live_maintenance_bytes(const AnchorLive*);
uint64_t anchor_live_deleted_count(const AnchorLive*);
int anchor_live_keys(AnchorLive* live, char* dst, int capacity);
roaring_bitmap_t* anchor_live_filter(const AnchorLive* live, const char* const* keys,
                                    const int* groups, int ngroups);
/* Merge exact live scores into a descending, unique top list. Caller holds lock. */
int anchor_live_search(const AnchorLive* live, const uint32_t* cells, int ncells,
                       const float* unit_query, const roaring_bitmap_t* allowed,
                       uint32_t* ids, float* scores, int count, int capacity, uint64_t* entries, uint64_t* bytes);
#endif
