/* Each query owns its ring and two bounded buffers. Read the next group of
 * cells while scoring the current group; scoring order remains unchanged.
 * Dense residual cells already require a single request each: aligned direct
 * reads include at most two boundary pages, without rewriting the payload. */
typedef struct {
    int first, end, n;
    uint64_t off[256], len[256], expected[256], count[256];
    uint8_t* dst[256];
    uint8_t* records[256];
    unsigned pending;
} ResidualBatch;

int anchor_query_residual_io(AnchorQuery* c,int width,int overlap,int direct) {
    if(!c||c->failed||c->index->residual_fd<0||width<1||width>256||
       (overlap!=0&&overlap!=1)||(direct!=0&&direct!=1)||
       (direct&&c->index->residual_direct_fd<0))return -1;
    c->res_width=width;c->res_overlap=overlap;
    c->res_direct=c->res_live_direct=direct;return 0;
}
static uint64_t residual_read_length(const AnchorQuery* c,uint32_t cell){
    const AnchorIndex* idx=c->index;int entry=residual_source_entry(idx);
    uint64_t off=idx->offs[cell]/entry*72,len=(idx->offs[cell+1]-idx->offs[cell])/entry*72;
    return c->res_direct&&len?(len+(off&4095)+4095)&~UINT64_C(4095):len;
}
static int residual_buffers(AnchorQuery* c,int probes){
    uint64_t largest=0;
    for(int j=0;j<probes;j++){
        uint64_t bytes=residual_read_length(c,c->cand[j].id);
        if(bytes>INT_MAX)return -2;
        if(bytes>largest)largest=bytes;
    }
    unsigned buffers=c->res_overlap?2:1;
    uint64_t budget=(c->block_limit/buffers)&~UINT64_C(4095);
    uint64_t capacity=8u*1024*1024;
    if(capacity>budget)capacity=budget;
    largest=(largest+4095)&~UINT64_C(4095);
    if(capacity<largest)capacity=largest;
    if(capacity>budget||!capacity)return -2;
    if(c->res_capacity!=capacity||(!c->res_overlap&&c->res_buffer[1])||
       (c->res_overlap&&!c->res_buffer[1])){
        free(c->res_buffer[0]);free(c->res_buffer[1]);
        c->res_buffer[0]=c->res_buffer[1]=NULL;c->res_capacity=0;
        for(unsigned i=0;i<buffers;i++)if(posix_memalign((void**)&c->res_buffer[i],4096,capacity))return -2;
        c->res_capacity=capacity;
    }
    return 0;
}
static int residual_prepare(AnchorQuery* c,ResidualBatch* b,int first,int probes,uint8_t* buffer){
    const AnchorIndex* idx=c->index;int entry=residual_source_entry(idx);
    b->first=b->end=first;b->n=0;b->pending=0;uint64_t used=0;
    while(b->end<probes&&b->end-first<c->res_width){
        uint32_t cell=c->cand[b->end].id;
        uint64_t off=idx->offs[cell]/entry*72,len=(idx->offs[cell+1]-idx->offs[cell])/entry*72;
        uint64_t size=residual_read_length(c,cell);
        if(size>c->res_capacity-used)break;
        int j=b->end-first;b->count[j]=len/72;b->records[j]=buffer+used;
        if(len){
            int r=b->n++;
            b->off[r]=c->res_direct?off&~UINT64_C(4095):off;
            b->len[r]=size;b->dst[r]=buffer+used;
            b->expected[r]=size;
            if(size>idx->residual_bytes-b->off[r])b->expected[r]=idx->residual_bytes-b->off[r];
            if(c->res_direct)b->records[j]+=off&4095;
            used+=size;
        }
        b->end++;
    }
    return b->end>first?0:-2;
}
/* Closing a broken ring cancels/drains its operations before buffer release.
 * The public search entry poisons the context on any error. No process abort. */
static int residual_ring_error(AnchorQuery* c){
    if(c->ring_ok){io_uring_queue_exit(&c->ring);c->ring_ok=0;}
    c->failed=1;return -1;
}
static int residual_submit(AnchorQuery* c,ResidualBatch* b,AnchorStats* st){
    if(!c->ring_ok||b->pending||io_uring_sq_ready(&c->ring)||io_uring_cq_ready(&c->ring)||
       io_uring_sq_space_left(&c->ring)<(unsigned)b->n)return residual_ring_error(c);
    double t=now_ms();
    for(int j=0;j<b->n;j++){
        struct io_uring_sqe* sqe=io_uring_get_sqe(&c->ring);
        if(!sqe)return residual_ring_error(c);
        io_uring_prep_read(sqe,c->res_direct?c->index->residual_direct_fd:c->index->residual_fd,
                          b->dst[j],(unsigned)b->len[j],b->off[j]);
        sqe->user_data=b->expected[j];st->bytes+=b->len[j];
    }
    b->pending=b->n;int submitted=0;
    while(submitted<b->n){
        int rc=io_uring_submit(&c->ring);if(rc==-EINTR)continue;
        if(rc<=0||rc>b->n-submitted)return residual_ring_error(c);
        submitted+=rc;
    }
    st->io_ms+=now_ms()-t;return 0;
}
static int residual_wait(AnchorQuery* c,ResidualBatch* b,AnchorStats* st){
    double t=now_ms();int failed=0;
    while(b->pending){
        struct io_uring_cqe* cqe;int rc;
        do{rc=io_uring_wait_cqe(&c->ring,&cqe);}while(rc==-EINTR);
        if(rc<0)return residual_ring_error(c);
        if(cqe->res<0||(uint64_t)cqe->res!=cqe->user_data)failed=1;
        io_uring_cqe_seen(&c->ring,cqe);b->pending--;
    }
    st->io_ms+=now_ms()-t;return failed?-1:0;
}
static int residual_score_batch(AnchorQuery* c,const ResidualBatch* b,ResidualLiveScore* state,int* hn,AnchorStats* st){
    const AnchorIndex* idx=c->index;double t=now_ms();
    for(int j=b->first;j<b->end;j++){
        uint32_t cell=c->cand[j].id;int slot=j-b->first;
        st->entries+=b->count[slot];
        for(uint64_t e=0;e<b->count[slot];e++){
            const uint8_t* p=b->records[slot]+e*72;uint32_t id;memcpy(&id,p,4);
            if((uint64_t)id>=(uint64_t)idx->meta.n)return -1;
            if(anchor_live_is_deleted(idx->live,id)||anchor_live_is_overridden(idx->live,id)||
               (c->allowed&&!roaring_bitmap_contains(c->allowed,id)))continue;
            anchor_trace_mark(c,id,0);float value=residual_score_live(state,cell,p+4);
            if(!isfinite(value))return -1;
            residual_heap(c->heap,hn,idx->meta.M*c->rerank,(ScId){value,id});
        }
    }
    st->score_ms+=now_ms()-t;return 0;
}
static int residual_read_score(AnchorQuery* c,int probes,ResidualLiveScore* state,int* hn,AnchorStats* st){
    int rc=residual_buffers(c,probes);if(rc)return rc;
    ResidualBatch b[2]={0};int current=0;
    rc=residual_prepare(c,&b[0],0,probes,c->res_buffer[0]);if(rc)return rc;
    if(residual_submit(c,&b[0],st)||residual_wait(c,&b[0],st))return -1;
    for(;;){
        int next=1-current,first=b[current].end;
        int more=first<probes&&!anchor_adapt_expired(c,b[current].first);
        if(more&&c->res_overlap){
            rc=residual_prepare(c,&b[next],first,probes,c->res_buffer[next]);if(rc)return rc;
            if(residual_submit(c,&b[next],st))return -1;
        }
        rc=residual_score_batch(c,&b[current],state,hn,st);
        if(more&&c->res_overlap){int failed=residual_wait(c,&b[next],st);if(failed)return -1;}
        if(rc)return rc;
        c->policy_used=first;
        if(!more)return 0;
        if(!c->res_overlap){
            rc=residual_prepare(c,&b[next],first,probes,c->res_buffer[0]);if(rc)return rc;
            if(residual_submit(c,&b[next],st)||residual_wait(c,&b[next],st))return -1;
        }
        current=next;
    }
}
