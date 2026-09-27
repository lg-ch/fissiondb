/* A cell owns one contiguous, power-of-two run of allocation chunks. The used
 * records form a dense prefix; unused capacity belongs to the same cell. Runs
 * retired by growth/publication become reusable only after a durable checkpoint.
 * Free-run bins are reconstructed from ownership, never persisted as authority.
 */
static void fission_runs_reset(LiveFission*f){
    f->free_head=FISSION_NONE;
    for(unsigned i=0;i<32;i++)f->free_runs[i]=FISSION_NONE;
}
static uint64_t fission_slot_capacity(const FissionCell*c){
    return c->head==FISSION_NONE?0:((uint64_t)c->tail-c->head+1)*FISSION_CHUNK;
}
static int fission_slot_valid(const LiveFission*f,const FissionCell*c){
    if(c->head==FISSION_NONE)return c->tail==FISSION_NONE&&!c->count;
    if(c->tail<c->head||c->tail>=f->J||c->count>fission_slot_capacity(c))return 0;
    uint32_t blocks=c->tail-c->head+1;if(blocks&(blocks-1))return 0;
    uint32_t left=c->count;
    for(uint32_t b=c->head;;b++){
        uint32_t n=left<FISSION_CHUNK?left:FISSION_CHUNK;
        if(f->chunks[b].used!=n||f->chunks[b].next!=(b==c->tail?FISSION_NONE:b+1))return 0;
        left-=n;if(b==c->tail)break;
    }
    return !left;
}
static void fission_slot_count(LiveFission*f,FissionCell*c,uint32_t count){
    uint32_t left=count;
    if(c->head!=FISSION_NONE)for(uint32_t b=c->head;;b++){
        uint32_t n=left<FISSION_CHUNK?left:FISSION_CHUNK;
        f->chunks[b].used=n;left-=n;if(b==c->tail)break;
    }
    c->count=count;
}
static int fission_slot_reserve(AnchorLive*l,uint32_t count,FissionCell*out){
    LiveFission*f=l->fission;uint64_t need=((uint64_t)count+FISSION_CHUNK-1)/FISSION_CHUNK;
    unsigned order=0;while((UINT64_C(1)<<order)<need)order++;
    if(order>=32)return -1;
    unsigned found=order;while(found<32&&f->free_runs[found]==FISSION_NONE)found++;
    uint32_t first,blocks=UINT32_C(1)<<order;
    if(found<32){
        first=f->free_runs[found];f->free_runs[found]=f->chunks[first].next;
        while(found>order){
            found--;uint32_t other=first+(UINT32_C(1)<<found);
            f->chunks[other]=(FissionChunk){f->free_runs[found],0};f->free_runs[found]=other;
        }
    }else{
        if((uint64_t)f->J+blocks>UINT32_MAX||fission_chunks(f,f->J+blocks))return -1;
        first=f->J;
        if(ftruncate(f->fd,(off_t)(((uint64_t)f->J+blocks)*FISSION_CHUNK*FISSION_ENTRY)))return -1;
        f->J+=blocks;
    }
    for(uint32_t i=0;i<blocks;i++)f->chunks[first+i]=(FissionChunk){i+1<blocks?first+i+1:FISSION_NONE,0};
    *out=(FissionCell){first,first+blocks-1,0};return 0;
}
/* Streaming copy: legacy chains and new slots have identical record bytes.
 * No vector, representative, checksum or membership is recomputed here. */
static int fission_slot_copy(const LiveFission*f,const FissionCell*source,int destination,uint32_t head){
    uint8_t buffer[FISSION_CHUNK*FISSION_ENTRY];uint64_t copied=0;uint32_t visited=0;
    for(uint32_t b=source->head;b!=FISSION_NONE;b=f->chunks[b].next){
        if(b>=f->J||++visited>f->J||f->chunks[b].used>FISSION_CHUNK)return -1;
        size_t bytes=(size_t)f->chunks[b].used*FISSION_ENTRY;
        if(bytes&&(read_all(f->fd,buffer,bytes,(uint64_t)b*FISSION_CHUNK*FISSION_ENTRY)||
            write_all(destination,buffer,bytes,((uint64_t)head*FISSION_CHUNK+copied)*FISSION_ENTRY)))return -1;
        copied+=f->chunks[b].used;
    }
    return copied!=source->count?-1:0;
}
