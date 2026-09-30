/* Same bounded IO policy as frozen residual cells: two buffers, batches of
 * independent reads, next batch in flight during scoring, batched reranking.
 * Caller pins published descriptors with live.lock for this entire operation.
 * Unpublished split writes never modify the used slots copied here. */
typedef struct {
    int n;
    unsigned pending;
    uint64_t offset[256],length[256],expected[256];
    uint8_t* dst[256];
    uint8_t* records[256];
    uint32_t cell[256],count[256];
} FissionReadBatch;

static int fission_io_fail(AnchorLiveIO*io){
    /* queue_exit cancels/drains outstanding requests before buffer reuse/free. */
    if(*io->ring_ok){io_uring_queue_exit(io->ring);*io->ring_ok=0;}
    return -1;
}
static int fission_io_submit(AnchorLiveIO*io,FissionReadBatch*b,int fd){
    if(!*io->ring_ok||b->pending||io_uring_sq_ready(io->ring)||io_uring_cq_ready(io->ring)||
       io_uring_sq_space_left(io->ring)<(unsigned)b->n)return fission_io_fail(io);
    double t=fission_now();
    for(int i=0;i<b->n;i++){
        struct io_uring_sqe*sqe=io_uring_get_sqe(io->ring);if(!sqe)return fission_io_fail(io);
        io_uring_prep_read(sqe,fd,b->dst[i],(unsigned)b->length[i],b->offset[i]);
        sqe->user_data=b->expected[i];
    }
    b->pending=b->n;int submitted=0;
    while(submitted<b->n){int n=io_uring_submit(io->ring);if(n==-EINTR)continue;
        if(n<=0||n>b->n-submitted)return fission_io_fail(io);
        submitted+=n;}
    io->submits++;io->reads+=b->n;if((unsigned)b->n>io->max_pending)io->max_pending=b->n;
    io->io_ms+=fission_now()-t;return 0;
}
static int fission_io_wait(AnchorLiveIO*io,FissionReadBatch*b){
    double t=fission_now();int failed=0;
    while(b->pending){struct io_uring_cqe*cqe;int rc;
        do{rc=io_uring_wait_cqe(io->ring,&cqe);}while(rc==-EINTR);
        if(rc<0)return fission_io_fail(io);
        if(cqe->res<0||(uint64_t)cqe->res!=cqe->user_data)failed=1;
        io_uring_cqe_seen(io->ring,cqe);b->pending--;
    }
    io->io_ms+=fission_now()-t;return failed?-1:0;
}
/* Advance over contiguous cells in ranked order, preserving score/tie order
 * for width1 and overlapping batches alike. Direct reads may include boundary
 * pages, but only the captured used slots are scored and checksum-validated.
 * A normal cell is one read. Only a cell larger than the entire bounded buffer
 * is split across reads; allocation-chunk boundaries never cause extra IO. */
static int fission_io_prepare(const LiveFission*f,const PackHit*routes,int rn,int*route,uint32_t*block,
                             AnchorLiveIO*io,FissionReadBatch*b,uint8_t*buffer,uint64_t*bytes,uint64_t*entries){
    b->n=0;b->pending=0;size_t used=0;uint64_t file_bytes=(uint64_t)f->J*FISSION_CHUNK*FISSION_ENTRY;
    while(*route<rn&&b->n<io->width){
        uint32_t cell=routes[*route].id;
        if(*block==FISSION_NONE){(*route)++;if(*route<rn)*block=f->cells[routes[*route].id].head;continue;}
        const FissionCell*c=f->cells+cell;
        if(*block>=f->J||*block<c->head||*block>c->tail)return -1;
        uint64_t skipped=((uint64_t)*block-c->head)*FISSION_CHUNK;
        if(skipped>=c->count){*block=FISSION_NONE;continue;}
        uint32_t count=c->count-(uint32_t)skipped;
        uint64_t offset=(uint64_t)*block*FISSION_CHUNK*FISSION_ENTRY,length=(uint64_t)count*FISSION_ENTRY;
        uint64_t aligned=io->direct?offset&~UINT64_C(4095):offset;
        uint64_t read_size=io->direct?(length+(offset-aligned)+4095)&~UINT64_C(4095):length;
        size_t allocation=(size_t)((read_size+4095)&~UINT64_C(4095));
        if(allocation>io->capacity){
            if(b->n)break;
            if(io->capacity<=offset-aligned)return -1;
            uint64_t blocks=(io->capacity-(offset-aligned))/(FISSION_CHUNK*FISSION_ENTRY);
            if(!blocks)return -1;
            count=(uint32_t)(blocks*FISSION_CHUNK);length=(uint64_t)count*FISSION_ENTRY;
            read_size=io->direct?(length+(offset-aligned)+4095)&~UINT64_C(4095):length;
            allocation=(size_t)((read_size+4095)&~UINT64_C(4095));
        }
        if(allocation>io->capacity-used)break;
        int i=b->n++;b->offset[i]=aligned;b->length[i]=read_size;
        b->expected[i]=read_size<file_bytes-aligned?read_size:file_bytes-aligned;
        b->dst[i]=buffer+used;b->records[i]=buffer+used+(offset-aligned);b->cell[i]=cell;b->count[i]=count;
        used+=allocation;*bytes+=read_size;*entries+=count;
        *block=skipped+count<c->count?*block+count/FISSION_CHUNK:FISSION_NONE;
    }
    return b->n?1:(*route>=rn?0:-1);
}
static int fission_io_score(const AnchorLive*l,const FissionReadBatch*b,AnchorLiveIO*io,
                           const roaring_bitmap_t*allowed,PackHit*hits,int*hn,int capacity,
                           AnchorFissionScore score_code,void*user){
    const LiveFission*f=l->fission;double t=fission_now();
    for(int i=0;i<b->n;i++)for(uint32_t j=0;j<b->count[i];j++){
        const uint8_t*p=b->records[i]+(size_t)j*FISSION_ENTRY;uint32_t id,crc;uint64_t offset;
        memcpy(&id,p,4);memcpy(&offset,p+4,8);memcpy(&crc,p+80,4);
        if(XXH32(p,80,b->cell[i])!=crc)return -1;
        if(!fission_current(l,id,offset)||(allowed&&!roaring_bitmap_contains(allowed,id)))continue;
        float score=score_code(user,f->centers+(size_t)b->cell[i]*f->ops.center_dim,p+12);
        if(!isfinite(score))return -1;
        pack_offer(hits,hn,2*capacity,(PackHit){score,id,offset});
    }
    io->score_ms+=fission_now()-t;return 0;
}
int anchor_live_search_fission(const AnchorLive*l,int probes,const float*q,const roaring_bitmap_t*allowed,
        uint32_t*ids,float*scores,int count,int capacity,uint64_t*entries,uint64_t*bytes,
        AnchorFissionScore score_code,void*user,AnchorLiveIO*io){
    const LiveFission*f=l->fission;if(!f||!io||!io->buffer[0]||io->width<1||io->width>256)return -1;
    if(!f->K)return count;
    if(probes>(int)f->K)probes=f->K;
    PackHit*routes=malloc((size_t)probes*sizeof(*routes)),*hits=malloc((size_t)capacity*2*sizeof(*hits));
    uint32_t*selected=malloc((size_t)capacity*4);FissionReadBatch batch[2]={{0}};
    int rc=-1,rn=0,hn=0;if(!routes||!hits||!selected)goto done;
    double t=fission_now();int8_t q8[1024];float mx=0;
    for(int d=0;d<l->dim;d++)mx=fmaxf(mx,fabsf(q[d]));
    for(int d=0;d<l->dim;d++)q8[d]=(int8_t)lrintf(q[d]*127/fmaxf(mx,1e-30f));
    for(uint32_t k=0;k<f->K;k++)if(f->cells[k].count)
        pack_offer(routes,&rn,probes,(PackHit){f->ops.dot(q8,f->centers+(size_t)k*f->ops.center_dim,f->ops.center_dim)*f->scales[k],k,0});
    qsort(routes,rn,sizeof(*routes),pack_hit_cmp);io->route_ms=fission_now()-t;
    io->direct=io->direct&&f->direct_fd>=0;
    int fd=io->direct?f->direct_fd:f->fd,route=0,current=0;
    uint32_t block=rn?f->cells[routes[0].id].head:FISSION_NONE;
    int more=fission_io_prepare(f,routes,rn,&route,&block,io,&batch[0],io->buffer[0],bytes,entries);
    if(more<0)goto done;
    if(more&&(fission_io_submit(io,&batch[0],fd)||fission_io_wait(io,&batch[0])))goto done;
    while(more){
        io->code_reads+=batch[current].n;if(io->direct)io->direct_reads+=batch[current].n;
        int next=1-current;
        if(io->overlap){
            more=fission_io_prepare(f,routes,rn,&route,&block,io,&batch[next],io->buffer[next],bytes,entries);
            if(more<0)goto done;
            if(more){if(fission_io_submit(io,&batch[next],fd))goto done;io->overlaps++;}
        }
        if(fission_io_score(l,&batch[current],io,allowed,hits,&hn,capacity,score_code,user))goto done;
        if(io->overlap){if(more&&fission_io_wait(io,&batch[next]))goto done;current=next;}
        else{
            more=fission_io_prepare(f,routes,rn,&route,&block,io,&batch[current],io->buffer[current],bytes,entries);
            if(more<0)goto done;
            if(more&&(fission_io_submit(io,&batch[current],fd)||fission_io_wait(io,&batch[current])))goto done;
        }
    }
    t=fission_now();qsort(hits,hn,sizeof(*hits),pack_hit_cmp);int ns=0;
    for(int i=0;i<hn&&ns<capacity;i++){
        int dup=0;for(int j=0;j<ns;j++)if(hits[selected[j]].id==hits[i].id){dup=1;break;}
        if(!dup)selected[ns++]=i;
    }
    for(int first=0;first<ns;){
        FissionReadBatch*b=&batch[0];b->n=0;b->pending=0;int begin=first;
        while(first<ns&&b->n<io->width&&(size_t)(b->n+1)*l->dim*4<=io->capacity){
            int i=b->n++;b->offset[i]=hits[selected[first++]].offset+sizeof(LiveRecord);
            b->length[i]=b->expected[i]=(uint64_t)l->dim*4;b->dst[i]=io->buffer[0]+(size_t)i*l->dim*4;
        }
        if(!b->n||fission_io_submit(io,b,l->log_fd)||fission_io_wait(io,b))goto done;
        io->rerank_reads+=b->n;*bytes+=(uint64_t)b->n*l->dim*4;
        for(int i=0;i<b->n;i++){
            uint32_t id=hits[selected[begin+i]].id;int dup=0;for(int j=0;j<count;j++)if(ids[j]==id){dup=1;break;}
            if(dup)continue;
            const float*v=(const float*)b->dst[i];float score=0;for(int d=0;d<l->dim;d++)score+=q[d]*v[d];
            if(count<capacity||score>scores[count-1]){int j=count<capacity?count++:capacity-1;
                while(j>0&&score>scores[j-1]){scores[j]=scores[j-1];ids[j]=ids[j-1];j--;}scores[j]=score;ids[j]=id;}
        }
    }
    io->rerank_ms=fission_now()-t;
    if(f->active!=FISSION_NONE)__atomic_add_fetch((uint64_t*)&f->queries_during_prepare,1,__ATOMIC_RELAXED);
    rc=count;
done:
    if(batch[0].pending||batch[1].pending)fission_io_fail(io);
    free(routes);free(hits);free(selected);return rc;
}
