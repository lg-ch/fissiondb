/* Bounded durable append transaction. Foreground writers are serialized by
 * writer_mutex; publish_mutex pins representative identities and excludes arena
 * reclamation/journal replacement. Readers continue during routing, encoding,
 * writes and fsync. Only arena reservation and publication take live.write.
 * Lock order: writer_mutex -> publish_mutex -> live.lock -> work_mutex. Pressure
 * is awaited BEFORE publish_mutex, so the split worker can relieve it. The
 * worker may still prepare daughters while an append transaction is staged.
 * Existing published code slots are immutable; a partial tail is extended only
 * beyond its captured used count. No on-disk format change is required. */
typedef struct {
    uint32_t cell,old_tail,old_used,new_block,added;
} FissionAppendCell;
typedef struct {
    uint32_t plan,slot;
    uint64_t position;
    uint8_t code[FISSION_ENTRY];
} FissionAppendCode;
typedef struct {
    LiveRecord* records[256];
    uint64_t offsets[256];
    FissionAppendCell cells[512];
    FissionAppendCode codes[512];
    int n,ncells,ncodes,accepted,terminal;
    size_t bytes;
    uint8_t* journal;
} FissionAppend;

static void fission_append_free(FissionAppend*t){
    if(!t)return;
    for(int i=0;i<t->n;i++)free(t->records[i]);
    free(t->journal);free(t);
}
static int fission_append_plan(AnchorLive*l,FissionAppend*t,const float*v,uint32_t id,uint64_t offset){
    LiveFission*f=l->fission;int8_t q[1024];float mx=0;
    for(int d=0;d<l->dim;d++)mx=fmaxf(mx,fabsf(v[d]));
    for(int d=0;d<l->dim;d++)q[d]=(int8_t)lrintf(v[d]*127/fmaxf(mx,1e-30f));
    float best[2]={-INFINITY,-INFINITY};uint32_t pick[2]={0,1};
    for(uint32_t k=0;k<f->K;k++){
        float score=f->ops.dot(q,f->centers+(size_t)k*l->dim,l->dim)*f->scales[k];
        if(score>best[1]){if(score>best[0]){best[1]=best[0];pick[1]=pick[0];best[0]=score;pick[0]=k;}else{best[1]=score;pick[1]=k;}}
    }
    if(pick[0]==pick[1])return -1;
    for(int s=0;s<2;s++){
        int j=0;while(j<t->ncells&&t->cells[j].cell!=pick[s])j++;
        if(j==t->ncells){
            if(t->ncells>=512)return -1;
            uint32_t tail=f->cells[pick[s]].tail;
            t->cells[t->ncells++]=(FissionAppendCell){pick[s],tail,tail==FISSION_NONE?0:f->chunks[tail].used,FISSION_NONE,0};
        }
        if(t->ncodes>=512)return -1;
        FissionAppendCode*c=t->codes+t->ncodes++;
        c->plan=j;c->slot=t->cells[j].added++;memcpy(c->code,&id,4);memcpy(c->code+4,&offset,8);
        if(f->ops.encode(f->user,v,f->centers+(size_t)pick[s]*l->dim,c->code+12))return -1;
        uint32_t crc=XXH32(c->code,80,pick[s]);memcpy(c->code+80,&crc,4);
    }
    return 0;
}
/* Read lock protects the published journal/metadata while computing the plan.
 * Once released, writer_mutex and publish_mutex keep the plan valid until commit. */
static int fission_append_prepare(AnchorLive*l,FissionAppend*t,int n,const float*vectors,
        const uint32_t*cells,const char*const*keys,const int*counts,const uint8_t*const*tokens,
        const uint8_t*const*digests,uint32_t*ids){
    size_t vb=(size_t)l->dim*4;int key=0;
    for(int i=0;i<n;i++){
        const uint8_t*token=tokens?tokens[i]:NULL;const uint8_t*digest=token?digests[i]:NULL;
        if(token){
            Request*old=request_find(l,token);
            int found=old!=NULL,conflict=old&&memcmp(old->digest,digest,32);uint32_t id=old?old->id:0;
            for(int j=0;j<t->n&&!found;j++)if(t->records[j]->kind==6){
                const uint8_t*saved=(uint8_t*)(t->records[j]+1)+vb;
                if(!memcmp(saved,token,32)){found=1;conflict=memcmp(saved+32,digest,32);id=t->records[j]->id;}
            }
            if(conflict){t->terminal=-3;break;}
            if(found){ids[i]=id;t->accepted++;key+=counts[i];continue;}
        }
        if(l->base_n+l->count+(uint64_t)t->n>=UINT32_MAX){t->terminal=-1;break;}
        size_t kb=0;for(int j=0;j<counts[i];j++)kb+=strlen(keys[key+j])+1;
        size_t size=sizeof(LiveRecord)+vb+kb+(token?64:0);
        LiveRecord*r=calloc(1,size);if(!r)return -1;
        int record=t->n++;t->records[record]=r;t->offsets[record]=l->end+t->bytes;t->bytes+=size;
        r->magic=RECORD_MAGIC;r->bytes=size;r->kind=token?6:1;
        r->id=(uint32_t)(l->base_n+l->count+record);r->key_bytes=kb;r->nkeys=counts[i];
        memcpy(r+1,vectors+(size_t)i*l->dim,vb);
        double norm=0;for(int d=0;d<l->dim;d++){float x=((float*)(r+1))[d];if(!isfinite(x))return -1;norm+=(double)x*x;}
        if(norm<.999||norm>1.001)return -1;
        for(int s=0;s<l->copies;s++){
            uint32_t cell=cells[(size_t)i*l->copies+s];if(cell>=(uint32_t)l->cells)return -1;
            for(int j=0;j<s;j++)if(r->cells[j]==cell)return -1;
            r->cells[s]=cell;r->previous[s]=l->heads[cell];int found=0;
            for(int j=record-1;j>=0&&!found;j--)for(int k=0;k<l->copies;k++)if(t->records[j]->cells[k]==cell){r->previous[s]=t->offsets[j];found=1;break;}
        }
        uint8_t*dest=(uint8_t*)(r+1)+vb;
        if(token){memcpy(dest,token,32);memcpy(dest+32,digest,32);dest+=64;}
        for(int j=0;j<counts[i];j++){size_t z=strlen(keys[key+j])+1;memcpy(dest,keys[key+j],z);dest+=z;}
        key+=counts[i];r->checksum=XXH64(r,size,0);
        if(fission_append_plan(l,t,(float*)(r+1),r->id,t->offsets[record]))return -1;
        ids[i]=r->id;t->accepted++;
    }
    if(t->bytes){t->journal=malloc(t->bytes);if(!t->journal)return -1;
        for(int i=0;i<t->n;i++)memcpy(t->journal+t->offsets[i]-l->end,t->records[i],t->records[i]->bytes);}
    return 0;
}
static int fission_append_reserve(AnchorLive*l,FissionAppend*t){
    LiveFission*f=l->fission;uint32_t before=f->J;
    for(int i=0;i<t->ncells;i++){
        FissionAppendCell*c=t->cells+i;
        uint32_t available=c->old_tail==FISSION_NONE?0:FISSION_CHUNK-c->old_used;
        if(c->added<=available)continue;
        uint32_t b;
        if(f->free_head!=FISSION_NONE){b=f->free_head;f->free_head=f->chunks[b].next;}
        else{b=f->J;if(b==UINT32_MAX||fission_chunks(f,b+1))return -1;f->J++;}
        f->chunks[b]=(FissionChunk){FISSION_NONE,0};c->new_block=b;
    }
    if(before!=f->J&&ftruncate(f->fd,(off_t)((uint64_t)f->J*FISSION_CHUNK*FISSION_ENTRY)))return -1;
    for(int i=0;i<t->ncodes;i++){
        FissionAppendCode*code=t->codes+i;FissionAppendCell*c=t->cells+code->plan;
        uint32_t available=c->old_tail==FISSION_NONE?0:FISSION_CHUNK-c->old_used;
        uint32_t block=code->slot<available?c->old_tail:c->new_block;
        uint32_t slot=code->slot<available?c->old_used+code->slot:code->slot-available;
        if(block==FISSION_NONE||slot>=FISSION_CHUNK)return -1;
        code->position=((uint64_t)block*FISSION_CHUNK+slot)*FISSION_ENTRY;
    }
    return 0;
}
static int fission_append_publish(AnchorLive*l,FissionAppend*t){
    LiveFission*f=l->fission;
    for(int i=0;i<t->n;i++)if(apply_record_impl(l,t->records[i],t->offsets[i],1))return -1;
    for(int i=0;i<t->ncells;i++){
        FissionAppendCell*c=t->cells+i;FissionCell*published=f->cells+c->cell;
        uint32_t available=c->old_tail==FISSION_NONE?0:FISSION_CHUNK-c->old_used;
        if(c->old_tail!=FISSION_NONE)f->chunks[c->old_tail].used=c->old_used+(c->added<available?c->added:available);
        if(c->new_block!=FISSION_NONE){
            f->chunks[c->new_block].used=c->added-available;
            if(c->old_tail==FISSION_NONE)published->head=c->new_block;
            else f->chunks[c->old_tail].next=c->new_block;
            published->tail=c->new_block;
        }
        if(published->count<=f->cap+256&&published->count+c->added>f->cap+256)f->pressure++;
        published->count+=c->added;fission_enqueue(f,c->cell);
    }
    l->end+=t->bytes;f->records+=t->n;return 0;
}
static int live_append_fission_batch(AnchorLive*l,int n,const float*vectors,const uint32_t*cells,
        const char*const*keys,const int*counts,const uint8_t*const*tokens,const uint8_t*const*digests,
        uint32_t*ids,int*committed){
    int rc=-1;FissionAppend*t=NULL;LiveFission*f=l->fission;
    pthread_mutex_lock(&l->writer_mutex);
    for(;;){
        if(anchor_live_read_lock(l))goto writer_done;
        if(!f->pressure||f->K>=f->maximum)break;
        __atomic_add_fetch(&f->pressure_waits,1,__ATOMIC_RELAXED);fission_wait_locked(l);
    }
    /* Bootstrap representatives use the existing one-time empty-index path. */
    if(!f->K){pthread_rwlock_unlock(&l->lock);pthread_rwlock_wrlock(&l->lock);
        rc=live_append_batch_locked(l,n,vectors,cells,keys,counts,tokens,digests,ids,committed);
        pthread_rwlock_unlock(&l->lock);goto writer_done;}
    pthread_rwlock_unlock(&l->lock);
    pthread_mutex_lock(&f->publish_mutex);
    t=calloc(1,sizeof(*t));if(!t)goto done;
    if(anchor_live_read_lock(l))goto done;
    int prepared=fission_append_prepare(l,t,n,vectors,cells,keys,counts,tokens,digests,ids);
    anchor_live_read_unlock(l);
    if(prepared)goto done;
    if(!t->n){*committed=t->accepted;rc=t->terminal;goto done;}
    if(fission_test_barrier(3))goto done;
    pthread_rwlock_wrlock(&l->lock);
    int reserved=fission_append_reserve(l,t);
    if(reserved)l->poisoned=1;
    pthread_rwlock_unlock(&l->lock);
    if(reserved)goto done;
    /* New journal bytes, row IDs and code slots are not published yet. */
    if(write_all(l->log_fd,t->journal,t->bytes,l->end))goto poison;
    for(int i=0;i<t->n;i++)if(write_all(l->rows_fd,t->records[i]+1,(size_t)l->dim*4,
                    (t->records[i]->id-l->base_n)*(uint64_t)l->dim*4))goto poison;
    for(int i=0;i<t->ncodes;i++)if(write_all(f->fd,t->codes[i].code,FISSION_ENTRY,t->codes[i].position))goto poison;
    if(fsync(l->log_fd)||fsync(l->rows_fd)||fission_test_barrier(4))goto poison;
    pthread_rwlock_wrlock(&l->lock);
    if(fission_append_publish(l,t)){l->poisoned=1;rc=-1;}
    else{*committed=t->accepted;rc=t->terminal;}
    pthread_rwlock_unlock(&l->lock);goto done;
poison:
    pthread_rwlock_wrlock(&l->lock);l->poisoned=1;pthread_rwlock_unlock(&l->lock);
done:
    fission_append_free(t);pthread_mutex_unlock(&f->publish_mutex);
writer_done:
    pthread_mutex_unlock(&l->writer_mutex);return rc;
}
