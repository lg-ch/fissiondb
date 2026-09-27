/* Bounded durable append transaction. Foreground writers are serialized by
 * writer_mutex; publish_mutex pins representative identities and excludes arena
 * reclamation/journal replacement. Readers continue during routing, encoding,
 * writes and fsync. Only arena reservation and publication take live.write.
 * Lock order: writer_mutex -> publish_mutex -> live.lock -> work_mutex. Pressure
 * is awaited BEFORE publish_mutex, so the split worker can relieve it. The
 * worker may still prepare daughters while an append transaction is staged.
 * Existing published code slots are immutable; a partial tail is extended only
 * beyond its captured used count. Slot growth copies into an unpublished run;
 * the old run remains valid until publication and a durable checkpoint. */
typedef struct {
    uint32_t cell,added;
    FissionCell old,replacement;
    int moved;
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
/* Routing/encoding is independent per record. Publication order and cell ties
 * remain identical for 1 or many ingestion threads. Callbacks only read their
 * immutable codec state and representatives pinned by publish_mutex. */
static int fission_append_encode(AnchorLive*l,FissionAppend*t,int row,uint32_t pick[2]){
    LiveFission*f=l->fission;LiveRecord*r=t->records[row];const float*v=(float*)(r+1);
    int8_t q[1024];float mx=0;
    for(int d=0;d<l->dim;d++)mx=fmaxf(mx,fabsf(v[d]));
    for(int d=0;d<l->dim;d++)q[d]=(int8_t)lrintf(v[d]*127/fmaxf(mx,1e-30f));
    float best[2]={-INFINITY,-INFINITY};pick[0]=0;pick[1]=1;
    for(uint32_t k=0;k<f->K;k++){
        float score=f->ops.dot(q,f->centers+(size_t)k*l->dim,l->dim)*f->scales[k];
        if(score>best[1]){if(score>best[0]){best[1]=best[0];pick[1]=pick[0];best[0]=score;pick[0]=k;}else{best[1]=score;pick[1]=k;}}
    }
    if(pick[0]==pick[1])return -1;
    for(int side=0;side<2;side++){
        uint8_t*code=t->codes[row*2+side].code;
        memcpy(code,&r->id,4);memcpy(code+4,t->offsets+row,8);
        if(f->ops.encode(f->user,v,f->centers+(size_t)pick[side]*l->dim,code+12))return -1;
        uint32_t crc=XXH32(code,80,pick[side]);memcpy(code+80,&crc,4);
    }
    return 0;
}
static int fission_append_plan(AnchorLive*l,FissionAppend*t){
    LiveFission*f=l->fission;uint32_t picks[256][2];int failed=0;
    int threads=f->ingest_threads<t->n?f->ingest_threads:t->n;
    if(threads<1)threads=1;
    #pragma omp parallel for num_threads(threads) if(threads>1 && t->n>=16) schedule(static) reduction(|:failed)
    for(int row=0;row<t->n;row++)failed|=fission_append_encode(l,t,row,picks[row])!=0;
    if(failed)return -1;
    for(int row=0;row<t->n;row++)for(int side=0;side<2;side++){
        uint32_t cell=picks[row][side];int j=0;
        while(j<t->ncells&&t->cells[j].cell!=cell)j++;
        if(j==t->ncells){
            if(t->ncells>=512)return -1;
            t->cells[t->ncells++]=(FissionAppendCell){.cell=cell,.old=f->cells[cell],.replacement=f->cells[cell]};
        }
        FissionAppendCode*c=t->codes+t->ncodes++;c->plan=j;c->slot=t->cells[j].added++;
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
        ids[i]=r->id;t->accepted++;
    }
    if(fission_append_plan(l,t))return -1;
    if(t->bytes){t->journal=malloc(t->bytes);if(!t->journal)return -1;
        for(int i=0;i<t->n;i++)memcpy(t->journal+t->offsets[i]-l->end,t->records[i],t->records[i]->bytes);}
    return 0;
}
static int fission_append_reserve(AnchorLive*l,FissionAppend*t){
    for(int i=0;i<t->ncells;i++){
        FissionAppendCell*c=t->cells+i;
        uint64_t count=(uint64_t)c->old.count+c->added;if(count>UINT32_MAX)return -1;
        if(count>fission_slot_capacity(&c->old)){
            if(fission_slot_reserve(l,(uint32_t)count,&c->replacement))return -1;
            c->moved=1;
        }
    }
    for(int i=0;i<t->ncodes;i++){
        FissionAppendCode*code=t->codes+i;FissionAppendCell*c=t->cells+code->plan;
        if(c->replacement.head==FISSION_NONE)return -1;
        code->position=((uint64_t)c->replacement.head*FISSION_CHUNK+c->old.count+code->slot)*FISSION_ENTRY;
    }
    return 0;
}
static int fission_append_publish(AnchorLive*l,FissionAppend*t){
    LiveFission*f=l->fission;
    for(int i=0;i<t->n;i++)if(apply_record_impl(l,t->records[i],t->offsets[i],1))return -1;
    for(int i=0;i<t->ncells;i++){
        FissionAppendCell*c=t->cells+i;FissionCell*published=f->cells+c->cell;
        if(published->count<=f->cap+256&&published->count+c->added>f->cap+256)f->pressure++;
        *published=c->replacement;
        fission_slot_count(f,published,c->old.count+c->added);fission_enqueue(f,c->cell);
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
    for(int i=0;i<t->ncells;i++)if(t->cells[i].moved&&
        fission_slot_copy(f,&t->cells[i].old,f->fd,t->cells[i].replacement.head))goto poison;
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
