/* One bounded split job. The parent stays published (including concurrent
 * appends) until all of its records have been encoded into private daughters.
 * Lock order: job_mutex -> publish_mutex -> live lock -> work_mutex.
 * Only reservations/publication take publish_mutex. Never wait for progress
 * with live locked.
 * Checkpoint/compaction take job_mutex, so neither can recycle staging chunks
 * or change journal offsets during preparation. Readers only take live.lock.
 */
typedef struct {uint32_t block,start,count;} FissionSpan;
typedef struct {
    uint32_t parent,other,limit,consumed,used,count[2],written[2],blocks[2];
    int split;
    int8_t centers[2][1024];
    float scales[2];
    FissionSpan* spans;
    uint32_t* block_ids[2];
    uint8_t *incoming,*valid,*codes[2];
    float *raw,*sketch;
    FissionMember* members;
    uint64_t bytes;
} FissionJob;

static void fission_signal(LiveFission*f){
    pthread_mutex_lock(&f->work_mutex);
    f->wakeup=1;f->epoch++;pthread_cond_broadcast(&f->work_cond);
    pthread_mutex_unlock(&f->work_mutex);
}
static void fission_enqueue(LiveFission*f,uint32_t cell){
    if(f->K>=f->maximum||f->cells[cell].count<=f->cap||f->queued[cell]||f->active==cell)return;
    f->queued[cell]=1;f->queue_next[cell]=FISSION_NONE;
    if(f->queue_tail==FISSION_NONE)f->queue_head=cell;else f->queue_next[f->queue_tail]=cell;
    f->queue_tail=cell;f->pending++;fission_signal(f);
}
/* Consume a wakeup without losing a publication between unlock and wait. */
static void fission_wait_locked(AnchorLive*l){
    LiveFission*f=l->fission;pthread_mutex_lock(&f->work_mutex);uint64_t epoch=f->epoch;
    pthread_rwlock_unlock(&l->lock);
    while(epoch==f->epoch&&!f->stop)pthread_cond_wait(&f->work_cond,&f->work_mutex);
    pthread_mutex_unlock(&f->work_mutex);
}
static void live_vector_write_lock(AnchorLive*l){
    pthread_rwlock_wrlock(&l->lock);
    while(!l->poisoned&&l->fission&&l->fission->started&&l->fission->pressure&&l->fission->K<l->fission->maximum){
        l->fission->pressure_waits++;fission_wait_locked(l);pthread_rwlock_wrlock(&l->lock);
    }
}
static void live_fission_maintenance_lock(AnchorLive*l){if(l->fission){pthread_mutex_lock(&l->fission->job_mutex);pthread_mutex_lock(&l->fission->publish_mutex);}}
static void live_fission_maintenance_unlock(AnchorLive*l){if(l->fission){pthread_mutex_unlock(&l->fission->publish_mutex);pthread_mutex_unlock(&l->fission->job_mutex);}}

static void fission_job_free(FissionJob*j){
    free(j->spans);free(j->incoming);free(j->valid);free(j->raw);free(j->sketch);free(j->members);
    for(int s=0;s<2;s++){free(j->block_ids[s]);free(j->codes[s]);}free(j);
}
static FissionJob* fission_job_alloc(uint32_t cap,int dim){
    FissionJob*j=calloc(1,sizeof(*j));if(!j)return NULL;
    j->limit=cap+512;uint32_t n=j->limit,blocks=1;
    while((uint64_t)blocks*FISSION_CHUNK<n)blocks*=2;
#define JOB_ALLOC(field,size) do{j->field=malloc(size);if(!j->field)goto fail;j->bytes+=(size);}while(0)
    JOB_ALLOC(spans,(size_t)(blocks+1)*sizeof(*j->spans));
    JOB_ALLOC(incoming,(size_t)n*FISSION_ENTRY);JOB_ALLOC(valid,n);
    JOB_ALLOC(raw,(size_t)n*dim*4);JOB_ALLOC(sketch,(size_t)n*(dim<64?dim:64)*4);
    JOB_ALLOC(members,(size_t)n*sizeof(*j->members));
    for(int s=0;s<2;s++){JOB_ALLOC(block_ids[s],(size_t)blocks*4);JOB_ALLOC(codes[s],(size_t)n*FISSION_ENTRY);}
#undef JOB_ALLOC
    j->bytes+=sizeof(*j);return j;
fail:fission_job_free(j);return NULL;
}
/* Copy only descriptors while locked; all code IO and journal IO are outside.
 * The published parent only appends until this job swaps it. Existing offsets
 * and used slots never change. Appends after this snapshot become the next delta.
 */
static int fission_job_collect(AnchorLive*l,FissionJob*j,uint32_t*amount){
    LiveFission*f=l->fission;uint32_t count=0,skip=j->consumed,nspan=0,target;
    if(anchor_live_read_lock(l))return -1;
    target=f->cells[j->parent].count;
    if(target>j->limit||target<skip){anchor_live_read_unlock(l);return -1;}
    for(uint32_t b=f->cells[j->parent].head;b!=FISSION_NONE;b=f->chunks[b].next){
        uint32_t n=f->chunks[b].used;
        if(skip>=n){skip-=n;continue;}
        j->spans[nspan++]=(FissionSpan){b,skip,n-skip};count+=n-skip;skip=0;
    }
    anchor_live_read_unlock(l);
    if(count!=target-j->consumed)return -1;
    size_t offset=0;
    for(uint32_t i=0;i<nspan;i++){
        FissionSpan*s=j->spans+i;size_t bytes=(size_t)s->count*FISSION_ENTRY;
        if(read_all(f->fd,j->incoming+offset,bytes,((uint64_t)s->block*FISSION_CHUNK+s->start)*FISSION_ENTRY))return -1;
        offset+=bytes;
    }
    for(uint32_t i=0;i<count;i++){uint8_t*p=j->incoming+(size_t)i*FISSION_ENTRY;uint32_t crc;
        memcpy(&crc,p+80,4);if(crc!=XXH32(p,80,j->parent))return -1;}
    if(anchor_live_read_lock(l))return -1;
    for(uint32_t i=0;i<count;i++){uint8_t*p=j->incoming+(size_t)i*FISSION_ENTRY;uint32_t id;uint64_t off;
        memcpy(&id,p,4);memcpy(&off,p+4,8);j->valid[i]=fission_current(l,id,off);}
    anchor_live_read_unlock(l);j->consumed=target;*amount=count;return 0;
}
static int fission_job_encode(AnchorLive*l,FissionJob*j,int side,const uint8_t*source,const float*v){
    LiveFission*f=l->fission;if(j->count[side]>=j->limit)return -1;
    uint8_t*out=j->codes[side]+(size_t)j->count[side]*FISSION_ENTRY;
    memset(out,0,FISSION_ENTRY);memcpy(out,source,12);
    if(f->ops.encode(f->user,v,j->centers[side],out+12))return -1;
    uint32_t crc=XXH32(out,80,side?j->other:j->parent);memcpy(out+80,&crc,4);j->count[side]++;return 0;
}
static int fission_job_prepare(AnchorLive*l,FissionJob*j){
    LiveFission*f=l->fission;uint32_t n,D=l->dim,S=D<64?D:64;
    if(fission_job_collect(l,j,&n))return -1;
    uint32_t used=0;
    for(uint32_t i=0;i<n;i++)if(j->valid[i]){
        uint8_t*p=j->incoming+(size_t)i*FISSION_ENTRY;uint64_t off;memcpy(&off,p+4,8);
        float*v=j->raw+(size_t)used*D;
        if(read_all(l->log_fd,v,D*4,off+sizeof(LiveRecord)))return -1;
        /* Retain the source prefix in a packed order, matching raw/members. */
        memmove(j->incoming+(size_t)used*FISSION_ENTRY,p,12);
        float rotated[1024];f->ops.sketch(f->user,v,rotated);memcpy(j->sketch+(size_t)used*S,rotated,S*4);
        j->members[used]=(FissionMember){0,used};used++;
    }
    j->used=used;j->split=used>f->cap;
    if(!j->split){for(uint32_t i=0;i<used;i++)if(fission_job_encode(l,j,0,j->incoming+(size_t)i*FISSION_ENTRY,j->raw+(size_t)i*D))return -1;return 0;}
    uint64_t rng=UINT64_C(781923)+f->splits*31337+j->parent;float best=-1;uint32_t pa=0,pb=1;
    for(int pair=0;pair<16;pair++){uint32_t a=fission_random(&rng)%used,b=fission_random(&rng)%used;float distance=0;
        for(uint32_t d=0;d<S;d++){float v=j->sketch[(size_t)a*S+d]-j->sketch[(size_t)b*S+d];distance+=v*v;}
        if(distance>best){best=distance;pa=a;pb=b;}}
    for(uint32_t i=0;i<used;i++)for(uint32_t d=0;d<S;d++)j->members[i].score+=(j->sketch[(size_t)pa*S+d]-j->sketch[(size_t)pb*S+d])*j->sketch[(size_t)i*S+d];
    qsort(j->members,used,sizeof(*j->members),fission_member_cmp);uint32_t mid=used/2;float means[2][1024]={{0}};
    for(uint32_t i=0;i<used;i++)for(uint32_t d=0;d<D;d++)means[i>=mid][d]+=j->raw[(size_t)j->members[i].row*D+d];
    for(int s=0;s<2;s++){
        float mx=0;for(uint32_t d=0;d<D;d++)mx=fmaxf(mx,fabsf(means[s][d]));
        if(mx<1e-10f){memcpy(means[s],j->raw+(size_t)j->members[s?mid:0].row*D,D*4);mx=0;for(uint32_t d=0;d<D;d++)mx=fmaxf(mx,fabsf(means[s][d]));}
        double norm=0;for(uint32_t d=0;d<D;d++){int v=lrintf(fmaxf(-127,fminf(127,means[s][d]*127/mx)));j->centers[s][d]=v;norm+=v*v;}
        if(norm<=0)return -1;
        j->scales[s]=1/sqrt(norm);
    }
    for(uint32_t i=0;i<used;i++){uint32_t row=j->members[i].row;
        if(fission_job_encode(l,j,i>=mid,j->incoming+(size_t)row*FISSION_ENTRY,j->raw+(size_t)row*D))return -1;}
    return 0;
}
static int fission_job_delta(AnchorLive*l,FissionJob*j){
    LiveFission*f=l->fission;uint32_t n;if(fission_job_collect(l,j,&n))return -1;
    for(uint32_t i=0;i<n;i++)if(j->valid[i]){
        uint8_t*p=j->incoming+(size_t)i*FISSION_ENTRY;uint64_t off;memcpy(&off,p+4,8);float v[1024];
        if(read_all(l->log_fd,v,(size_t)l->dim*4,off+sizeof(LiveRecord)))return -1;
        int side=0;if(j->split){float score[2]={0};for(int s=0;s<2;s++)for(int d=0;d<l->dim;d++)score[s]+=v[d]*j->centers[s][d]*j->scales[s];side=score[1]>score[0];}
        if(fission_job_encode(l,j,side,p,v))return -1;
    }
    pthread_rwlock_wrlock(&l->lock);f->delta_records+=n;pthread_rwlock_unlock(&l->lock);return 0;
}
/* Only the split worker changes representatives while job_mutex is held.
 * Grow/copy their large array without the live lock. Mutable cell/queue
 * descriptors are copied at the short pointer swap; old readers have drained.
 */
static int fission_job_grow(AnchorLive*l,uint32_t need){
    LiveFission*f=l->fission;if(need<=f->cell_capacity)return 0;
    if(need>f->maximum)return -1;
    uint32_t n=f->cell_capacity;
    while(n<need){if(n>f->maximum/2){n=f->maximum;break;}n*=2;}
    FissionCell*c=calloc(n,sizeof(*c));int8_t*centers=calloc(n,(size_t)l->dim);
    float*scales=calloc(n,4);uint32_t*next=calloc(n,4);uint8_t*queued=calloc(n,1);
    if(!c||!centers||!scales||!next||!queued){free(c);free(centers);free(scales);free(next);free(queued);return -1;}
    memcpy(centers,f->centers,(size_t)f->K*l->dim);memcpy(scales,f->scales,(size_t)f->K*4);
    pthread_rwlock_wrlock(&l->lock);
    memcpy(c,f->cells,(size_t)f->K*sizeof(*c));memcpy(next,f->queue_next,(size_t)f->K*4);memcpy(queued,f->queued,f->K);
    FissionCell*oldc=f->cells;int8_t*oldcenters=f->centers;float*oldscales=f->scales;
    uint32_t*oldnext=f->queue_next;uint8_t*oldqueued=f->queued;
    f->cells=c;f->centers=centers;f->scales=scales;f->queue_next=next;f->queued=queued;f->cell_capacity=n;
    pthread_rwlock_unlock(&l->lock);
    free(oldc);free(oldcenters);free(oldscales);free(oldnext);free(oldqueued);return 0;
}
/* Reserve unpublished arena slots under the lock, then write their codes after
 * releasing it. job_mutex prevents checkpoint reclamation of reserved slots.
 */
static int fission_job_write(AnchorLive*l,FissionJob*j){
    LiveFission*f=l->fission;int rc=-1;
    pthread_mutex_lock(&f->publish_mutex);
    if(fission_job_grow(l,j->other+(j->split?1:0))){pthread_mutex_unlock(&f->publish_mutex);return -1;}
    pthread_rwlock_wrlock(&l->lock);
    if(l->poisoned)goto unlock;
    for(int s=0;s<=j->split;s++){
        uint32_t need=(j->count[s]+FISSION_CHUNK-1)/FISSION_CHUNK;
        if(j->blocks[s]<need){
            FissionCell run;if(fission_slot_reserve(l,j->count[s],&run))goto unlock;
            j->blocks[s]=run.tail-run.head+1;j->written[s]=0;
            for(uint32_t k=0;k<j->blocks[s];k++)j->block_ids[s][k]=run.head+k;
        }
        uint32_t left=j->count[s];
        for(uint32_t k=0;k<j->blocks[s];k++){uint32_t n=left<FISSION_CHUNK?left:FISSION_CHUNK;f->chunks[j->block_ids[s][k]].used=n;left-=n;}
    }
    rc=0;
unlock:pthread_rwlock_unlock(&l->lock);pthread_mutex_unlock(&f->publish_mutex);if(rc)return rc;
    for(int s=0;s<=j->split;s++)while(j->written[s]<j->count[s]){
        uint32_t pos=j->written[s],block=pos/FISSION_CHUNK,inside=pos%FISSION_CHUNK,n=j->count[s]-pos;
        if(n>FISSION_CHUNK-inside)n=FISSION_CHUNK-inside;
        if(write_all(f->fd,j->codes[s]+(size_t)pos*FISSION_ENTRY,(size_t)n*FISSION_ENTRY,((uint64_t)j->block_ids[s][block]*FISSION_CHUNK+inside)*FISSION_ENTRY))return -1;
        j->written[s]+=n;
    }
    return 0;
}


static int fission_run_job(AnchorLive*l,uint32_t cell){
    LiveFission*f=l->fission;double began=fission_now();int rc=-1;
    FissionJob*j=fission_job_alloc(f->cap,l->dim);if(!j)return -1;
    j->parent=cell;pthread_rwlock_wrlock(&l->lock);j->other=f->K;
    memcpy(j->centers[0],f->centers+(size_t)cell*l->dim,l->dim);j->scales[0]=f->scales[cell];
    f->scratch_bytes=j->bytes;if(j->bytes>f->peak_scratch_bytes)f->peak_scratch_bytes=j->bytes;
    pthread_rwlock_unlock(&l->lock);
    if(fission_job_prepare(l,j))goto done;
    if(fission_test_barrier(1))goto done;
    for(;;){
        if(fission_job_delta(l,j)||fission_job_write(l,j))goto done;
        if(fission_test_barrier(2))goto done;
        pthread_mutex_lock(&f->publish_mutex);
        pthread_rwlock_wrlock(&l->lock);
        if(l->poisoned){pthread_rwlock_unlock(&l->lock);pthread_mutex_unlock(&f->publish_mutex);goto done;}
        if(f->cells[cell].count!=j->consumed){pthread_rwlock_unlock(&l->lock);pthread_mutex_unlock(&f->publish_mutex);continue;}
        /* Acquiring the write lock also waits out readers of the old parent.
         * Its chunks remain untouched until a later durable checkpoint. */
        double published=fission_now();
        if(f->cells[cell].count>f->cap+256)f->pressure--;
        for(int s=0;s<=j->split;s++){
            uint32_t k=s?j->other:cell;
            f->cells[k]=(FissionCell){j->blocks[s]?j->block_ids[s][0]:FISSION_NONE,j->blocks[s]?j->block_ids[s][j->blocks[s]-1]:FISSION_NONE,j->count[s]};
            memcpy(f->centers+(size_t)k*l->dim,j->centers[s],l->dim);f->scales[k]=j->scales[s];
            if(j->count[s]>f->cap+256)f->pressure++;
        }
        if(j->split){f->K++;f->splits++;f->rewritten+=j->count[0]+j->count[1];}
        f->active=FISSION_NONE;fission_enqueue(f,cell);if(j->split)fission_enqueue(f,j->other);
        f->last_publish_ms=fission_now()-published;f->publish_ms+=f->last_publish_ms;f->publish_max_ms=fmax(f->publish_max_ms,f->last_publish_ms);f->publishes++;
        f->last_split_ms=fission_now()-began;f->split_ms+=f->last_split_ms;f->split_max_ms=fmax(f->split_max_ms,f->last_split_ms);
        fission_signal(f);pthread_rwlock_unlock(&l->lock);pthread_mutex_unlock(&f->publish_mutex);rc=0;break;
    }
done:fission_job_free(j);pthread_rwlock_wrlock(&l->lock);f->scratch_bytes=0;pthread_rwlock_unlock(&l->lock);return rc;
}
static void* fission_worker_main(void*arg){
    AnchorLive*l=arg;LiveFission*f=l->fission;
    for(;;){
        pthread_mutex_lock(&f->work_mutex);
        while(!f->wakeup&&!f->stop)pthread_cond_wait(&f->work_cond,&f->work_mutex);
        int stop=f->stop;f->wakeup=0;pthread_mutex_unlock(&f->work_mutex);if(stop)break;
        for(;;){
            pthread_mutex_lock(&f->job_mutex);pthread_rwlock_wrlock(&l->lock);
            uint32_t cell=f->queue_head;
            if(l->poisoned||cell==FISSION_NONE){
                int failed=l->poisoned;if(failed)fission_signal(f);
                pthread_rwlock_unlock(&l->lock);pthread_mutex_unlock(&f->job_mutex);
                if(failed)return NULL;
                break;
            }
            f->queue_head=f->queue_next[cell];if(f->queue_head==FISSION_NONE)f->queue_tail=FISSION_NONE;
            f->queued[cell]=0;f->pending--;
            int needed=f->K<f->maximum&&f->cells[cell].count>f->cap;
            if(needed)f->active=cell;
            if(!needed)fission_signal(f);
            pthread_rwlock_unlock(&l->lock);
            int failed=needed&&fission_run_job(l,cell);
            if(failed){pthread_rwlock_wrlock(&l->lock);l->poisoned=1;f->active=FISSION_NONE;fission_signal(f);pthread_rwlock_unlock(&l->lock);}
            pthread_mutex_unlock(&f->job_mutex);if(failed)return NULL;
        }
    }return NULL;
}
static int fission_worker_start(AnchorLive*l){
    LiveFission*f=l->fission;pthread_attr_t attr;if(pthread_attr_init(&attr))return -1;
    int rc=pthread_attr_setstacksize(&attr,1024*1024);
    if(!rc)rc=pthread_create(&f->worker,&attr,fission_worker_main,l);
    pthread_attr_destroy(&attr);if(rc)return -1;
    f->started=1;
    for(uint32_t k=0;k<f->K;k++){if(f->cells[k].count>f->cap+256)f->pressure++;fission_enqueue(f,k);}
    return 0;
}
int anchor_live_fission_flush(AnchorLive*l){
    if(!l||!l->fission)return -1;
    for(;;){
        if(anchor_live_read_lock(l))return -1;
        LiveFission*f=l->fission;
        if(!f->pending&&f->active==FISSION_NONE){anchor_live_read_unlock(l);return 0;}
        fission_wait_locked(l);
    }
}
static void fission_worker_stop(AnchorLive*l){
    LiveFission*f=l->fission;if(!f->started)return;
    anchor_live_fission_flush(l);
    pthread_mutex_lock(&f->work_mutex);f->stop=1;pthread_cond_broadcast(&f->work_cond);pthread_mutex_unlock(&f->work_mutex);
    pthread_join(f->worker,NULL);f->started=0;
}
int anchor_live_fission_progress(AnchorLive*l,uint64_t v[8],double t[3]){
    if(!l||!l->fission||!v||!t||anchor_live_read_lock(l))return -1;
    LiveFission*f=l->fission;
    uint64_t values[]={f->pending+(f->active!=FISSION_NONE),f->active!=FISSION_NONE,
        __atomic_load_n(&f->queries_during_prepare,__ATOMIC_RELAXED),f->delta_records,__atomic_load_n(&f->pressure_waits,__ATOMIC_RELAXED),f->scratch_bytes,f->peak_scratch_bytes,f->publishes};
    double times[]={f->publish_ms,f->publish_max_ms,f->last_publish_ms};memcpy(v,values,sizeof(values));memcpy(t,times,sizeof(times));anchor_live_read_unlock(l);return 0;
}
