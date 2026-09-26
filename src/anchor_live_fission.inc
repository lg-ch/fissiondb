/* Mutable, rebuildable live-cell index. The fsynced journal is authoritative.
 * Published cells use AnchorLive.lock; one background worker prepares private
 * daughters outside it. Retired chunks remain untouched until a newer checkpoint
 * is durable, so recovery can replay a tail after interrupted publication.
 */
#define FISSION_NONE UINT32_MAX
#define FISSION_CHUNK 512u
#define FISSION_ENTRY 84u
#define FISSION_MAGIC UINT64_C(0x314e4f4953534946)
typedef struct {uint32_t head,tail,count;} FissionCell;
typedef struct {uint32_t next,used;} FissionChunk;
typedef struct LiveFission {
    AnchorFissionOps ops;
    void* user;
    uint32_t cap,maximum,K,J,cell_capacity,chunk_capacity,free_head;
    FissionCell* cells;
    FissionChunk* chunks;
    int8_t* centers;
    float* scales;
    int fd,direct_fd;
    uint64_t checkpoint_end,splits,rewritten,records;
    double split_ms,split_max_ms,last_split_ms;
    uint32_t* queue_next;
    uint8_t* queued;
    uint32_t queue_head,queue_tail,pending,active,pressure;
    pthread_t worker;
    pthread_mutex_t work_mutex,job_mutex,publish_mutex;
    pthread_cond_t work_cond;
    int started,stop,wakeup,rebuilding;
    uint64_t epoch,queries_during_prepare,delta_records,pressure_waits,publishes;
    uint64_t scratch_bytes,peak_scratch_bytes;
    double publish_ms,publish_max_ms,last_publish_ms;
} LiveFission;
static void fission_enqueue(LiveFission*,uint32_t);
static void fission_signal(LiveFission*);
static int fission_worker_start(AnchorLive*);
static void fission_worker_stop(AnchorLive*);
typedef struct {
    uint64_t magic,version,fingerprint,log_device,log_inode,end,code_device,code_inode;
    uint64_t dim,cap,maximum,K,J,splits,rewritten,records;
} FissionHeader;
typedef struct {float score;uint32_t row;} FissionMember;
static double fission_now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static uint64_t fission_random(uint64_t* s){*s+=UINT64_C(0x9e3779b97f4a7c15);uint64_t z=*s;z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);return z^(z>>31);}
static int fission_member_cmp(const void* aa,const void* bb){const FissionMember*a=aa,*b=bb;if(a->score!=b->score)return a->score>b->score?-1:1;return(a->row>b->row)-(a->row<b->row);}
static int fission_cells(AnchorLive*l,uint32_t need){
    LiveFission*f=l->fission;if(need<=f->cell_capacity)return 0;if(need>f->maximum)return -1;
    uint32_t n=f->cell_capacity?f->cell_capacity:16;while(n<need){if(n>f->maximum/2){n=f->maximum;break;}n*=2;}if(n>f->maximum)n=f->maximum;
    FissionCell*c=calloc(n,sizeof(*c));int8_t*centers=calloc(n,(size_t)l->dim);float*scales=calloc(n,sizeof(*scales));
    uint32_t*next=calloc(n,4);uint8_t*queued=calloc(n,1);
    if(!c||!centers||!scales||!next||!queued){free(c);free(centers);free(scales);free(next);free(queued);return -1;}
    if(f->K){memcpy(c,f->cells,f->K*sizeof(*c));memcpy(centers,f->centers,(size_t)f->K*l->dim);memcpy(scales,f->scales,f->K*sizeof(*scales));}
    if(f->K){memcpy(next,f->queue_next,f->K*4);memcpy(queued,f->queued,f->K);}
    free(f->cells);free(f->centers);free(f->scales);free(f->queue_next);free(f->queued);
    f->cells=c;f->centers=centers;f->scales=scales;f->queue_next=next;f->queued=queued;f->cell_capacity=n;return 0;
}
static int fission_chunks(LiveFission*f,uint32_t need){
    if(need<=f->chunk_capacity)return 0;
    uint32_t n=f->chunk_capacity?f->chunk_capacity:32;while(n<need){if(n>UINT32_MAX/2)return -1;n*=2;}
    FissionChunk*p=realloc(f->chunks,(size_t)n*sizeof(*p));if(!p)return -1;f->chunks=p;f->chunk_capacity=n;return 0;
}
static int fission_set_center(AnchorLive*l,uint32_t k,const float*v){
    LiveFission*f=l->fission;float mx=0;for(int d=0;d<l->dim;d++)mx=fmaxf(mx,fabsf(v[d]));if(!isfinite(mx)||mx<=0)return -1;
    int8_t*c=f->centers+(size_t)k*l->dim;double norm=0;
    for(int d=0;d<l->dim;d++){c[d]=(int8_t)lrintf(fmaxf(-127,fminf(127,v[d]*127/mx)));norm+=(int)c[d]*(int)c[d];}
    if(norm<=0)return -1;
    f->scales[k]=1/sqrt(norm);return 0;
}
static int fission_append(AnchorLive*l,uint32_t k,uint32_t id,uint64_t offset,const float*v){
    LiveFission*f=l->fission;FissionCell*c=f->cells+k;uint8_t code[FISSION_ENTRY]={0};
    memcpy(code,&id,4);memcpy(code+4,&offset,8);
    if(f->ops.encode(f->user,v,f->centers+(size_t)k*l->dim,code+12))return -1;
    uint32_t crc=XXH32(code,80,k);memcpy(code+80,&crc,4);
    if(c->tail==FISSION_NONE||f->chunks[c->tail].used==FISSION_CHUNK){
        uint32_t b;
        if(f->free_head!=FISSION_NONE){b=f->free_head;f->free_head=f->chunks[b].next;}
        else{b=f->J;if(b==UINT32_MAX||fission_chunks(f,b+1))return -1;
            if(ftruncate(f->fd,(off_t)((uint64_t)(b+1)*FISSION_CHUNK*FISSION_ENTRY)))return -1;
            f->J++;}
        f->chunks[b]=(FissionChunk){FISSION_NONE,0};
        if(c->tail==FISSION_NONE)c->head=b;else f->chunks[c->tail].next=b;c->tail=b;
    }
    FissionChunk*b=f->chunks+c->tail;
    if(write_all(f->fd,code,sizeof(code),((uint64_t)c->tail*FISSION_CHUNK+b->used)*FISSION_ENTRY))return -1;
    b->used++;c->count++;
    if(f->started&&!f->rebuilding&&c->count==f->cap+257)f->pressure++;
    return 0;
}
static int fission_current(const AnchorLive*l,uint32_t id,uint64_t offset){
    if((uint64_t)id>=l->base_n+l->count||anchor_live_is_deleted(l,id))return 0;
    Override*u=override_find(l,id);return !u||u->offset==offset;
}
static int fission_split(AnchorLive*l,uint32_t cell){
    LiveFission*f=l->fission;uint32_t n=f->cells[cell].count,D=l->dim,S=D<64?D:64;
    if(n<=f->cap||f->K>=f->maximum)return 0;
    double began=fission_now();int rc=-1;uint32_t used=0;
    float*raw=malloc((size_t)n*D*4),*sketch=malloc((size_t)n*S*4),*means=calloc((size_t)2*D,4);
    uint32_t*ids=malloc((size_t)n*4);uint64_t*locations=malloc((size_t)n*8);
    FissionMember*members=malloc((size_t)n*sizeof(*members));uint8_t*buffer=malloc(FISSION_CHUNK*FISSION_ENTRY);
    if(!raw||!sketch||!means||!ids||!locations||!members||!buffer)goto done;
    for(uint32_t b=f->cells[cell].head;b!=FISSION_NONE;b=f->chunks[b].next){
        if(b>=f->J||f->chunks[b].used>FISSION_CHUNK)goto done;
        uint32_t m=f->chunks[b].used;if(read_all(f->fd,buffer,(size_t)m*FISSION_ENTRY,(uint64_t)b*FISSION_CHUNK*FISSION_ENTRY))goto done;
        for(uint32_t j=0;j<m;j++){
            uint8_t*p=buffer+(size_t)j*FISSION_ENTRY;uint32_t id,crc;uint64_t off;
            memcpy(&id,p,4);memcpy(&off,p+4,8);memcpy(&crc,p+80,4);if(XXH32(p,80,cell)!=crc)goto done;
            if(!fission_current(l,id,off))continue;
            if(used>=n||read_all(l->log_fd,raw+(size_t)used*D,(size_t)D*4,off+sizeof(LiveRecord)))goto done;
            ids[used]=id;locations[used]=off;members[used]=(FissionMember){0,used};
            float projected[1024];f->ops.sketch(f->user,raw+(size_t)used*D,projected);memcpy(sketch+(size_t)used*S,projected,S*4);used++;
        }
    }
    /* Old versions/deletions can make a physical overflow disappear. */
    if(used<=f->cap){
        f->cells[cell]=(FissionCell){FISSION_NONE,FISSION_NONE,0};
        for(uint32_t i=0;i<used;i++)if(fission_append(l,cell,ids[i],locations[i],raw+(size_t)i*D))goto done;
        rc=0;goto done;
    }
    uint64_t rng=UINT64_C(781923)+f->splits*31337+cell;float best=-1;uint32_t pa=0,pb=1;
    for(int pair=0;pair<16;pair++){uint32_t a=fission_random(&rng)%used,b=fission_random(&rng)%used;float distance=0;
        for(uint32_t d=0;d<S;d++){float v=sketch[(size_t)a*S+d]-sketch[(size_t)b*S+d];distance+=v*v;}
        if(distance>best){best=distance;pa=a;pb=b;}}
    for(uint32_t i=0;i<used;i++)for(uint32_t d=0;d<S;d++)members[i].score+=(sketch[(size_t)pa*S+d]-sketch[(size_t)pb*S+d])*sketch[(size_t)i*S+d];
    qsort(members,used,sizeof(*members),fission_member_cmp);uint32_t mid=used/2;
    for(uint32_t j=0;j<used;j++){int side=j>=mid;for(uint32_t d=0;d<D;d++)means[(size_t)side*D+d]+=raw[(size_t)members[j].row*D+d];}
    uint32_t other=f->K;if(fission_cells(l,other+1))goto done;
    /* Quantization only needs the mean direction; division by count cancels. */
    for(int side=0;side<2;side++){double norm=0;for(uint32_t d=0;d<D;d++)norm+=(double)means[(size_t)side*D+d]*means[(size_t)side*D+d];
        if(norm<1e-20)memcpy(means+(size_t)side*D,raw+(size_t)members[side?mid:0].row*D,D*4);}
    if(fission_set_center(l,cell,means)||fission_set_center(l,other,means+D))goto done;
    f->K++;f->cells[cell]=f->cells[other]=(FissionCell){FISSION_NONE,FISSION_NONE,0};
    for(uint32_t j=0;j<used;j++){uint32_t i=members[j].row;if(fission_append(l,j>=mid?other:cell,ids[i],locations[i],raw+(size_t)i*D))goto done;}
    f->splits++;f->rewritten+=used;f->last_split_ms=fission_now()-began;f->split_ms+=f->last_split_ms;f->split_max_ms=fmax(f->split_max_ms,f->last_split_ms);rc=0;
done:
    free(raw);free(sketch);free(means);free(ids);free(locations);free(members);free(buffer);return rc;
}
static int live_fission_observe(AnchorLive*l,const LiveRecord*r,uint64_t offset){
    LiveFission*f=l->fission;const float*v=(const float*)(r+1);
    if(!f->K){if(fission_cells(l,2))return -1;f->K=2;for(int i=0;i<2;i++){f->cells[i]=(FissionCell){FISSION_NONE,FISSION_NONE,0};if(fission_set_center(l,i,v))return -1;}}
    int8_t q[1024];float mx=0;for(int d=0;d<l->dim;d++)mx=fmaxf(mx,fabsf(v[d]));
    for(int d=0;d<l->dim;d++)q[d]=(int8_t)lrintf(v[d]*127/fmaxf(mx,1e-30f));
    float best[2]={-INFINITY,-INFINITY};uint32_t pick[2]={0,1};
    for(uint32_t k=0;k<f->K;k++){float score=f->ops.dot(q,f->centers+(size_t)k*l->dim,l->dim)*f->scales[k];
        if(score>best[1]){if(score>best[0]){best[1]=best[0];pick[1]=pick[0];best[0]=score;pick[0]=k;}else{best[1]=score;pick[1]=k;}}}
    if(pick[0]==pick[1])return -1;
    for(int i=0;i<2;i++)if(fission_append(l,pick[i],r->id,offset,v))return -1;
    if(f->started&&!f->rebuilding){for(int i=0;i<2;i++)fission_enqueue(f,pick[i]);}
    else for(int i=0;i<2;i++)if(fission_split(l,pick[i]))return -1;
    f->records++;return 0;
}
static int fission_collect_free(AnchorLive*l){
    LiveFission*f=l->fission;uint8_t*used=calloc(f->J?f->J:1,1);if(!used)return -1;
    for(uint32_t k=0;k<f->K;k++){uint64_t count=0;uint32_t last=FISSION_NONE;
        for(uint32_t b=f->cells[k].head;b!=FISSION_NONE;b=f->chunks[b].next){
            if(b>=f->J||used[b]||!f->chunks[b].used||f->chunks[b].used>FISSION_CHUNK){free(used);return -1;}
            used[b]=1;count+=f->chunks[b].used;last=b;
        }
        if(count!=f->cells[k].count||last!=f->cells[k].tail){free(used);return -1;}}
    f->free_head=FISSION_NONE;for(uint32_t b=0;b<f->J;b++)if(!used[b]){f->chunks[b]=(FissionChunk){f->free_head,0};f->free_head=b;}
    free(used);return 0;
}
static int fission_checkpoint_locked(AnchorLive*l){
    LiveFission*f=l->fission;if(!f||l->poisoned)return -1;struct stat log,code;
    if(fstat(l->log_fd,&log)||fstat(f->fd,&code)||fsync(l->log_fd)||fsync(f->fd))return -1;
    FissionHeader h={FISSION_MAGIC,1,l->header.fingerprint,log.st_dev,log.st_ino,l->end,code.st_dev,code.st_ino,l->dim,f->cap,f->maximum,f->K,f->J,f->splits,f->rewritten,f->records};
    int fd=openat(l->dir_fd,".fission.state.tmp",O_RDWR|O_CREAT|O_TRUNC|O_NOFOLLOW,0600);if(fd<0)return -1;
    XXH64_state_t*hash=XXH64_createState();if(!hash){close(fd);return -1;}XXH64_reset(hash,0);
    const void*blocks[]={&h,f->cells,f->centers,f->scales,f->chunks};
    size_t sizes[]={sizeof(h),(size_t)f->K*sizeof(*f->cells),(size_t)f->K*l->dim,(size_t)f->K*4,(size_t)f->J*sizeof(*f->chunks)};
    int rc=-1;uint64_t off=0;
    for(int i=0;i<5;i++){if(write_all(fd,blocks[i],sizes[i],off))goto done;XXH64_update(hash,blocks[i],sizes[i]);off+=sizes[i];}
    uint64_t sum=XXH64_digest(hash);if(write_all(fd,&sum,8,off)||fsync(fd)||renameat(l->dir_fd,".fission.state.tmp",l->dir_fd,"fission.state")||fsync(l->dir_fd))goto done;
    f->checkpoint_end=l->end;rc=fission_collect_free(l);
done:XXH64_freeState(hash);close(fd);return rc;
}
static int fission_validate_codes(AnchorLive*l){
    LiveFission*f=l->fission;uint32_t*owners=malloc((size_t)(f->J?f->J:1)*4);
    uint8_t*buffer=malloc(FISSION_CHUNK*FISSION_ENTRY);int rc=-1;
    if(!owners||!buffer)goto done;
    for(uint32_t b=0;b<f->J;b++)owners[b]=FISSION_NONE;
    for(uint32_t k=0;k<f->K;k++)for(uint32_t b=f->cells[k].head;b!=FISSION_NONE;b=f->chunks[b].next)owners[b]=k;
    for(uint32_t b=0;b<f->J;b++)if(owners[b]!=FISSION_NONE){
        uint32_t n=f->chunks[b].used;if(read_all(f->fd,buffer,(size_t)n*FISSION_ENTRY,(uint64_t)b*FISSION_CHUNK*FISSION_ENTRY))goto done;
        for(uint32_t i=0;i<n;i++){uint8_t*p=buffer+(size_t)i*FISSION_ENTRY;uint32_t crc,id;uint64_t offset;
            memcpy(&id,p,4);memcpy(&offset,p+4,8);memcpy(&crc,p+80,4);
            if(crc!=XXH32(p,80,owners[b])||(uint64_t)id>=l->base_n+l->count||offset<sizeof(LiveHeader)||offset>=f->checkpoint_end)goto done;
        }
    }rc=0;
done:free(owners);free(buffer);return rc;
}
static int fission_load(AnchorLive*l){
    LiveFission*f=l->fission;int fd=openat(l->dir_fd,"fission.state",O_RDONLY|O_NOFOLLOW);if(fd<0)return -1;
    int rc=-1;FissionHeader h;struct stat st,log,code;XXH64_state_t*hash=NULL;
    if(read_all(fd,&h,sizeof(h),0)||fstat(fd,&st)||fstat(l->log_fd,&log)||fstat(f->fd,&code)||h.magic!=FISSION_MAGIC||h.version!=1||
       h.fingerprint!=l->header.fingerprint||h.log_device!=(uint64_t)log.st_dev||h.log_inode!=(uint64_t)log.st_ino||h.code_device!=(uint64_t)code.st_dev||h.code_inode!=(uint64_t)code.st_ino||
       h.end<sizeof(LiveHeader)||h.end>l->end||h.dim!=(uint64_t)l->dim||h.cap!=f->cap||h.maximum!=f->maximum||h.K>f->maximum||h.J>UINT32_MAX||
       (h.K==1)||h.J>(uint64_t)code.st_size/(FISSION_CHUNK*FISSION_ENTRY))goto done;
    if(fission_cells(l,h.K)||fission_chunks(f,h.J))goto done;
    size_t sizes[]={sizeof(h),(size_t)h.K*sizeof(*f->cells),(size_t)h.K*l->dim,(size_t)h.K*4,(size_t)h.J*sizeof(*f->chunks)};
    void*blocks[]={&h,f->cells,f->centers,f->scales,f->chunks};uint64_t bytes=8;for(int i=0;i<5;i++)bytes+=sizes[i];if(bytes!=(uint64_t)st.st_size)goto done;
    hash=XXH64_createState();if(!hash)goto done;XXH64_reset(hash,0);uint64_t off=0;
    for(int i=0;i<5;i++){if(read_all(fd,blocks[i],sizes[i],off))goto done;XXH64_update(hash,blocks[i],sizes[i]);off+=sizes[i];}
    uint64_t sum;if(read_all(fd,&sum,8,off)||sum!=XXH64_digest(hash))goto done;
    f->K=h.K;f->J=h.J;f->splits=h.splits;f->rewritten=h.rewritten;f->records=h.records;f->checkpoint_end=h.end;
    for(uint32_t k=0;k<f->K;k++)if(!isfinite(f->scales[k])||f->scales[k]<=0)goto done;
    if(fission_collect_free(l)||fission_validate_codes(l))goto done;
    rc=0;
done:if(hash)XXH64_freeState(hash);close(fd);return rc;
}
static int fission_replay(AnchorLive*l,uint64_t from){
    size_t cap=sizeof(LiveRecord)+(size_t)l->dim*4+TAG_BYTES_MAX+64;LiveRecord*r=malloc(cap);if(!r)return -1;int rc=-1;
    for(uint64_t off=from;off<l->end;){
        if(read_all(l->log_fd,r,sizeof(*r),off)||r->magic!=RECORD_MAGIC||r->bytes<sizeof(*r)||r->bytes>cap||r->bytes>l->end-off||read_all(l->log_fd,r+1,r->bytes-sizeof(*r),off+sizeof(*r)))goto done;
        uint64_t sum=r->checksum;r->checksum=0;if(XXH64(r,r->bytes,0)!=sum)goto done;
        if((r->kind==1||r->kind==6||r->kind==7)&&fission_current(l,r->id,off)&&live_fission_observe(l,r,off))goto done;
        off+=r->bytes;
    }rc=0;
done:free(r);return rc;
}
static int live_fission_rebuild(AnchorLive*l){
    /* A crash after truncation must not revive an older checkpoint that names
       the same arena inode. Invalidate it durably before replacing any bytes. */
    if(unlinkat(l->dir_fd,"fission.state",0)&&errno!=ENOENT)return -1;
    if(fsync(l->dir_fd))return -1;
    LiveFission*f=l->fission;f->K=f->J=0;f->free_head=FISSION_NONE;f->checkpoint_end=0;f->splits=f->rewritten=f->records=0;
    f->queue_head=f->queue_tail=f->active=FISSION_NONE;f->pending=f->pressure=0;
    if(f->queued)memset(f->queued,0,f->cell_capacity);
    f->rebuilding=1;int rc=ftruncate(f->fd,0)||fission_replay(l,sizeof(LiveHeader));f->rebuilding=0;
    if(rc)return -1;
    return fission_checkpoint_locked(l);
}
int anchor_live_fission_config(AnchorLive*l,uint32_t*cap,uint32_t*maximum){
    if(!l)return -1;
    int fd=openat(l->dir_fd,"fission.config",O_RDONLY|O_NOFOLLOW);if(fd<0)return errno==ENOENT?0:-1;
    uint64_t h[5];struct stat st;int bad=read_all(fd,h,sizeof(h),0)||fstat(fd,&st)||st.st_size!=sizeof(h)||h[0]!=FISSION_MAGIC||h[1]!=1||h[4]!=XXH64(h,32,0)||h[2]<64||h[2]>65536||h[3]<2||h[3]>1000000;
    close(fd);if(bad)return -1;*cap=h[2];*maximum=h[3];return 1;
}
int anchor_live_enable_fission(AnchorLive*l,uint32_t cap,uint32_t maximum,const AnchorFissionOps*ops,void*user){
    if(!l||l->fission||l->dim>1024||cap<64||cap>65536||maximum<2||maximum>1000000||!ops||!ops->dot||!ops->encode||!ops->sketch)return -1;
    LiveFission*f=calloc(1,sizeof(*f));if(!f)return -1;f->fd=f->direct_fd=-1;f->free_head=FISSION_NONE;f->ops=*ops;f->user=user;f->cap=cap;f->maximum=maximum;
    f->queue_head=f->queue_tail=f->active=FISSION_NONE;
    pthread_mutex_init(&f->work_mutex,NULL);pthread_mutex_init(&f->job_mutex,NULL);pthread_mutex_init(&f->publish_mutex,NULL);pthread_cond_init(&f->work_cond,NULL);
    pthread_rwlock_wrlock(&l->lock);int rc=-1;if(l->poisoned)goto done;l->fission=f;
    f->fd=openat(l->dir_fd,"fission.codes",O_RDWR|O_CREAT|O_NOFOLLOW,0600);if(f->fd<0)goto done;
    f->direct_fd=openat(l->dir_fd,"fission.codes",O_RDONLY|O_DIRECT|O_NOFOLLOW);
    uint32_t saved_cap=0,saved_max=0;int config=anchor_live_fission_config(l,&saved_cap,&saved_max);
    if(config<0||(config&&(saved_cap!=cap||saved_max!=maximum)))goto done;
    if(fission_load(l)){if(live_fission_rebuild(l))goto done;}
    else if(fission_replay(l,f->checkpoint_end))goto done;
    if(!config){
        uint64_t h[5]={FISSION_MAGIC,1,cap,maximum,0};h[4]=XXH64(h,32,0);
        int fd=openat(l->dir_fd,".fission.config.tmp",O_WRONLY|O_CREAT|O_TRUNC|O_NOFOLLOW,0600);if(fd<0)goto done;
        int bad=write_all(fd,h,sizeof(h),0)||fsync(fd);close(fd);if(bad||renameat(l->dir_fd,".fission.config.tmp",l->dir_fd,"fission.config")||fsync(l->dir_fd))goto done;
    }
    if(fission_worker_start(l))goto done;
    rc=0;
done:
    if(rc){l->fission=NULL;if(f->fd>=0)close(f->fd);if(f->direct_fd>=0)close(f->direct_fd);free(f->cells);free(f->chunks);free(f->centers);free(f->scales);free(f->queue_next);free(f->queued);pthread_cond_destroy(&f->work_cond);pthread_mutex_destroy(&f->work_mutex);pthread_mutex_destroy(&f->job_mutex);pthread_mutex_destroy(&f->publish_mutex);free(f);}
    pthread_rwlock_unlock(&l->lock);return rc;
}
static void live_fission_close(AnchorLive*l){
    LiveFission*f=l->fission;if(!f)return;
    fission_worker_stop(l);
    if(!l->poisoned)fission_checkpoint_locked(l);
    if(f->fd>=0)close(f->fd);
    if(f->direct_fd>=0)close(f->direct_fd);
    free(f->cells);free(f->chunks);free(f->centers);free(f->scales);free(f->queue_next);free(f->queued);
    pthread_cond_destroy(&f->work_cond);pthread_mutex_destroy(&f->work_mutex);pthread_mutex_destroy(&f->job_mutex);pthread_mutex_destroy(&f->publish_mutex);free(f);l->fission=NULL;
}
int anchor_live_has_fission(const AnchorLive*l){return l&&l->fission;}
static uint64_t live_fission_end(const AnchorLive*l){return l->fission?l->fission->checkpoint_end:0;}
int anchor_live_fission_checkpoint(AnchorLive*l){if(!l||!l->fission)return -1;live_fission_maintenance_lock(l);pthread_rwlock_wrlock(&l->lock);int rc=fission_checkpoint_locked(l);pthread_rwlock_unlock(&l->lock);live_fission_maintenance_unlock(l);return rc;}
int anchor_live_fission_stats(AnchorLive*l,uint64_t v[8],double t[3]){
    if(!l||!l->fission||!v||!t||anchor_live_read_lock(l))return -1;
    LiveFission*f=l->fission;uint32_t largest=0;
    for(uint32_t k=0;k<f->K;k++)if(f->cells[k].count>largest)largest=f->cells[k].count;
    uint64_t a[]={f->K,f->splits,f->rewritten,f->records,largest,(uint64_t)f->cell_capacity*(l->dim+sizeof(FissionCell)+9)+(uint64_t)f->chunk_capacity*sizeof(FissionChunk)+sizeof(*f),(uint64_t)f->J*FISSION_CHUNK*FISSION_ENTRY,f->maximum};
    memcpy(v,a,sizeof(a));t[0]=f->split_ms;t[1]=f->split_max_ms;t[2]=f->last_split_ms;anchor_live_read_unlock(l);return 0;
}
#include "anchor_live_fission_io.inc"

#include "anchor_live_fission_async.inc"

#include "anchor_live_fission_ingest.inc"
