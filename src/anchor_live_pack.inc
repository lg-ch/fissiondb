/* Rebuildable compressed live snapshot. The journal remains authoritative.
 * Build from a pinned journal inode and captured cell heads without holding
 * the write lock. Concurrent appends are searched as an exact tail. */
typedef struct {
    uint64_t magic, version, fingerprint, device, inode, end, cells, bytes;
} PackHeader;
#define PACK_MAGIC UINT64_C(0x314b43415052474d)
#define PACK_ENTRY 84u
static void live_pack_clear(AnchorLive* l) {
    if(l->pack_fd>=0)close(l->pack_fd);
    l->pack_fd=-1;l->pack_end=0;free(l->pack_offsets);l->pack_offsets=NULL;
}
static void live_pack_open(AnchorLive* l) {
    int fd=openat(l->dir_fd,"live.pack",O_RDONLY|O_NOFOLLOW);if(fd<0)return;
    PackHeader h;struct stat st,log;
    if(read_all(fd,&h,sizeof(h),0)||fstat(fd,&st)||fstat(l->log_fd,&log)||h.magic!=PACK_MAGIC||h.version!=1||
       h.fingerprint!=l->header.fingerprint||h.device!=(uint64_t)log.st_dev||h.inode!=(uint64_t)log.st_ino||
       h.end>l->end||h.end<sizeof(LiveHeader)||h.cells!=(uint64_t)l->cells||h.bytes!=(uint64_t)st.st_size){close(fd);return;}
    uint64_t* offsets=malloc(((size_t)l->cells+1)*8);if(!offsets){close(fd);return;}
    int bad=read_all(fd,offsets,((size_t)l->cells+1)*8,sizeof(h))||offsets[0]!=sizeof(h)+((size_t)l->cells+1)*8;
    for(int c=0;c<l->cells&&!bad;c++)bad=offsets[c+1]<offsets[c]||(offsets[c+1]-offsets[c])%PACK_ENTRY;
    if(bad||offsets[l->cells]!=h.bytes){free(offsets);close(fd);return;}
    l->pack_fd=fd;l->pack_offsets=offsets;l->pack_end=h.end;
}
uint64_t anchor_live_unpacked_bytes(AnchorLive* l) {
    if(!l||anchor_live_read_lock(l))return UINT64_MAX;
    uint64_t cutoff=l->fission?live_fission_end(l):l->pack_end;
    uint64_t n=l->end-(cutoff?cutoff:sizeof(LiveHeader));anchor_live_read_unlock(l);return n;
}
int anchor_live_pack(AnchorLive* l,AnchorLiveEncode encode,void* user) {
    if(!l||!encode||l->dim>1024||pthread_mutex_trylock(&l->pack_lock))return -1;
    int rc=-1,source=-1,fd=-1;uint64_t *heads=NULL,*offsets=NULL;LiveRecord* r=NULL;
    PackHeader h={PACK_MAGIC,1,0,0,0,0,0,0};struct stat st;
    size_t cap=sizeof(LiveRecord)+(size_t)l->dim*4+TAG_BYTES_MAX+64;
    heads=malloc((size_t)l->cells*8);offsets=malloc(((size_t)l->cells+1)*8);r=malloc(cap);
    if(!heads||!offsets||!r||anchor_live_read_lock(l))goto done;
    source=dup(l->log_fd);
    if(source<0||fstat(source,&st)){anchor_live_read_unlock(l);goto done;}
    memcpy(heads,l->heads,(size_t)l->cells*8);
    h.fingerprint=l->header.fingerprint;h.device=st.st_dev;h.inode=st.st_ino;h.end=l->end;h.cells=l->cells;
    anchor_live_read_unlock(l);
    fd=openat(l->dir_fd,".live.pack.tmp",O_RDWR|O_CREAT|O_TRUNC|O_NOFOLLOW,0600);if(fd<0)goto done;
    uint64_t end=sizeof(h)+((size_t)l->cells+1)*8;
    uint8_t batch[PACK_ENTRY*1024];
    for(int cell=0;cell<l->cells;cell++){
        offsets[cell]=end;uint64_t off=heads[cell];size_t used=0;
        while(off){
            if(off<sizeof(LiveHeader)||off>=h.end||read_all(source,r,sizeof(*r),off)||r->magic!=RECORD_MAGIC||
               (r->kind!=1&&r->kind!=6&&r->kind!=7)||r->bytes<sizeof(*r)+(size_t)l->dim*4||r->bytes>cap||r->bytes>h.end-off)goto done;
            if(read_all(source,r+1,r->bytes-sizeof(*r),off+sizeof(*r)))goto done;
            uint64_t hash=r->checksum;r->checksum=0;if(XXH64(r,r->bytes,0)!=hash)goto done;
            int slot=-1;for(int i=0;i<l->copies;i++)if(r->cells[i]==(uint32_t)cell)slot=i;
            if(slot<0||r->previous[slot]>=off)goto done;
            uint8_t* p=batch+used;memcpy(p,&r->id,4);memcpy(p+4,&off,8);
            if(encode(user,(uint32_t)cell,(const float*)(r+1),p+12))goto done;
            uint32_t checksum=XXH32(p,80,(uint32_t)cell);memcpy(p+80,&checksum,4);
            used+=PACK_ENTRY;
            if(used==sizeof(batch)){if(write_all(fd,batch,used,end))goto done;end+=used;used=0;}
            off=r->previous[slot];
        }
        if(used){if(write_all(fd,batch,used,end))goto done;end+=used;}
    }
    offsets[l->cells]=end;h.bytes=end;
    if(write_all(fd,&h,sizeof(h),0)||write_all(fd,offsets,((size_t)l->cells+1)*8,sizeof(h))||fsync(fd))goto done;
    pthread_rwlock_wrlock(&l->lock);
    if(l->poisoned||fstat(l->log_fd,&st)||h.device!=(uint64_t)st.st_dev||h.inode!=(uint64_t)st.st_ino){pthread_rwlock_unlock(&l->lock);goto done;}
    if(renameat(l->dir_fd,".live.pack.tmp",l->dir_fd,"live.pack")||fsync(l->dir_fd)){pthread_rwlock_unlock(&l->lock);goto done;}
    live_pack_clear(l);l->pack_fd=fd;fd=-1;l->pack_offsets=offsets;offsets=NULL;l->pack_end=h.end;
    pthread_rwlock_unlock(&l->lock);rc=0;
done:
    if(fd>=0)close(fd);
    if(source>=0)close(source);
    free(heads);free(offsets);free(r);pthread_mutex_unlock(&l->pack_lock);return rc;
}

typedef struct {float score;uint32_t id;uint64_t offset;} PackHit;
static void pack_offer(PackHit* h,int* n,int cap,PackHit x){
    if(*n<cap){int i=(*n)++;while(i&&h[(i-1)/2].score>x.score){h[i]=h[(i-1)/2];i=(i-1)/2;}h[i]=x;return;}
    if(x.score<=h[0].score)return;
    int i=0;while(2*i+1<cap){int j=2*i+1;if(j+1<cap&&h[j+1].score<h[j].score)j++;
        if(h[j].score>=x.score)break;
        h[i]=h[j];i=j;}h[i]=x;
}
static int pack_hit_cmp(const void* a,const void* b){
    float x=((const PackHit*)a)->score,y=((const PackHit*)b)->score;return (y>x)-(y<x);
}
int anchor_live_search_packed(const AnchorLive* l,const uint32_t* cells,int ncells,const float* q,const roaring_bitmap_t* allowed,
    uint32_t* ids,float* scores,int count,int capacity,uint64_t* entries,uint64_t* bytes,AnchorLiveScore score_code,void* user){
    if(!l||l->pack_fd<0)return live_search_tail(l,cells,ncells,q,allowed,ids,scores,count,capacity,entries,bytes,0);
    int hn=0,rc=-1,heapcap=l->copies*capacity;
    PackHit* heap=malloc((size_t)heapcap*sizeof(*heap));
    float* vector=malloc((size_t)l->dim*4);uint32_t* seen=malloc((size_t)capacity*4);
    uint8_t* buffer=malloc(PACK_ENTRY*4096);
    if(!heap||!vector||!buffer||!seen)goto done;
    for(int c=0;c<ncells;c++){
        uint32_t cell=cells[c];if(cell>=(uint32_t)l->cells)goto done;
        for(uint64_t off=l->pack_offsets[cell];off<l->pack_offsets[cell+1];){
            size_t size=l->pack_offsets[cell+1]-off;if(size>PACK_ENTRY*4096)size=PACK_ENTRY*4096;
            if(read_all(l->pack_fd,buffer,size,off))goto done;
            *bytes+=size;*entries+=size/PACK_ENTRY;
            for(size_t pos=0;pos<size;pos+=PACK_ENTRY){const uint8_t* p=buffer+pos;uint32_t id;uint64_t location;
                uint32_t checksum;memcpy(&checksum,p+80,4);if(XXH32(p,80,cell)!=checksum)goto done;
                memcpy(&id,p,4);memcpy(&location,p+4,8);
                if((uint64_t)id>=l->base_n+l->count||location<sizeof(LiveHeader)||location>=l->pack_end)goto done;
                Override* latest=override_find(l,id);
                if((latest&&latest->offset!=location)||anchor_live_is_deleted(l,id)||(allowed&&!roaring_bitmap_contains(allowed,id)))continue;
                float score=score_code(user,cell,p+12);if(!isfinite(score))goto done;
                pack_offer(heap,&hn,heapcap,(PackHit){score,id,location});
            }
            off+=size;
        }
    }
    qsort(heap,hn,sizeof(*heap),pack_hit_cmp);int ns=0;
    for(int h=0;h<hn&&ns<capacity;h++){
        int dup=0;for(int i=0;i<ns;i++)if(seen[i]==heap[h].id){dup=1;break;}if(dup)continue;
        seen[ns++]=heap[h].id;
        for(int i=0;i<count;i++)if(ids[i]==heap[h].id){dup=1;break;}
        if(dup)continue;
        if(read_all(l->log_fd,vector,(size_t)l->dim*4,heap[h].offset+sizeof(LiveRecord)))goto done;
        *bytes+=(uint64_t)l->dim*4;float score=0;for(int d=0;d<l->dim;d++)score+=q[d]*vector[d];
        if(count<capacity||score>scores[count-1]){int i=count<capacity?count++:capacity-1;
            while(i>0&&score>scores[i-1]){scores[i]=scores[i-1];ids[i]=ids[i-1];i--;}
            scores[i]=score;ids[i]=heap[h].id;}
    }
    rc=live_search_tail(l,cells,ncells,q,allowed,ids,scores,count,capacity,entries,bytes,l->pack_end);
done:
    free(heap);free(vector);free(seen);free(buffer);return rc;
}
