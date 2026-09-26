#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "anchor_live.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xxhash.h>
#include <time.h>
#include <liburing.h>

#define LIVE_MAGIC UINT64_C(0x314C564152474E4D)
#define RECORD_MAGIC 0x31524c41u
#define TAG_BYTES_MAX 65536u
#define TAG_COUNT_MAX 256
#define TAG_BUCKETS 4096
#define REQUEST_BUCKETS 65536
typedef struct Request {
    struct Request* next;
    uint8_t token[32], digest[32];
    uint32_t id;
} Request;
typedef struct Override {
    struct Override* next;
    uint64_t offset, pending_offset;
    uint32_t id;
} Override;

typedef struct Tag {
    struct Tag* next;
    char* key;
    roaring_bitmap_t* ids;
} Tag;
typedef struct {
    uint64_t magic, version, fingerprint, base_n, dim, cells, copies, reserved;
} LiveHeader;
typedef struct {
    uint32_t magic, bytes, kind, id;
    uint32_t cells[4];
    uint64_t previous[4];
    uint32_t key_bytes, nkeys;
    uint64_t checksum;
} LiveRecord;
_Static_assert(sizeof(LiveHeader)==64 && sizeof(LiveRecord)==80, "journal layout");
struct AnchorLive {
    int log_fd, rows_fd, dir_fd, lock_fd, dim, cells, copies, poisoned;
    uint64_t base_n, count, end, maintenance_bytes, request_count;
    LiveHeader header;
    uint64_t* heads;
    Tag* tags[TAG_BUCKETS];
    Request* requests[REQUEST_BUCKETS];
    roaring_bitmap_t* deleted;
    Override* overrides[4096];
    uint64_t override_count;
    int pack_fd;
    uint64_t pack_end;
    uint64_t* pack_offsets;
    pthread_mutex_t pack_lock,writer_mutex;
    pthread_rwlock_t lock;
    struct LiveFission* fission;
};
static void live_pack_clear(AnchorLive*);
static void live_pack_open(AnchorLive*);
static int live_fission_observe(AnchorLive*,const LiveRecord*,uint64_t);
static void live_fission_close(AnchorLive*);
static int live_fission_rebuild(AnchorLive*);
static uint64_t live_fission_end(const AnchorLive*);
static void live_vector_write_lock(AnchorLive*);
static void live_fission_maintenance_lock(AnchorLive*);
static void live_fission_maintenance_unlock(AnchorLive*);
static int live_append_fission_batch(AnchorLive*,int,const float*,const uint32_t*,const char* const*,const int*,const uint8_t* const*,const uint8_t* const*,uint32_t*,int*);
static int write_all(int fd, const void* data, size_t n, uint64_t off) {
    const char* p=data;
    while(n) {
        ssize_t w=pwrite(fd,p,n,(off_t)off);
        if(w<0 && errno==EINTR)continue;
        if(w<=0)return -1;
        p+=w;n-=w;off+=(uint64_t)w;
    }
    return 0;
}
static int read_all(int fd, void* data, size_t n, uint64_t off) {
    char* p=data;
    while(n) {
        ssize_t r=pread(fd,p,n,(off_t)off);
        if(r<0 && errno==EINTR)continue;
        if(r<=0)return -1;
        p+=r;n-=r;off+=(uint64_t)r;
    }
    return 0;
}
static Tag* tag_find(const AnchorLive* l,const char* key) {
    unsigned b=(unsigned)(XXH64(key,strlen(key),0)%TAG_BUCKETS);
    for(Tag* t=l->tags[b];t;t=t->next)if(!strcmp(t->key,key))return t;
    return NULL;
}
static Tag* tag_get(AnchorLive* l,const char* key) {
    Tag* t=tag_find(l,key);if(t)return t;
    t=calloc(1,sizeof(*t));if(!t)return NULL;
    t->key=strdup(key);t->ids=roaring_bitmap_create();
    if(!t->key||!t->ids){free(t->key);if(t->ids)roaring_bitmap_free(t->ids);free(t);return NULL;}
    unsigned b=(unsigned)(XXH64(key,strlen(key),0)%TAG_BUCKETS);
    t->next=l->tags[b];l->tags[b]=t;return t;
}
static int key_valid(const char* key) {
    if(!key)return 0;
    size_t n=strnlen(key,256);if(!n||n>=256)return 0;
    for(size_t i=0;i<n;i++)if((unsigned char)key[i]<32 || key[i]==127)return 0;
    return 1;
}
static int payload_keys(const LiveRecord* r,const char* payload,const char** keys) {
    size_t pos=0;
    if(r->nkeys>TAG_COUNT_MAX || r->key_bytes>TAG_BYTES_MAX)return -1;
    for(uint32_t i=0;i<r->nkeys;i++) {
        if(pos>=r->key_bytes)return -1;
        const char* end=memchr(payload+pos,0,r->key_bytes-pos);
        if(!end || end-(payload+pos)>=256)return -1;
        keys[i]=payload+pos;if(!key_valid(keys[i]))return -1;
        pos=(size_t)(end-payload)+1;
    }
    return pos==r->key_bytes?0:-1;
}
static int replace_tags(AnchorLive* l,uint32_t id,const char* const* keys,int n,int replacing) {
    /* Allocate missing dictionary entries before changing membership. */
    for(int i=0;i<n;i++)if(!tag_get(l,keys[i]))return -1;
    if(replacing)for(int b=0;b<TAG_BUCKETS;b++)for(Tag* t=l->tags[b];t;t=t->next)
        roaring_bitmap_remove(t->ids,id);
    for(int i=0;i<n;i++)roaring_bitmap_add(tag_find(l,keys[i])->ids,id);
    if(replacing)for(int b=0;b<TAG_BUCKETS;b++) {
        Tag** link=&l->tags[b];
        while(*link) {
            Tag* tag=*link;
            if(roaring_bitmap_is_empty(tag->ids)) {
                *link=tag->next;roaring_bitmap_free(tag->ids);free(tag->key);free(tag);
            } else link=&tag->next;
        }
    }
    return 0;
}
static Request* request_find(const AnchorLive* l,const uint8_t* token) {
    unsigned b=XXH64(token,32,0)%REQUEST_BUCKETS;
    for(Request* r=l->requests[b];r;r=r->next)if(!memcmp(r->token,token,32))return r;
    return NULL;
}
static int request_add(AnchorLive* l,uint32_t id,const uint8_t* token,const uint8_t* digest) {
    Request* r=request_find(l,token);
    if(r)return r->id==id&&!memcmp(r->digest,digest,32)?0:-1;
    r=malloc(sizeof(*r));if(!r)return -1;
    memcpy(r->token,token,32);memcpy(r->digest,digest,32);r->id=id;
    unsigned b=XXH64(token,32,0)%REQUEST_BUCKETS;
    r->next=l->requests[b];l->requests[b]=r;l->request_count++;return 0;
}
static Override* override_find(const AnchorLive* l,uint32_t id) {
    if(!l)return NULL;
    for(Override* v=l->overrides[id%4096];v;v=v->next)if(v->id==id)return v;
    return NULL;
}
int anchor_live_is_overridden(const AnchorLive* l,uint32_t id){return override_find(l,id)!=NULL;}
int anchor_live_override_vector(const AnchorLive* l,uint32_t id,float* dst) {
    Override* v=override_find(l,id);if(!v)return 0;
    return read_all(l->log_fd,dst,(size_t)l->dim*4,v->offset+sizeof(LiveRecord))?-1:1;
}
int anchor_live_is_deleted(const AnchorLive* l,uint32_t id) {
    return l&&roaring_bitmap_contains(l->deleted,id);
}
void anchor_live_exclude_deleted(const AnchorLive* l,roaring_bitmap_t* ids) {
    if(l&&ids)roaring_bitmap_andnot_inplace(ids,l->deleted);
}
int anchor_live_lookup(AnchorLive* l,const uint8_t* token,const uint8_t* digest,uint32_t* id) {
    if(!l||!token||!digest||!id||anchor_live_read_lock(l))return -1;
    Request* r=request_find(l,token);
    int rc=0;
    if(r){rc=memcmp(r->digest,digest,32)?-3:1;if(rc==1)*id=r->id;}
    anchor_live_read_unlock(l);return rc;
}
static int apply_record_impl(AnchorLive* l,LiveRecord* r,uint64_t offset,int prepared) {
    const char* keys[TAG_COUNT_MAX];
    if(r->kind==8) {
        /* Allocation tombstone: retain ID continuity without deleted vectors. */
        if(r->bytes!=sizeof(*r)||r->id!=l->base_n+l->count||r->id==UINT32_MAX)return -1;
        roaring_bitmap_add(l->deleted,r->id);l->count++;return 0;
    }
    if(r->kind==4) {
        l->maintenance_bytes+=r->bytes;
        unsigned n=r->cells[0];
        if(!n||n>8192||r->bytes!=sizeof(*r)+(size_t)n*4)return -1;
        for(unsigned i=0;i<n;i++) {
            uint32_t id;memcpy(&id,(char*)(r+1)+(size_t)i*4,4);
            if((uint64_t)id>=l->base_n+l->count)return -1;
            roaring_bitmap_add(l->deleted,id);
        }
        return 0;
    }
    if(r->kind==5) {
        if(r->bytes!=sizeof(*r)+64||r->id<l->base_n||r->id>=l->base_n+l->count)return -1;
        return request_add(l,r->id,(uint8_t*)(r+1),(uint8_t*)(r+1)+32);
    }
    if(r->kind==3) {
        /* Version 2 checkpoint: one key followed by a bounded uint32 ID batch. */
        uint32_t n=r->cells[0];
        if(r->nkeys!=1 || !n || n>8192 || r->key_bytes>256 ||
           r->bytes!=sizeof(*r)+r->key_bytes+(size_t)n*4 ||
           payload_keys(r,(const char*)(r+1),keys))return -1;
        const char* ids=(const char*)(r+1)+r->key_bytes;
        for(uint32_t i=0;i<n;i++) {
            uint32_t id;memcpy(&id,ids+(size_t)i*4,4);
            if((uint64_t)id>=l->base_n+l->count)return -1;
        }
        Tag* tag=tag_get(l,keys[0]);if(!tag)return -1;
        for(uint32_t i=0;i<n;i++) {
            uint32_t id;memcpy(&id,ids+(size_t)i*4,4);roaring_bitmap_add(tag->ids,id);
        }
        return 0;
    }
    int is_vector=r->kind==1||r->kind==6||r->kind==7;
    size_t vector_bytes=is_vector?(size_t)l->dim*4:0, extra=r->kind==6?64:0;
    if((!is_vector && r->kind!=2) || r->bytes!=sizeof(*r)+vector_bytes+extra+r->key_bytes ||
        payload_keys(r,(char*)(r+1)+vector_bytes+extra,keys))return -1;
    if(is_vector) {
        if(r->id==UINT32_MAX)return -1;
        if(r->kind==7){if(r->id>=l->base_n+l->count||anchor_live_is_deleted(l,r->id))return -1;}
        else if(r->id!=l->base_n+l->count)return -1;
        const float* vector=(const float*)(r+1);
        double norm=0;
        for(int d=0;d<l->dim;d++){if(!isfinite(vector[d]))return -1;norm+=(double)vector[d]*vector[d];}
        if(norm<0.999 || norm>1.001)return -1;
        for(int i=0;i<l->copies;i++) {
            if(r->cells[i]>=(uint32_t)l->cells || r->previous[i]!=l->heads[r->cells[i]])return -1;
            for(int j=0;j<i;j++)if(r->cells[i]==r->cells[j])return -1;
        }
        if(r->kind==7){
            Override* u=override_find(l,r->id);
            if(!u){u=calloc(1,sizeof(*u));if(!u)return -1;u->id=r->id;u->next=l->overrides[r->id%4096];l->overrides[r->id%4096]=u;l->override_count++;}
            u->offset=offset;
        }else if(!prepared&&write_all(l->rows_fd,vector,vector_bytes,l->count*vector_bytes))return -1;
        for(int i=0;i<l->copies;i++)l->heads[r->cells[i]]=offset;
        if(r->kind!=7)l->count++;
    } else if(r->id>=l->base_n+l->count)return -1;
    if(extra&&request_add(l,r->id,(uint8_t*)(r+1)+vector_bytes,(uint8_t*)(r+1)+vector_bytes+32))return -1;
    int rc=replace_tags(l,r->id,keys,(int)r->nkeys,r->kind==2||r->kind==7);
    if(!rc&&(r->kind==2||r->kind==7))l->maintenance_bytes+=r->bytes;
    if(!rc&&is_vector&&l->fission&&!prepared)rc=live_fission_observe(l,r,offset);
    return rc;
}
static int apply_record(AnchorLive*l,LiveRecord*r,uint64_t offset){return apply_record_impl(l,r,offset,0);}
void anchor_live_close(AnchorLive* l) {
    if(!l)return;
    live_fission_close(l);
    live_pack_clear(l);pthread_mutex_destroy(&l->pack_lock);
    if(l->rows_fd>=0)close(l->rows_fd);
    if(l->log_fd>=0)close(l->log_fd);
    if(l->lock_fd>=0)close(l->lock_fd);
    if(l->dir_fd>=0)close(l->dir_fd);
    for(int b=0;b<TAG_BUCKETS;b++)for(Tag* t=l->tags[b];t;) {
        Tag* next=t->next;roaring_bitmap_free(t->ids);free(t->key);free(t);t=next;
    }
    for(int b=0;b<REQUEST_BUCKETS;b++)for(Request* r=l->requests[b];r;) {
        Request* next=r->next;free(r);r=next;
    }
    for(int b=0;b<4096;b++)for(Override* u=l->overrides[b];u;){Override* next=u->next;free(u);u=next;}
    if(l->deleted)roaring_bitmap_free(l->deleted);
    free(l->heads);pthread_rwlock_destroy(&l->lock);pthread_mutex_destroy(&l->writer_mutex);free(l);
}
AnchorLive* anchor_live_open(const char* dir,uint64_t fingerprint,uint64_t base_n,int dim,int cells,int copies) {
    if(!dir||dim<8||cells<1||copies<1||copies>4||copies>cells||base_n>=UINT32_MAX)return NULL;
    if(mkdir(dir,0755) && errno!=EEXIST)return NULL;
    AnchorLive* l=calloc(1,sizeof(*l));if(!l)return NULL;
    l->log_fd=l->rows_fd=l->dir_fd=l->lock_fd=l->pack_fd=-1;pthread_rwlock_init(&l->lock,NULL);pthread_mutex_init(&l->pack_lock,NULL);pthread_mutex_init(&l->writer_mutex,NULL);
    l->base_n=base_n;l->dim=dim;l->cells=cells;l->copies=copies;
    l->heads=calloc((size_t)cells,8);l->deleted=roaring_bitmap_create();
    if(!l->heads||!l->deleted)goto fail;
    l->dir_fd=open(dir,O_RDONLY|O_DIRECTORY);if(l->dir_fd<0)goto fail;
    l->lock_fd=openat(l->dir_fd,"live.lock",O_RDWR|O_CREAT,0644);
    if(l->lock_fd<0||flock(l->lock_fd,LOCK_EX|LOCK_NB))goto fail;
    char path[4096];
    if(snprintf(path,sizeof(path),"%s/live.log",dir)>=(int)sizeof(path))goto fail;
    l->log_fd=open(path,O_RDWR|O_CREAT,0644);
    if(l->log_fd<0||flock(l->log_fd,LOCK_EX|LOCK_NB))goto fail;
    LiveHeader expected={LIVE_MAGIC,1,fingerprint,base_n,(uint64_t)dim,(uint64_t)cells,(uint64_t)copies,0}, header;
    struct stat st;if(fstat(l->log_fd,&st))goto fail;
    if(st.st_size==0) {
        if(write_all(l->log_fd,&expected,sizeof(expected),0)||fsync(l->log_fd))goto fail;
        st.st_size=sizeof(header);
    }
    if(read_all(l->log_fd,&header,sizeof(header),0))goto fail;
    if(header.version<1||header.version>4)goto fail;
    expected.version=header.version;
    if(memcmp(&header,&expected,sizeof(header)))goto fail;
    l->header=header;
    snprintf(path,sizeof(path),"%s/live.rows",dir);
    l->rows_fd=open(path,O_RDWR|O_CREAT|O_TRUNC,0644);if(l->rows_fd<0)goto fail;
    size_t cap=sizeof(LiveRecord)+(size_t)dim*4+TAG_BYTES_MAX+64;
    LiveRecord* record=malloc(cap);if(!record)goto fail;
    uint64_t off=sizeof(header), end=(uint64_t)st.st_size;
    int bad=0;
    while(off<end) {
        if(end-off<sizeof(*record))break; /* uncommitted torn tail */
        if(read_all(l->log_fd,record,sizeof(*record),off)){bad=1;break;}
        if(((record->kind==7||record->kind==8)&&header.version<4)||(record->kind==3&&header.version<2)||record->magic!=RECORD_MAGIC||record->bytes<sizeof(*record)||record->bytes>cap){bad=1;break;}
        if(record->bytes>end-off)break;
        if(read_all(l->log_fd,(char*)(record+1),record->bytes-sizeof(*record),off+sizeof(*record))){bad=1;break;}
        uint64_t checksum=record->checksum;record->checksum=0;
        if(XXH64(record,record->bytes,0)!=checksum||apply_record(l,record,off)){bad=1;break;}
        off+=record->bytes;
    }
    free(record);
    if(bad)goto fail; /* Complete corrupt records are never silently discarded. */
    if(ftruncate(l->log_fd,(off_t)off)||fsync(l->log_fd)||fsync(l->rows_fd))goto fail;
    l->end=off;live_pack_open(l);
    int dfd=open(dir,O_RDONLY|O_DIRECTORY);if(dfd<0)goto fail;
    int rc=fsync(dfd);close(dfd);if(rc)goto fail;
    snprintf(path,sizeof(path),"%s/..",dir);dfd=open(path,O_RDONLY|O_DIRECTORY);
    if(dfd<0)goto fail;
    rc=fsync(dfd);close(dfd);if(rc)goto fail;
    return l;
fail:
    anchor_live_close(l);return NULL;
}
int anchor_live_read_lock(AnchorLive* l) {
    if(!l)return 0;
    pthread_rwlock_rdlock(&l->lock);
    if(l->poisoned){pthread_rwlock_unlock(&l->lock);return -1;}return 0;
}
void anchor_live_read_unlock(AnchorLive* l){if(l)pthread_rwlock_unlock(&l->lock);}
uint64_t anchor_live_count(const AnchorLive* l){return l?l->count:0;}
int anchor_live_rows_fd(const AnchorLive* l){return l?l->rows_fd:-1;}
static int commit_record(AnchorLive* l,const float* vector,const uint32_t* cells,uint32_t id,const char* const* keys,int nkeys,uint32_t* output,const uint8_t* token,const uint8_t* digest,int update,int locked,int durable) {
    if(!l||nkeys<0||nkeys>TAG_COUNT_MAX||(nkeys&&!keys))return -1;
    size_t key_bytes=0;
    for(int i=0;i<nkeys;i++){if(!key_valid(keys[i]))return -1;key_bytes+=strlen(keys[i])+1;}
    if(key_bytes>TAG_BYTES_MAX)return -1;
    size_t vector_bytes=vector?(size_t)l->dim*4:0, bytes=sizeof(LiveRecord)+vector_bytes+key_bytes+(token?64:0);
    LiveRecord* r=calloc(1,bytes);if(!r)return -1;
    r->magic=RECORD_MAGIC;r->bytes=(uint32_t)bytes;r->kind=vector?(update?7:(token?6:1)):2;r->key_bytes=(uint32_t)key_bytes;r->nkeys=(uint32_t)nkeys;
    if(vector)memcpy(r+1,vector,vector_bytes);
    if(token){memcpy((char*)(r+1)+vector_bytes,token,32);memcpy((char*)(r+1)+vector_bytes+32,digest,32);}
    char* dest=(char*)(r+1)+vector_bytes+(token?64:0);
    for(int i=0;i<nkeys;i++){size_t n=strlen(keys[i])+1;memcpy(dest,keys[i],n);dest+=n;}
    if(!locked){pthread_mutex_lock(&l->writer_mutex);if(vector)live_vector_write_lock(l);else pthread_rwlock_wrlock(&l->lock);}
    int rc=-1;
    if(l->poisoned)goto done;
    if(token) {
        Request* found=request_find(l,token);
        if(found){rc=memcmp(found->digest,digest,32)?-3:0;if(!rc)*output=found->id;goto done;}
    }
    if(vector) {
        if(!cells||(!update&&l->base_n+l->count>=UINT32_MAX))goto done;
        if(update&&(id>=l->base_n+l->count||anchor_live_is_deleted(l,id)))goto done;
        r->id=update?id:(uint32_t)(l->base_n+l->count);
        for(int i=0;i<l->copies;i++) {
            if(cells[i]>=(uint32_t)l->cells)goto done;
            for(int j=0;j<i;j++)if(cells[i]==cells[j])goto done;
            r->cells[i]=cells[i];r->previous[i]=l->heads[cells[i]];
        }
    } else {if(id>=l->base_n+l->count||anchor_live_is_deleted(l,id))goto done;r->id=id;}
    if(update&&l->header.version<4){
        l->header.version=4;
        if(write_all(l->log_fd,&l->header,sizeof(l->header),0)||fsync(l->log_fd)){l->poisoned=1;goto done;}
    }
    r->checksum=XXH64(r,bytes,0);
    /* The journal is authoritative. Any ambiguous failure poisons this handle
       until reopen; recovery reconstructs rows and metadata from valid records. */
    if(write_all(l->log_fd,r,bytes,l->end)||(durable&&fsync(l->log_fd))||apply_record(l,r,l->end)||(durable&&fsync(l->rows_fd))) {
        l->poisoned=1;goto done;
    }
    l->end+=bytes;if(output)*output=r->id;rc=0;
done:
    if(!locked){pthread_rwlock_unlock(&l->lock);pthread_mutex_unlock(&l->writer_mutex);}
    free(r);return rc;
}
int anchor_live_append(AnchorLive* l,const float* vector,const uint32_t* cells,const char* const* keys,int nkeys,uint32_t* id) {
    if(!vector||!id)return -1;
    int committed=0;return anchor_live_append_batch(l,1,vector,cells,keys,&nkeys,NULL,NULL,id,&committed);
}
/* Bounded posting import, visible under one write lock. A crash can retain
 * a prefix of key records; idempotent set union makes a full retry safe. */
int anchor_live_add_tag(AnchorLive* l,const uint32_t* ids,int n,const char* const* keys,int nkeys) {
    if(!l||!ids||n<1||n>8192||!keys||nkeys<1||nkeys>TAG_COUNT_MAX)return -1;
    size_t total=0;
    for(int k=0;k<nkeys;k++){
        if(!key_valid(keys[k]))return -1;
        total+=sizeof(LiveRecord)+strlen(keys[k])+1+(size_t)n*4;
    }
    uint8_t* data=calloc(1,total);if(!data)return -1;
    size_t pos=0;
    /* Records need not be aligned after variable-length keys. */
    for(int k=0;k<nkeys;k++){
        LiveRecord header={0};size_t kb=strlen(keys[k])+1,bytes=sizeof(header)+kb+(size_t)n*4;
        header.magic=RECORD_MAGIC;header.bytes=bytes;header.kind=3;header.cells[0]=n;
        header.nkeys=1;header.key_bytes=kb;memcpy(data+pos,&header,sizeof(header));
        memcpy(data+pos+sizeof(header),keys[k],kb);
        memcpy(data+pos+sizeof(header)+kb,ids,(size_t)n*4);
        header.checksum=XXH64(data+pos,bytes,0);memcpy(data+pos,&header,sizeof(header));pos+=bytes;
    }
    LiveRecord* record=malloc(sizeof(LiveRecord)+256+(size_t)n*4);
    if(!record){free(data);return -1;}
    pthread_mutex_lock(&l->writer_mutex);
    pthread_rwlock_wrlock(&l->lock);int rc=-1;
    if(l->poisoned)goto done;
    for(int i=0;i<n;i++)if(ids[i]>=l->base_n+l->count||anchor_live_is_deleted(l,ids[i]))goto done;
    if(l->header.version<2){
        l->header.version=2;
        if(write_all(l->log_fd,&l->header,sizeof(l->header),0)||fsync(l->log_fd)){
            l->poisoned=1;goto done;
        }
    }
    if(write_all(l->log_fd,data,total,l->end)||fsync(l->log_fd)){l->poisoned=1;goto done;}
    pos=0;
    while(pos<total){
        memcpy(record,data+pos,sizeof(*record));memcpy(record,data+pos,record->bytes);
        if(apply_record(l,record,l->end+pos)){l->poisoned=1;goto done;}
        pos+=record->bytes;
    }
    l->end+=total;rc=0;
done:
    pthread_rwlock_unlock(&l->lock);pthread_mutex_unlock(&l->writer_mutex);free(record);free(data);return rc;
}
int anchor_live_set_tags(AnchorLive* l,uint32_t id,const char* const* keys,int nkeys) {
    return commit_record(l,NULL,NULL,id,keys,nkeys,NULL,NULL,NULL,0,0,1);
}
int anchor_live_append_once(AnchorLive* l,const float* v,const uint32_t* cells,
    const char* const* keys,int nkeys,const uint8_t* token,const uint8_t* digest,uint32_t* id) {
    if(!v||!token||!digest||!id)return -1;
    const uint8_t*ts[]={token},*ds[]={digest};int committed=0;
    return anchor_live_append_batch(l,1,v,cells,keys,&nkeys,ts,ds,id,&committed);
}
int anchor_live_update(AnchorLive* l,uint32_t id,const float* vector,const uint32_t* cells,const char* const* keys,int nkeys) {
    if(!vector)return -1;
    return commit_record(l,vector,cells,id,keys,nkeys,NULL,NULL,NULL,1,0,1);
}
static int live_append_batch_locked(AnchorLive* l,int n,const float* vectors,const uint32_t* cells,const char* const* keys,const int* counts,const uint8_t* const* tokens,const uint8_t* const* digests,uint32_t* ids,int* committed) {
    int rc=-1,applied=0,total=0;
    if(l->poisoned)goto done;
    for(int i=0;i<n;i++){
        rc=commit_record(l,vectors+(size_t)i*l->dim,cells+(size_t)i*l->copies,0,keys?keys+total:NULL,counts[i],ids+i,tokens?tokens[i]:NULL,digests?digests[i]:NULL,0,1,0);
        if(rc)break;
        applied++;total+=counts[i];
    }
    if(l->poisoned)goto done;
    if(fsync(l->log_fd)||fsync(l->rows_fd)){l->poisoned=1;rc=-1;goto done;}
    *committed=applied;
done:
    return rc;
}
int anchor_live_append_batch(AnchorLive* l,int n,const float* vectors,const uint32_t* cells,const char* const* keys,const int* counts,const uint8_t* const* tokens,const uint8_t* const* digests,uint32_t* ids,int* committed) {
    if(!l||n<1||n>256||!vectors||!cells||!counts||!ids||!committed)return -1;
    *committed=0;int total=0;
    for(int i=0;i<n;i++){if(counts[i]<0||counts[i]>TAG_COUNT_MAX||(counts[i]&&!keys))return -1;
        if(tokens&&tokens[i]&&(!digests||!digests[i]))return -1;
        size_t bytes=0;for(int j=0;j<counts[i];j++){if(!key_valid(keys[total+j]))return -1;bytes+=strlen(keys[total+j])+1;}
        if(bytes>TAG_BYTES_MAX)return -1;
        total+=counts[i];}
    if(l->fission)return live_append_fission_batch(l,n,vectors,cells,keys,counts,tokens,digests,ids,committed);
    pthread_mutex_lock(&l->writer_mutex);live_vector_write_lock(l);
    int rc=live_append_batch_locked(l,n,vectors,cells,keys,counts,tokens,digests,ids,committed);
    pthread_rwlock_unlock(&l->lock);pthread_mutex_unlock(&l->writer_mutex);return rc;
}

int anchor_live_delete(AnchorLive* l,uint32_t id) {
    if(!l)return -1;
    pthread_mutex_lock(&l->writer_mutex);
    pthread_rwlock_wrlock(&l->lock);int rc=-1;
    if(l->poisoned||(uint64_t)id>=l->base_n+l->count)goto done;
    if(anchor_live_is_deleted(l,id)){rc=0;goto done;}
    _Alignas(LiveRecord) uint8_t buffer[sizeof(LiveRecord)+4];memset(buffer,0,sizeof(buffer));
    LiveRecord* r=(LiveRecord*)buffer;
    r->magic=RECORD_MAGIC;r->bytes=sizeof(buffer);r->kind=4;r->cells[0]=1;
    memcpy(r+1,&id,4);r->checksum=XXH64(buffer,sizeof(buffer),0);
    if(write_all(l->log_fd,buffer,sizeof(buffer),l->end)||fsync(l->log_fd)||apply_record(l,r,l->end)) {
        l->poisoned=1;goto done;
    }
    l->end+=sizeof(buffer);rc=0;
done:
    pthread_rwlock_unlock(&l->lock);pthread_mutex_unlock(&l->writer_mutex);return rc;
}
uint64_t anchor_live_journal_bytes(const AnchorLive* l){return l?l->end:0;}
uint64_t anchor_live_request_count(const AnchorLive* l){return l?l->request_count:0;}
uint64_t anchor_live_maintenance_bytes(const AnchorLive* l){return l?l->maintenance_bytes:0;}
uint64_t anchor_live_deleted_count(const AnchorLive* l){return l?roaring_bitmap_get_cardinality(l->deleted):0;}
/* A stable directory lock protects journal replacement. The write lock keeps
   all contexts on one generation. Scratch space is O(cells + one record). */
int anchor_live_compact(AnchorLive* l,uint64_t* before,uint64_t* after) {
    if(!l||!before||!after)return -1;
    pthread_mutex_lock(&l->writer_mutex);
    live_fission_maintenance_lock(l);
    pthread_rwlock_wrlock(&l->lock);
    int rc=-1,fd=-1,rows_new=-1,published=0;
    uint64_t* heads=NULL;LiveRecord* record=NULL;
    if(l->poisoned)goto done;
    *before=l->end;*after=l->end;
    uint64_t estimated=sizeof(LiveHeader)+l->count*(sizeof(LiveRecord)+(uint64_t)l->dim*4)
        +l->request_count*(sizeof(LiveRecord)+64)
        +l->override_count*(sizeof(LiveRecord)+(uint64_t)l->dim*4);
    uint64_t nd=roaring_bitmap_get_cardinality(l->deleted),deleted_live=0,deleted_overrides=0;
    roaring_uint32_iterator_t di;roaring_init_iterator(l->deleted,&di);
    while(di.has_value){if(di.current_value>=l->base_n)deleted_live++;roaring_advance_uint32_iterator(&di);}
    for(int b=0;b<4096;b++)for(Override* u=l->overrides[b];u;u=u->next)if(anchor_live_is_deleted(l,u->id))deleted_overrides++;
    estimated-=deleted_live*(uint64_t)l->dim*4+deleted_overrides*(sizeof(LiveRecord)+(uint64_t)l->dim*4);
    estimated+=nd*4+((nd+8191)/8192)*sizeof(LiveRecord);
    for(int b=0;b<TAG_BUCKETS;b++)for(Tag* tag=l->tags[b];tag;tag=tag->next) {
        uint64_t n=roaring_bitmap_get_cardinality(tag->ids);
        estimated+=n*4+((n+8191)/8192)*(sizeof(LiveRecord)+strlen(tag->key)+1);
    }
    if(estimated>=l->end){rc=0;goto done;}
    size_t cap=sizeof(LiveRecord)+(size_t)l->dim*4+TAG_BYTES_MAX+64;
    heads=calloc((size_t)l->cells,sizeof(*heads));record=malloc(cap);
    if(!heads||!record)goto done;
    /* A leftover temporary file is never authoritative. Only its exact name
       is removed, under the stable lock, before starting another attempt. */
    if(unlinkat(l->dir_fd,".live.compact",0)&&errno!=ENOENT)goto done;
    fd=openat(l->dir_fd,".live.compact",O_RDWR|O_CREAT|O_EXCL,0644);
    if(fd<0||flock(fd,LOCK_EX|LOCK_NB))goto done;
    if(unlinkat(l->dir_fd,".rows.compact",0)&&errno!=ENOENT)goto done;
    rows_new=openat(l->dir_fd,".rows.compact",O_RDWR|O_CREAT|O_EXCL,0644);
    if(rows_new<0)goto done;
    LiveHeader header;
    if(read_all(l->log_fd,&header,sizeof(header),0)||memcmp(&header,&l->header,sizeof(header)))goto done;
    header.version=(l->override_count||deleted_live)?4:3;
    if(write_all(fd,&header,sizeof(header),0))goto done;
    uint64_t off=sizeof(header),end=sizeof(header),count=0;
    while(off<l->end) {
        if(l->end-off<sizeof(*record)||read_all(l->log_fd,record,sizeof(*record),off)||
           record->magic!=RECORD_MAGIC||record->bytes<sizeof(*record)||
           record->bytes>cap||record->bytes>l->end-off)goto done;
        size_t old_bytes=record->bytes;
        if(read_all(l->log_fd,record+1,old_bytes-sizeof(*record),off+sizeof(*record)))goto done;
        uint64_t hash=record->checksum;record->checksum=0;
        if(XXH64(record,old_bytes,0)!=hash)goto done;
        if(record->kind==1||record->kind==6||record->kind==8) {
            if(record->id!=l->base_n+count || (record->kind!=8&&old_bytes<sizeof(*record)+(size_t)l->dim*4))goto done;
            if(anchor_live_is_deleted(l,record->id)){
                uint32_t id=record->id;memset(record,0,sizeof(*record));
                record->magic=RECORD_MAGIC;record->bytes=sizeof(*record);record->kind=8;record->id=id;
                record->checksum=XXH64(record,record->bytes,0);
                if(write_all(fd,record,record->bytes,end))goto done;
                end+=record->bytes;count++;off+=old_bytes;continue;
            }
            if(record->kind==8)goto done;
            for(int i=0;i<l->copies;i++) {
                if(record->cells[i]>=(uint32_t)l->cells)goto done;
                record->previous[i]=heads[record->cells[i]];
            }
            if(write_all(rows_new,record+1,(size_t)l->dim*4,count*(uint64_t)l->dim*4))goto done;
            record->kind=1;record->nkeys=record->key_bytes=0;
            record->bytes=sizeof(*record)+(size_t)l->dim*4;
            record->checksum=XXH64(record,record->bytes,0);
            if(write_all(fd,record,record->bytes,end))goto done;
            for(int i=0;i<l->copies;i++)heads[record->cells[i]]=end;
            end+=record->bytes;count++;
        } else if(record->kind!=2&&record->kind!=3&&record->kind!=4&&record->kind!=5&&record->kind!=7)goto done;
        off+=old_bytes;
    }
    if(count!=l->count)goto done;
    /* Persist only the newest replacement per stable ID, before tag/deletion
       checkpoints. Pending offsets become visible only after publication. */
    for(int b=0;b<4096;b++)for(Override* u=l->overrides[b];u;u=u->next){
        if(anchor_live_is_deleted(l,u->id))continue;
        if(read_all(l->log_fd,record,sizeof(*record),u->offset)||record->kind!=7||record->id!=u->id||record->bytes>cap||record->bytes<sizeof(*record)+(size_t)l->dim*4)goto done;
        if(read_all(l->log_fd,record+1,record->bytes-sizeof(*record),u->offset+sizeof(*record)))goto done;
        uint64_t hash=record->checksum;record->checksum=0;if(XXH64(record,record->bytes,0)!=hash)goto done;
        record->nkeys=record->key_bytes=0;record->bytes=sizeof(*record)+(size_t)l->dim*4;
        for(int i=0;i<l->copies;i++){if(record->cells[i]>=(uint32_t)l->cells)goto done;record->previous[i]=heads[record->cells[i]];}
        record->checksum=XXH64(record,record->bytes,0);
        if(write_all(fd,record,record->bytes,end))goto done;
        for(int i=0;i<l->copies;i++)heads[record->cells[i]]=end;
        u->pending_offset=end;end+=record->bytes;
    }
    /* Snapshot by key, not by document: no dense per-document map and no
       scan of every key for every document. Replay accepts subsequent updates. */
    for(int b=0;b<TAG_BUCKETS;b++)for(Tag* tag=l->tags[b];tag;tag=tag->next) {
        roaring_uint32_iterator_t it;roaring_init_iterator(tag->ids,&it);
        while(it.has_value) {
            memset(record,0,sizeof(*record));
            record->magic=RECORD_MAGIC;record->kind=3;record->nkeys=1;
            record->key_bytes=(uint32_t)strlen(tag->key)+1;
            memcpy(record+1,tag->key,record->key_bytes);
            char* dest=(char*)(record+1)+record->key_bytes;
            while(it.has_value&&record->cells[0]<8192) {
                memcpy(dest+(size_t)record->cells[0]*4,&it.current_value,4);
                record->cells[0]++;roaring_advance_uint32_iterator(&it);
            }
            record->bytes=sizeof(*record)+record->key_bytes+record->cells[0]*4;
            record->checksum=XXH64(record,record->bytes,0);
            if(write_all(fd,record,record->bytes,end))goto done;
            end+=record->bytes;
        }
    }
    roaring_uint32_iterator_t deleted;roaring_init_iterator(l->deleted,&deleted);
    while(deleted.has_value) {
        memset(record,0,sizeof(*record));record->magic=RECORD_MAGIC;record->kind=4;
        while(deleted.has_value&&record->cells[0]<8192) {
            memcpy((char*)(record+1)+(size_t)record->cells[0]*4,&deleted.current_value,4);
            record->cells[0]++;roaring_advance_uint32_iterator(&deleted);
        }
        record->bytes=sizeof(*record)+record->cells[0]*4;
        record->checksum=XXH64(record,record->bytes,0);
        if(write_all(fd,record,record->bytes,end))goto done;
        end+=record->bytes;
    }
    for(int b=0;b<REQUEST_BUCKETS;b++)for(Request* req=l->requests[b];req;req=req->next) {
        memset(record,0,sizeof(*record));record->magic=RECORD_MAGIC;record->kind=5;
        record->id=req->id;record->bytes=sizeof(*record)+64;
        memcpy(record+1,req->token,32);memcpy((char*)(record+1)+32,req->digest,32);
        record->checksum=XXH64(record,record->bytes,0);
        if(write_all(fd,record,record->bytes,end))goto done;
        end+=record->bytes;
    }
    if(end>=l->end){rc=0;goto done;} /* No history to reclaim. */
    if(ftruncate(rows_new,(off_t)(l->count*(uint64_t)l->dim*4))||fsync(rows_new)||fsync(fd))goto done;
    if(renameat(l->dir_fd,".live.compact",l->dir_fd,"live.log"))goto done;
    published=1;
    /* No allocations or fallible reads after publication. Readers cannot see
       new offsets with the old descriptor. A failed directory fsync is ambiguous. */
    live_pack_clear(l);
    int old_fd=l->log_fd;l->log_fd=fd;fd=-1;close(old_fd);
    for(int b=0;b<4096;b++){
        Override** link=&l->overrides[b];
        while(*link){Override* u=*link;
            if(anchor_live_is_deleted(l,u->id)){*link=u->next;free(u);l->override_count--;}
            else{u->offset=u->pending_offset;link=&u->next;}
        }
    }
    free(l->heads);l->heads=heads;heads=NULL;l->end=end;l->header=header;l->maintenance_bytes=0;
    if(renameat(l->dir_fd,".rows.compact",l->dir_fd,"live.rows")){l->poisoned=1;goto done;}
    int old_rows=l->rows_fd;l->rows_fd=rows_new;rows_new=-1;close(old_rows);
    if(fsync(l->dir_fd)){l->poisoned=1;goto done;}
    if(l->fission&&live_fission_rebuild(l)){l->poisoned=1;goto done;}
    *after=end;rc=0;
done:
    if(fd>=0)close(fd);
    if(!published&&l->dir_fd>=0)unlinkat(l->dir_fd,".live.compact",0);
    if(rows_new>=0)close(rows_new);
    if(l->dir_fd>=0)unlinkat(l->dir_fd,".rows.compact",0);
    free(heads);free(record);pthread_rwlock_unlock(&l->lock);live_fission_maintenance_unlock(l);pthread_mutex_unlock(&l->writer_mutex);return rc;
}

int anchor_live_keys(AnchorLive* l,char* dst,int cap) {
    if(!l||cap<0||(cap&&!dst)||anchor_live_read_lock(l))return -1;
    size_t used=0;
    for(int b=0;b<TAG_BUCKETS;b++)for(Tag* t=l->tags[b];t;t=t->next) {
        if(roaring_bitmap_is_empty(t->ids))continue;
        size_t n=strlen(t->key);
        if(used+n+1<(size_t)cap){memcpy(dst+used,t->key,n);dst[used+n]='\n';}
        used+=n+1;
    }
    if(cap)dst[used<(size_t)cap?used:(size_t)cap-1]=0;
    anchor_live_read_unlock(l);return used<INT_MAX?(int)used:-1;
}
roaring_bitmap_t* anchor_live_filter(const AnchorLive* l,const char* const* keys,const int* groups,int ngroups) {
    roaring_bitmap_t* result=NULL;int pos=0;
    for(int g=0;g<ngroups;g++) {
        roaring_bitmap_t* group=roaring_bitmap_create();if(!group)goto fail;
        for(int j=0;j<groups[g];j++) {
            Tag* t=l?tag_find(l,keys[pos]):NULL;pos++;
            if(t)roaring_bitmap_or_inplace(group,t->ids);
        }
        if(!result)result=group;
        else {roaring_bitmap_and_inplace(result,group);roaring_bitmap_free(group);}
    }
    return result;
fail:
    if(result)roaring_bitmap_free(result);
    return NULL;
}
static int live_search_tail(const AnchorLive* l,const uint32_t* cells,int ncells,const float* q,const roaring_bitmap_t* allowed,uint32_t* ids,float* scores,int count,int capacity,uint64_t* entries,uint64_t* bytes,uint64_t cutoff) {
    if(!l||(!l->count&&!l->override_count))return count;
    size_t cap=sizeof(LiveRecord)+(size_t)l->dim*4+TAG_BYTES_MAX+64;
    LiveRecord* r=malloc(cap);if(!r)return -1;
    for(int c=0;c<ncells;c++) {
        if(cells[c]>=(uint32_t)l->cells){free(r);return -1;}
        uint64_t off=l->heads[cells[c]];
        while(off&&off>=cutoff) {
            if(off<sizeof(LiveHeader)||off>=l->end||read_all(l->log_fd,r,sizeof(*r),off)||
               r->magic!=RECORD_MAGIC||(r->kind!=1&&r->kind!=6&&r->kind!=7)||r->bytes>cap||r->bytes<sizeof(*r)+(size_t)l->dim*4||r->bytes>l->end-off) {free(r);return -1;}
            if(read_all(l->log_fd,r+1,r->bytes-sizeof(*r),off+sizeof(*r))){free(r);return -1;}
            uint64_t hash=r->checksum;r->checksum=0;
            if(XXH64(r,r->bytes,0)!=hash){free(r);return -1;}
            (*entries)++;*bytes+=r->bytes;
            int slot=-1;for(int i=0;i<l->copies;i++)if(r->cells[i]==cells[c])slot=i;
            if(slot<0||r->previous[slot]>=off){free(r);return -1;}
            uint64_t next=r->previous[slot];
            Override* latest=override_find(l,r->id);
            if((!latest||latest->offset==off)&&!anchor_live_is_deleted(l,r->id)&&(!allowed||roaring_bitmap_contains(allowed,r->id))) {
                int dup=0;for(int i=0;i<count;i++)if(ids[i]==r->id){dup=1;break;}
                if(!dup) {
                    const float* v=(const float*)(r+1);float score=0;
                    for(int d=0;d<l->dim;d++)score+=q[d]*v[d];
                    if(count<capacity || score>scores[count-1]) {
                        int i=count<capacity?count++:capacity-1;
                        while(i>0&&score>scores[i-1]){scores[i]=scores[i-1];ids[i]=ids[i-1];i--;}
                        scores[i]=score;ids[i]=r->id;
                    }
                }
            }
            off=next;
        }
    }
    free(r);return count;
}

#include "anchor_live_pack.inc"
#include "anchor_live_fission.inc"
int anchor_live_search(const AnchorLive* l,const uint32_t* cells,int ncells,const float* q,const roaring_bitmap_t* allowed,uint32_t* ids,float* scores,int count,int capacity,uint64_t* entries,uint64_t* bytes) {
    return live_search_tail(l,cells,ncells,q,allowed,ids,scores,count,capacity,entries,bytes,0);
}

#include "anchor_live_backup.inc"
