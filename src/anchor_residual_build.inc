/* Bounded, resumable conversion. Publication is the fsynced manifest rename.
 * Existing published output is immutable; writers serialize on a stable lock. */
#include <sys/file.h>
static int residual_write_all(int fd,const void* data,size_t bytes) {
    const uint8_t* p=data;
    while(bytes){ssize_t n=write(fd,p,bytes);if(n<0&&errno==EINTR)continue;if(n<=0)return -1;p+=n;bytes-=n;}
    return 0;
}
static int residual_publish_text(int dirfd,const char* temporary,const char* target,const char* text) {
    int fd=openat(dirfd,temporary,O_WRONLY|O_CREAT|O_TRUNC|O_NOFOLLOW,0600);if(fd<0)return -1;
    int rc=residual_write_all(fd,text,strlen(text));if(!rc&&fsync(fd))rc=-1;
    if(close(fd))rc=-1;
    if(!rc&&renameat(dirfd,temporary,dirfd,target))rc=-1;
    if(!rc&&fsync(dirfd))rc=-1;
    return rc;
}
int anchor_index_build_residual(AnchorIndex* idx,const char* output) {
    if(!idx||!output||!idx->a8_mode||idx->meta.dim>1024||idx->bfd<0||idx->basefd<0||idx->residual_fd>=0||
       idx->live||__atomic_load_n(&idx->contexts,__ATOMIC_SEQ_CST))return -1;
    int dim=idx->meta.dim,input_dim=idx->meta.input_dim,entry=residual_source_entry(idx);
    int legacy=dim==1024&&input_dim==1024&&idx->meta.cdim==1024&&idx->meta.tqbits==1&&idx->meta.M==2;
    const char* checkpoint_magic=legacy?"MGR512CHECK1":"MGRRESCHECK2";
    int rc=-1,dirfd=-1,lock=-1,fd=-1,ring_ok=0;struct io_uring ring;
    uint8_t *old=NULL,*codes=NULL;uint16_t* raw=NULL;
    uint64_t off[1024],len[1024];uint8_t* dst[1024];char text[512];
    struct stat st;uint32_t header[2];
    if(fstat(idx->basefd,&st)||pread(idx->basefd,header,8,0)!=8||header[0]<(uint64_t)idx->meta.n||header[1]!=(uint32_t)input_dim||
       st.st_size<0||(uint64_t)st.st_size!=8+(uint64_t)header[0]*input_dim*2)return -1;
    uint64_t fp=residual_fingerprint(idx);
    uint64_t identity[]={st.st_dev,st.st_ino,st.st_size,st.st_mtim.tv_sec,st.st_mtim.tv_nsec,st.st_ctim.tv_sec,st.st_ctim.tv_nsec};
    uint64_t source=XXH64(identity,sizeof(identity),fp);
    dirfd=open(output,O_RDONLY|O_DIRECTORY);if(dirfd<0)goto done;
    lock=openat(dirfd,"residual.lock",O_RDWR|O_CREAT|O_NOFOLLOW,0600);
    if(lock<0||flock(lock,LOCK_EX|LOCK_NB))goto done;
    if(fstatat(dirfd,"residual.meta",&st,AT_SYMLINK_NOFOLLOW)==0||errno!=ENOENT)goto done;
    int first=0,cpfd=openat(dirfd,"residual.checkpoint",O_RDONLY|O_NOFOLLOW);
    if(cpfd>=0){FILE* f=fdopen(cpfd,"r");if(!f){close(cpfd);goto done;}
        char magic[32];unsigned long long saved;int valid=fscanf(f,"%31s %llu %d",magic,&saved,&first)==3;
        int ch;do{ch=fgetc(f);}while(ch==' '||ch=='\n'||ch=='\r'||ch=='\t');
        valid=valid&&ch==EOF&&!ferror(f);fclose(f);
        if(!valid||strcmp(magic,checkpoint_magic)||saved!=source||first<0||first>idx->meta.K)goto done;
    }else {
        if(errno!=ENOENT)goto done;
        /* An unrecognized existing payload must never be truncated. */
        if(fstatat(dirfd,"res512.bin",&st,AT_SYMLINK_NOFOLLOW)==0||errno!=ENOENT)goto done;
        snprintf(text,sizeof(text),"%s %llu 0\n",checkpoint_magic,(unsigned long long)source);
        if(residual_publish_text(dirfd,"residual.checkpoint.tmp","residual.checkpoint",text))goto done;
    }
    fd=openat(dirfd,"res512.bin",O_RDWR|O_CREAT|O_NOFOLLOW,0600);if(fd<0)goto done;
    uint64_t committed=idx->offs[first]/entry*72;
    if(fstat(fd,&st)||st.st_size<0||(uint64_t)st.st_size<committed||ftruncate(fd,committed)||lseek(fd,committed,SEEK_SET)<0)goto done;
    size_t maxn=0;for(int k=0;k<idx->meta.K;k++){size_t n=(idx->offs[k+1]-idx->offs[k])/entry;if(n>maxn)maxn=n;}
    old=malloc(maxn*entry+1);codes=malloc(maxn*72+1);raw=malloc((size_t)1024*input_dim*2);
    if(!old||!codes||!raw||io_uring_queue_init(1024,&ring,0))goto done;
    ring_ok=1;
    for(int cell=first;cell<idx->meta.K;cell++){
        size_t n=(idx->offs[cell+1]-idx->offs[cell])/entry;
        if(pread(idx->bfd,old,n*entry,idx->offs[cell])!=(ssize_t)(n*entry))goto done;
        float a[1024],ar[1024],v[1024];
        if(residual_anchor(idx,cell,a,ar))goto done;
        for(size_t begin=0;begin<n;begin+=1024){int count=n-begin<1024?(int)(n-begin):1024;
            for(int j=0;j<count;j++){uint32_t id;memcpy(&id,old+(begin+j)*entry,4);if((uint64_t)id>=(uint64_t)idx->meta.n)goto done;
                off[j]=8+(uint64_t)id*input_dim*2;len[j]=(uint64_t)input_dim*2;dst[j]=(uint8_t*)(raw+(size_t)j*input_dim);}
            if(anchor_reads(&ring,idx->basefd,off,len,dst,count))goto done;
            for(int j=0;j<count;j++){
                row_f16_padded(raw+(size_t)j*input_dim,v,input_dim,dim);
                uint8_t* p=codes+(begin+j)*72;memcpy(p,old+(begin+j)*entry,4);
                if(residual_encode_vector(idx,v,a,ar,p+4))goto done;
            }
        }
        if(residual_write_all(fd,codes,n*72))goto done;
        if((cell+1)%1000==0||cell+1==idx->meta.K){
            if(fsync(fd))goto done;
            snprintf(text,sizeof(text),"%s %llu %d\n",checkpoint_magic,(unsigned long long)source,cell+1);
            if(residual_publish_text(dirfd,"residual.checkpoint.tmp","residual.checkpoint",text))goto done;
            fprintf(stderr,"residual512 %d/%d\n",cell+1,idx->meta.K);
        }
    }
    if(fsync(fd))goto done;
    if(fstat(idx->basefd,&st))goto done;
    uint64_t final_identity[]={st.st_dev,st.st_ino,st.st_size,st.st_mtim.tv_sec,st.st_mtim.tv_nsec,st.st_ctim.tv_sec,st.st_ctim.tv_nsec};
    if(XXH64(final_identity,sizeof(final_identity),fp)!=source)goto done;
    if(legacy)snprintf(text,sizeof(text),"MGR512V1 %llu %d 1024 %llu %llu\n",(unsigned long long)idx->meta.n,idx->meta.K,(unsigned long long)idx->meta.seed,(unsigned long long)fp);
    else snprintf(text,sizeof(text),"MGRRESV2 %llu %d %d %llu %llu %d %d %d\n",(unsigned long long)idx->meta.n,idx->meta.K,dim,(unsigned long long)idx->meta.seed,(unsigned long long)fp,input_dim,residual_bits_for(dim),residual_dim_for(dim));
    if(residual_publish_text(dirfd,"residual.meta.tmp","residual.meta",text))goto done;
    rc=0;
done:
    if(ring_ok)io_uring_queue_exit(&ring);
    free(old);free(codes);free(raw);
    if(fd>=0)close(fd);
    if(lock>=0)close(lock);
    if(dirfd>=0)close(dirfd);
    return rc;
}
