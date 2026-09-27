/* Convert only the derived code arena. A hard link retains the old inode until
 * the new checkpoint is durable. After interruption, the checkpoint's validated
 * inode selects the old or new arena; no journal replay/repartition is needed
 * merely to complete/roll back a layout conversion. */
#define FISSION_SLOTS_TEMP ".fission.codes.slots.tmp"
#define FISSION_SLOTS_OLD ".fission.codes.before-contiguous"

static int fission_state_header(AnchorLive*l,FissionHeader*h){
    int fd=openat(l->dir_fd,"fission.state",O_RDONLY|O_NOFOLLOW);if(fd<0)return -1;
    struct stat st;int rc=-1;XXH64_state_t*hash=NULL;
    if(fstat(fd,&st)||st.st_size<(off_t)(sizeof(*h)+8)||read_all(fd,h,sizeof(*h),0)||
       h->magic!=FISSION_MAGIC||(h->version!=1&&h->version!=2))goto done;
    hash=XXH64_createState();if(!hash)goto done;XXH64_reset(hash,0);
    uint8_t buffer[65536];uint64_t end=(uint64_t)st.st_size-8;
    for(uint64_t off=0;off<end;){size_t n=end-off<sizeof(buffer)?(size_t)(end-off):sizeof(buffer);
        if(read_all(fd,buffer,n,off))goto done;
        XXH64_update(hash,buffer,n);off+=n;}
    uint64_t saved;if(read_all(fd,&saved,8,end)||saved!=XXH64_digest(hash))goto done;
    rc=0;
done:if(hash)XXH64_freeState(hash);close(fd);return rc;
}
static int fission_migration_recover(AnchorLive*l){
    struct stat old,current;
    if(fstatat(l->dir_fd,FISSION_SLOTS_OLD,&old,AT_SYMLINK_NOFOLLOW)){
        if(errno!=ENOENT)return -1;
        if(unlinkat(l->dir_fd,FISSION_SLOTS_TEMP,0)&&errno!=ENOENT)return -1;
        return 0;
    }
    FissionHeader h;if(!S_ISREG(old.st_mode)||fission_state_header(l,&h))return -1;
    int present=!fstatat(l->dir_fd,"fission.codes",&current,AT_SYMLINK_NOFOLLOW);
    if(present&&S_ISREG(current.st_mode)&&h.code_device==(uint64_t)current.st_dev&&h.code_inode==(uint64_t)current.st_ino){
        if(unlinkat(l->dir_fd,FISSION_SLOTS_OLD,0))return -1;
    }else if(h.code_device==(uint64_t)old.st_dev&&h.code_inode==(uint64_t)old.st_ino){
        if(renameat(l->dir_fd,FISSION_SLOTS_OLD,l->dir_fd,"fission.codes"))return -1;
    }else return -1;
    if(unlinkat(l->dir_fd,FISSION_SLOTS_TEMP,0)&&errno!=ENOENT)return -1;
    return fsync(l->dir_fd);
}
static int fission_slots_migrate(AnchorLive*l){
    LiveFission*f=l->fission;LiveFission staged={0};AnchorLive temporary={0};temporary.fission=&staged;
    staged.fd=-1;fission_runs_reset(&staged);
    FissionCell*cells=calloc(f->K?f->K:1,sizeof(*cells));if(!cells)return -1;
    int rc=-1,renamed=0;
    staged.fd=openat(l->dir_fd,FISSION_SLOTS_TEMP,O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    if(staged.fd<0)goto done;
    for(uint32_t k=0;k<f->K;k++){
        cells[k]=(FissionCell){FISSION_NONE,FISSION_NONE,0};
        if(!f->cells[k].count)continue;
        if(fission_slot_reserve(&temporary,f->cells[k].count,cells+k)||
           fission_slot_copy(f,f->cells+k,staged.fd,cells[k].head))goto done;
        fission_slot_count(&staged,cells+k,f->cells[k].count);
    }
    if(fsync(staged.fd)||linkat(l->dir_fd,"fission.codes",l->dir_fd,FISSION_SLOTS_OLD,0)||fsync(l->dir_fd))goto done;
    if(fission_test_barrier(5))goto done;
    if(renameat(l->dir_fd,FISSION_SLOTS_TEMP,l->dir_fd,"fission.codes"))goto done;
    renamed=1;if(fsync(l->dir_fd))goto done;
    if(fission_test_barrier(6))goto done;
    if(f->direct_fd>=0)close(f->direct_fd);
    close(f->fd);f->fd=staged.fd;staged.fd=-1;
    f->direct_fd=openat(l->dir_fd,"fission.codes",O_RDONLY|O_DIRECT|O_NOFOLLOW);
    free(f->chunks);f->chunks=staged.chunks;staged.chunks=NULL;f->J=staged.J;f->chunk_capacity=staged.chunk_capacity;
    if(f->K)memcpy(f->cells,cells,(size_t)f->K*sizeof(*cells));
    f->layout_version=2;fission_runs_reset(f);
    /* Preserve the saved prefix: the journal tail is replayed AFTER conversion. */
    if(fission_checkpoint_at(l,f->checkpoint_end))goto done;
    if(fission_test_barrier(7))goto done;
    if(unlinkat(l->dir_fd,FISSION_SLOTS_OLD,0)||fsync(l->dir_fd))goto done;
    rc=0;
done:
    if(staged.fd>=0)close(staged.fd);
    free(staged.chunks);free(cells);
    if(!renamed)unlinkat(l->dir_fd,FISSION_SLOTS_TEMP,0);
    return rc;
}
