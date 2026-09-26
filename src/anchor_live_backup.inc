/* A committed journal prefix is a complete logical live snapshot. Copy from a
 * pinned inode so append and compaction can continue after the snapshot point. */
int anchor_live_snapshot(AnchorLive* l,const char* destination) {
    if(!l||!destination)return -1;
    int rc=-1,source=-1,dir=-1,fd=-1;uint8_t* buffer=NULL;uint64_t end,config[5]={0};LiveHeader header;
    if(anchor_live_read_lock(l))return -1;
    source=dup(l->log_fd);end=l->end;header=l->header;
    if(l->fission){config[0]=FISSION_MAGIC;config[1]=1;config[2]=l->fission->cap;config[3]=l->fission->maximum;config[4]=XXH64(config,32,0);}
    anchor_live_read_unlock(l);if(source<0)goto done;
    dir=open(destination,O_RDONLY|O_DIRECTORY);if(dir<0)goto done;
    struct stat st;
    if(fstatat(dir,"live.log",&st,AT_SYMLINK_NOFOLLOW)==0||errno!=ENOENT)goto done;
    fd=openat(dir,".snapshot.tmp",O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(fd<0)goto done;
    buffer=malloc(1048576);if(!buffer||write_all(fd,&header,sizeof(header),0))goto done;
    for(uint64_t off=sizeof(header);off<end;){size_t size=end-off<1048576?(size_t)(end-off):1048576;
        if(read_all(source,buffer,size,off)||write_all(fd,buffer,size,off))goto done;
        off+=size;}
    if(config[0]){
        int cfg=openat(dir,"fission.config",O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(cfg<0)goto done;
        int bad=write_all(cfg,config,sizeof(config),0)||fsync(cfg);close(cfg);if(bad)goto done;
    }
    if(fsync(fd)||renameat(dir,".snapshot.tmp",dir,"live.log")||fsync(dir))goto done;
    rc=0;
done:
    if(source>=0)close(source);
    if(fd>=0)close(fd);
    if(dir>=0)close(dir);
    free(buffer);return rc;
}
