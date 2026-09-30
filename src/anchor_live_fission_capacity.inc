/* A threshold change preserves routing and encoded bytes until normal splits
 * publish replacements. Serialize with writers and the current split job. */
uint32_t anchor_live_fission_capacity(AnchorLive*l){
    if(!l||!l->fission||anchor_live_read_lock(l))return 0;
    uint32_t cap=l->fission->cap;anchor_live_read_unlock(l);return cap;
}
int anchor_live_fission_set_capacity(AnchorLive*l,uint32_t cap){
    if(!l||!l->fission||cap<64||cap>65536)return -1;
    pthread_mutex_lock(&l->writer_mutex);live_fission_maintenance_lock(l);
    pthread_rwlock_wrlock(&l->lock);
    LiveFission*f=l->fission;int rc=-1,fd=-1;
    if(l->poisoned)goto done;
    if(cap==f->cap){rc=0;goto done;}
    if(f->K<f->maximum)
        for(uint32_t k=0;k<f->K;k++)if(f->cells[k].count>cap+512){rc=-2;goto done;}
    uint64_t h[5]={FISSION_MAGIC,1,cap,f->maximum,0};h[4]=XXH64(h,32,0);
    fd=openat(l->dir_fd,".fission.config.tmp",O_WRONLY|O_CREAT|O_TRUNC|O_NOFOLLOW,0600);
    if(fd<0||write_all(fd,h,sizeof(h),0)||fsync(fd))goto done;
    close(fd);fd=-1;
    if(fission_test_barrier(8))goto done;
    if(renameat(l->dir_fd,".fission.config.tmp",l->dir_fd,"fission.config"))goto done;
    if(fsync(l->dir_fd)||fission_test_barrier(9)){l->poisoned=1;fission_signal(f);goto done;}
    f->cap=cap;f->pressure=0;f->queue_head=f->queue_tail=FISSION_NONE;f->pending=0;
    memset(f->queued,0,f->cell_capacity);
    for(uint32_t k=0;k<f->K;k++){
        if(f->cells[k].count>cap+256)f->pressure++;
        fission_enqueue(f,k);
    }
    fission_signal(f);rc=0;
done:
    if(fd>=0)close(fd);
    pthread_rwlock_unlock(&l->lock);live_fission_maintenance_unlock(l);
    pthread_mutex_unlock(&l->writer_mutex);return rc;
}
