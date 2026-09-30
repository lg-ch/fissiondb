/* Keep the existing checkpoint format byte-compatible. Zero tails exist only
 * in this bounded IO buffer, never in the resident representative array. */
static int fission_centers_io_count(AnchorLive*l,int fd,uint64_t off,XXH64_state_t*hash,int writing,uint32_t count){
    LiveFission*f=l->fission;int native=f->ops.center_dim,stored=l->dim;
    if(native==stored){
        size_t bytes=(size_t)count*stored;
        if(writing?write_all(fd,f->centers,bytes,off):read_all(fd,f->centers,bytes,off))return -1;
        XXH64_update(hash,f->centers,bytes);return 0;
    }
    int8_t buffer[65536];uint32_t step=sizeof(buffer)/(unsigned)stored;
    for(uint32_t first=0;first<count;first+=step){
        uint32_t n=count-first<step?count-first:step;size_t bytes=(size_t)n*stored;
        if(writing){
            memset(buffer,0,bytes);
            for(uint32_t i=0;i<n;i++)memcpy(buffer+(size_t)i*stored,f->centers+(size_t)(first+i)*native,native);
            if(write_all(fd,buffer,bytes,off+(uint64_t)first*stored))return -1;
        }else{
            if(read_all(fd,buffer,bytes,off+(uint64_t)first*stored))return -1;
            for(uint32_t i=0;i<n;i++){
                for(int d=native;d<stored;d++)if(buffer[(size_t)i*stored+d])return -1;
                memcpy(f->centers+(size_t)(first+i)*native,buffer+(size_t)i*stored,native);
            }
        }
        XXH64_update(hash,buffer,bytes);
    }
    return 0;
}
static int fission_centers_io(AnchorLive*l,int fd,uint64_t off,XXH64_state_t*hash,int writing){
    return fission_centers_io_count(l,fd,off,hash,writing,l->fission->K);
}
int anchor_live_fission_center_info(AnchorLive*l,uint64_t v[3]){
    if(!l||!l->fission||!v||anchor_live_read_lock(l))return -1;
    LiveFission*f=l->fission;
    v[0]=f->ops.center_dim;v[1]=(uint64_t)f->K*f->ops.center_dim;
    v[2]=(uint64_t)f->cell_capacity*f->ops.center_dim;
    anchor_live_read_unlock(l);return 0;
}
