/* Residual records retain a fixed 72-byte stride: ID, beta/gain f16 and
 * at most 64 bytes of rotated residual. V1 (1024d) remains byte-compatible.
 * V2 uses 4 bits through padded128, 2 through256 and 1 through1024. */
static int residual_bits_for(int dim){return dim<=128?4:dim<=256?2:1;}
static int residual_dim_for(int dim){return dim<512?dim:512;}
static uint64_t residual_fingerprint(const AnchorIndex* idx){
    return XXH64(idx->A8,(size_t)idx->meta.K*idx->meta.dim,idx->meta.seed)^XXH64(idx->offs,((size_t)idx->meta.K+1)*8,17);
}
static int residual_source_entry(const AnchorIndex* idx){return 4+idx->meta.cdim*idx->meta.tqbits/8;}
int anchor_index_enable_residual(AnchorIndex* idx,const char* dir) {
    if(!idx||!dir||idx->residual_fd>=0||!idx->a8_mode||idx->meta.dim>1024||
       __atomic_load_n(&idx->contexts,__ATOMIC_SEQ_CST))return -1;
    char path[2048],magic[32];unsigned long long n,seed,fingerprint;
    int K,dim,input_dim=0,bits=0,cdim=0;
    if(snprintf(path,sizeof(path),"%s/residual.meta",dir)>=(int)sizeof(path))return -1;
    FILE* f=fopen(path,"r");if(!f)return -1;
    int valid=fscanf(f,"%31s %llu %d %d %llu %llu",magic,&n,&K,&dim,&seed,&fingerprint)==6;
    if(valid&&!strcmp(magic,"MGRRESV2"))valid=fscanf(f,"%d %d %d",&input_dim,&bits,&cdim)==3;
    else if(valid&&!strcmp(magic,"MGR512V1")){
        input_dim=1024;bits=1;cdim=512;
        valid=idx->meta.dim==1024&&idx->meta.cdim==1024&&idx->meta.tqbits==1&&idx->meta.M==2;
    }else valid=0;
    int ch;do{ch=fgetc(f);}while(ch==' '||ch=='\n'||ch=='\r'||ch=='\t');
    valid=valid&&ch==EOF&&!ferror(f);fclose(f);
    if(!valid||n!=(uint64_t)idx->meta.n||K!=idx->meta.K||dim!=idx->meta.dim||input_dim!=idx->meta.input_dim||
       bits!=residual_bits_for(dim)||cdim!=residual_dim_for(dim)||seed!=idx->meta.seed||fingerprint!=residual_fingerprint(idx))return -1;
    if(snprintf(path,sizeof(path),"%s/res512.bin",dir)>=(int)sizeof(path))return -1;
    int fd=open(path,O_RDONLY);struct stat st;if(fd<0)return -1;
    uint64_t bytes=idx->offs[K]/residual_source_entry(idx)*72;
    if(fstat(fd,&st)||st.st_size<0||(uint64_t)st.st_size!=bytes){close(fd);return -1;}
    idx->residual_fd=fd;idx->residual_bytes=bytes;idx->residual_bits=bits;idx->residual_cdim=cdim;
    /* Optional: unsupported filesystems retain the buffered reader. Explicit
       query configuration can require direct IO, reporting absence as error. */
    idx->residual_direct_fd=open(path,O_RDONLY|O_DIRECT);
    return 0;
}

static void residual_heap(ScId* h,int* n,int cap,ScId x) {
    if(*n<cap){int i=(*n)++;while(i&&h[(i-1)/2].s>x.s){h[i]=h[(i-1)/2];i=(i-1)/2;}h[i]=x;return;}
    if(x.s<=h[0].s)return;
    int i=0;while(2*i+1<cap){int j=2*i+1;if(j+1<cap&&h[j+1].s<h[j].s)j++;
        if(h[j].s>=x.s)break;
        h[i]=h[j];i=j;}h[i]=x;
}

static int residual_anchor(const AnchorIndex* idx,uint32_t cell,float* a,float* ar){
    int dim=idx->meta.dim;float an=0;
    for(int d=0;d<dim;d++){a[d]=idx->A8[(size_t)cell*dim+d];an+=a[d]*a[d];}
    if(an<=0)return -1;
    float inv=1/sqrtf(an);for(int d=0;d<dim;d++)a[d]*=inv;
    rot_seeded(a,ar,idx->sgn,dim);return 0;
}
static int residual_encode_vector(const AnchorIndex* idx,const float* v,const float* a,const float* ar,uint8_t* code){
    int dim=idx->meta.dim,bits=residual_bits_for(dim),cd=residual_dim_for(dim),levels=(1<<bits)-1;
    float vr[1024],beta=dotf(v,a,dim),n2=0,denom=0,mx=0;
    rot_seeded(v,vr,idx->sgn,dim);memset(code,0,68);
    for(int d=0;d<dim;d++){vr[d]-=beta*ar[d];n2+=vr[d]*vr[d];if(d<cd)mx=fmaxf(mx,fabsf(vr[d]));}
    for(int d=0;d<cd;d++){
        int u=bits==1?(vr[d]>=0):(int)lrintf((vr[d]/fmaxf(mx,1e-30f)+1)*levels*.5f);
        if(u<0)u=0;
        if(u>levels)u=levels;
        denom+=vr[d]*(2*u-levels);
        /* TQ4 kernel expects signed two's-complement nibbles. TQ2
           expects unsigned u, translated by two inside the kernel. */
        int stored=bits==4?(u-8)&15:u;
        code[4+d*bits/8]|=(uint8_t)(stored<<((d*bits)%8));
    }
    float gain=n2/fmaxf(denom,1e-20f);
    if(!isfinite(beta)||!isfinite(gain)||gain>65504)return -1;
    _Float16 b=(_Float16)beta,g=(_Float16)gain;memcpy(code,&b,2);memcpy(code+2,&g,2);return 0;
}
static int residual_encode_live(void* user,uint32_t cell,const float* v,uint8_t* code) {
    const AnchorIndex* idx=user;float a[1024],ar[1024];
    if(residual_anchor(idx,cell,a,ar))return -1;
    return residual_encode_vector(idx,v,a,ar,code);
}
static int residual_center_dim(const AnchorIndex*idx){
#if defined(FISSIONDB_TESTING) && defined(FISSIONDB_TEST_PADDED_CENTERS)
    return idx->meta.dim; /* reference layout for exact-equivalence tests */
#else
    return idx->meta.input_dim;
#endif
}
static int residual_encode_fission(void*user,const float*v,const int8_t*center,uint8_t*code){
    const AnchorIndex*idx=user;float a[1024],ar[1024];double norm=0;
    int native=residual_center_dim(idx);
    for(int d=0;d<native;d++)norm+=(int)center[d]*(int)center[d];
    if(norm<=0)return -1;
    float scale=1/sqrt(norm);for(int d=0;d<native;d++)a[d]=center[d]*scale;
    for(int d=native;d<idx->meta.dim;d++)a[d]=0;
    rot_seeded(a,ar,idx->sgn,idx->meta.dim);return residual_encode_vector(idx,v,a,ar,code);
}
static void residual_sketch_fission(void*user,const float*v,float*out){const AnchorIndex*idx=user;rot_seeded(v,out,idx->sgn,idx->meta.dim);}
int anchor_index_enable_fission(AnchorIndex*idx,uint32_t cap,uint32_t maximum){
    if(!idx||!idx->live||idx->residual_fd<0||__atomic_load_n(&idx->contexts,__ATOMIC_SEQ_CST))return -1;
    if(anchor_live_has_fission(idx->live)){uint32_t saved_cap=0,saved_max=0;return anchor_live_fission_config(idx->live,&saved_cap,&saved_max)==1&&cap==saved_cap&&maximum==saved_max?0:-1;}
    AnchorFissionOps ops={doti8,residual_encode_fission,residual_sketch_fission,residual_center_dim(idx)};
    return anchor_live_enable_fission(idx->live,cap,maximum,&ops,idx);
}
int anchor_index_fission_stats(AnchorIndex*idx,uint64_t values[8],double timings[3]){return idx?anchor_live_fission_stats(idx->live,values,timings):-1;}
int anchor_index_fission_flush(AnchorIndex*idx){return idx?anchor_live_fission_flush(idx->live):-1;}
int anchor_index_fission_set_capacity(AnchorIndex*idx,uint32_t cap){return idx?anchor_live_fission_set_capacity(idx->live,cap):-1;}
uint32_t anchor_index_fission_capacity(AnchorIndex*idx){return idx?anchor_live_fission_capacity(idx->live):0;}
int anchor_index_fission_center_info(AnchorIndex*idx,uint64_t values[3]){return idx?anchor_live_fission_center_info(idx->live,values):-1;}
int anchor_index_fission_progress(AnchorIndex*idx,uint64_t values[8],double timings[3]){return idx?anchor_live_fission_progress(idx->live,values,timings):-1;}
int anchor_index_pack_live(AnchorIndex* idx) {
    if(!idx||idx->residual_fd<0||!idx->live)return -1;
    if(anchor_live_has_fission(idx->live))return anchor_live_fission_checkpoint(idx->live);
    return anchor_live_pack(idx->live,residual_encode_live,idx);
}
uint64_t anchor_index_unpacked_bytes(AnchorIndex* idx){return idx?anchor_live_unpacked_bytes(idx->live):UINT64_MAX;}
typedef struct {AnchorQuery* query;uint32_t cell;float qa,qs;int qsum;} ResidualLiveScore;
static float residual_score_live(void* user,uint32_t cell,const uint8_t* code) {
    ResidualLiveScore* state=user;AnchorQuery* c=state->query;const AnchorIndex* idx=c->index;
    int dim=idx->meta.dim,cd=idx->residual_cdim,bits=idx->residual_bits,streams=8/bits,stride=cd/streams;
    if(state->cell!=cell){
        float ar[1024],weights[512],mx=0;
        if(residual_anchor(idx,cell,c->vf,ar))return NAN;
        state->qa=dotf(c->qn,c->vf,dim);
        for(int d=0;d<cd;d++){weights[d]=c->qr[d]-state->qa*ar[d];mx=fmaxf(mx,fabsf(weights[d]));}
        state->qs=mx/127;state->qsum=0;
        for(int d=0;d<cd;d++){int8_t v=(int8_t)lrintf(weights[d]/(mx+1e-9f)*127);
            state->qsum+=v;c->c8[(d%streams)*stride+d/streams]=v;}
        state->cell=cell;
    }
    uint16_t b,g;memcpy(&b,code,2);memcpy(&g,code+2,2);
    int dot;
    if(bits==1)dot=2*score_tq1(code+4,c->c8,stride,cd)-state->qsum;
    else if(bits==2)dot=2*score_tq2(code+4,c->c8,c->c8+stride,c->c8+2*stride,c->c8+3*stride,cd)+state->qsum;
    else dot=2*score_tq4(code+4,c->c8,c->c8+stride,cd)+state->qsum;
    return dot*state->qs*h2f(g)+h2f(b)*state->qa;
}
typedef struct {ResidualLiveScore score;const int8_t* center;} ResidualFissionScore;
static float residual_score_fission(void*user,const int8_t*center,const uint8_t*code){
    ResidualFissionScore*s=user;ResidualLiveScore*state=&s->score;AnchorQuery*c=state->query;const AnchorIndex*idx=c->index;
    if(s->center!=center){
        int dim=idx->meta.dim,cd=idx->residual_cdim,streams=8/idx->residual_bits,stride=cd/streams;
        float ar[1024],weights[512],mx=0;double norm=0;
        int native=residual_center_dim(idx);
        for(int d=0;d<native;d++)norm+=(int)center[d]*(int)center[d];
        if(norm<=0)return NAN;
        float scale=1/sqrt(norm);for(int d=0;d<native;d++)c->vf[d]=center[d]*scale;
        for(int d=native;d<dim;d++)c->vf[d]=0;
        rot_seeded(c->vf,ar,idx->sgn,dim);state->qa=dotf(c->qn,c->vf,dim);
        for(int d=0;d<cd;d++){weights[d]=c->qr[d]-state->qa*ar[d];mx=fmaxf(mx,fabsf(weights[d]));}
        state->qs=mx/127;state->qsum=0;
        for(int d=0;d<cd;d++){int8_t v=(int8_t)lrintf(weights[d]/(mx+1e-9f)*127);state->qsum+=v;c->c8[(d%streams)*stride+d/streams]=v;}
        s->center=center;state->cell=0;
    }
    return residual_score_live(state,0,code);
}

#include "anchor_residual_io.inc"

static int anchor_residual_search(AnchorQuery* c,const float* q,int top_k,uint32_t* ids,float* scores,AnchorStats* st) {
    const AnchorIndex* idx=c->index;int dim=idx->meta.dim,input_dim=idx->meta.input_dim;double start=now_ms();float n2=0,mx=0;
    for(int d=0;d<dim;d++)n2+=q[d]*q[d];
    if(!isfinite(n2)||n2<=0)return -1;
    for(int d=0;d<dim;d++){c->qn[d]=q[d]/sqrtf(n2);mx=fmaxf(mx,fabsf(c->qn[d]));}
    for(int d=0;d<dim;d++)c->qn8[d]=(int8_t)lrintf(c->qn[d]*127/(mx+1e-9f));
    int kept=0;
    for(int k=0;k<idx->meta.K;k++)residual_heap(c->cand,&kept,c->nprobe,(ScId){(float)doti8(c->qn8,idx->A8+(size_t)k*dim,dim),(uint32_t)k});
    qsort(c->cand,kept,sizeof(ScId),scid_cmp);
    int probes=anchor_adapt_probes(c);
    rot_seeded(c->qn,c->qr,idx->sgn,dim);
    ResidualLiveScore frozen_state={c,UINT32_MAX,0,0,0};
    st->anchor_ms=now_ms()-start;int hn=0;
    int rc=residual_read_score(c,probes,&frozen_state,&hn,st);
    if(rc<0)return rc;
    probes=c->policy_used;
    double t=now_ms();qsort(c->heap,hn,sizeof(ScId),scid_cmp);int nf=0;
    for(int i=0;i<hn&&nf<c->rerank;i++){int dup=0;for(int j=0;j<nf;j++)if(c->fin_all[j].id==c->heap[i].id){dup=1;break;}
        if(!dup){c->fin_all[nf++]=c->heap[i];anchor_trace_mark(c,c->heap[i].id,1);}}
    for(int i=0;i<nf;i++){c->read_off[i]=8+(uint64_t)c->fin_all[i].id*(input_dim*2);c->read_len[i]=(input_dim*2);c->read_dst[i]=c->rows_all+(size_t)i*(input_dim*2);}
    if(anchor_reads(&c->ring,idx->basefd,c->read_off,c->read_len,c->read_dst,nf))return -1;
    for(int i=0;i<nf;i++){row_f16_padded((uint16_t*)c->read_dst[i],c->vf,input_dim,dim);c->fin_all[i].s=dotf(c->qn,c->vf,dim);}
    qsort(c->fin_all,nf,sizeof(ScId),scid_cmp);
    if(idx->live){
        for(int i=0;i<nf;i++){c->merge_ids[i]=c->fin_all[i].id;c->merge_scores[i]=c->fin_all[i].s;}
        uint32_t* cells=(uint32_t*)c->read_off;for(int i=0;i<probes;i++)cells[i]=c->cand[i].id;
        ResidualLiveScore state={c,UINT32_MAX,0,0,0};
        if(anchor_live_has_fission(idx->live)){
            ResidualFissionScore dynamic={{c,0,0,0,0},NULL};
            c->live_io=(AnchorLiveIO){.ring=&c->ring,.ring_ok=&c->ring_ok,
                .buffer={c->res_buffer[0],c->res_buffer[1]},.capacity=c->res_capacity,
                .width=c->res_width,.overlap=c->res_overlap,.direct=c->res_live_direct};
            nf=anchor_live_search_fission(idx->live,c->nprobe,c->qn,c->allowed,c->merge_ids,c->merge_scores,nf,c->rerank,&st->entries,&st->bytes,residual_score_fission,&dynamic,&c->live_io);
        }else nf=anchor_live_search_packed(idx->live,cells,probes,c->qn,c->allowed,c->merge_ids,c->merge_scores,nf,c->rerank,&st->entries,&st->bytes,residual_score_live,&state);
        if(nf<0)return -1;
        for(int i=0;i<nf;i++)c->fin_all[i]=(ScId){c->merge_scores[i],c->merge_ids[i]};
    }
    int count=nf<top_k?nf:top_k;
    for(int i=0;i<top_k;i++){ids[i]=i<count?c->fin_all[i].id:UINT32_MAX;if(scores)scores[i]=i<count?c->fin_all[i].s:-INFINITY;}
    st->rerank_ms=now_ms()-t;st->total_ms=now_ms()-start;return count;
}
