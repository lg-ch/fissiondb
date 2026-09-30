/* Standalone exactness and guard-page checks for runtime-dispatched kernels. */
#define _GNU_SOURCE
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "../src/anchor_x86.inc"

#if defined(__x86_64__) && defined(__GNUC__)
static void guard_check(int dim,int wide){
    size_t page=(size_t)sysconf(_SC_PAGESIZE);
    int stride=(dim+7)/8;
    size_t lengths[4]={(size_t)dim,(size_t)dim,(size_t)stride,(size_t)stride*8};
    uint8_t*maps[4],*p[4];
    for(int k=0;k<4;k++){
        assert(lengths[k]<=page);
        maps[k]=mmap(NULL,2*page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        assert(maps[k]!=MAP_FAILED&&mprotect(maps[k]+page,page,PROT_NONE)==0);
        p[k]=maps[k]+page-lengths[k];
        memset(p[k],k==2?255:128,lengths[k]);
    }
    assert(anchor_doti8_avx2((const int8_t*)p[0],(const int8_t*)p[1],dim)==dim*16384);
    assert(anchor_score_tq1_avx2(p[2],(const int8_t*)p[3],stride,dim)==dim*-128);
    if(wide){
        assert(anchor_doti8_avx512((const int8_t*)p[0],(const int8_t*)p[1],dim)==dim*16384);
        assert(anchor_score_tq1_avx512(p[2],(const int8_t*)p[3],stride,dim)==dim*-128);
    }
    for(int k=0;k<4;k++)assert(munmap(maps[k],2*page)==0);
}
#endif
int main(void){
#if defined(__x86_64__) && defined(__GNUC__)
    unsigned cases=0;uint32_t seed=761;
    int wide=!!(__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw"));
    int8_t abuf[2112],bbuf[2112],wbuf[4096];uint8_t cbuf[320];
    if(__builtin_cpu_supports("avx2")){
        for(int pass=0;pass<20;pass++){
            int8_t*a=abuf+pass,*b=bbuf+pass,*weights=wbuf+pass;uint8_t*code=cbuf+pass;
            for(int i=0;i<2048;i++){
                seed=seed*1664525u+1013904223u;a[i]=(int8_t)(seed>>24);
                seed=seed*1664525u+1013904223u;b[i]=(int8_t)(seed>>24);
                if(pass>=16){a[i]=pass&1?-128:127;b[i]=pass&2?-128:127;}
            }
            for(int i=0;i<8*260;i++)weights[i]=b[i%2048];
            memcpy(code,a,256);
            for(int dim=1;dim<=2048;dim++){
                int32_t expected=0;for(int d=0;d<dim;d++)expected+=(int32_t)a[d]*b[d];
                assert(anchor_doti8_avx2(a,b,dim)==expected);cases++;
                if(wide){assert(anchor_doti8_avx512(a,b,dim)==expected);cases++;}
                int stride=(dim+7)/8+pass%5;expected=0;
                for(int d=0;d<dim;d++)if((code[d/8]>>(d%8))&1)expected+=weights[(d%8)*stride+d/8];
                assert(anchor_score_tq1_avx2(code,weights,stride,dim)==expected);cases++;
                if(wide){assert(anchor_score_tq1_avx512(code,weights,stride,dim)==expected);cases++;}
            }
        }
        const int dims[]={1,7,8,31,32,33,63,64,65,127,128,129,255,256,257,511,512,513,767,768,769,1000,1023,1024,1025,2048};
        for(unsigned i=0;i<sizeof(dims)/sizeof(*dims);i++)guard_check(dims[i],wide);
    }else{puts("AVX2 unavailable: not verified");return 77;}
    if(__builtin_cpu_supports("f16c")){
        for(unsigned h=0;h<65536;h++){
            unsigned sign=h>>15,e=(h>>10)&31,m=h&1023;float expected;
            if(!e)expected=(sign?-1.f:1.f)*(float)m*5.9604645e-8f;
            else if(e==31)expected=sign?-65504.f:65504.f;
            else{uint32_t raw=(sign<<31)|((e+112)<<23)|(m<<13);memcpy(&expected,&raw,4);}
            float got=anchor_h2f_f16c(h);assert(!memcmp(&got,&expected,4));cases++;
        }
    }else{puts("F16C unavailable: not verified");return 77;}
    printf("%u exact kernel comparisons passed; avx512=%d selected_avx512=%d; guard pages passed\n",cases,wide,anchor_x86_use_avx512);return 0;
#else
    puts("Requires x86-64 GCC/Clang");return 77;
#endif
}
