/* Standalone exactness test; gcc -O2 -std=c11 tests/check_x86_kernels.c -o /tmp/check-x86 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../src/anchor_x86.inc"
int main(void){
#if defined(__x86_64__) && defined(__GNUC__)
    unsigned cases=0;uint32_t seed=761;
    int8_t a[2048],b[2048],weights[2048];uint8_t code[256];
    if(__builtin_cpu_supports("avx2")){
        for(int pass=0;pass<16;pass++){
            for(int i=0;i<2048;i++){
                seed=seed*1664525u+1013904223u;a[i]=(int8_t)(seed>>24);
                seed=seed*1664525u+1013904223u;b[i]=(int8_t)(seed>>24);
                weights[i]=b[i];
            }
            memcpy(code,a,sizeof(code));
            for(int dim=1;dim<=2048;dim++){
                int32_t expected=0;for(int d=0;d<dim;d++)expected+=(int32_t)a[d]*b[d];
                assert(anchor_doti8_avx2(a,b,dim)==expected);cases++;
                if(dim%8==0){
                    int stride=dim/8;expected=0;
                    for(int d=0;d<dim;d++)if((code[d/8]>>(d%8))&1)expected+=weights[(d%8)*stride+d/8];
                    assert(anchor_score_tq1_avx2(code,weights,stride,dim)==expected);cases++;
                }
            }
        }
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
    printf("%u exact kernel comparisons passed\n",cases);return 0;
#else
    puts("Requires x86-64 GCC/Clang");return 77;
#endif
}
