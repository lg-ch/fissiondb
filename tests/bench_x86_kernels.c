/* Optional CPU-only comparison; does not estimate end-to-end retrieval speed.
 * gcc -O3 -std=c11 -march=x86-64 tests/bench_x86_kernels.c -o /tmp/bench-simd */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "../src/anchor_x86.inc"

#if defined(__x86_64__) && defined(__GNUC__)
static volatile int64_t checksum;
static double seconds(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void){
    if(!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("avx512bw"))return 77;
    const size_t rows=200000,codes=1048576;const int dim=1024;
    int8_t*a=malloc(rows*dim),q[1024],weights[512];uint8_t*c=malloc(codes*64);
    if(!a||!c)return 1;
    uint32_t seed=123;
    for(size_t i=0;i<rows*dim;i++){seed=seed*1664525u+1013904223u;a[i]=(int8_t)(seed>>24);}
    for(size_t i=0;i<codes*64;i++){seed=seed*1664525u+1013904223u;c[i]=(uint8_t)(seed>>24);}
    memcpy(q,a,sizeof(q));memcpy(weights,a+37,sizeof(weights));
    for(int pass=0;pass<8;pass++){
        /* A/B/B/A order repeated; function pointers inhibit loop hoisting. */
        int wide=(pass%4==1||pass%4==2);
        int32_t(*volatile dot)(const int8_t*,const int8_t*,int)=wide?anchor_doti8_avx512:anchor_doti8_avx2;
        int32_t(*volatile score)(const uint8_t*,const int8_t*,int,int)=wide?anchor_score_tq1_avx512:anchor_score_tq1_avx2;
        int64_t sum=0;double start=seconds();
        for(size_t i=0;i<rows;i++)sum+=dot(q,a+i*dim,dim);
        double route=seconds()-start;checksum=sum;
        sum=0;start=seconds();
        for(size_t i=0;i<codes;i++)sum+=score(c+i*64,weights,64,512);
        double scan=seconds()-start;checksum=sum;
        printf("{\"pass\":%d,\"backend\":\"%s\",\"anchors\":%zu,\"dim\":%d,\"routing_ms\":%.6f,\"codes\":%zu,\"score_ms\":%.6f,\"checksum\":%lld}\n",
               pass,wide?"avx512bw":"avx2",rows,dim,route*1e3,codes,scan*1e3,(long long)sum);
    }
    free(a);free(c);return 0;
}
#else
int main(void){return 77;}
#endif
