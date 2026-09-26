/* Runtime-dispatched exact x86 kernels. Generic binaries stay compatible
 * with CPUs without AVX2/F16C; ARM uses the existing NEON implementations. */
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
__attribute__((target("avx2"))) static int32_t anchor_doti8_avx2(const int8_t*a,const int8_t*b,int d){
    __m256i acc0=_mm256_setzero_si256(),acc1=acc0;int i=0;
    for(;i+32<=d;i+=32){
        __m256i a0=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a+i)));
        __m256i b0=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b+i)));
        __m256i a1=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a+i+16)));
        __m256i b1=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b+i+16)));
        acc0=_mm256_add_epi32(acc0,_mm256_madd_epi16(a0,b0));
        acc1=_mm256_add_epi32(acc1,_mm256_madd_epi16(a1,b1));
    }
    __m256i acc=_mm256_add_epi32(acc0,acc1);
    __m128i sum=_mm_add_epi32(_mm256_castsi256_si128(acc),_mm256_extracti128_si256(acc,1));
    sum=_mm_add_epi32(sum,_mm_srli_si128(sum,8));sum=_mm_add_epi32(sum,_mm_srli_si128(sum,4));
    int32_t result=_mm_cvtsi128_si32(sum);for(;i<d;i++)result+=(int32_t)a[i]*b[i];return result;
}
__attribute__((target("f16c"))) static float anchor_h2f_f16c(uint16_t h){
    /* Preserve the x86 baseline's saturation of exponent31, including NaNs. */
    if((h&0x7c00)==0x7c00)return h&0x8000?-65504.f:65504.f;
    return _cvtsh_ss(h);
}
__attribute__((target("avx2"))) static int32_t anchor_score_tq1_avx2(const uint8_t*code,
                                         const int8_t*weights,
                                         int stride,int dim){
    const __m256i one8=_mm256_set1_epi8(1);
    const __m256i one16=_mm256_set1_epi16(1);
    __m256i acc=_mm256_setzero_si256();
    int i=0,nb=dim/8;
    for(;i+32<=nb;i+=32){
        __m256i code32=_mm256_loadu_si256((const __m256i*)(code+i));
        __m256i partial=_mm256_setzero_si256();
        /* Each pair sums at most two signed bytes selected by 0/1.
         * Eight planes fit int16: [-2048,2032], so neither the saturating
         * pair multiply nor the subsequent additions can lose precision. */
#define CLOUD_TQ1_PLANE(bit) do { \
        __m256i bits=_mm256_and_si256(_mm256_srli_epi16(code32,bit),one8); \
        __m256i q=_mm256_loadu_si256((const __m256i*)(weights+(bit)*stride+i)); \
        partial=_mm256_add_epi16(partial,_mm256_maddubs_epi16(bits,q)); \
    } while(0)
        CLOUD_TQ1_PLANE(0);CLOUD_TQ1_PLANE(1);
        CLOUD_TQ1_PLANE(2);CLOUD_TQ1_PLANE(3);
        CLOUD_TQ1_PLANE(4);CLOUD_TQ1_PLANE(5);
        CLOUD_TQ1_PLANE(6);CLOUD_TQ1_PLANE(7);
#undef CLOUD_TQ1_PLANE
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(partial,one16));
    }
    int32_t lanes[8];_mm256_storeu_si256((__m256i*)lanes,acc);
    int32_t sum=0;for(int lane=0;lane<8;lane++)sum+=lanes[lane];
    for(int d=i*8;d<dim;d++)
        sum+=((code[d>>3]>>(d&7))&1)*weights[(d&7)*stride+(d>>3)];
    return sum;
}
#endif
