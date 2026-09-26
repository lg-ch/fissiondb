/* Index ANCRES : cellules de Voronoi sur K ancres echantillonnees dans la
   base (equilibre statistique gratuit), spill frontiere (un doc rejoint
   les ancres a distance <= (1+eps) de la plus proche, max M), blocs par
   cellule [u32 id + code TQ] colocalises — l'unite de lecture requete.
   Requete : descente ancres (RAM) -> nprobe blocs (io_uring) -> scoring
   TQ asymetrique (SDOT int8) -> top-R -> rerank exact (pread base f16).
   Mode cosinus : docs, ancres et requetes normalises ; pas de normes.
   Rotation : FWHT + signes seedes (recomputable), scale int par dim.

   Fichiers (out_dir) : meta.txt, anchors.bin (K x dim f32),
   offs.bin ((K+1) x u64), blocks.bin, scale.bin (dim f32).            */
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <errno.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <liburing.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define ANC_NEON 1
#else
#define ANC_NEON 0
#endif

/* ---------- mode S3 : vagues de range-GETs paralleles (libcurl multi,
   SigV4 natif). Cles UNIQUEMENT via l environnement AWS_ACCESS_KEY_ID /
   AWS_SECRET_ACCESS_KEY (jamais en argv ni en fichier). ---------- */
#include <curl/curl.h>
typedef struct {
    uint8_t* dst;
    size_t cap, got;
    uint64_t off;
    int range_ok, done, success;
} S3Buf;

static size_t s3_write_cb(void* p, size_t s, size_t n, void* u) {
    S3Buf* b = (S3Buf*)u;
    if (s && n > SIZE_MAX / s) return 0;
    size_t k = s * n;
    if (k > b->cap - b->got) return 0; /* reject oversized bodies */
    memcpy(b->dst + b->got, p, k); b->got += k;
    return k;
}

static size_t s3_header_cb(char* p, size_t s, size_t n, void* u) {
    S3Buf* b = (S3Buf*)u;
    if (s && n > SIZE_MAX / s) return 0;
    size_t k = s * n;
    if (k >= 5 && !strncasecmp(p, "HTTP/", 5)) b->range_ok = 0;
    if (k >= 14 && !strncasecmp(p, "Content-Range:", 14)) {
        char line[256];
        if (k >= sizeof(line)) return 0;
        memcpy(line, p, k); line[k] = 0;
        unsigned long long first, last, total;
        b->range_ok = sscanf(line + 14, " bytes %llu-%llu/%llu", &first, &last, &total) == 3
            && first == b->off && last >= first && last - first + 1 == b->cap && total > last;
    }
    return k;
}

typedef struct {
    CURLM* multi;
    char userpwd[512];
    struct curl_slist* headers;
    int has_auth, force_http1;
    int hedge_ms;       /* 0 = pas de hedging */
    int last_hedged;    /* GETs doubles lors de la derniere vague */
    long last_newconn;  /* connexions TCP ouvertes (0 = tout reutilise) */
} S3Ctx;

static double now_ms_s3(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static pthread_once_t s3_once = PTHREAD_ONCE_INIT;
static CURLcode s3_global_status = CURLE_FAILED_INIT;
static void s3_global_init_once(void) { s3_global_status = curl_global_init(CURL_GLOBAL_DEFAULT); }
static int s3_init(S3Ctx* c) {
    pthread_once(&s3_once, s3_global_init_once);
    if (s3_global_status != CURLE_OK) return -1;
    c->multi = curl_multi_init();
    if (!c->multi) return -1;
    curl_multi_setopt(c->multi, CURLMOPT_MAX_HOST_CONNECTIONS, 512L);
    curl_multi_setopt(c->multi, CURLMOPT_MAX_TOTAL_CONNECTIONS, 512L);
    curl_multi_setopt(c->multi, CURLMOPT_MAXCONNECTS, 512L);
    const char* http1=getenv("FISSIONDB_S3_HTTP1");
    if(http1&&strcmp(http1,"0")&&strcmp(http1,"1"))goto fail;
    c->force_http1=http1&&!strcmp(http1,"1");
    const char* k = getenv("AWS_ACCESS_KEY_ID");
    const char* s = getenv("AWS_SECRET_ACCESS_KEY");
    if((k&&!s)||(!k&&s))goto fail;
    c->has_auth = (k && s);
    if(c->has_auth&&snprintf(c->userpwd,sizeof(c->userpwd),"%s:%s",k,s)>=(int)sizeof(c->userpwd))goto fail;
    const char* token=getenv("AWS_SESSION_TOKEN");
    if(token&&*token) {
        if(!c->has_auth||strlen(token)>8192||strchr(token,'\r')||strchr(token,'\n'))goto fail;
        char header[8256];snprintf(header,sizeof(header),"x-amz-security-token: %s",token);
        c->headers=curl_slist_append(NULL,header);if(!c->headers)goto fail;
    }
    return 0;
fail:
    curl_slist_free_all(c->headers);c->headers=NULL;
    curl_multi_cleanup(c->multi);c->multi=NULL;return -1;
}

static void s3_close(S3Ctx* c) {
    if(c->multi)curl_multi_cleanup(c->multi);
    curl_slist_free_all(c->headers);c->multi=NULL;c->headers=NULL;
}
static CURL* s3_request(S3Ctx* c, const char* url, S3Buf* b) {
    CURL* h = curl_easy_init();
    if (!h) return NULL;
    char range[64], sig[128];
    const char* region = getenv("AWS_REGION");
    if (!region) region = "us-east-1";
    snprintf(sig, sizeof(sig), "aws:amz:%s:s3", region);
    snprintf(range, sizeof(range), "%llu-%llu", (unsigned long long)b->off,
             (unsigned long long)(b->off + b->cap - 1));
#define S3_OPT(opt, val) do { if (curl_easy_setopt(h, opt, val) != CURLE_OK) goto fail; } while (0)
    S3_OPT(CURLOPT_URL, url); S3_OPT(CURLOPT_RANGE, range);
    S3_OPT(CURLOPT_WRITEFUNCTION, s3_write_cb); S3_OPT(CURLOPT_WRITEDATA, b);
    S3_OPT(CURLOPT_HEADERFUNCTION, s3_header_cb); S3_OPT(CURLOPT_HEADERDATA, b);
    S3_OPT(CURLOPT_PRIVATE, b); S3_OPT(CURLOPT_TCP_KEEPALIVE, 1L);
    S3_OPT(CURLOPT_CONNECTTIMEOUT_MS, 5000L); S3_OPT(CURLOPT_TIMEOUT_MS, 30000L);
    S3_OPT(CURLOPT_NOSIGNAL, 1L);
    if(c->force_http1)S3_OPT(CURLOPT_HTTP_VERSION, (long)CURL_HTTP_VERSION_1_1);
    if(c->headers)S3_OPT(CURLOPT_HTTPHEADER,c->headers);
    if (c->has_auth) {
        S3_OPT(CURLOPT_AWS_SIGV4, sig); S3_OPT(CURLOPT_USERPWD, c->userpwd);
    }
#undef S3_OPT
    if (curl_multi_add_handle(c->multi, h) != CURLM_OK) goto fail;
    return h;
fail:
    curl_easy_cleanup(h); return NULL;
}

/* Only completed, successful HTTP 206 responses with the exact range count.
   A hedge wins after validation, never merely because its body buffer is full. */
static int s3_wave(S3Ctx* c, const char* url, const uint64_t* off,
                   const uint64_t* len, uint8_t* const* dst, int n) {
    if (n <= 0) return n == 0 ? 0 : -1;
    CURL** hs = calloc((size_t)n * 2, sizeof(*hs));
    S3Buf* b = calloc((size_t)n * 2, sizeof(*b));
    int ok = -1, hedged = 0, hedge_started = 0;
    long newconn = 0;
    if (!hs || !b) goto cleanup;
    for (int i = 0; i < n; i++) {
        b[i].dst = dst[i]; b[i].cap = len[i]; b[i].off = off[i];
        if (!len[i]) { b[i].done = b[i].success = 1; continue; }
        if (len[i] > SIZE_MAX || off[i] > UINT64_MAX - len[i]) goto cleanup;
        hs[i] = s3_request(c, url, &b[i]);
        if (!hs[i]) goto cleanup;
    }
    double start = now_ms_s3();
    for (;;) {
        int running = 0;
        if (curl_multi_perform(c->multi, &running) != CURLM_OK) goto cleanup;
        int left;
        CURLMsg* msg;
        while ((msg = curl_multi_info_read(c->multi, &left))) {
            if (msg->msg != CURLMSG_DONE) continue;
            S3Buf* state = NULL; long status = 0;
            curl_easy_getinfo(msg->easy_handle, CURLINFO_PRIVATE, &state);
            curl_easy_getinfo(msg->easy_handle, CURLINFO_RESPONSE_CODE, &status);
            if (state) {
                state->done = 1;
                state->success = msg->data.result == CURLE_OK && status == 206 &&
                    state->range_ok && state->got == state->cap;
            }
        }
        int complete = 1;
        for (int i = 0; i < n; i++)
            if (!b[i].success && !b[n+i].success) { complete = 0; break; }
        if (complete || !running) break;
        if (!hedge_started && c->hedge_ms > 0 && now_ms_s3() - start >= c->hedge_ms) {
            hedge_started = 1;
            for (int i = 0; i < n; i++) {
                if (b[i].done) continue;
                b[n+i].cap = len[i]; b[n+i].off = off[i];
                b[n+i].dst = malloc(len[i]);
                if (!b[n+i].dst) goto cleanup;
                hs[n+i] = s3_request(c, url, &b[n+i]);
                if (!hs[n+i]) goto cleanup;
                hedged++;
            }
        }
        if (curl_multi_poll(c->multi, NULL, 0, 20, NULL) != CURLM_OK) goto cleanup;
    }
    ok = 0;
    for (int i = 0; i < n; i++) {
        if (b[i].success) ok++;
        else if (b[n+i].success) { memcpy(dst[i], b[n+i].dst, len[i]); ok++; }
    }
cleanup:
    if (hs) for (int i = 0; i < 2*n; i++) if (hs[i]) {
        long nc = 0; curl_easy_getinfo(hs[i], CURLINFO_NUM_CONNECTS, &nc); newconn += nc;
        curl_multi_remove_handle(c->multi, hs[i]); curl_easy_cleanup(hs[i]);
    }
    if (b) for (int i = n; i < 2*n; i++) free(b[i].dst);
    free(hs); free(b);
    c->last_hedged = hedged; c->last_newconn = newconn;
    return ok;
}

/* ---------- PRNG splitmix64 ---------- */
static uint64_t asm64(uint64_t* st) {
    uint64_t z = (*st += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ---------- FWHT (dim = puissance de 2) + signes seedes ---------- */
static void fwht(float* x, int d) {
    for (int h = 1; h < d; h <<= 1)
        for (int i = 0; i < d; i += h << 1)
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
    float s = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) x[i] *= s;
}

static void rot_seeded(const float* in, float* out, const int8_t* sgn,
                       int d) {
    for (int i = 0; i < d; i++) out[i] = in[i] * (float)sgn[i];
    fwht(out, d);
}

static void make_signs(int8_t* sgn, int d, uint64_t seed) {
    uint64_t st = seed;
    for (int i = 0; i < d; i++)
        sgn[i] = (asm64(&st) & 1) ? 1 : -1;
}

#include "anchor_x86.inc"

/* ---------- f16 -> f32 ---------- */
static inline float h2f(uint16_t h) {
#if defined(__x86_64__) && defined(__GNUC__)
    if(__builtin_cpu_supports("f16c"))return anchor_h2f_f16c(h);
#endif
#if ANC_NEON
    __fp16 x;
    memcpy(&x, &h, 2);
    return (float)x;
#else
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    if (e == 0) return (s ? -1.f : 1.f) * (float)m * 5.9604645e-8f;
    if (e == 31) return s ? -65504.f : 65504.f;
    union { uint32_t u; float f; } v;
    v.u = (s << 31) | ((e + 112) << 23) | (m << 13);
    return v.f;
#endif
}

static void row_f16_to_unit(const uint16_t* src, float* dst, int d) {
    float n2 = 0.0f;
    for (int i = 0; i < d; i++) { dst[i] = h2f(src[i]); n2 += dst[i] * dst[i]; }
    float inv = 1.0f / (sqrtf(n2) + 1e-9f);
    for (int i = 0; i < d; i++) dst[i] *= inv;
}

static void row_f16_padded(const uint16_t* src,float* dst,int input_dim,int dim) {
    row_f16_to_unit(src,dst,input_dim);
    memset(dst+input_dim,0,(size_t)(dim-input_dim)*4);
}
static int anchor_read_padded(FILE* f,uint16_t* raw,size_t count,int input_dim,int dim) {
    if(fread(raw,(size_t)input_dim*2,count,f)!=count)return -1;
    if(input_dim!=dim)for(size_t i=count;i-->0;){
        memmove(raw+i*dim,raw+i*input_dim,(size_t)input_dim*2);
        memset(raw+i*dim+input_dim,0,(size_t)(dim-input_dim)*2);
    }
    return 0;
}

/* ---------- dot f32 NEON ---------- */
static inline float dotf(const float* a, const float* b, int d) {
#if ANC_NEON
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= d; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float s = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < d; i++) s += a[i] * b[i];
    return s;
#else
    float s = 0;
    for (int i = 0; i < d; i++) s += a[i] * b[i];
    return s;
#endif
}

/* ---------- dot int8 (SDOT si dispo, sinon scalaire) ---------- */
static inline int32_t doti8(const int8_t* a, const int8_t* b, int d) {
#if defined(__x86_64__) && defined(__GNUC__)
    if(__builtin_cpu_supports("avx2"))return anchor_doti8_avx2(a,b,d);
#endif
#if ANC_NEON && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    int i = 0;
    for (; i + 16 <= d; i += 16)
        acc = vdotq_s32(acc, vld1q_s8(a + i), vld1q_s8(b + i));
    int32_t s = vaddvq_s32(acc);
    for (; i < d; i++) s += (int32_t)a[i] * b[i];
    return s;
#else
    int32_t s = 0;
    for (int i = 0; i < d; i++) s += (int32_t)a[i] * b[i];
    return s;
#endif
}

/* Scoring TQ4 sans depaquetage : nibbles bas/hauts sign-etendus en NEON
   (shl4+shr4 / shr4 arithmetique), SDOT contre la requete REORDONNEE
   (qlo = dims paires, qhi = dims impaires). ~2 instr / 32 dims.        */
static inline int32_t score_tq4(const uint8_t* code, const int8_t* qlo,
                                const int8_t* qhi, int dim) {
#if ANC_NEON && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    int nb = dim / 2;
    int i = 0;
    for (; i + 16 <= nb; i += 16) {
        int8x16_t b = vld1q_s8((const int8_t*)code + i);
        int8x16_t lo = vshrq_n_s8(vshlq_n_s8(b, 4), 4);
        int8x16_t hi = vshrq_n_s8(b, 4);
        acc = vdotq_s32(acc, lo, vld1q_s8(qlo + i));
        acc = vdotq_s32(acc, hi, vld1q_s8(qhi + i));
    }
    int32_t sum = vaddvq_s32(acc);
    for (; i < nb; i++) {
        int lo = code[i] & 15, hi = code[i] >> 4;
        if (lo > 7) lo -= 16;
        if (hi > 7) hi -= 16;
        sum += lo*qlo[i] + hi*qhi[i];
    }
    return sum;
#else
    int32_t s = 0;
    for (int d = 0; d < dim; d += 2) {
        uint8_t b = code[d >> 1];
        int q0 = (int)(b & 15); if (q0 > 7) q0 -= 16;
        int q1 = (int)(b >> 4); if (q1 > 7) q1 -= 16;
        s += q0 * qlo[d >> 1] + q1 * qhi[d >> 1];
    }
    return s;
#endif
}

static inline float score_tq_l2(const uint8_t* code,const float* lookup,int bytes) {
    /* Independent sums avoid a dependency chain for every code byte. */
    float s0=0,s1=0,s2=0,s3=0;
    int i=0;
    for(;i+3<bytes;i+=4) {
        s0+=lookup[(size_t)i*256+code[i]];
        s1+=lookup[(size_t)(i+1)*256+code[i+1]];
        s2+=lookup[(size_t)(i+2)*256+code[i+2]];
        s3+=lookup[(size_t)(i+3)*256+code[i+3]];
    }
    for(;i<bytes;i++)s0+=lookup[(size_t)i*256+code[i]];
    return (s0+s1)+(s2+s3);
}

/* TQ2 : 2 bits/dim (4 dims/octet, valeur stockee (q+2)&3, q in [-2,1]).
   Decodage NEON par champs de 2 bits + 4 SDOT contre la requete
   deinterlacee en 4 flux (dims = 0,1,2,3 mod 4).                        */
static inline int32_t score_tq2(const uint8_t* code, const int8_t* q0,
                                const int8_t* q1, const int8_t* q2,
                                const int8_t* q3, int dim) {
#if ANC_NEON && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    int8x16_t three = vdupq_n_s8(3), two = vdupq_n_s8(2);
    int nb = dim / 4;
    int i = 0;
    for (; i + 16 <= nb; i += 16) {
        uint8x16_t b = vld1q_u8(code + i);
        int8x16_t d0 = vsubq_s8(vandq_s8(vreinterpretq_s8_u8(b), three), two);
        int8x16_t d1 = vsubq_s8(vandq_s8(
            vreinterpretq_s8_u8(vshrq_n_u8(b, 2)), three), two);
        int8x16_t d2 = vsubq_s8(vandq_s8(
            vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), three), two);
        int8x16_t d3 = vsubq_s8(
            vreinterpretq_s8_u8(vshrq_n_u8(b, 6)), two);
        acc = vdotq_s32(acc, d0, vld1q_s8(q0 + i));
        acc = vdotq_s32(acc, d1, vld1q_s8(q1 + i));
        acc = vdotq_s32(acc, d2, vld1q_s8(q2 + i));
        acc = vdotq_s32(acc, d3, vld1q_s8(q3 + i));
    }
    int32_t sum = vaddvq_s32(acc);
    for (; i < nb; i++) {
        unsigned b = code[i];
        sum += ((int)(b&3)-2)*q0[i] + ((int)((b>>2)&3)-2)*q1[i]
             + ((int)((b>>4)&3)-2)*q2[i] + ((int)((b>>6)&3)-2)*q3[i];
    }
    return sum;
#else
    int32_t s = 0;
    for (int d = 0; d < dim; d += 4) {
        uint8_t b = code[d >> 2];
        s += ((int)(b & 3) - 2) * q0[d >> 2]
           + ((int)((b >> 2) & 3) - 2) * q1[d >> 2]
           + ((int)((b >> 4) & 3) - 2) * q2[d >> 2]
           + ((int)((b >> 6) & 3) - 2) * q3[d >> 2];
    }
    return s;
#endif
}

/* TQ1 : somme masquee des q8 aux bits leves (offset constant par requete,
   sans effet sur l ordre). 8 dims par octet via vtst.                   */
/* TQ1 sur le motif TQ2 (qui est rapide) : 8 flux de bits extraits par
   decalage+AND, SDOT contre la requete deinterlacee en 8 flux
   (flux b = dims congrues a b mod 8). qs1 : 8 flux contigus de `stride`
   octets chacun ; `dim` = dims a traiter (prefixe ou tout).           */
static inline int32_t score_tq1(const uint8_t* code, const int8_t* qs1,
                                int stride, int dim) {
#if defined(__x86_64__) && defined(__GNUC__)
    if(__builtin_cpu_supports("avx2"))return anchor_score_tq1_avx2(code,qs1,stride,dim);
#endif
#if ANC_NEON && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    int8x16_t one = vdupq_n_s8(1);
    int nb = dim / 8;
    int i = 0;
    for (; i + 16 <= nb; i += 16) {
        uint8x16_t b = vld1q_u8(code + i);
        /* deroule explicitement les 8 decalages (vshrq_n exige une cst) */
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vandq_u8(b, vreinterpretq_u8_s8(one))), vld1q_s8(qs1 + 0 * stride + i));
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(b, 1), vreinterpretq_u8_s8(one))), vld1q_s8(qs1 + 1 * stride + i));
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(b, 2), vreinterpretq_u8_s8(one))), vld1q_s8(qs1 + 2 * stride + i));
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(b, 3), vreinterpretq_u8_s8(one))), vld1q_s8(qs1 + 3 * stride + i));
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(b, 4), vreinterpretq_u8_s8(one))), vld1q_s8(qs1 + 4 * stride + i));
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(b, 5), vreinterpretq_u8_s8(one))), vld1q_s8(qs1 + 5 * stride + i));
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(b, 6), vreinterpretq_u8_s8(one))), vld1q_s8(qs1 + 6 * stride + i));
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vshrq_n_u8(b, 7)), vld1q_s8(qs1 + 7 * stride + i));
    }
    int32_t sum = vaddvq_s32(acc);
    for (; i < nb; i++) for (int bit = 0; bit < 8; bit++)
        if ((code[i] >> bit) & 1) sum += qs1[bit*stride+i];
    return sum;
#else
    int32_t s = 0;
    for (int d = 0; d < dim; d++)
        if ((code[d >> 3] >> (d & 7)) & 1) s += qs1[(d & 7) * stride + (d >> 3)];
    return s;
#endif
}

typedef struct {
    int K, dim, M, tqbits;
    float eps;
    int64_t n;
    uint64_t seed;
    int input_dim; /* Original vector width; dim is the padded rotation width. */
    int cdim;   /* dims couvertes par le code stocke (layout leger) */
} AMeta;

static int meta_load(const char* dir, AMeta* m) {
    char p[1024];
    snprintf(p, sizeof(p), "%s/meta.txt", dir);
    FILE* f = fopen(p, "r");
    if (!f) return -1;
    long long n = 0; unsigned long long sd = 0;
    int cd = 0, input_dim = 0;
    int r = fscanf(f, "%d %d %d %d %f %lld %llu %d %d", &m->K, &m->dim, &m->M,
                   &m->tqbits, &m->eps, &n, &sd, &cd, &input_dim);
    fclose(f);
    m->n = n; m->seed = sd;
    m->input_dim = r >= 9 ? input_dim : m->dim;
    m->cdim = (r >= 8 && cd > 0) ? cd : m->dim;
    return r >= 7 && m->input_dim >= 1 && m->input_dim <= m->dim && m->K > 0 && m->dim >= 8 && m->dim <= 65536 &&
        !(m->dim & (m->dim-1)) && m->M >= 1 && m->M <= 4 && m->M <= m->K &&
        m->n >= m->K && isfinite(m->eps) && m->eps >= 0 &&
        (m->tqbits == 1 || m->tqbits == 2 || m->tqbits == 4) &&
        m->cdim >= 8 && m->cdim <= m->dim && m->cdim % 8 == 0 ? 0 : -1;
}

/* Cache identity is intentionally conservative: replacing/touching the base
   invalidates it. Legacy assignment payloads remain readable as coarse input. */
static uint64_t cache_mix(uint64_t h, uint64_t v) {
    for (int i = 0; i < 8; i++) { h = (h ^ (v & 255)) * 1099511628211ULL; v >>= 8; }
    return h;
}
static uint64_t cache_file(uint64_t h, const char* path) {
    struct stat st;
    if (stat(path, &st)) return 0;
    h = cache_mix(h, st.st_dev); h = cache_mix(h, st.st_ino);
    h = cache_mix(h, st.st_size); h = cache_mix(h, st.st_mtim.tv_sec);
    return cache_mix(h, st.st_mtim.tv_nsec);
}

/* Bounded waves prevent CQ overflow. All completions are consumed before
   callers reuse buffers, including after a short read or a failed request. */
static int anchor_reads(struct io_uring* ring, int fd, const uint64_t* off,
                        const uint64_t* len, uint8_t* const* dst, int n) {
    for (int first = 0; first < n; first += 1024) {
        int end = first + 1024 < n ? first + 1024 : n, count = 0;
        for (int i = first; i < end; i++) {
            if (!len[i]) continue;
            if (len[i] > INT_MAX || off[i] > INT64_MAX) return -1;
            struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
            if (!sqe) return -1;
            io_uring_prep_read(sqe, fd, dst[i], (unsigned)len[i], off[i]);
            sqe->user_data = len[i];
            count++;
        }
        int submitted = 0;
        while (submitted < count) {
            int rc = io_uring_submit(ring);
            if (rc == -EINTR) continue;
            if (rc <= 0) return -1;
            submitted += rc;
        }
        int failed = 0;
        for (int i = 0; i < count; i++) {
            struct io_uring_cqe* cqe;
            int rc;
            do { rc = io_uring_wait_cqe(ring, &cqe); } while (rc == -EINTR);
            if (rc < 0) return -1;
            if (cqe->res < 0 || (uint64_t)cqe->res != cqe->user_data) failed = 1;
            io_uring_cqe_seen(ring, cqe);
        }
        if (failed) { fprintf(stderr, "anchor: failed or short read\n"); return -1; }
    }
    return 0;
}

typedef struct { uint64_t offset, entry; } AnchorWrite;
static int anchor_write_cmp(const void* a, const void* b) {
    uint64_t x = ((const AnchorWrite*)a)->offset, y = ((const AnchorWrite*)b)->offset;
    return (x > y) - (x < y);
}
static int anchor_write_all(int fd, const uint8_t* data, size_t n, uint64_t off) {
    while (n) {
        ssize_t w = pwrite(fd, data, n, off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return -1;
        data += w; n -= w; off += w;
    }
    return 0;
}

/* Build scratch is file-backed so the kernel can reclaim assignment pages
   under a cgroup limit. Unlink immediately: completion, errors and SIGKILL
   release the temporary file without leaving stale scratch names behind. */
static void* anchor_build_scratch(const char* directory,size_t bytes) {
    char path[1200];
    if(!bytes||bytes>INT64_MAX||snprintf(path,sizeof(path),"%s/.assign-scratch-XXXXXX",directory)>=(int)sizeof(path))return NULL;
    int fd=mkstemp(path);if(fd<0)return NULL;
    if(unlink(path)){close(fd);return NULL;}
    int err=posix_fallocate(fd,0,(off_t)bytes);
    if(err){close(fd);errno=err;perror("assignment scratch allocation");return NULL;}
    void* memory=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    close(fd);return memory==MAP_FAILED?NULL:memory;
}

static int anchor_save_file(const char* path, const void* data, size_t size) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    int ok = fwrite(data, 1, size, f) == size;
    if (fflush(f) != 0 || fsync(fileno(f)) != 0) ok = 0;
    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}

/* ================= BUILD ================= */
int cmd_anchor_build(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: fissiondb-engine abuild <base.f16bin> <out_dir> <K> "
                        "[--eps 0.20] [--m 3] [--tqbits 4] [--seed 42] "
                        "[--nmax 0]\n");
        return 1;
    }
    const char* base_path = argv[2];
    const char* out = argv[3];
    int K = atoi(argv[4]);
    float eps = 0.20f;
    int M = 3, tqbits = 4;
    uint64_t seed = 42;
    int64_t nmax = 0;
    int cdim = 0;
    const char* coarse = NULL;
    int neighbors = 64;
    for (int i = 5; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--eps")) eps = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--m")) M = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--tqbits")) tqbits = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--seed")) seed = strtoull(argv[i + 1], 0, 10);
        else if (!strcmp(argv[i], "--nmax")) nmax = atoll(argv[i + 1]);
        else if (!strcmp(argv[i], "--neighbors")) neighbors = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--coarse")) coarse = argv[i + 1];
        else if (!strcmp(argv[i], "--cdim")) cdim = atoi(argv[i + 1]);
    }
    if (M < 1 || M > 4 || K < M || neighbors < 1 || neighbors > 4096 ||
        (tqbits != 1 && tqbits != 2 && tqbits != 4) || eps < 0) return 1;
    FILE* bf = fopen(base_path, "rb");
    if (!bf) { perror("base"); return 1; }
    uint32_t hdr[2];
    if (fread(hdr, 4, 2, bf) != 2) { fclose(bf); return 1; }
    int64_t n = hdr[0];
    int input_dim = (int)hdr[1], dim = 8;
    if(input_dim < 1 || input_dim > 65536){fclose(bf);return 1;}
    while(dim < input_dim)dim *= 2;
    if (nmax > 0 && nmax < n) n = nmax;
    if (n < K || dim < 8 || (dim & (dim - 1)) || dim > 65536 ||
        (cdim && (cdim > dim || cdim < 8 || cdim % 8))) return 1;
    fprintf(stderr, "abuild: n=%lld dim=%d K=%d eps=%.2f M=%d tq%d\n",
            (long long)n, dim, K, eps, M, tqbits);

    /* --- ancres : K ids sans remise (seed) --- */
    uint8_t* taken = (uint8_t*)calloc((size_t)n, 1);
    int64_t* aids = (int64_t*)malloc((size_t)K * 8);
    uint64_t st = seed;
    for (int k = 0; k < K; k++) {
        int64_t id;
        do { id = (int64_t)(asm64(&st) % (uint64_t)n); } while (taken[id]);
        taken[id] = 1;
        aids[k] = id;
    }
    free(taken);
    float* A = (float*)malloc((size_t)K * dim * 4);
    uint16_t* rowbuf = (uint16_t*)malloc((size_t)dim * 2);
    for (int k = 0; k < K; k++) {
        fseeko(bf, 8 + aids[k] * (int64_t)input_dim * 2, SEEK_SET);
        memset(rowbuf,0,(size_t)dim*2);
        if (fread(rowbuf, 2, input_dim, bf) != (size_t)input_dim) return 1;
        row_f16_to_unit(rowbuf, A + (size_t)k * dim, dim);
    }
    fprintf(stderr, "abuild: ancres chargees\n");

    int8_t* sgn = (int8_t*)malloc(dim);
    make_signs(sgn, dim, seed ^ 0x51CA);

    /* Ancres en int8 pour l assignation : SDOT = 4x les ops/cycle du FMA
       f32, et 41 Mo au lieu de 164 — tuilables en L2. La quantization ne
       perturbe le choix des 2 plus proches que sur des ex aequo (spill
       rang-2 : sans consequence). argmax(dot) invariant par echelle.    */
    float amax = 0.0f;
    for (size_t i = 0; i < (size_t)K * dim; i++) {
        float v = fabsf(A[i]);
        if (v > amax) amax = v;
    }
    float aq = 127.0f / (amax + 1e-9f);
    int8_t* A8 = (int8_t*)malloc((size_t)K * dim);
    for (size_t i = 0; i < (size_t)K * dim; i++)
        A8[i] = (int8_t)lrintf(A[i] * aq);

    /* --- passe 1 : assignation top-M + scale (chunks streames) --- */
    const int64_t CHUNK = 8192;
    uint16_t* raw = (uint16_t*)malloc((size_t)CHUNK * dim * 2);
    size_t assignment_bytes=(size_t)n*M*4;
    int32_t* topm = anchor_build_scratch(out,assignment_bytes);
    float* topd = anchor_build_scratch(out,assignment_bytes);
    if(!raw||!topm||!topd)return 1;
    fprintf(stderr,"abuild: file-backed assignments %.1f MB; input buffer %.1f MB\n",
            2.0*assignment_bytes/1e6,(double)CHUNK*dim*2/1e6);
    double* scale_acc = (double*)calloc(dim, 8);
    int64_t scale_n = 0;
    double t0 = omp_get_wtime();
    /* cache d assignation : la passe 1 (3h a 40M) est independante de
       tqbits/eps — reutilisable pour rebuilder avec d autres codes.
       Layout : [i64 n][i64 M][topm nxM i32][topd nxM f32][sigmean dim f32] */
    char apath[1024];
    snprintf(apath, sizeof(apath), "%s/assign.bin", out);
    float* sigmean = (float*)malloc((size_t)dim * 4);
    int skip_pass1 = 0;
    char identity_path[1050];
    snprintf(identity_path, sizeof(identity_path), "%s.identity", apath);
    uint64_t identity = cache_file(1469598103934665603ULL, base_path);
    identity = cache_mix(identity, 2); /* assignment/rotation algorithm version */
    identity = cache_mix(identity, n); identity = cache_mix(identity, K);
    identity = cache_mix(identity, dim); identity = cache_mix(identity, M);
    identity = cache_mix(identity, input_dim);
    identity = cache_mix(identity, seed); identity = cache_mix(identity, neighbors);
    if (coarse) {
        char cp[1024];
        snprintf(cp, sizeof(cp), "%s/assign.bin", coarse);
        identity = cache_file(identity, cp);
        snprintf(cp, sizeof(cp), "%s/anchors.bin", coarse);
        identity = cache_file(identity, cp);
    }
    uint64_t stored_identity = 0;
    FILE* idf = fopen(identity_path, "rb");
    if (idf) { if (fread(&stored_identity, 8, 1, idf) != 1) stored_identity = 0; fclose(idf); }
    {
        FILE* af = fopen(apath, "rb");
        if (af && identity && stored_identity == identity) {
            int64_t ah[2];
            if (fread(ah, 8, 2, af) == 2 && ah[0] == n && ah[1] == M
                && fread(topm, 4, (size_t)n * M, af) == (size_t)n * M
                && fread(topd, 4, (size_t)n * M, af) == (size_t)n * M
                && fread(sigmean, 4, dim, af) == (size_t)dim) {
                skip_pass1 = 1;
                fprintf(stderr, "abuild: assignation reprise du cache\n");
            }
        }
        if (af) fclose(af);
    }
    int loaded_cache = skip_pass1;
    /* --- assignation HIERARCHIQUE via un index existant (--coarse) :
       les plus proches parmi les K nouvelles ancres se cherchent dans le
       voisinage des anciennes ancres du doc. O(N x ~96) au lieu de
       O(N x K) — c est aussi le chemin de production a 1B.             */
    if (!skip_pass1 && coarse) {
        AMeta om;
        char op[1024];
        if (meta_load(coarse, &om) != 0 || om.dim != dim || om.input_dim != input_dim || om.n != n) {
            fprintf(stderr, "coarse: meta incompatible\n");
            return 1;
        }
        int Ko = om.K, Mo = om.M;
        snprintf(op, sizeof(op), "%s/anchors.bin", coarse);
        FILE* f = fopen(op, "rb");
        float* Ao = (float*)malloc((size_t)Ko * dim * 4);
        if (!f || fread(Ao, 4, (size_t)Ko * dim, f) != (size_t)Ko * dim)
            return 1;
        fclose(f);
        snprintf(op, sizeof(op), "%s/assign.bin", coarse);
        f = fopen(op, "rb");
        int64_t ah[2];
        int32_t* topm_o = anchor_build_scratch(out,(size_t)n*Mo*4);
        if (!f || !topm_o || fread(ah, 8, 2, f) != 2 || ah[0] != n || ah[1] != Mo
            || fread(topm_o, 4, (size_t)n * Mo, f) != (size_t)n * Mo) {
            fprintf(stderr, "coarse: assign.bin absent/incompatible\n");
            return 1;
        }
        for (int64_t i = 0; i < n * Mo; i++)
            if (topm_o[i] < 0 || topm_o[i] >= Ko) { fclose(f); return 1; }
        fseeko(f, (off_t)((size_t)n * Mo * 4), SEEK_CUR); /* saute topd */
        if (fread(sigmean, 4, dim, f) != (size_t)dim) return 1;
        fclose(f);
        /* table de voisinage ancienne -> nouvelles : NBR plus proches */
        const int NBR = neighbors < K ? neighbors : K;
        int32_t* nbrs = (int32_t*)malloc((size_t)Ko * NBR * 4);
        double t1 = omp_get_wtime();
        #pragma omp parallel
        {
            int8_t* v8 = (int8_t*)malloc(dim);
            #pragma omp for schedule(dynamic, 16)
            for (int ko = 0; ko < Ko; ko++) {
                const float* v = Ao + (size_t)ko * dim;
                float vmax = 0;
                for (int d = 0; d < dim; d++) {
                    float x = fabsf(v[d]);
                    if (x > vmax) vmax = x;
                }
                float vq = 127.0f / (vmax + 1e-9f);
                for (int d = 0; d < dim; d++)
                    v8[d] = (int8_t)lrintf(v[d] * vq);
                float bd[NBR];
                int32_t bi[NBR];
                for (int j = 0; j < NBR; j++) bd[j] = -2e9f;
                for (int k = 0; k < K; k++) {
                    float s = (float)doti8(v8, A8 + (size_t)k * dim, dim);
                    if (s > bd[NBR - 1]) {
                        int j = NBR - 1;
                        while (j > 0 && s > bd[j - 1]) {
                            bd[j] = bd[j - 1]; bi[j] = bi[j - 1]; j--;
                        }
                        bd[j] = s; bi[j] = k;
                    }
                }
                memcpy(nbrs + (size_t)ko * NBR, bi, NBR * 4);
            }
            free(v8);
        }
        fprintf(stderr, "abuild: voisinage %dx%d en %.0fs\n", Ko, NBR,
                omp_get_wtime() - t1);
        /* passe docs : candidats = union des voisinages des Mo anciennes */
        fseeko(bf, 8, SEEK_SET);
        t1 = omp_get_wtime();
        for (int64_t off = 0; off < n; off += CHUNK) {
            int64_t c = n - off < CHUNK ? n - off : CHUNK;
            if (anchor_read_padded(bf,raw,c,input_dim,dim)) return 1;
            #pragma omp parallel
            {
                float* v = (float*)malloc((size_t)dim * 4);
                int8_t* v8 = (int8_t*)malloc(dim);
                int32_t cnd[Mo * NBR];
                #pragma omp for schedule(dynamic, 64)
                for (int64_t i = 0; i < c; i++) {
                    int64_t g = off + i;
                    row_f16_to_unit(raw + (size_t)i * dim, v, dim);
                    float vmax = 0;
                    for (int d = 0; d < dim; d++) {
                        float x = fabsf(v[d]);
                        if (x > vmax) vmax = x;
                    }
                    float vq = 127.0f / (vmax + 1e-9f);
                    for (int d = 0; d < dim; d++)
                        v8[d] = (int8_t)lrintf(v[d] * vq);
                    int nc = 0;
                    for (int mo = 0; mo < Mo; mo++) {
                        int32_t ko = topm_o[g * Mo + mo];
                        if (ko < 0) continue;
                        const int32_t* nb = nbrs + (size_t)ko * NBR;
                        for (int j = 0; j < NBR; j++) {
                            int32_t k = nb[j];
                            int dup = 0;
                            for (int x = 0; x < nc; x++)
                                if (cnd[x] == k) { dup = 1; break; }
                            if (!dup) cnd[nc++] = k;
                        }
                    }
                    float bd[4] = {2e9f, 2e9f, 2e9f, 2e9f};
                    int32_t bi[4] = {-1, -1, -1, -1};
                    for (int x = 0; x < nc; x++) {
                        float d2 = -(float)doti8(
                            v8, A8 + (size_t)cnd[x] * dim, dim);
                        if (d2 < bd[M - 1]) {
                            int j = M - 1;
                            while (j > 0 && d2 < bd[j - 1]) {
                                bd[j] = bd[j - 1]; bi[j] = bi[j - 1]; j--;
                            }
                            bd[j] = d2; bi[j] = cnd[x];
                        }
                    }
                    float inv = 1.0f / (vq * aq);
                    for (int j = 0; j < M; j++) {
                        topm[g * M + j] = bi[j];
                        topd[g * M + j] = 2.0f + 2.0f * bd[j] * inv;
                    }
                }
                free(v); free(v8);
            }
            if (off % 4000000 == 0)
                fprintf(stderr, "abuild: coarse-assign %lld/%lld (%.0fs)\n",
                        (long long)off, (long long)n, omp_get_wtime() - t1);
        }
        free(Ao); munmap(topm_o,(size_t)n*Mo*4); free(nbrs);
        skip_pass1 = 1;
        fprintf(stderr, "abuild: assignation hierarchique OK\n");
    }
    fseeko(bf, 8, SEEK_SET);
    for (int64_t off = 0; skip_pass1 == 0 && off < n; off += CHUNK) {
        int64_t c = n - off < CHUNK ? n - off : CHUNK;
        if (anchor_read_padded(bf,raw,c,input_dim,dim)) return 1;
        #pragma omp parallel
        {
            float* v = (float*)malloc((size_t)dim * 4);
            float* r = (float*)malloc((size_t)dim * 4);
            int8_t* v8 = (int8_t*)malloc(dim);
            #pragma omp for schedule(dynamic, 64)
            for (int64_t i = 0; i < c; i++) {
                row_f16_to_unit(raw + (size_t)i * dim, v, dim);
                float vmax = 0.0f;
                for (int d = 0; d < dim; d++) {
                    float x = fabsf(v[d]);
                    if (x > vmax) vmax = x;
                }
                float vq = 127.0f / (vmax + 1e-9f);
                for (int d = 0; d < dim; d++)
                    v8[d] = (int8_t)lrintf(v[d] * vq);
                float bd[4] = {2e9f, 2e9f, 2e9f, 2e9f};
                int32_t bi[4] = {-1, -1, -1, -1};
                for (int k = 0; k < K; k++) {
                    float d2 = -(float)doti8(v8, A8 + (size_t)k * dim, dim);
                    if (d2 < bd[M - 1]) {
                        int j = M - 1;
                        while (j > 0 && d2 < bd[j - 1]) {
                            bd[j] = bd[j - 1]; bi[j] = bi[j - 1]; j--;
                        }
                        bd[j] = d2; bi[j] = k;
                    }
                }
                /* topd en ||q-a||^2 approx via cos int8 renormalise */
                float inv = 1.0f / (vq * aq);
                for (int j = 0; j < M; j++) {
                    topm[(off + i) * M + j] = bi[j];
                    topd[(off + i) * M + j] = 2.0f + 2.0f * bd[j] * inv;
                }
                if (((off + i) & 63) == 0) {
                    rot_seeded(v, r, sgn, dim);
                    #pragma omp critical
                    {
                        for (int d = 0; d < dim; d++)
                            scale_acc[d] += fabsf(r[d]);
                        scale_n++;
                    }
                }
            }
            free(v); free(r); free(v8);
        }
        fprintf(stderr, "abuild: assign %lld/%lld (%.0fs)\n",
                (long long)(off + c), (long long)n, omp_get_wtime() - t0);
    }
    if (!skip_pass1) {
        for (int d = 0; d < dim; d++)
            sigmean[d] = (float)(scale_acc[d] / (double)(scale_n ? scale_n : 1));
    }
    if (!loaded_cache) {
        /* Invalidate first: a failed rewrite must never retain a valid identity. */
        if (unlink(identity_path) != 0 && errno != ENOENT) return 1;
        FILE* af = fopen(apath, "wb");
        int64_t ah[2] = {n, M};
        if (!af) return 1;
        int ok = fwrite(ah, 8, 2, af) == 2 &&
            fwrite(topm, 4, (size_t)n * M, af) == (size_t)n * M &&
            fwrite(topd, 4, (size_t)n * M, af) == (size_t)n * M &&
            fwrite(sigmean, 4, dim, af) == (size_t)dim;
        if (fflush(af) || fsync(fileno(af))) ok = 0;
        if (fclose(af)) ok = 0;
        if (!ok) return 1;
        idf = fopen(identity_path, "wb");
        if (!idf) return 1;
        ok = fwrite(&identity, 8, 1, idf) == 1;
        if (fflush(idf) || fsync(fileno(idf))) ok = 0;
        if (fclose(idf)) ok = 0;
        if (!ok) return 1;
    }
    for (int64_t i = 0; i < n * M; i++)
        if (topm[i] < 0 || topm[i] >= K || !isfinite(topd[i])) return 1;
    /* scale : ~3 sigma de |x| moyen (demi-normale) par dim */
    float* scale = (float*)malloc((size_t)dim * 4);
    float qlevels = (float)((1 << (tqbits - 1)) - 1) + 0.5f;
    for (int d = 0; d < dim; d++) {
        float sig = sigmean[d] * 1.2533f;
        scale[d] = qlevels / (3.0f * sig + 1e-9f);
    }
    free(scale_acc); free(sigmean);

    /* --- comptage cellules avec spill --- */
    int64_t* cnt = (int64_t*)calloc((size_t)K + 1, 8);
    int64_t entries = 0;
    for (int64_t i = 0; i < n; i++) {
        float lim = (1.0f + eps) * (1.0f + eps) * topd[i * M];
        for (int j = 0; j < M; j++) {
            if (j > 0 && topd[i * M + j] > lim) break;
            cnt[topm[i * M + j]]++;
            entries++;
        }
    }
    if (cdim <= 0 || cdim > dim) cdim = dim;
    int code_b = cdim * tqbits / 8;
    int ent_b = 4 + code_b;
    uint64_t* offs = (uint64_t*)malloc(((size_t)K + 1) * 8);
    offs[0] = 0;
    for (int k = 0; k < K; k++)
        offs[k + 1] = offs[k] + (uint64_t)cnt[k] * ent_b;
    fprintf(stderr, "abuild: %lld entrees (x%.2f), blocks %.1f Go\n",
            (long long)entries, (double)entries / n,
            (double)offs[K] / 1e9);

    /* --- passe 2 : rotation + quantization + ecriture blocs --- */
    char p[1024];
    snprintf(p, sizeof(p), "%s/blocks.bin", out);
    int bfd = open(p, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (bfd < 0 || ftruncate(bfd, (off_t)offs[K]) != 0) {
        perror("blocks"); return 1;
    }
    uint64_t* cur = (uint64_t*)malloc((size_t)K * 8);
    memcpy(cur, offs, (size_t)K * 8);
    fseeko(bf, 8, SEEK_SET);
    t0 = omp_get_wtime();
    uint8_t* entbuf = (uint8_t*)malloc((size_t)CHUNK * M * ent_b);
    int64_t* entoff = (int64_t*)malloc((size_t)CHUNK * M * 8);
    AnchorWrite* writes = malloc((size_t)CHUNK * M * sizeof(*writes));
    const size_t write_cap = 1024 * 1024;
    uint8_t* write_buf = malloc(write_cap);
    if (!entbuf || !entoff || !writes || !write_buf) return 1;
    uint64_t write_calls = 0;
    for (int64_t off = 0; off < n; off += CHUNK) {
        int64_t c = n - off < CHUNK ? n - off : CHUNK;
        if (anchor_read_padded(bf,raw,c,input_dim,dim)) return 1;
        int64_t ne = 0;
        /* offsets sequentiels (ordre doc) + index premiere entree/doc */
        int64_t* first_ent = (int64_t*)malloc((size_t)c * 8);
        for (int64_t i = 0; i < c; i++) {
            int64_t g = off + i;
            first_ent[i] = ne;
            float lim = (1.0f + eps) * (1.0f + eps) * topd[g * M];
            for (int j = 0; j < M; j++) {
                if (j > 0 && topd[g * M + j] > lim) break;
                int32_t cell = topm[g * M + j];
                entoff[ne] = (int64_t)cur[cell];
                cur[cell] += ent_b;
                ne++;
            }
        }
        #pragma omp parallel
        {
            float* v = (float*)malloc((size_t)dim * 4);
            float* r = (float*)malloc((size_t)dim * 4);
            uint8_t* code = (uint8_t*)malloc(code_b);
            #pragma omp for schedule(dynamic, 64)
            for (int64_t i = 0; i < c; i++) {
                int64_t g = off + i;
                row_f16_to_unit(raw + (size_t)i * dim, v, dim);
                rot_seeded(v, r, sgn, dim);
                if (tqbits == 4) {
                    for (int d = 0; d < cdim; d += 2) {
                        int q0 = (int)lrintf(r[d] * scale[d]);
                        int q1 = (int)lrintf(r[d + 1] * scale[d + 1]);
                        if (q0 < -8) q0 = -8;
                        if (q0 > 7) q0 = 7;
                        if (q1 < -8) q1 = -8;
                        if (q1 > 7) q1 = 7;
                        code[d >> 1] = (uint8_t)((q0 & 15) | ((q1 & 15) << 4));
                    }
                } else if (tqbits == 2) {
                    for (int d = 0; d < cdim; d += 4) {
                        uint8_t b = 0;
                        for (int j = 0; j < 4; j++) {
                            int q = (int)lrintf(r[d + j] * scale[d + j]);
                            if (q < -2) q = -2;
                            if (q > 1) q = 1;
                            b |= (uint8_t)((q + 2) & 3) << (2 * j);
                        }
                        code[d >> 2] = b;
                    }
                } else { /* tq1 : bit de signe */
                    memset(code, 0, code_b);
                    for (int d = 0; d < cdim; d++)
                        if (r[d] >= 0) code[d >> 3] |= (uint8_t)(1 << (d & 7));
                }
                /* copie vers chaque entree du doc */
                float lim = (1.0f + eps) * (1.0f + eps) * topd[g * M];
                int64_t idx = first_ent[i];
                for (int j = 0; j < M; j++) {
                    if (j > 0 && topd[g * M + j] > lim) break;
                    uint8_t* e = entbuf + (idx + j) * ent_b;
                    uint32_t id32 = (uint32_t)g;
                    memcpy(e, &id32, 4);
                    memcpy(e + 4, code, (size_t)code_b);
                }
            }
            free(v); free(r); free(code);
        }
        for (int64_t e = 0; e < ne; e++) {
            writes[e].offset = entoff[e]; writes[e].entry = e;
        }
        qsort(writes, (size_t)ne, sizeof(*writes), anchor_write_cmp);
        for (int64_t e = 0; e < ne;) {
            uint64_t start = writes[e].offset;
            size_t used = 0;
            do {
                memcpy(write_buf + used, entbuf + writes[e].entry * ent_b, ent_b);
                used += ent_b; e++;
            } while (e < ne && writes[e].offset == start + used && used + ent_b <= write_cap);
            if (anchor_write_all(bfd, write_buf, used, start)) return 1;
            write_calls++;
        }
        free(first_ent);
        fprintf(stderr, "abuild: blocs %lld/%lld (%.0fs)\n",
                (long long)(off + c), (long long)n, omp_get_wtime() - t0);
    }
    if (fsync(bfd) != 0 || close(bfd) != 0) return 1;
    fprintf(stderr, "abuild: grouped writes %llu for %lld entries\n",
            (unsigned long long)write_calls, (long long)entries);
    free(entbuf); free(entoff); free(writes); free(write_buf);

    snprintf(p, sizeof(p), "%s/anchors.bin", out);
    if (anchor_save_file(p, A, (size_t)K * dim * 4)) return 1;
    snprintf(p, sizeof(p), "%s/offs.bin", out);
    if (anchor_save_file(p, offs, ((size_t)K + 1) * 8)) return 1;
    snprintf(p, sizeof(p), "%s/scale.bin", out);
    if (anchor_save_file(p, scale, (size_t)dim * 4)) return 1;
    snprintf(p, sizeof(p), "%s/meta.txt", out);
    char metadata[256];
    int meta_len = snprintf(metadata, sizeof(metadata), "%d %d %d %d %.4f %lld %llu %d %d\n",
        K, dim, M, tqbits, eps, (long long)n, (unsigned long long)seed, cdim, input_dim);
    if (meta_len < 0 || meta_len >= (int)sizeof(metadata) || anchor_save_file(p, metadata, meta_len)) return 1;
    int dfd = open(out, O_RDONLY | O_DIRECTORY);
    if (dfd < 0) return 1;
    int sync_rc = fsync(dfd); close(dfd);
    if (sync_rc) return 1;
    fprintf(stderr, "abuild: DONE\n");
    fclose(bf);
    free(A); free(A8); free(aids); free(raw);
    munmap(topm,assignment_bytes);munmap(topd,assignment_bytes);
    free(cnt); free(offs); free(cur); free(scale); free(sgn); free(rowbuf);
    return 0;
}

/* ================= BENCH (query + latences) ================= */
typedef struct { float s; uint32_t id; } ScId;
typedef struct { float s; const uint8_t* ent; } ScEnt;

static int scid_cmp(const void* a, const void* b) {
    float d = ((const ScId*)b)->s - ((const ScId*)a)->s;
    return d > 0 ? 1 : (d < 0 ? -1 : 0);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}


#include "anchor.h"
#include "anchor_live.h"
#include <xxhash.h>
struct AnchorIndex {
    AMeta meta;
    int a8_mode, bfd, basefd, residual_fd, residual_direct_fd;
    int residual_bits, residual_cdim;
    uint64_t residual_bytes;
    float *A, *scale;
    int8_t *A8, *sgn;
    uint64_t *offs, bytes;
    char directory[1024];
    AnchorLive* live;
    unsigned contexts;
};
struct AnchorQuery {
    const AnchorIndex* index;
    int nprobe, rerank, threads, ring_ok, failed, registered;
    int policy_min, policy_filtered_min, policy_used, policy_limited;
    float policy_gap;
    uint64_t policy_bytes;
    double policy_ms, policy_start, policy_observed_gap;
    const roaring_bitmap_t* allowed;
    int trace_n;
    uint32_t trace_ids[64];
    uint64_t trace_routed, trace_candidates;
    roaring_bitmap_t* trace_unique;
    uint64_t trace_entries;
    uint32_t* merge_ids;
    float* merge_scores;
    uint64_t block_limit;
    struct io_uring ring;
    S3Ctx s3;
    char *s3url, s3_blocks[1024], s3_base[1024];
    float *qn, *qr, *vf, *input;
    int8_t *q8, *c8, *qn8;
    ScId *cand, *heap, *lh_all, *fin_all, *route_all;
    ScEnt* ph_all;
    float* score_lut;
    size_t ph_stride, blk_cap;
    uint8_t *blk, *rows_all, *survivor_codes;
    uint8_t *res_buffer[2];
    size_t res_capacity;
    int res_width, res_overlap, res_direct;
    uint64_t *boff, *blen, *read_off, *read_len;
    uint8_t** read_dst;
};
/* Offline diagnostic only. Bitsets count unique watched document IDs even
   when an index stores multiple assignments. Never use traced timings as
   production latency measurements. Live overlays are deliberately excluded. */
static void anchor_trace_mark(AnchorQuery* c,uint32_t id,int candidate) {
    if(!c->trace_n)return;
    if(!candidate&&c->trace_unique){c->trace_entries++;roaring_bitmap_add(c->trace_unique,id);}
    int lo=0,hi=c->trace_n;
    while(lo<hi){int mid=(lo+hi)/2;if(c->trace_ids[mid]<id)lo=mid+1;else hi=mid;}
    if(lo<c->trace_n&&c->trace_ids[lo]==id){
        if(candidate)c->trace_candidates|=UINT64_C(1)<<lo;
        else c->trace_routed|=UINT64_C(1)<<lo;
    }
}
int anchor_query_trace(AnchorQuery* c,const uint32_t* ids,int n) {
    if(!c||c->failed||n<0||n>64||(n&&!ids)||
       (n&&(c->threads!=1||c->index->live||c->s3url)))return -1;
    for(int i=0;i<n;i++)if(ids[i]>=(uint64_t)c->index->meta.n||(i&&ids[i]<=ids[i-1]))return -1;
    if(n)memcpy(c->trace_ids,ids,(size_t)n*4);
    c->trace_n=n;c->trace_routed=0;c->trace_candidates=0;return 0;
}
int anchor_query_trace_stats(AnchorQuery* c,uint64_t out[3]) {
    if(!c||c->failed||!out)return -1;
    out[0]=c->trace_n;out[1]=__builtin_popcountll(c->trace_routed);
    out[2]=__builtin_popcountll(c->trace_candidates);return 0;
}
int anchor_query_unique(AnchorQuery* c,int enable) {
    if(!c||c->failed||(enable&&!c->trace_n))return -1;
    if(enable&&!c->trace_unique){c->trace_unique=roaring_bitmap_create();if(!c->trace_unique)return -1;}
    if(!enable&&c->trace_unique){roaring_bitmap_free(c->trace_unique);c->trace_unique=NULL;}
    c->trace_entries=0;return 0;
}
int anchor_query_unique_stats(AnchorQuery* c,uint64_t out[2]) {
    if(!c||c->failed||!out||!c->trace_unique)return -1;
    out[0]=c->trace_entries;out[1]=roaring_bitmap_get_cardinality(c->trace_unique);return 0;
}
void anchor_index_close(AnchorIndex* idx) {
    if (!idx) return;
    anchor_live_close(idx->live);
    if (idx->residual_fd >= 0) close(idx->residual_fd);
    if (idx->residual_direct_fd >= 0) close(idx->residual_direct_fd);
    if (idx->bfd >= 0) close(idx->bfd);
    if (idx->basefd >= 0) close(idx->basefd);
    free(idx->A); free(idx->A8); free(idx->offs); free(idx->scale); free(idx->sgn); free(idx);
}
int anchor_index_dim(const AnchorIndex* idx) { return idx ? idx->meta.input_dim : 0; }
uint64_t anchor_index_bytes(const AnchorIndex* idx) { return idx ? idx->bytes : 0; }
static int anchor_load_file(const char* dir, const char* name, void* dst, size_t bytes) {
    char path[1024]; snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    int ok = fread(dst, 1, bytes, f) == bytes && fgetc(f) == EOF && !ferror(f);
    fclose(f); return ok ? 0 : -1;
}
AnchorIndex* anchor_index_open(const char* dir, const char* base_path, int a8_mode) {
    if (!dir) return NULL;
    AnchorIndex* idx = calloc(1, sizeof(*idx));
    if (!idx) return NULL;
    idx->bfd = idx->basefd = idx->residual_fd = idx->residual_direct_fd = -1; idx->a8_mode = !!a8_mode;
    if (snprintf(idx->directory,sizeof(idx->directory),"%s",dir)>=(int)sizeof(idx->directory)) { free(idx); return NULL; }
    FILE* f = NULL;
    if (meta_load(dir, &idx->meta)) goto fail;
    int dim = idx->meta.dim, K = idx->meta.K, cd = idx->meta.cdim;
    int bits = idx->meta.tqbits;
    if (K < 1 || dim < 8 || dim > 65536 || (dim & (dim-1)) || cd < 8 || cd > dim || cd % 8 ||
        (bits != 1 && bits != 2 && bits != 4) || idx->meta.n < K) goto fail;
    size_t total = (size_t)K * dim;
    char path[1024]; snprintf(path, sizeof(path), "%s/anchors.bin", dir);
    if (a8_mode) {
        const size_t chunk = 262144;
        float* scratch = malloc(chunk * sizeof(float));
        idx->A8 = malloc(total);
        f = fopen(path, "rb");
        if (!f || !scratch || !idx->A8) { free(scratch); goto fail; }
        float max = 0; int bad = 0;
        for (size_t off = 0; off < total && !bad; off += chunk) {
            size_t n = total-off < chunk ? total-off : chunk;
            if (fread(scratch, 4, n, f) != n) { bad = 1; break; }
            for (size_t i = 0; i < n; i++) {
                if (!isfinite(scratch[i])) { bad = 1; break; }
                if (fabsf(scratch[i]) > max) max = fabsf(scratch[i]);
            }
        }
        if (fseeko(f, 0, SEEK_SET)) bad = 1;
        float scale = 127.0f / (max + 1e-9f);
        for (size_t off = 0; off < total && !bad; off += chunk) {
            size_t n = total-off < chunk ? total-off : chunk;
            if (fread(scratch, 4, n, f) != n) { bad = 1; break; }
            for (size_t i = 0; i < n; i++) idx->A8[off+i] = (int8_t)lrintf(scratch[i] * scale);
        }
        free(scratch); fclose(f); f = NULL;
        if (bad) goto fail;
    } else {
        idx->A = malloc(total * 4);
        if (!idx->A || anchor_load_file(dir, "anchors.bin", idx->A, total*4)) goto fail;
        for (size_t i = 0; i < total; i++) if (!isfinite(idx->A[i])) goto fail;
    }
    idx->offs = malloc(((size_t)K+1)*8); idx->scale = malloc((size_t)dim*4); idx->sgn = malloc(dim);
    if (!idx->offs || !idx->scale || !idx->sgn ||
        anchor_load_file(dir, "offs.bin", idx->offs, ((size_t)K+1)*8) ||
        anchor_load_file(dir, "scale.bin", idx->scale, (size_t)dim*4)) goto fail;
    int ent_b = 4 + cd*bits/8;
    if (idx->offs[0] != 0) goto fail;
    for (int k = 0; k < K; k++) if (idx->offs[k+1] < idx->offs[k] ||
        (idx->offs[k+1]-idx->offs[k]) % ent_b || idx->offs[k+1]-idx->offs[k] > INT_MAX) goto fail;
    for (int d = 0; d < dim; d++) if (!isfinite(idx->scale[d]) || idx->scale[d] <= 0) goto fail;
    make_signs(idx->sgn, dim, idx->meta.seed ^ 0x51CA);
    snprintf(path, sizeof(path), "%s/blocks.bin", dir);
    idx->bfd = open(path, O_RDONLY);
    if (base_path) idx->basefd = open(base_path, O_RDONLY);
    idx->bytes = sizeof(*idx) + total*(a8_mode ? 1 : 4) + ((uint64_t)K+1)*8 + dim*5;
    return idx;
fail:
    if (f) fclose(f);
    anchor_index_close(idx); return NULL;
}

int anchor_index_enable_live(AnchorIndex* idx,const char* dir) {
    if(!idx||!dir||idx->live||__atomic_load_n(&idx->contexts,__ATOMIC_SEQ_CST))return -1;
    char path[1200];snprintf(path,sizeof(path),"%s/anchors.bin",idx->directory);
    FILE* f=fopen(path,"rb");if(!f)return -1;
    XXH64_state_t* hash=XXH64_createState();
    uint8_t* buffer=malloc(1048576);
    if(!hash||!buffer){if(hash)XXH64_freeState(hash);free(buffer);fclose(f);return -1;}
    XXH64_reset(hash,0);
    size_t n;while((n=fread(buffer,1,1048576,f)))XXH64_update(hash,buffer,n);
    int failed=ferror(f);fclose(f);free(buffer);
    XXH64_update(hash,idx->offs,((size_t)idx->meta.K+1)*8);
    uint64_t config[]={idx->meta.seed,idx->meta.n,idx->meta.K,idx->meta.dim,idx->meta.M};
    XXH64_update(hash,config,sizeof(config));
    if(idx->meta.input_dim!=idx->meta.dim)XXH64_update(hash,&idx->meta.input_dim,sizeof(int));
    uint64_t fingerprint=XXH64_digest(hash);XXH64_freeState(hash);
    if(failed)return -1;
    idx->live=anchor_live_open(dir,fingerprint,idx->meta.n,idx->meta.dim,idx->meta.K,idx->meta.M);
    if(!idx->live)return -1;
    idx->bytes+=(uint64_t)idx->meta.K*16+(8192+65536)*sizeof(void*);
    return 0;
}
uint64_t anchor_index_count(AnchorIndex* idx) {
    if(!idx||anchor_live_read_lock(idx->live))return 0;
    uint64_t count=idx->meta.n+anchor_live_count(idx->live);
    anchor_live_read_unlock(idx->live);return count;
}
static int anchor_insert_impl(AnchorIndex* idx,const float* vector,const char* const* keys,int nkeys,uint32_t* id,const uint8_t* token,const uint8_t* digest,int update) {
    if(!idx||!idx->live||!vector||!id)return -1;
    if(token){int found=anchor_live_lookup(idx->live,token,digest,id);if(found)return found==1?0:found;}
    int dim=idx->meta.dim,M=idx->meta.M,input_dim=idx->meta.input_dim;
    float* unit=malloc((size_t)dim*4);int8_t* q8=malloc(dim);
    if(!unit||!q8){free(unit);free(q8);return -1;}
    double norm=0;for(int d=0;d<input_dim;d++) {
        if(!isfinite(vector[d])){free(unit);free(q8);return -1;}
        norm+=(double)vector[d]*vector[d];
    }
    if(norm==0){free(unit);free(q8);return -1;}
    double magnitude=sqrt(norm);
    float mx=0;for(int d=0;d<dim;d++){unit[d]=d<input_dim?(float)(vector[d]/magnitude):0;if(fabsf(unit[d])>mx)mx=fabsf(unit[d]);}
    for(int d=0;d<dim;d++)q8[d]=(int8_t)lrintf(unit[d]*(127.f/(mx+1e-9f)));
    uint32_t cells[4]={0};float scores[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY};
    for(int k=0;k<idx->meta.K;k++) {
        float score=idx->a8_mode?(float)doti8(q8,idx->A8+(size_t)k*dim,dim):dotf(unit,idx->A+(size_t)k*dim,dim);
        if(score>scores[M-1]){int j=M-1;while(j>0&&score>scores[j-1]){scores[j]=scores[j-1];cells[j]=cells[j-1];j--;}
            scores[j]=score;cells[j]=(uint32_t)k;}
    }
    int rc=update?anchor_live_update(idx->live,*id,unit,cells,keys,nkeys):token?anchor_live_append_once(idx->live,unit,cells,keys,nkeys,token,digest,id):anchor_live_append(idx->live,unit,cells,keys,nkeys,id);
    free(unit);free(q8);return rc;
}
int anchor_index_insert(AnchorIndex* idx,const float* v,const char* const* keys,int n,uint32_t* id) {
    return anchor_insert_impl(idx,v,keys,n,id,NULL,NULL,0);
}
int anchor_index_insert_once(AnchorIndex* idx,const float* v,const char* const* keys,int n,
    const uint8_t* token,const uint8_t* digest,uint32_t* id) {
    if(!token||!digest)return -1;
    return anchor_insert_impl(idx,v,keys,n,id,token,digest,0);
}
int anchor_index_insert_batch(AnchorIndex* idx,int n,const float* vectors,const char* const* keys,const int* counts,
    const uint8_t* const* tokens,const uint8_t* const* digests,uint32_t* ids,int* committed) {
    if(committed)*committed=0;
    if(!idx||!idx->live||n<1||n>256||!vectors||!ids||!committed)return -1;
    int dim=idx->meta.dim,M=idx->meta.M,input_dim=idx->meta.input_dim,rc=-1;
    float* units=malloc((size_t)n*dim*4);int8_t* q8=malloc(dim);uint32_t* cells=malloc((size_t)n*M*4);
    if(!units||!q8||!cells)goto done;
    /* Route before taking the journal write lock, so queries continue during
       the expensive scan of all anchors. Only publication needs exclusivity. */
    for(int row=0;row<n;row++){
        const float* v=vectors+(size_t)row*input_dim;float* unit=units+(size_t)row*dim;
        double norm=0;for(int d=0;d<input_dim;d++){if(!isfinite(v[d]))goto done;norm+=(double)v[d]*v[d];}
        if(norm<=0)goto done;
        double magnitude=sqrt(norm);float mx=0;
        for(int d=0;d<dim;d++){unit[d]=d<input_dim?(float)(v[d]/magnitude):0;mx=fmaxf(mx,fabsf(unit[d]));}
        for(int d=0;d<dim;d++)q8[d]=(int8_t)lrintf(unit[d]*(127.f/(mx+1e-9f)));
        float scores[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY};uint32_t* chosen=cells+(size_t)row*M;
        for(int k=0;k<idx->meta.K;k++){
            float score=idx->a8_mode?(float)doti8(q8,idx->A8+(size_t)k*dim,dim):dotf(unit,idx->A+(size_t)k*dim,dim);
            if(score>scores[M-1]){int j=M-1;while(j>0&&score>scores[j-1]){scores[j]=scores[j-1];chosen[j]=chosen[j-1];j--;}
                scores[j]=score;chosen[j]=(uint32_t)k;}
        }
    }
    rc=anchor_live_append_batch(idx->live,n,units,cells,keys,counts,tokens,digests,ids,committed);
done:
    free(units);free(q8);free(cells);return rc;
}
int anchor_index_update(AnchorIndex* idx,uint32_t id,const float* v,const char* const* keys,int n){return anchor_insert_impl(idx,v,keys,n,&id,NULL,NULL,1);}
int anchor_index_snapshot_live(AnchorIndex* idx,const char* out){return idx?anchor_live_snapshot(idx->live,out):-1;}
int anchor_index_delete(AnchorIndex* idx,uint32_t id){return idx?anchor_live_delete(idx->live,id):-1;}
int anchor_index_live_stats(AnchorIndex* idx,uint64_t values[5]) {
    if(!idx||!values||anchor_live_read_lock(idx->live))return -1;
    values[0]=idx->meta.n+anchor_live_count(idx->live);
    values[1]=anchor_live_deleted_count(idx->live);
    values[2]=anchor_live_maintenance_bytes(idx->live);
    values[3]=anchor_live_journal_bytes(idx->live);
    values[4]=anchor_live_request_count(idx->live);
    anchor_live_read_unlock(idx->live);return 0;
}
uint64_t anchor_index_maintenance_bytes(AnchorIndex* idx) {
    if(!idx||anchor_live_read_lock(idx->live))return UINT64_MAX;
    uint64_t n=anchor_live_maintenance_bytes(idx->live);anchor_live_read_unlock(idx->live);return n;
}
uint64_t anchor_index_deleted_count(AnchorIndex* idx) {
    if(!idx||anchor_live_read_lock(idx->live))return UINT64_MAX;
    uint64_t n=anchor_live_deleted_count(idx->live);anchor_live_read_unlock(idx->live);return n;
}
int anchor_index_add_tag(AnchorIndex* idx,const uint32_t* ids,int n,const char* const* keys,int nkeys) {
    return idx&&idx->live?anchor_live_add_tag(idx->live,ids,n,keys,nkeys):-1;
}
int anchor_index_set_tags(AnchorIndex* idx,uint32_t id,const char* const* keys,int nkeys) {
    return idx&&idx->live?anchor_live_set_tags(idx->live,id,keys,nkeys):-1;
}
int anchor_index_compact(AnchorIndex* idx,uint64_t* before,uint64_t* after) {
    return idx&&idx->live?anchor_live_compact(idx->live,before,after):-1;
}
int anchor_index_tag_keys(AnchorIndex* idx,char* out,int cap) {
    return idx&&idx->live?anchor_live_keys(idx->live,out,cap):-1;
}

void anchor_query_close(AnchorQuery* ctx) {
    if(ctx&&ctx->trace_unique)roaring_bitmap_free(ctx->trace_unique);
    if (!ctx) return;
    if(ctx->registered)__atomic_sub_fetch((unsigned*)&ctx->index->contexts,1,__ATOMIC_SEQ_CST);
    free(ctx->merge_ids);free(ctx->merge_scores);free(ctx->score_lut);
    /* Tear down the ring before releasing any possible in-flight IO buffers. */
    if (ctx->ring_ok) io_uring_queue_exit(&ctx->ring);
    s3_close(&ctx->s3);
    free(ctx->s3url); free(ctx->qn); free(ctx->qr); free(ctx->vf);
    free(ctx->input);free(ctx->res_buffer[0]);free(ctx->res_buffer[1]);
    free(ctx->q8); free(ctx->c8); free(ctx->qn8); free(ctx->cand); free(ctx->heap);
    free(ctx->lh_all); free(ctx->fin_all); free(ctx->route_all); free(ctx->ph_all);
    free(ctx->survivor_codes); free(ctx->blk); free(ctx->rows_all); free(ctx->boff); free(ctx->blen);
    free(ctx->read_off); free(ctx->read_len); free(ctx->read_dst); free(ctx);
}
AnchorQuery* anchor_query_create(const AnchorIndex* idx, int np, int rr, int threads,
                                uint64_t memory, const char* url, int hedge) {
    if (idx && idx->residual_fd >= 0 && (threads != 1 || url)) return NULL;
    if (!idx || np < 1 || np > idx->meta.K || rr < 1 || rr > 1000000 ||
        threads < 1 || threads > 256 || (!url && ((idx->bfd < 0 && idx->residual_fd < 0) || idx->basefd < 0))) return NULL;
    AnchorQuery* ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->index = idx; ctx->nprobe = np; ctx->rerank = rr; ctx->threads = threads;
    ctx->res_width=64;ctx->res_overlap=1;ctx->res_direct=idx->residual_direct_fd>=0;
    ctx->ph_stride = rr*4 > 16384 ? (size_t)rr*4 : 16384;
    uint64_t owned = idx->bytes + sizeof(*ctx);
    int dim = idx->meta.dim, reads = np > rr ? np : rr;
#define ANC_ALLOC(field, count) do { \
    size_t bytes = (size_t)(count) * sizeof(*ctx->field); \
    if (memory && (owned > memory || bytes > memory-owned)) goto fail; \
    ctx->field = calloc((size_t)(count), sizeof(*ctx->field)); \
    if (!ctx->field) goto fail; \
    owned += bytes; \
} while (0)
    if (idx->residual_fd < 0 && idx->meta.tqbits != 1)
        ANC_ALLOC(score_lut, (size_t)idx->meta.cdim*idx->meta.tqbits/8*256);
    ANC_ALLOC(input, dim); ANC_ALLOC(qn, dim); ANC_ALLOC(qr, dim); ANC_ALLOC(vf, dim);
    ANC_ALLOC(q8, dim); ANC_ALLOC(c8, dim); ANC_ALLOC(qn8, dim);
    ANC_ALLOC(merge_ids, rr); ANC_ALLOC(merge_scores, rr);
    ANC_ALLOC(cand, np); ANC_ALLOC(heap, idx->residual_fd >= 0 ? (size_t)rr*idx->meta.M : (size_t)rr+1); ANC_ALLOC(fin_all, rr);
    if(idx->residual_fd<0){
        ANC_ALLOC(lh_all, (size_t)threads*rr); ANC_ALLOC(route_all, (size_t)threads*np);
        ANC_ALLOC(ph_all, (size_t)threads*ctx->ph_stride);
    } ANC_ALLOC(rows_all, (size_t)rr*dim*2+16);
    if (threads == 1 && idx->residual_fd<0) ANC_ALLOC(survivor_codes,
        ctx->ph_stride*(4 + idx->meta.cdim*idx->meta.tqbits/8));
    ANC_ALLOC(boff, np); ANC_ALLOC(blen, np); ANC_ALLOC(read_off, reads);
    ANC_ALLOC(read_len, reads); ANC_ALLOC(read_dst, reads);
#undef ANC_ALLOC
    if(idx->residual_fd>=0&&idx->live){
        uint64_t scratch=(uint64_t)rr*(idx->meta.M*16+4)+84*4096+4096;
        if(memory&&(owned>memory||scratch>memory-owned))goto fail;
        owned+=scratch;
    }
    ctx->block_limit = memory ? memory-owned : SIZE_MAX;
    if (url) {
        ctx->s3url = strdup(url);
        if (!ctx->s3url || s3_init(&ctx->s3)) goto fail;
        ctx->s3.hedge_ms = hedge;
        snprintf(ctx->s3_blocks, sizeof(ctx->s3_blocks), "%s/blocks.bin", url);
        snprintf(ctx->s3_base, sizeof(ctx->s3_base), "%s/base.f16bin", url);
    } else {
        if (io_uring_queue_init(1024, &ctx->ring, 0)) goto fail;
        ctx->ring_ok = 1;
    }
    size_t initial = (size_t)np*(idx->offs[idx->meta.K]/idx->meta.K*2+4096);
    if (threads == 1 && initial > 128u*1024*1024) initial = 128u*1024*1024;
    if (initial > ctx->block_limit) initial = ctx->block_limit;
    if(idx->residual_fd>=0)initial=0;
    ctx->blk = calloc(initial ? initial : 1, 1); ctx->blk_cap = initial;
    if (!ctx->blk) goto fail;
    ctx->registered=1;__atomic_add_fetch((unsigned*)&idx->contexts,1,__ATOMIC_SEQ_CST);
    return ctx;
fail:
    anchor_query_close(ctx); return NULL;
}
/* Quiescent-context configuration. Limits are soft, checked between IO waves. */
int anchor_query_adapt(AnchorQuery* c,int minimum,int filtered_minimum,float gap,
                       uint64_t bytes,double milliseconds) {
    if(!c||c->failed||c->threads!=1||minimum<1||minimum>c->nprobe||
       filtered_minimum<minimum||filtered_minimum>c->nprobe||!isfinite(gap)||gap<0||
       !isfinite(milliseconds)||milliseconds<0)return -1;
    c->policy_min=minimum;c->policy_filtered_min=filtered_minimum;c->policy_gap=gap;
    c->policy_bytes=bytes;c->policy_ms=milliseconds;return 0;
}
int anchor_query_adapt_stats(AnchorQuery* c,double out[3]) {
    if(!c||!out)return -1;
    out[0]=c->policy_used;out[1]=c->policy_limited;out[2]=c->policy_observed_gap;return 0;
}
static int anchor_adapt_probes(AnchorQuery* c) {
    if(!c->policy_min){c->policy_used=c->nprobe;return c->nprobe;}
    int n=c->allowed?c->policy_filtered_min:c->policy_min;
    float denominator=fmaxf(fabsf(c->cand[0].s),1e-9f);
    c->policy_observed_gap=(c->cand[0].s-c->cand[n-1].s)/denominator;
    while(n<c->nprobe&&(c->cand[0].s-c->cand[n-1].s)/denominator<c->policy_gap)
        n=n>c->nprobe/2?c->nprobe:n*2;
    uint64_t used=0;int end=0;
    for(;end<n;end++) {
        uint32_t cell=c->cand[end].id;
        uint64_t size=c->index->offs[cell+1]-c->index->offs[cell];
        if(c->index->residual_fd>=0)size=size/(4+c->index->meta.cdim*c->index->meta.tqbits/8)*72;
        if(c->policy_bytes&&end&&size>c->policy_bytes-used){c->policy_limited=1;break;}
        used+=size;
        if(c->policy_bytes&&used>=c->policy_bytes){end++;c->policy_limited=end<n||used>c->policy_bytes;break;}
    }
    c->policy_used=end;return end;
}
static int anchor_adapt_expired(AnchorQuery* c,int completed) {
    if(c->policy_min&&completed&&c->policy_ms&&now_ms()-c->policy_start>=c->policy_ms) {
        c->policy_limited=1;c->policy_used=completed;return 1;
    }
    return 0;
}
#include "anchor_residual.inc"
#include "anchor_residual_build.inc"

static int anchor_search_impl(AnchorQuery* ctx, const float* vector, int top_k,
                             uint32_t* ids, float* scores, AnchorStats* stats) {
    const AnchorIndex* idx = ctx->index;
    int dim = idx->meta.dim, K = idx->meta.K, cdim = idx->meta.cdim;
    int has_deleted=anchor_live_deleted_count(idx->live)!=0;
    int ent_b = 4 + cdim*idx->meta.tqbits/8;
    int nprobe = ctx->nprobe, rerank = ctx->rerank;
        double T0 = now_ms();
        /* normalise + descente ancres */
        memcpy(ctx->qn, vector, (size_t)dim * 4);
        float n2 = 0;
        for (int d = 0; d < dim; d++) n2 += ctx->qn[d] * ctx->qn[d];
        float inv = 1.0f / (sqrtf(n2) + 1e-9f);
        for (int d = 0; d < dim; d++) ctx->qn[d] *= inv;
        for (int c = 0; c < nprobe; c++) ctx->cand[c] = (ScId){-2e9f, UINT32_MAX};

        {
            float qmx = 0.0f;
            for (int d = 0; d < dim; d++) if (fabsf(ctx->qn[d]) > qmx) qmx = fabsf(ctx->qn[d]);
            float qq = 127.0f / (qmx + 1e-9f);
            for (int d = 0; d < dim; d++) ctx->qn8[d] = (int8_t)lrintf(ctx->qn[d] * qq);
        }
        #pragma omp parallel num_threads(ctx->threads)
        {
            ScId* loc = ctx->route_all + (size_t)omp_get_thread_num() * nprobe;
            for (int c = 0; c < nprobe; c++) loc[c] = (ScId){-2e9f, UINT32_MAX};
            #pragma omp for schedule(static)
            for (int k = 0; k < K; k++) {
                float s = idx->a8_mode ? (float)doti8(ctx->qn8, idx->A8 + (size_t)k * dim, dim)
                                  : dotf(ctx->qn, idx->A + (size_t)k * dim, dim);
                if (s > loc[nprobe - 1].s) {
                    int j = nprobe - 1;
                    while (j > 0 && s > loc[j - 1].s) {
                        loc[j] = loc[j - 1]; j--;
                    }
                    loc[j].s = s; loc[j].id = (uint32_t)k;
                }
            }
            #pragma omp critical
            for (int c = 0; c < nprobe; c++) {
                float s = loc[c].s;
                if (s > ctx->cand[nprobe - 1].s) {
                    int j = nprobe - 1;
                    while (j > 0 && s > ctx->cand[j - 1].s) {
                        ctx->cand[j] = ctx->cand[j - 1]; j--;
                    }
                    ctx->cand[j].s = s; ctx->cand[j].id = loc[c].id;
                }
            }

        }

        nprobe=anchor_adapt_probes(ctx);
        double T1 = now_ms();
        /* scoring TQ : rotation requete + int8, top-rerank */
        rot_seeded(ctx->qn, ctx->qr, idx->sgn, dim);
        /* Original vectors are normalized: cosine ranking equals squared-L2
           ranking. Dequantize each code coordinate and retain its norm term;
           rounding changes vector norms, so a plain dot product is biased. */
        if (idx->meta.tqbits != 1) {
            int bits=idx->meta.tqbits, per_byte=8/bits;
            for (int byte=0;byte<cdim*bits/8;byte++) for (int value=0;value<256;value++) {
                float score=0;
                for (int j=0;j<per_byte;j++) {
                    int quant=(value>>(j*bits))&((1<<bits)-1);
                    quant=bits==2?quant-2:(quant>=8?quant-16:quant);
                    int d=byte*per_byte+j;
                    float decoded=quant/idx->scale[d];
                    score+=ctx->qr[d]*decoded-.5f*decoded*decoded;
                }
                if (!isfinite(score)) return -1;
                ctx->score_lut[(size_t)byte*256+value]=score;
            }
        }
        float qmax = 0;
        for (int d = 0; d < dim; d++) {
            float v = fabsf(ctx->qr[d]);
            if (v > qmax) qmax = v;
        }
        for (int d = 0; d < dim; d++) {
            int q = (int)lrintf(ctx->qr[d] / (qmax + 1e-9f) * 127.0f);
            ctx->q8[d] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
        }
        /* requete reordonnee selon le mode : tq4 = 2 flux pair/impair,
           tq2 = 4 flux (dims mod 4), tq1 = q8 direct.                   */
        int8_t* qlo = ctx->c8;               /* reutilise le scratch */
        int8_t* qhi = ctx->c8 + dim / 2;
        int8_t* qs2[4] = {ctx->c8, ctx->c8 + dim / 4, ctx->c8 + dim / 2,
                          ctx->c8 + 3 * (dim / 4)};
        if (idx->meta.tqbits == 4) {
            for (int d = 0; d < dim; d += 2) {
                qlo[d >> 1] = ctx->q8[d];
                qhi[d >> 1] = ctx->q8[d + 1];
            }
        } else if (idx->meta.tqbits == 2) {
            for (int d = 0; d < dim; d++)
                qs2[d & 3][d >> 2] = ctx->q8[d];
        } else { /* tq1 : 8 flux de cdim/8 octets (flux b = dims = b mod 8) */
            for (int d = 0; d < cdim; d++)
                ctx->c8[(d & 7) * (cdim / 8) + (d >> 3)] = ctx->q8[d];
        }
        /* SCORING PROGRESSIF : pre-score sur les DIM_PRE premieres dims
           tournees (la FWHT egalise l energie -> le prefixe porte
           DIM_PRE/dim de la variance), preselection top-PRE_KEEP par
           thread, puis score COMPLET des seuls survivants. CPU ~/4.    */
        /* min-tas binaire sur s : remplacement du minimum en O(log n)
           (l insertion decalee coutait O(n) — mur a 16k en mono-thread) */
        #define PH_SIFT(ph, n) do {                                      \
            int _i = 0;                                                  \
            for (;;) {                                                   \
                int _l = 2 * _i + 1, _r = _l + 1, _m = _i;               \
                if (_l < (n) && (ph)[_l].s < (ph)[_m].s) _m = _l;        \
                if (_r < (n) && (ph)[_r].s < (ph)[_m].s) _m = _r;        \
                if (_m == _i) break;                                     \
                ScEnt _t = (ph)[_i]; (ph)[_i] = (ph)[_m];                \
                (ph)[_m] = _t; _i = _m;                                  \
            }                                                            \
        } while (0)

        int hn = 0;
        int64_t docs_seen = 0;
        uint64_t need = 0;
        double T2;
        if (ctx->threads == 1) {
            const int DIM_PRE = cdim >= 512 ? 256 : cdim;
            const int PRE_KEEP = (int)ctx->ph_stride;
            ScEnt* ph = ctx->ph_all;
            int pn = 0, first = 0;
            double io_elapsed = 0;
            uint64_t target = 128u*1024*1024;
            if (target > ctx->block_limit) target = ctx->block_limit;
            while (first < nprobe) {
                if(anchor_adapt_expired(ctx,first)){nprobe=first;break;}
                uint64_t bytes = 0;
                int end = first;
                while (end < nprobe) {
                    if(ctx->policy_min&&end-first>=32)break;
                    uint32_t k = ctx->cand[end].id;
                    uint64_t len = idx->offs[k+1] - idx->offs[k];
                    if (len > ctx->block_limit) return -2;
                    if (end > first && bytes + len > target) break;
                    bytes += len; end++;
                    if (bytes >= target) break;
                }
                if (bytes > ctx->blk_cap) {
                    free(ctx->blk); ctx->blk = NULL; ctx->blk_cap = 0;
                    ctx->blk = malloc(bytes ? bytes : 1);
                    if (!ctx->blk) return -2;
                    ctx->blk_cap = bytes;
                }
                uint64_t offset = 0;
                for (int c = first; c < end; c++) {
                    int local = c-first; uint32_t k = ctx->cand[c].id;
                    ctx->boff[local] = offset;
                    ctx->blen[local] = idx->offs[k+1] - idx->offs[k];
                    ctx->read_off[local] = idx->offs[k];
                    ctx->read_dst[local] = ctx->blk + offset;
                    offset += ctx->blen[local];
                }
                double started = now_ms();
                if (ctx->s3url) {
                    if (s3_wave(&ctx->s3, ctx->s3_blocks, ctx->read_off, ctx->blen,
                                ctx->read_dst, end-first) != end-first) return -1;
                } else if (anchor_reads(&ctx->ring, idx->bfd, ctx->read_off, ctx->blen,
                                         ctx->read_dst, end-first)) return -1;
                io_elapsed += now_ms()-started;
                need += bytes;
                for (int c = 0; c < end-first; c++) {
                    const uint8_t* base = ctx->blk + ctx->boff[c];
                    int64_t ne = ctx->blen[c]/ent_b;
                    docs_seen += ne;
                    for (int64_t e = 0; e < ne; e++) {
                        const uint8_t* ent = base + e*ent_b;
                        uint32_t doc;memcpy(&doc,ent,4);
                        if((has_deleted&&anchor_live_is_deleted(idx->live,doc))||anchor_live_is_overridden(idx->live,doc)||(ctx->allowed&&!roaring_bitmap_contains(ctx->allowed,doc)))continue;
                        anchor_trace_mark(ctx,doc,0);
                        const uint8_t* code = ent+4;
                        float score = idx->meta.tqbits != 1
                            ? score_tq_l2(code,ctx->score_lut,DIM_PRE*idx->meta.tqbits/8)
                            : score_tq1(code,ctx->c8,cdim/8,DIM_PRE);
                        if (pn < PRE_KEEP) {
                            int i = pn++;
                            uint8_t* saved = ctx->survivor_codes + (size_t)i*ent_b;
                            memcpy(saved,ent,ent_b);
                            ph[i] = (ScEnt){score,saved};
                            while(i>0) {
                                int parent=(i-1)/2;
                                if(ph[parent].s<=ph[i].s)break;
                                ScEnt swap=ph[parent];ph[parent]=ph[i];ph[i]=swap;i=parent;
                            }
                        } else if (score > ph[0].s) {
                            memcpy((void*)ph[0].ent,ent,ent_b);
                            ph[0].s=score;
                            PH_SIFT(ph,PRE_KEEP);
                        }
                    }
                }
                first=end;
            }
            ScId* lh = ctx->heap;
            int ln=0;
            for (int e = 0; e < pn; e++) {
                const uint8_t* ent = ph[e].ent;
                const uint8_t* code = ent + 4;
                float s = idx->meta.tqbits != 1
                    ? score_tq_l2(code,ctx->score_lut,cdim*idx->meta.tqbits/8)
                    : (float)score_tq1(code,ctx->c8,cdim/8,cdim);
                if (ln < rerank) {
                    lh[ln].s = s;
                    memcpy(&lh[ln].id, ent, 4);
                    ln++;
                    if (ln == rerank)
                        qsort(lh, ln, sizeof(ScId), scid_cmp);
                } else if (s > lh[rerank - 1].s) {
                    int j = rerank - 1;
                    while (j > 0 && s > lh[j - 1].s) {
                        lh[j] = lh[j - 1]; j--;
                    }
                    lh[j].s = s;
                    memcpy(&lh[j].id, ent, 4);
                }
            }
            if (ln < rerank) qsort(lh, ln, sizeof(ScId), scid_cmp);

            hn=ln;
            T2=T1+io_elapsed;
        } else {

        /* vague io_uring : nprobe blocs */
        need = 0;
        for (int c = 0; c < nprobe; c++)
            need += idx->offs[ctx->cand[c].id + 1] - idx->offs[ctx->cand[c].id];
        if (need > ctx->block_limit) return -2;
        if (need > ctx->blk_cap) {
            free(ctx->blk); ctx->blk = NULL; ctx->blk_cap = 0;
            uint8_t* nb = (uint8_t*)malloc(need);
            if (!nb) { fprintf(stderr, "OOM blocs req\n"); return -1; }
            ctx->blk = nb; ctx->blk_cap = need;
        }
        uint64_t bo = 0;
        if (ctx->s3url) {
            uint64_t* soff = ctx->read_off;
            uint8_t** sdst = ctx->read_dst;
            for (int c = 0; c < nprobe; c++) {
                int k = (int)ctx->cand[c].id;
                ctx->boff[c] = bo;
                ctx->blen[c] = idx->offs[k + 1] - idx->offs[k];
                soff[c] = idx->offs[k];
                sdst[c] = ctx->blk + bo;
                bo += ctx->blen[c];
            }
            int ok = s3_wave(&ctx->s3, ctx->s3_blocks, soff, ctx->blen, sdst, nprobe);


            if (ok != nprobe) { fprintf(stderr, "S3: incomplete block wave\n"); return -1; }

        } else {
        uint64_t* roff = ctx->read_off;
        uint8_t** rdst = ctx->read_dst;
        for (int c = 0; c < nprobe; c++) {
            int k = (int)ctx->cand[c].id;
            ctx->boff[c] = bo; ctx->blen[c] = idx->offs[k + 1] - idx->offs[k];
            roff[c] = idx->offs[k]; rdst[c] = ctx->blk + bo; bo += ctx->blen[c];
        }
        if (anchor_reads(&ctx->ring, idx->bfd, roff, ctx->blen, rdst, nprobe)) return -1;
        }
        T2 = now_ms();
        const int DIM_PRE = (cdim >= 512) ? 256 : cdim;


        #pragma omp parallel num_threads(ctx->threads) reduction(+ : docs_seen)
        {
            int PRE_KEEP = 16384 / omp_get_num_threads();
            if (PRE_KEEP < rerank * 4) PRE_KEEP = rerank * 4;
            /* scratch persistant par thread (pas de malloc par requete) */
            ScEnt* ph = ctx->ph_all + (size_t)omp_get_thread_num() * ctx->ph_stride;
            int pn = 0;
            #pragma omp for schedule(dynamic, 1)
            for (int c = 0; c < nprobe; c++) {
                const uint8_t* base = ctx->blk + ctx->boff[c];
                int64_t ne = (int64_t)(ctx->blen[c] / ent_b);
                docs_seen += ne;
                for (int64_t e = 0; e < ne; e++) {
                    const uint8_t* ent = base + e * ent_b;
                    uint32_t doc;memcpy(&doc,ent,4);
                    if((has_deleted&&anchor_live_is_deleted(idx->live,doc))||anchor_live_is_overridden(idx->live,doc)||(ctx->allowed&&!roaring_bitmap_contains(ctx->allowed,doc)))continue;
                    const uint8_t* code = ent + 4;
                    float s = idx->meta.tqbits != 1
                    ? score_tq_l2(code,ctx->score_lut,DIM_PRE*idx->meta.tqbits/8)
                    : (float)score_tq1(code,ctx->c8,cdim/8,DIM_PRE);
                    if (pn < PRE_KEEP) {
                        /* construction : sift-up */
                        int i2 = pn++;
                        ph[i2].s = s; ph[i2].ent = ent;
                        while (i2 > 0) {
                            int p2 = (i2 - 1) / 2;
                            if (ph[p2].s <= ph[i2].s) break;
                            ScEnt t = ph[p2]; ph[p2] = ph[i2];
                            ph[i2] = t; i2 = p2;
                        }
                    } else if (s > ph[0].s) {
                        ph[0].s = s; ph[0].ent = ent;
                        PH_SIFT(ph, PRE_KEEP);
                    }
                }
            }
            /* score complet des survivants locaux -> top-rerank local */
            ScId* lh = ctx->lh_all + (size_t)omp_get_thread_num() * rerank;
            int ln = 0;
            for (int e = 0; e < pn; e++) {
                const uint8_t* ent = ph[e].ent;
                const uint8_t* code = ent + 4;
                float s = idx->meta.tqbits != 1
                    ? score_tq_l2(code,ctx->score_lut,cdim*idx->meta.tqbits/8)
                    : (float)score_tq1(code,ctx->c8,cdim/8,cdim);
                if (ln < rerank) {
                    lh[ln].s = s;
                    memcpy(&lh[ln].id, ent, 4);
                    ln++;
                    if (ln == rerank)
                        qsort(lh, ln, sizeof(ScId), scid_cmp);
                } else if (s > lh[rerank - 1].s) {
                    int j = rerank - 1;
                    while (j > 0 && s > lh[j - 1].s) {
                        lh[j] = lh[j - 1]; j--;
                    }
                    lh[j].s = s;
                    memcpy(&lh[j].id, ent, 4);
                }
            }
            if (ln < rerank) qsort(lh, ln, sizeof(ScId), scid_cmp);
            #pragma omp critical
            for (int e = 0; e < ln; e++) {
                float s = lh[e].s;
                if (hn < rerank) {
                    ctx->heap[hn++] = lh[e];
                    if (hn == rerank)
                        qsort(ctx->heap, hn, sizeof(ScId), scid_cmp);
                } else if (s > ctx->heap[rerank - 1].s) {
                    int j = rerank - 1;
                    while (j > 0 && s > ctx->heap[j - 1].s) {
                        ctx->heap[j] = ctx->heap[j - 1]; j--;
                    }
                    ctx->heap[j] = lh[e];
                } else break; /* lh trie : plus rien a inserer */
            }
        }
        if (hn < rerank) qsort(ctx->heap, hn, sizeof(ScId), scid_cmp);
        stats->entries = docs_seen; stats->bytes = need;

        }
        stats->entries=docs_seen; stats->bytes=need;
        double T3=now_ms();

        /* rerank exact : pread f16 */
        /* rerank exact en UNE vague io_uring (les preads sequentiels
           coutaient 12-13 ms pour 300 lignes ; en vague ~2-3 ms) */
        int nr = hn < rerank ? hn : rerank;
        ScId* fin = ctx->fin_all;
        int nf = 0;
        for (int e = 0; e < nr; e++) {
            uint32_t id = ctx->heap[e].id;
            if ((uint64_t)id >= (uint64_t)idx->meta.n) return -1;
            int dup = 0;
            for (int x = 0; x < nf; x++)
                if (fin[x].id == id) { dup = 1; break; }
            if (!dup) {fin[nf++].id = id;anchor_trace_mark(ctx,id,1);}
        }
        uint8_t* rows = ctx->rows_all;
        if (ctx->s3url) {
            uint64_t* roff = ctx->read_off;
            uint64_t* rlen = ctx->read_len;
            uint8_t** rdst = ctx->read_dst;
            for (int e = 0; e < nf; e++) {
                roff[e] = 8 + (uint64_t)fin[e].id * idx->meta.input_dim * 2;
                rlen[e] = (uint64_t)idx->meta.input_dim * 2;
                rdst[e] = rows + (size_t)e * dim * 2;
            }
            if (s3_wave(&ctx->s3, ctx->s3_base, roff, rlen, rdst, nf) != nf) return -1;


        } else {
            uint64_t *roff = ctx->read_off, *rlen = ctx->read_len;
            uint8_t** rdst = ctx->read_dst;
            for (int e = 0; e < nf; e++) {
                roff[e] = 8 + (uint64_t)fin[e].id * idx->meta.input_dim * 2;
                rlen[e] = (uint64_t)idx->meta.input_dim * 2;
                rdst[e] = rows + (size_t)e * dim * 2;
            }
            if (anchor_reads(&ctx->ring, idx->basefd, roff, rlen, rdst, nf)) return -1;
        }
        for (int e = 0; e < nf; e++) {
            row_f16_padded((const uint16_t*)(rows + (size_t)e * dim * 2),
                            ctx->vf, idx->meta.input_dim, dim);
            fin[e].s = dotf(ctx->qn, ctx->vf, dim);
        }
        qsort(fin, nf, sizeof(ScId), scid_cmp);
        if(idx->live) {
            for(int i=0;i<nf;i++){ctx->merge_ids[i]=fin[i].id;ctx->merge_scores[i]=fin[i].s;}
            uint32_t* cells=(uint32_t*)ctx->read_off;
            for(int i=0;i<nprobe;i++)cells[i]=ctx->cand[i].id;
            nf=anchor_live_search(idx->live,cells,nprobe,ctx->qn,ctx->allowed,
                                 ctx->merge_ids,ctx->merge_scores,nf,rerank,&stats->entries,&stats->bytes);
            if(nf<0)return -1;
            for(int i=0;i<nf;i++)fin[i]=(ScId){ctx->merge_scores[i],ctx->merge_ids[i]};
        }

        int count = top_k < nf ? top_k : nf;
        for (int e = 0; e < top_k; e++) {
            ids[e] = e < nf ? fin[e].id : UINT32_MAX;
            if (scores) scores[e] = e < nf ? fin[e].s : -INFINITY;
        }
        double T4 = now_ms();
        stats->anchor_ms = T1 - T0; stats->io_ms = T2 - T1;
        stats->score_ms = T3 - T2; stats->rerank_ms = T4 - T3;
        stats->total_ms = T4 - T0;
        return count;
}


static void anchor_offer(ScId* hits,int* count,int capacity,uint32_t id,float score) {
    if(*count==capacity&&score<=hits[capacity-1].s)return;
    int i=*count<capacity?(*count)++:capacity-1;
    while(i>0&&score>hits[i-1].s){hits[i]=hits[i-1];i--;}
    hits[i]=(ScId){score,id};
}
static int anchor_exact_filter(AnchorQuery* ctx,const float* vector,int top_k,uint32_t* ids,float* scores,AnchorStats* stats) {
    const AnchorIndex* idx=ctx->index;int dim=idx->meta.dim,count=0;
    double start=now_ms(),norm=0;
    for(int d=0;d<dim;d++)norm+=(double)vector[d]*vector[d];
    for(int d=0;d<dim;d++)ctx->qn[d]=(float)(vector[d]/(sqrt(norm)+1e-30));
    roaring_uint32_iterator_t it;roaring_init_iterator(ctx->allowed,&it);
    while(it.has_value) {
        int nr=0;
        while(it.has_value && nr<ctx->rerank) {
            uint32_t id=it.current_value;roaring_advance_uint32_iterator(&it);
            if(anchor_live_is_deleted(idx->live,id))continue;
            if((uint64_t)id>=(uint64_t)idx->meta.n+anchor_live_count(idx->live))continue;
            anchor_trace_mark(ctx,id,0);anchor_trace_mark(ctx,id,1);
            int overridden=anchor_live_override_vector(idx->live,id,ctx->vf);
            if(overridden<0)return -1;
            if(overridden){
                stats->bytes+=(uint64_t)dim*4;stats->entries++;
                anchor_offer(ctx->fin_all,&count,top_k,id,dotf(ctx->qn,ctx->vf,dim));
            } else if((uint64_t)id<(uint64_t)idx->meta.n) {
                ctx->merge_ids[nr]=id;
                ctx->read_off[nr]=8+(uint64_t)id*idx->meta.input_dim*2;
                ctx->read_len[nr]=(uint64_t)idx->meta.input_dim*2;
                ctx->read_dst[nr]=ctx->rows_all+(size_t)nr*dim*2;
                nr++;
            } else {
                size_t len=(size_t)dim*4;uint64_t off=((uint64_t)id-idx->meta.n)*len;
                ssize_t got;
                do {got=pread(anchor_live_rows_fd(idx->live),ctx->vf,len,off);}while(got<0&&errno==EINTR);
                if(got!=(ssize_t)len)return -1;
                stats->bytes+=len;stats->entries++;
                anchor_offer(ctx->fin_all,&count,top_k,id,dotf(ctx->qn,ctx->vf,dim));
            }
        }
        if(nr) {
            if(ctx->s3url) {
                if(s3_wave(&ctx->s3,ctx->s3_base,ctx->read_off,ctx->read_len,ctx->read_dst,nr)!=nr)return -1;
            } else if(anchor_reads(&ctx->ring,idx->basefd,ctx->read_off,ctx->read_len,ctx->read_dst,nr))return -1;
            for(int i=0;i<nr;i++) {
                row_f16_padded((const uint16_t*)ctx->read_dst[i],ctx->vf,idx->meta.input_dim,dim);
                anchor_offer(ctx->fin_all,&count,top_k,ctx->merge_ids[i],dotf(ctx->qn,ctx->vf,dim));
            }
            stats->bytes+=(uint64_t)nr*idx->meta.input_dim*2;stats->entries+=nr;
        }
    }
    for(int i=0;i<top_k;i++){ids[i]=i<count?ctx->fin_all[i].id:UINT32_MAX;if(scores)scores[i]=i<count?ctx->fin_all[i].s:-INFINITY;}
    stats->rerank_ms=stats->total_ms=now_ms()-start;return count;
}
int anchor_query_search_filtered(AnchorQuery* ctx,const float* vector,int top_k,
    const uint32_t* allowed_ids,int nallowed,const char* const* keys,const int* groups,int ngroups,
    uint32_t* ids,float* scores,AnchorStats* stats) {
    if(!ctx||ctx->failed||!vector||!ids||top_k<1||top_k>ctx->rerank||nallowed< -1||
       (nallowed>0&&!allowed_ids)||ngroups<0||(ngroups&&!groups))return -1;
    int input_dim=ctx->index->meta.input_dim;
    for(int d=0;d<input_dim;d++)if(!isfinite(vector[d]))return -1;
    memcpy(ctx->input,vector,(size_t)input_dim*4);
    memset(ctx->input+input_dim,0,(size_t)(ctx->index->meta.dim-input_dim)*4);
    vector=ctx->input;
    int total=0;for(int g=0;g<ngroups;g++) {
        if(groups[g]<0||groups[g]>65536||total>65536-groups[g])return -1;
        for(int j=0;j<groups[g];j++)if(!keys||!keys[total+j])return -1;
        total+=groups[g];
    }
    AnchorStats local;if(!stats)stats=&local;memset(stats,0,sizeof(*stats));
    double call_start=now_ms();
    ctx->trace_routed=0;ctx->trace_candidates=0;
    ctx->trace_entries=0;if(ctx->trace_unique)roaring_bitmap_clear(ctx->trace_unique);
    ctx->policy_start=call_start;ctx->policy_used=0;ctx->policy_limited=0;ctx->policy_observed_gap=0;
    AnchorLive* live=ctx->index->live;
    if(anchor_live_read_lock(live))return -1;
    roaring_bitmap_t* filter=NULL;
    int rc=-1;
    if(nallowed>=0) {filter=nallowed?roaring_bitmap_of_ptr((size_t)nallowed,allowed_ids):roaring_bitmap_create();if(!filter)goto done;}
    if(ngroups) {
        roaring_bitmap_t* tags=anchor_live_filter(live,keys,groups,ngroups);
        if(!tags)goto done;
        if(filter){roaring_bitmap_and_inplace(filter,tags);roaring_bitmap_free(tags);}else filter=tags;
    }
    anchor_live_exclude_deleted(live,filter);
    ctx->allowed=filter;
    if(filter&&roaring_bitmap_get_cardinality(filter)<=4096)rc=anchor_exact_filter(ctx,vector,top_k,ids,scores,stats);
    else if(ctx->index->residual_fd>=0)rc=anchor_residual_search(ctx,vector,top_k,ids,scores,stats);
    else rc=anchor_search_impl(ctx,vector,top_k,ids,scores,stats);
    if(rc<0)ctx->failed=1;
done:
    ctx->allowed=NULL;if(filter)roaring_bitmap_free(filter);
    anchor_live_read_unlock(live);stats->total_ms=now_ms()-call_start;return rc;
}
int anchor_query_search(AnchorQuery* ctx,const float* vector,int top_k,uint32_t* ids,float* scores,AnchorStats* stats) {
    return anchor_query_search_filtered(ctx,vector,top_k,NULL,-1,NULL,NULL,0,ids,scores,stats);
}

int cmd_anchor_bench(int argc, char** argv) {
    if (argc < 8) return 1;
    int nq = atoi(argv[5]), np = atoi(argv[6]), rr = atoi(argv[7]);
    int a8 = 1, hedge = 0, threads = omp_get_max_threads();
    const char *out = NULL, *drop = NULL, *url = NULL;
    uint64_t memory = 0;
    for (int i = 8; i+1 < argc; i += 2) {
        if (!strcmp(argv[i], "--out")) out = argv[i+1];
        else if (!strcmp(argv[i], "--drop")) drop = argv[i+1];
        else if (!strcmp(argv[i], "--s3")) url = argv[i+1];
        else if (!strcmp(argv[i], "--hedge")) hedge = atoi(argv[i+1]);
        else if (!strcmp(argv[i], "--a8")) a8 = atoi(argv[i+1]);
        else if (!strcmp(argv[i], "--threads")) threads = atoi(argv[i+1]);
        else if (!strcmp(argv[i], "--memory-mb")) memory = strtoull(argv[i+1], NULL, 10)*1000000ULL;
        else return 1;
    }
    if (nq < 1 || rr < 11) return 1;
    int rc = 1;
    AnchorIndex* idx = anchor_index_open(argv[2], argv[3], a8);
    AnchorQuery* ctx = NULL;
    FILE *qf = NULL, *fo = NULL;
    float* vector = NULL;
    double* times = NULL;
    if (!idx) goto done;
    ctx = anchor_query_create(idx, np, rr, threads, memory, url, hedge);
    if (!ctx) goto done;
    qf = fopen(argv[4], "rb");
    uint32_t hdr[2];
    if (!qf || fread(hdr, 4, 2, qf) != 2 || hdr[1] != (uint32_t)idx->meta.input_dim) goto done;
    if (hdr[0] < (uint32_t)nq) nq = hdr[0];
    if (!nq) goto done;
    vector = malloc((size_t)idx->meta.dim*4);
    times = malloc((size_t)nq*5*sizeof(double));
    if (!vector || !times) goto done;
    if (out && !(fo = fopen(out, "wb"))) goto done;
    uint64_t entries = 0;
    for (int qi = 0; qi < nq; qi++) {
        if (fread(vector, 4, idx->meta.input_dim, qf) != (size_t)idx->meta.input_dim) goto done;
        if (drop && system(drop) != 0) goto done;
        uint32_t ids[11]; AnchorStats st;
        int nr = anchor_query_search(ctx, vector, 11, ids, NULL, &st);
        if (nr < 0) { fprintf(stderr, "anchor query failed: %d\n", nr); goto done; }
        if (fo && fwrite(ids, 4, 11, fo) != 11) goto done;
        double v[5] = {st.anchor_ms, st.io_ms, st.score_ms, st.rerank_ms, st.total_ms};
        for (int j = 0; j < 5; j++) times[j*nq+qi] = v[j];
        entries += st.entries;
    }
    for (int j = 0; j < 5; j++) {
        double* t = times + j*nq;
        for (int i = 1; i < nq; i++) {
            double v = t[i]; int k = i-1;
            while (k >= 0 && t[k] > v) { t[k+1] = t[k]; k--; } t[k+1] = v;
        }
        printf("%-7s p50 %7.2f ms  p99 %7.2f ms\n",
               (const char*[]){"ancres", "io", "score", "rerank", "TOTAL"}[j], t[nq/2], t[(int)(nq*.99)]);
    }
    printf("docs vus/req : %llu\n", (unsigned long long)(entries/nq));
    rc = 0;
done:
    if (qf) fclose(qf);
    if (fo && fclose(fo)) rc = 1;
    free(vector); free(times); anchor_query_close(ctx); anchor_index_close(idx);
    return rc;
}
