#include "../src/anchor.c"
int main(int argc, char** argv) {
    if (argc == 1) {
        uint8_t code[128]; int8_t q[1024];
        for (int i=0;i<128;i++) code[i]=(uint8_t)(i*37+13);
        for (int i=0;i<1024;i++) q[i]=(int8_t)((i*17)%255-127);
        for (int dim=8;dim<=1024;dim+=8) {
            int s1=0,s2=0,s4=0;
            for(int d=0;d<dim;d++) if((code[d/8]>>(d%8))&1) s1+=q[(d%8)*(dim/8)+d/8];
            if(score_tq1(code,q,dim/8,dim)!=s1)return 10;
            if(dim<=512){
                for(int i=0;i<dim/4;i++)for(int b=0;b<4;b++)s2+=((int)((code[i]>>(2*b))&3)-2)*q[b*(dim/4)+i];
                if(score_tq2(code,q,q+dim/4,q+dim/2,q+3*dim/4,dim)!=s2)return 11;
            }
            if(dim<=256){
                for(int i=0;i<dim/2;i++){int a=code[i]&15,b=code[i]>>4;if(a>7)a-=16;if(b>7)b-=16;s4+=a*q[i]+b*q[dim/2+i];}
                if(score_tq4(code,q,q+dim/2,dim)!=s4)return 12;
            }
        }
        return 0;
    }
    if (argc != 2) return 2;
    S3Ctx c = {0}; if (s3_init(&c)) return 3;
    uint64_t off = 1024, len = 16;
    uint8_t buf[16] = {0}; uint8_t* dst = buf;
    int result = s3_wave(&c, argv[1], &off, &len, &dst, 1);
    s3_close(&c);
    return result == 1 ? 0 : 1;
}
