#ifdef FISSIONDB_TESTING
/* Deterministic test barrier; absent from production builds. */
static int fission_test_gate[3]={-1,-1,0};
void anchor_test_fission_gate(int reached,int resume,int phase){fission_test_gate[0]=reached;fission_test_gate[1]=resume;fission_test_gate[2]=phase;}
static int fission_test_barrier(int phase){
    if(fission_test_gate[0]<0||fission_test_gate[2]!=phase)return 0;
    int reached=fission_test_gate[0],resume=fission_test_gate[1];fission_test_gate[0]=-1;
    char c='s';if(write(reached,&c,1)!=1)return -1;
    ssize_t n;do{n=read(resume,&c,1);}while(n<0&&errno==EINTR);
    return n!=1||c=='!'?-1:0;
}
#else
#define fission_test_barrier(phase) 0
#endif
