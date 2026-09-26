#include <stdio.h>
#include <string.h>

int cmd_anchor_build(int argc,char** argv);
int cmd_anchor_bench(int argc,char** argv);

int main(int argc,char** argv) {
    if(argc>1&&!strcmp(argv[1],"abuild"))return cmd_anchor_build(argc,argv);
    if(argc>1&&!strcmp(argv[1],"abench"))return cmd_anchor_bench(argc,argv);
    if(argc>1&&!strcmp(argv[1],"--version")){puts("fissiondb-engine 0.3.0-dev");return 0;}
    fprintf(stderr,"Usage: fissiondb-engine abuild BASE INDEX K [options]\n"
                   "       fissiondb-engine abench INDEX BASE QUERIES NQ NPROBE RERANK [options]\n"
                   "Use python -m fissiondb.cli for residual conversion and service management.\n");
    return 2;
}
