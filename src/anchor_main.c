#include <stdio.h>
#include <string.h>

int cmd_anchor_build(int argc,char** argv);
int cmd_anchor_bench(int argc,char** argv);

int main(int argc,char** argv) {
    if(argc>1&&!strcmp(argv[1],"abuild"))return cmd_anchor_build(argc,argv);
    if(argc>1&&!strcmp(argv[1],"abench"))return cmd_anchor_bench(argc,argv);
    if(argc>1&&!strcmp(argv[1],"--version")){puts("mangrove-engine 0.3.0-dev");return 0;}
    fprintf(stderr,"Usage: mangrove-engine abuild BASE INDEX K [options]\n"
                   "       mangrove-engine abench INDEX BASE QUERIES NQ NPROBE RERANK [options]\n"
                   "Use python -m mangrove.cli for residual conversion and service management.\n");
    return 2;
}
